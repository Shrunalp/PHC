"""
Reads cell centroids from QuPath GeoJSON exports, so detected cells can be used as a point cloud.

Contents
--------
cell_centroids : function
    Centroid of every usable cell in a parsed GeoJSON FeatureCollection, Feature or list.
read_cell_centroids : function
    Loads a GeoJSON file written by QuPath and returns its cell centroids.
feature_centroids : function
    Centroid (or NaN) and id of every feature, by feature index, for per-cell results.
read_cell_features : function
    Loads a GeoJSON file written by QuPath and returns every feature's centroid and id.
"""

import json

import numpy as np

CHUNK_FEATURES = 100000      # features converted to one NumPy array at a time (bounds memory)
POLYGON_TYPES = ("Polygon", "MultiPolygon")
POINT_TYPES = ("Point", "MultiPoint")
GEOMETRY_KEYS = ("nucleusGeometry", "geometry")  # preferred first: nucleus, then whole cell


def _polygon_rings(geometry: dict) -> list[tuple[list, bool]]:

    """
    Splits a Polygon or MultiPolygon into closed rings tagged as exterior or hole, so their
    areas can be added or subtracted when the centroid is computed.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry with keys "type" ("Polygon" or "MultiPolygon") and "coordinates".

    Returns
    -------
    rings : list of tuple[list, bool]
        (closed ring as a list of [x, y], is_hole) for every ring with at least one vertex;
        the first ring of each polygon is its exterior and the rest are holes.
    """

    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    else:
        polygons = geometry["coordinates"]

    rings = []
    for polygon in polygons:
        for ring_id, ring in enumerate(polygon):
            if len(ring) == 0:
                continue
            if ring[0] != ring[-1]:
                ring = list(ring) + [ring[0]]  # GeoJSON rings should be closed; close if not
            rings.append((ring, ring_id > 0))
    return rings


def _usable_geometry(feature: dict) -> dict | None:

    """
    Picks the geometry that best represents a cell's position: the nucleus when QuPath
    stored one, otherwise the cell or detection outline.

    Parameters
    ----------
    feature : dict
        GeoJSON Feature, optionally with a top-level "nucleusGeometry" member.

    Returns
    -------
    geometry : dict | None
        The first of "nucleusGeometry" or "geometry" that is a non-empty Point, MultiPoint,
        Polygon or MultiPolygon, or None when neither is usable.
    """

    geometry = None
    for key in GEOMETRY_KEYS:
        candidate = feature.get(key)
        if (isinstance(candidate, dict)
                and candidate.get("type") in POLYGON_TYPES + POINT_TYPES
                and candidate.get("coordinates")):
            geometry = candidate
            break
    return geometry


def _polygon_centroids(chunk_rings: list[list[tuple[list, bool]]]) -> np.ndarray:

    """
    Computes area centroids of many polygons at once with the shoelace formula, vectorized
    over every vertex of the chunk. Holes subtract their area, and polygons with no area
    (e.g. collinear vertices) fall back to the mean of their exterior vertices.

    Parameters
    ----------
    chunk_rings : list of list of tuple[list, bool] - length k
        Rings of each polygon, as returned by `_polygon_rings`; each list must be non-empty.

    Returns
    -------
    centroids : np.ndarray of float - size (k, 2)
        (x, y) area centroid of each polygon.
    """

    ### Flatten every ring of the chunk into one vertex array ###
    vertices, ring_lengths, ring_owner, ring_sign = [], [], [], []
    for polygon_id, rings in enumerate(chunk_rings):
        for ring, is_hole in rings:
            vertices.extend(ring)
            ring_lengths.append(len(ring))
            ring_owner.append(polygon_id)
            ring_sign.append(-1.0 if is_hole else 1.0)
    vertices = np.asarray(vertices, dtype=float)[:, :2]  # drop any z coordinate
    ring_lengths = np.asarray(ring_lengths)
    ring_owner = np.asarray(ring_owner)
    ring_sign = np.asarray(ring_sign)
    ring_starts = np.concatenate(([0], np.cumsum(ring_lengths)[:-1]))
    ring_ends = ring_starts + ring_lengths - 1

    ### Shoelace terms for consecutive vertices, zeroed where one ring meets the next ###
    #   vertex i pairs with i+1; the last vertex of a ring (its closing copy) has no partner
    x_cur, y_cur = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x_cur, -1), np.roll(y_cur, -1)
    cross = x_cur * y_next - x_next * y_cur
    cross[ring_ends] = 0.0
    twice_area = np.add.reduceat(cross, ring_starts)  # signed, orientation dependent
    moment_x = np.add.reduceat((x_cur + x_next) * cross, ring_starts)
    moment_y = np.add.reduceat((y_cur + y_next) * cross, ring_starts)

    # A ring's centroid does not depend on its orientation; only its area sign does, so the
    # sign is set from the exterior/hole role instead of trusting the winding order
    has_area = twice_area != 0
    safe_area = np.where(has_area, twice_area, 1.0)
    ring_cx = np.where(has_area, moment_x / (3.0 * safe_area), 0.0)
    ring_cy = np.where(has_area, moment_y / (3.0 * safe_area), 0.0)
    ring_weight = ring_sign * np.abs(twice_area) / 2.0

    n_polygons = len(chunk_rings)
    total_area = np.bincount(ring_owner, weights=ring_weight, minlength=n_polygons)
    sum_x = np.bincount(ring_owner, weights=ring_weight * ring_cx, minlength=n_polygons)
    sum_y = np.bincount(ring_owner, weights=ring_weight * ring_cy, minlength=n_polygons)

    ### Fallback for degenerate polygons: mean of exterior vertices (closing copy excluded) ###
    exterior = ring_sign > 0
    open_lengths = np.maximum(ring_lengths - 1, 1)
    is_closed = ring_lengths > 1
    vertex_x = np.add.reduceat(x_cur, ring_starts) - np.where(is_closed, x_cur[ring_ends], 0)
    vertex_y = np.add.reduceat(y_cur, ring_starts) - np.where(is_closed, y_cur[ring_ends], 0)
    n_vertices = np.bincount(ring_owner, weights=exterior * open_lengths, minlength=n_polygons)
    mean_x = np.bincount(ring_owner, weights=exterior * vertex_x, minlength=n_polygons)
    mean_y = np.bincount(ring_owner, weights=exterior * vertex_y, minlength=n_polygons)
    n_vertices = np.maximum(n_vertices, 1)

    has_total_area = total_area > 0
    safe_total = np.where(has_total_area, total_area, 1.0)
    centroids = np.column_stack((
        np.where(has_total_area, sum_x / safe_total, mean_x / n_vertices),
        np.where(has_total_area, sum_y / safe_total, mean_y / n_vertices),
    ))
    return centroids


def _features(geojson: dict | list) -> list:

    """
    Unwraps the supported GeoJSON containers into one list of features, so every reader
    handles FeatureCollections, single Features and plain lists the same way.

    Parameters
    ----------
    geojson : dict | list
        Parsed GeoJSON: a FeatureCollection, a single Feature, or a list of Features.

    Returns
    -------
    features : list
        The features, in file order (entries need not be valid Features).

    Raises
    ------
    ValueError
        If `geojson` is not a FeatureCollection, Feature or list of Features.
    """

    if isinstance(geojson, list):
        features = geojson
    elif isinstance(geojson, dict) and geojson.get("type") == "FeatureCollection":
        features = geojson.get("features") or []
    elif isinstance(geojson, dict) and geojson.get("type") == "Feature":
        features = [geojson]
    else:
        raise ValueError("Unknown GeoJSON content; expected a 'FeatureCollection', a 'Feature' "
                         "or a list of Features.")
    return features


def _all_centroids(features: list) -> tuple[np.ndarray, np.ndarray]:

    """
    Computes the centroid of every feature at its own index, marking the ones without a
    usable geometry, so callers can either drop them or report them per feature.

    Parameters
    ----------
    features : list - length N
        GeoJSON Features (non-dict entries are treated as unusable).

    Returns
    -------
    centroids : np.ndarray of float - size (N, 2)
        (x, y) centroid of each feature; NaN rows for unusable features.

    usable : np.ndarray of bool - size (N,)
        True where the feature had a usable geometry.
    """

    centroids = np.full((len(features), 2), np.nan)
    usable = np.zeros(len(features), dtype=bool)
    for chunk_start in range(0, len(features), CHUNK_FEATURES):
        chunk_points, chunk_rings = [], []  # input order is kept within each geometry kind
        order = []                          # (feature index, is_polygon, index within kind)
        for offset, feature in enumerate(features[chunk_start:chunk_start + CHUNK_FEATURES]):
            geometry = _usable_geometry(feature) if isinstance(feature, dict) else None
            if geometry is None:
                continue
            if geometry["type"] in POLYGON_TYPES:
                rings = _polygon_rings(geometry)
                if rings:
                    order.append((chunk_start + offset, True, len(chunk_rings)))
                    chunk_rings.append(rings)
            elif geometry["type"] == "Point":
                order.append((chunk_start + offset, False, len(chunk_points)))
                chunk_points.append(geometry["coordinates"][:2])
            else:  # MultiPoint: mean of its points
                order.append((chunk_start + offset, False, len(chunk_points)))
                chunk_points.append(np.asarray(geometry["coordinates"], float)[:, :2].mean(0))

        polygon_centroids = (_polygon_centroids(chunk_rings) if chunk_rings
                             else np.empty((0, 2)))
        point_centroids = np.asarray(chunk_points, dtype=float).reshape(-1, 2)
        for feature_id, is_polygon, i in order:
            centroids[feature_id] = polygon_centroids[i] if is_polygon else point_centroids[i]
            usable[feature_id] = True
    return centroids, usable


def cell_centroids(geojson: dict | list) -> np.ndarray:

    """
    Turns QuPath detections into a 2D point cloud of cell positions, the input for alpha
    complex persistence on cells. Each cell is placed at the area centroid of its nucleus
    when present, otherwise of its outline; Points are used as they are.

    Parameters
    ----------
    geojson : dict | list
        Parsed GeoJSON: a FeatureCollection, a single Feature, or a list of Features.
        Coordinates are taken as (x, y) slide pixels.

    Returns
    -------
    centroids : np.ndarray of float - size (n, 2)
        (x, y) centroid of each feature with a usable geometry, in input order; features
        without one (missing, empty, or a line) are skipped.

    Raises
    ------
    ValueError
        If `geojson` is not a FeatureCollection, Feature or list of Features.
    """

    all_centroids, usable = _all_centroids(_features(geojson))
    centroids = all_centroids[usable]
    return centroids


def feature_centroids(geojson: dict | list) -> tuple[np.ndarray, list[str | None]]:

    """
    Gives every feature its centroid and id at its own index, so per-cell results can be
    written back to exactly the objects QuPath exported, in the same order. Centroids follow
    the same rules as `cell_centroids`.

    Parameters
    ----------
    geojson : dict | list
        Parsed GeoJSON: a FeatureCollection, a single Feature, or a list of Features.

    Returns
    -------
    centroids : np.ndarray of float - size (N, 2)
        (x, y) centroid of each of the N features; NaN rows for features without a usable
        geometry.

    ids : list of str | None - length N
        The feature's "id" member as a string, or None when it has none.

    Raises
    ------
    ValueError
        If `geojson` is not a FeatureCollection, Feature or list of Features.
    """

    features = _features(geojson)
    centroids, _ = _all_centroids(features)
    ids = [str(feature["id"]) if isinstance(feature, dict) and feature.get("id") is not None
           else None for feature in features]
    return centroids, ids


def _load_geojson(path: str) -> dict | list:

    """
    Reads a GeoJSON file written by QuPath, shared by the file-based readers below.

    Parameters
    ----------
    path : str
        Path to a GeoJSON file (FeatureCollection, Feature or list of Features).

    Returns
    -------
    geojson : dict | list
        The parsed JSON content.
    """

    with open(path) as geojson_file:
        geojson = json.load(geojson_file)
    return geojson


def read_cell_centroids(path: str) -> np.ndarray:

    """
    Loads the detections QuPath exported with PathIO.exportObjectsAsGeoJSON and returns the
    cell positions, ready for `PHC.convolve_points`.

    Parameters
    ----------
    path : str
        Path to a GeoJSON file (FeatureCollection, Feature or list of Features).

    Returns
    -------
    centroids : np.ndarray of float - size (n, 2)
        (x, y) centroid of each usable cell, in slide pixels.
    """

    geojson = _load_geojson(path)
    centroids = cell_centroids(geojson)
    return centroids


def read_cell_features(path: str) -> tuple[np.ndarray, list[str | None]]:

    """
    Loads the detections QuPath exported with PathIO.exportObjectsAsGeoJSON and returns the
    centroid and id of every feature by index, for per-cell PHC results.

    Parameters
    ----------
    path : str
        Path to a GeoJSON file (FeatureCollection, Feature or list of Features).

    Returns
    -------
    centroids : np.ndarray of float - size (N, 2)
        (x, y) centroid of each feature in slide pixels; NaN rows where unusable.

    ids : list of str | None - length N
        Each feature's "id", or None.
    """

    geojson = _load_geojson(path)
    centroids, ids = feature_centroids(geojson)
    return centroids, ids
