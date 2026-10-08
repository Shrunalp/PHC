"""
A collection of filtration based methods for computing persistence.

Contents
--------
alphacomplex : function
    Persistence diagram from the alpha complex of an image's foreground point cloud.
alpha_pointcloud : function
    Persistence diagram, in point units, from the alpha complex of a given point cloud.
lower_star : function
    Lower star filtration on a barycentric subdivision of the pixel grid.
adj_complex : function
    Filtration on the pixel adjacency graph, edges weighted by pixel magnitude.
cubicalcomplex : function
    Lower star persistence of the cubical complex built on the pixel grid.
"""

import numpy as np
import gudhi as gd
from gudhi import CubicalComplex, AlphaComplex

from .utils import diagram_in_dimension, pointcloud2D

MIN_ALPHA_POINTS = 3  # fewer points span no triangle, so the window carries no structure
ALPHA_REL_TOL = 1e-9  # intervals shorter than this fraction of their death are rounding error
CUBICAL_FIELD = 2     # homology coefficients of the cubical complex, Z/2Z
CUBICAL_MIN_PERSISTENCE = 0.01  # cubical intervals shorter than this are dropped


def alphacomplex(
        img: np.ndarray,
        alpha_value: float | None = None,
        dim: int = 1,
        persistence_type: str | None = None
        ) -> list[np.ndarray]:

    """
    Detects cell formations from the shape of the foreground pixels: builds the alpha complex
    of the point cloud of non-zero pixels, so rings of boundary pixels (cells) become
    1-dimensional features. Use it on thresholded and dilated slides.

    Parameters
    ----------
    img : np.ndarray of float - size (n, m)
        Preprocessed greyscale pathology slide; every pixel above 0 becomes a point.

    alpha_value : float | None
        Largest alpha radius kept in the complex, used to smooth out topological noise,
        default None (no limit).

    dim : int
        Persistent homology dimension, default 1.

    persistence_type : str | None
        None (ordinary persistence) or "Extended" (extended persistence), default None.

    Returns
    -------
    dgm : list of an np.ndarray of float - size (k, 2)
        Single persistence diagram in dimension `dim`, filtration values being squared alpha
        radii, without pixel-level noise points.

    Raises
    ------
    ValueError
        If `persistence_type` is not None or "Extended".
    """

    points = pointcloud2D(img)
    alpha_complex = AlphaComplex(points=points)

    if alpha_value is None:
        simplex_tree = alpha_complex.create_simplex_tree()
    else:
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=alpha_value**2)

    dgm = diagram_in_dimension(simplex_tree, dim=dim, persistence_type=persistence_type)
    return [dgm]


def alpha_pointcloud(points: np.ndarray, dim: int = 1) -> list[np.ndarray]:

    """
    Measures the shape of a point cloud, such as detected cell centroids, with the alpha
    complex: dimension 0 tracks how clusters of points merge and dimension 1 the rings of
    points (e.g. cells around a gland lumen). Unlike `alphacomplex` it takes the points
    directly and keeps every finite interval, since there is no pixel grid noise to remove.

    Parameters
    ----------
    points : np.ndarray of float - size (n, 2)
        (x, y) coordinates of the points; translating them does not change the diagram.

    dim : int
        Persistent homology dimension, default 1.

    Returns
    -------
    dgm : list of an np.ndarray of float - size (m, 2)
        Single finite (birth, death) diagram in the units of `points`, i.e. alpha radii (the
        square roots of gudhi's squared-radius filtration values). The infinite dimension 0
        interval and zero-length (rounding error) intervals are dropped, and the diagram is
        empty (size (0, 2)) for fewer than MIN_ALPHA_POINTS points.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < MIN_ALPHA_POINTS:
        dgm = np.empty((0, 2))
    else:
        # Shift to the origin so large slide coordinates do not cost floating point precision
        alpha_complex = AlphaComplex(points=points - points.min(axis=0))
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.persistence()
        squared_dgm = np.asarray(simplex_tree.persistence_intervals_in_dimension(dim),
                                 dtype=float).reshape(-1, 2)
        finite_dgm = squared_dgm[np.isfinite(squared_dgm[:, 1])]
        # Cocircular points (common in regular layouts) give zero-length intervals that
        # floating point leaves slightly positive; they carry no structure
        lengths = finite_dgm[:, 1] - finite_dgm[:, 0]
        finite_dgm = finite_dgm[lengths > ALPHA_REL_TOL * np.abs(finite_dgm[:, 1])]
        dgm = np.sqrt(np.maximum(finite_dgm, 0.0))  # squared radii -> radii (point units)
    return [dgm]


def lower_star(
        img: np.ndarray,
        dim: int = 1,
        persistence_type: str | None = None
        ) -> list[np.ndarray]:

    """
    Builds a lower star filtration on a barycentric subdivision of the pixel grid, so that
    pixel intensities drive the order in which cells appear. Use it for greyscale images
    where intensity carries the signal.

    Each pixel is split into four triangles around a centre vertex:

        TL -- TR      corners and boundary edges take the smallest value of the pixels
        |  \\/  |      they touch; the centre, its edges and the four triangles take the
        |  /\\  |      pixel's own value
        BL -- BR

    Parameters
    ----------
    img : np.ndarray of float - size (n, m)
        Greyscale pathology slide, must be depth one or zero.

    dim : int
        Persistent homology dimension, default 1.

    persistence_type : str | None
        None (ordinary persistence) or "Extended" (extended persistence), default None.

    Returns
    -------
    dgm : list of an np.ndarray of float - size (k, 2)
        Single persistence diagram in dimension `dim`, without pixel-level noise points.

    Raises
    ------
    ValueError
        If `persistence_type` is not None or "Extended".
    """

    rows, cols = img.shape
    pixel_vals = np.asarray(img, dtype=float)

    ### Vertex ids ###
    # Centre vertices: 0 to (rows*cols - 1), corner vertices follow them row by row
    centre_ids = np.arange(rows * cols).reshape(rows, cols)
    corner_ids = rows * cols + np.arange((rows + 1) * (cols + 1)).reshape(rows + 1, cols + 1)

    # Padding with +inf lets every corner and edge take a plain minimum over its neighbours
    padded = np.pad(pixel_vals, 1, constant_values=np.inf)
    corner_vals = np.minimum.reduce([padded[:-1, :-1], padded[:-1, 1:],
                                     padded[1:, :-1], padded[1:, 1:]])
    vertical_vals = np.minimum(padded[1:-1, :-1], padded[1:-1, 1:])    # left / right pixel
    horizontal_vals = np.minimum(padded[:-1, 1:-1], padded[1:, 1:-1])  # up / down pixel

    ### Corners, then boundary edges (vertical, horizontal), then pixel interiors ###
    simplex_tree = gd.SimplexTree()
    simplex_tree.insert_batch(corner_ids.reshape(1, -1), corner_vals.ravel())
    vertical_edges = np.stack([corner_ids[:-1, :].ravel(), corner_ids[1:, :].ravel()])
    simplex_tree.insert_batch(vertical_edges, vertical_vals.ravel())
    horizontal_edges = np.stack([corner_ids[:, :-1].ravel(), corner_ids[:, 1:].ravel()])
    simplex_tree.insert_batch(horizontal_edges, horizontal_vals.ravel())

    simplex_tree.insert_batch(centre_ids.reshape(1, -1), pixel_vals.ravel())
    centre = centre_ids.ravel()
    tl, tr = corner_ids[:-1, :-1].ravel(), corner_ids[:-1, 1:].ravel()
    bl, br = corner_ids[1:, :-1].ravel(), corner_ids[1:, 1:].ravel()
    for first, second in ((tl, tr), (tr, br), (br, bl), (bl, tl)):  # top, right, bottom, left
        simplex_tree.insert_batch(np.stack([centre, first, second]), pixel_vals.ravel())

    dgm = diagram_in_dimension(simplex_tree, dim=dim, persistence_type=persistence_type)
    return [dgm]


def adj_complex(
        img: np.ndarray,
        dim: int = 1,
        persistence_type: str | None = None
        ) -> list[np.ndarray]:

    """
    Filters the pixel adjacency graph by pixel magnitude: every pixel is a vertex with its
    own value, and it is joined to its 8 neighbours by edges valued at the larger of the two
    pixels. Use it to follow how bright regions connect as the threshold rises.

    Parameters
    ----------
    img : np.ndarray of float - size (n, m)
        Greyscale pathology slide, must be depth one or zero.

    dim : int
        Persistent homology dimension, default 1.

    persistence_type : str | None
        None (ordinary persistence) or "Extended" (extended persistence), default None.

    Returns
    -------
    dgm : list of an np.ndarray of float - size (k, 2)
        Single persistence diagram in dimension `dim`, without pixel-level noise points.

    Raises
    ------
    ValueError
        If `persistence_type` is not None or "Extended".
    """

    rows, cols = img.shape
    pixel_vals = np.asarray(img, dtype=float)
    ids = np.arange(rows * cols).reshape(rows, cols)

    simplex_tree = gd.SimplexTree()
    simplex_tree.insert_batch(ids.reshape(1, -1), pixel_vals.ravel())

    # (first pixel, second pixel) slices of each edge direction
    neighbours = (
        (np.s_[:-1, :], np.s_[1:, :]),     # vertical: pixel and the one below
        (np.s_[:, :-1], np.s_[:, 1:]),     # horizontal: pixel and the one to its right
        (np.s_[:-1, :-1], np.s_[1:, 1:]),  # diagonal: top-left to bottom-right
        (np.s_[:-1, 1:], np.s_[1:, :-1]),  # diagonal: top-right to bottom-left
    )
    for first, second in neighbours:
        edges = np.stack([ids[first].ravel(), ids[second].ravel()])
        edge_vals = np.maximum(pixel_vals[first], pixel_vals[second])  # brighter endpoint
        simplex_tree.insert_batch(edges, edge_vals.ravel())

    dgm = diagram_in_dimension(simplex_tree, dim=dim, persistence_type=persistence_type)
    return [dgm]


def cubicalcomplex(img: np.ndarray, dim: int = 1) -> list[np.ndarray]:

    """
    Computes lower star persistence on the cubical complex of the pixel grid, with one square
    per pixel. Faster than `lower_star` because no subdivision is needed.

    Parameters
    ----------
    img : np.ndarray of float - size (n, m)
        Greyscale pathology slide, must be depth one or zero.

    dim : int
        Persistent homology dimension, default 1.

    Returns
    -------
    dgm : list of an np.ndarray of float - size (k, 2)
        Single persistence diagram in dimension `dim`, without intervals shorter than
        CUBICAL_MIN_PERSISTENCE.
    """

    cubical_complex = CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
    cubical_complex.persistence(homology_coeff_field=CUBICAL_FIELD,
                                min_persistence=CUBICAL_MIN_PERSISTENCE)
    dgm = np.array(cubical_complex.persistence_intervals_in_dimension(dim))
    return [dgm]
