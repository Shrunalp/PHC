"""
Spatially constrained clustering of per-cell persistence vectors: cells may only merge with
neighbours in a Delaunay triangulation of their centroids, so clusters are contiguous regions.

Contents
--------
delaunay_adjacency : function
    Symmetric sparse adjacency of the Delaunay edges between points, optionally length-capped.
connect_components : function
    Joins the connected components of an adjacency graph by their nearest spatial pairs.
spatial_agglomerative_clusters : function
    Agglomerative clustering of vectors with merges restricted to Delaunay neighbours.
"""

import os

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import Delaunay, QhullError
from sklearn.cluster import AgglomerativeClustering

from .clustering import LINKAGES, _order_by_norm

try:  # numba is optional: without it (or if its self-check fails) sklearn is used
    from .constrained_linkage import (CONSTRAINED_LINKAGES, constrained_linkage_labels,
                                      numba_backend_available)
except ImportError:
    CONSTRAINED_LINKAGES = ()

MIN_TRIANGULATION_POINTS = 3    # Qhull needs a triangle; fewer points are joined along a line
TRIANGLE_EDGES = ((0, 1), (1, 2), (0, 2))
MST_WEIGHT_OFFSET = 1.0         # added to edge lengths so zero-length edges stay in the MST
BACKEND_ENV = "PHC_LINKAGE_BACKEND"  # "auto" (default): numba when exact, or "sklearn"


def _edge_matrix(edges: np.ndarray, n: int) -> csr_matrix:

    """
    Turns a list of undirected edges into the symmetric 0/1 adjacency sklearn expects as a
    connectivity matrix, dropping repeated edges and self loops.

    Parameters
    ----------
    edges : np.ndarray of int - size (m, 2)
        (i, j) point indices of each edge, in any order, repeats allowed.

    n : int
        Number of points (graph nodes).

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix of float64 - size (n, n)
        Symmetric, 1.0 for each edge, zero diagonal.
    """

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return adjacency


def _line_edges(points: np.ndarray) -> np.ndarray:

    """
    Joins points that cannot be triangulated (fewer than three, or all on one line) into a
    path, in their order along the line, so the graph is still connected.

    Parameters
    ----------
    points : np.ndarray of float - size (m, 2)
        Distinct points.

    Returns
    -------
    edges : np.ndarray of int - size (max(m - 1, 0), 2)
        Consecutive pairs of point indices along the principal axis.
    """

    if len(points) < 2:
        edges = np.empty((0, 2), dtype=np.int64)  # nothing to join
    else:
        centred = points - points.mean(axis=0)
        direction = np.linalg.svd(centred, full_matrices=False)[2][0]  # principal axis
        order = np.argsort(centred @ direction, kind="stable")
        edges = np.column_stack([order[:-1], order[1:]])
    return edges


def _triangulation_edges(points: np.ndarray) -> np.ndarray:

    """
    Lists the Delaunay edges between distinct points, falling back to a path along the line
    when Qhull cannot triangulate them, and attaching points Qhull left out as coplanar
    (near-duplicates) to their nearest triangulation vertex.

    Parameters
    ----------
    points : np.ndarray of float - size (m, 2)
        Distinct points.

    Returns
    -------
    edges : np.ndarray of int - size (e, 2)
        (i, j) point indices of each edge, possibly with repeats; every point is covered.
    """

    if len(points) < MIN_TRIANGULATION_POINTS:
        edges = _line_edges(points)
    else:
        try:
            triangulation = Delaunay(points)
        except QhullError:
            triangulation = None  # all points on one line
        if triangulation is None:
            edges = _line_edges(points)
        else:
            simplices = triangulation.simplices
            edges = np.concatenate([simplices[:, [a, b]] for a, b in TRIANGLE_EDGES])
            if len(triangulation.coplanar):  # (point, facet, nearest vertex) per dropped point
                edges = np.concatenate([edges, triangulation.coplanar[:, [0, 2]]])
    return edges


def delaunay_adjacency(points: np.ndarray, max_edge_length: float = 0) -> csr_matrix:

    """
    Builds the neighbourhood graph that keeps spatially constrained clusters contiguous: two
    cells are adjacent when they share an edge of the Delaunay triangulation of the centroids.
    Duplicate centroids are joined to the point they coincide with, and points that cannot be
    triangulated (fewer than three, or collinear) are joined along their line.

    Parameters
    ----------
    points : np.ndarray of float - size (n, 2)
        (x, y) centroid of each cell.

    max_edge_length : float
        Edges longer than this are dropped (e.g. across gaps in the tissue), default 0 (keep
        every edge). Edges between duplicate points have length 0 and are always kept.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix of float64 - size (n, n)
        Symmetric 0/1 adjacency with a zero diagonal; connected when max_edge_length is 0.

    Raises
    ------
    ValueError
        If `points` is not of size (n, 2) or `max_edge_length` is negative.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have size (n, 2), got {points.shape}.")
    if max_edge_length < 0:
        raise ValueError(f"max_edge_length must be at least 0, got {max_edge_length}.")

    ### Triangulate the distinct points; duplicates join their representative ###
    distinct, first, inverse = np.unique(points, axis=0, return_index=True,
                                         return_inverse=True)
    inverse = inverse.reshape(-1)
    edges = first[_triangulation_edges(distinct)]
    duplicate_edges = np.column_stack([np.arange(len(points)), first[inverse]])
    edges = np.concatenate([edges.reshape(-1, 2), duplicate_edges])

    ### Drop long edges ###
    if max_edge_length > 0:
        lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        edges = edges[lengths <= max_edge_length]
    adjacency = _edge_matrix(edges, len(points))
    return adjacency


def connect_components(adjacency: csr_matrix, points: np.ndarray) -> tuple[csr_matrix, int]:

    """
    Makes a neighbourhood graph connected, so sklearn never has to patch it, by adding the
    fewest and shortest bridging edges: the minimum spanning tree of the components, where
    two components are joined by their nearest pair of points along a Delaunay edge (the
    Euclidean minimum spanning tree is a subgraph of the Delaunay triangulation, so this is
    the shortest possible set of bridges). Runs in O(n log n), fine for 10^5 points.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix - size (n, n)
        Symmetric adjacency, e.g. from `delaunay_adjacency` with a maximum edge length.

    points : np.ndarray of float - size (n, 2)
        (x, y) position of each node.

    Returns
    -------
    connection : tuple[scipy.sparse.csr_matrix, int]
        (connected, n_components): connected is a symmetric 0/1 csr_matrix of float64 - size
        (n, n) holding every edge of `adjacency` plus n_components - 1 bridging edges;
        n_components is the number of connected components of `adjacency` before bridging.

    Raises
    ------
    ValueError
        If `adjacency` is not square of the same size as `points`.
    """

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if adjacency.shape != (n, n):
        raise ValueError(f"adjacency has size {adjacency.shape} but there are {n} points.")
    n_components, component = connected_components(adjacency, directed=False)

    if n_components <= 1:
        connected = _edge_matrix(np.column_stack(adjacency.nonzero()), n)
    else:
        ### Candidate bridges: full Delaunay edges between different components ###
        candidates = delaunay_adjacency(points)
        rows, cols = candidates.nonzero()
        between = (rows < cols) & (component[rows] != component[cols])
        rows, cols = rows[between], cols[between]
        lengths = np.linalg.norm(points[rows] - points[cols], axis=1)

        ### Shortest candidate per component pair, then the MST over components ###
        pair = np.sort(np.column_stack([component[rows], component[cols]]), axis=1)
        by_length = np.argsort(lengths, kind="stable")
        _, shortest = np.unique(pair[by_length], axis=0, return_index=True)
        best = by_length[shortest]
        component_graph = coo_matrix((lengths[best] + MST_WEIGHT_OFFSET,
                                      (pair[best, 0], pair[best, 1])),
                                     shape=(n_components, n_components)).tocsr()
        tree = minimum_spanning_tree(component_graph).tocoo()
        best_of_pair = {(a, b): k for k, (a, b) in zip(best, pair[best])}
        bridges = np.array([best_of_pair[(a, b)] for a, b in zip(tree.row, tree.col)],
                           dtype=np.int64)
        edges = np.concatenate([np.column_stack(adjacency.nonzero()),
                                np.column_stack([rows[bridges], cols[bridges]])])
        connected = _edge_matrix(edges, n)
    connection = (connected, int(n_components))
    return connection


def _linkage_backend(linkage: str) -> str:

    """
    Chooses who builds the constrained merge tree: the numba kernels of
    `constrained_linkage` (the same labels as sklearn, much faster) for Ward and average
    linkage when numba is installed and its self-check against sklearn passes, otherwise
    sklearn. Setting the environment variable PHC_LINKAGE_BACKEND to "sklearn" forces sklearn.

    Parameters
    ----------
    linkage : str
        One of "ward", "average", "complete" or "single".

    Returns
    -------
    backend : str
        "numba" or "sklearn".

    Raises
    ------
    ValueError
        If PHC_LINKAGE_BACKEND is set to anything but "auto" or "sklearn".
    """

    requested = os.environ.get(BACKEND_ENV, "auto").strip().lower() or "auto"
    if requested == "sklearn":
        backend = "sklearn"
    elif requested == "auto":
        exact = linkage in CONSTRAINED_LINKAGES and numba_backend_available()
        backend = "numba" if exact else "sklearn"
    else:
        raise ValueError(f"Unknown {BACKEND_ENV} {requested!r}; expected 'auto' or 'sklearn'.")
    return backend


def spatial_agglomerative_clusters(
        vectors: np.ndarray,
        points: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "ward",
        max_edge_length: float = 0
        ) -> tuple[np.ndarray, dict]:

    """
    Groups cells whose persistence vectors are close in L2 distance into spatially contiguous
    regions: agglomerative clustering where two clusters may only merge if they touch in the
    Delaunay graph of the centroids (bridged into one component first). All vectors are
    clustered; labels are ordered like `agglomerative_clusters` (cluster 0 = lowest mean L2
    norm). With a graph that is connected before bridging, every cluster is a connected
    subgraph of it.

    Parameters
    ----------
    vectors : np.ndarray of float - size (n, d)
        One flattened persistence image or silhouette per cell.

    points : np.ndarray of float - size (n, 2)
        (x, y) centroid of each cell, in the same order as `vectors`.

    n_clusters : int
        Number of clusters, clipped to n when there are fewer cells, default 4.

    linkage : str
        One of "ward", "average", "complete" or "single", default "ward".

    max_edge_length : float
        Delaunay edges longer than this are dropped before bridging, default 0 (no limit).

    Returns
    -------
    clustering : tuple[np.ndarray, dict]
        (labels, info): labels is an np.ndarray of int - size (n,) in [0, min(n_clusters,
        n)); info is a dict with keys "adjacency" (scipy.sparse.csr_matrix - size (n, n),
        the connected graph used), "n_edges" (int, its undirected edges), "n_components"
        (int, connected components before bridging) and "backend" (str, "numba" or
        "sklearn" for whoever built the merge tree, "none" when one cluster needs no tree).

    Raises
    ------
    ValueError
        If `linkage` is not a recognised option, `n_clusters` is below 1, `vectors` and
        `points` differ in length, or the PHC_LINKAGE_BACKEND environment variable is
        neither "auto" nor "sklearn".
    """

    if linkage not in LINKAGES:
        raise ValueError(
            f"Unknown linkage {linkage!r}; expected 'ward', 'average', 'complete' or 'single'."
        )
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}.")
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) != len(points):
        raise ValueError(f"{len(vectors)} vectors but {len(points)} points.")

    ### Neighbourhood graph, made connected explicitly ###
    adjacency = delaunay_adjacency(points, max_edge_length)
    adjacency, n_components = connect_components(adjacency, points)

    n_clusters = min(n_clusters, len(vectors))
    if n_clusters == 1:
        raw_labels = np.zeros(len(vectors), dtype=int)  # sklearn needs at least 2 samples
        backend = "none"
    else:
        backend = _linkage_backend(linkage)
        if backend == "numba":  # the same labels as sklearn's, much faster
            raw_labels = constrained_linkage_labels(vectors, adjacency, n_clusters, linkage)
        else:
            raw_labels = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean",
                                                 linkage=linkage,
                                                 connectivity=adjacency).fit_predict(vectors)

    labels = _order_by_norm(vectors, raw_labels, n_clusters)
    info = {"adjacency": adjacency, "n_edges": int(adjacency.nnz // 2),
            "n_components": n_components, "backend": backend}
    clustering = (labels, info)
    return clustering
