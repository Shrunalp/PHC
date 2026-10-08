"""
Compares and groups the vectorized windows produced by PHC.convolve.

Contents
--------
window_dissimilarity : function
    L2 norm of each window and its L2 distance to the mean window.
agglomerative_clusters : function
    Groups windows by L2 distance with agglomerative clustering, ordered by signal strength.
agglomerative_clusters_capped : function
    Agglomerative clustering fitted on at most a fixed number of vectors, the rest assigned
    to the nearest cluster mean; for inputs too large for O(n^2) clustering.
l2_dissimilarity : function
    Pairwise L2 distance matrix between window vectors.
classical_mds : function
    Classical (Torgerson) MDS coordinates of a dissimilarity matrix.
kruskal_stress : function
    Kruskal stress-1 of an embedding against a dissimilarity matrix.
mds_embedding : function
    Metric MDS (SMACOF) embedding of a dissimilarity matrix, started from classical MDS.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

LINKAGES = ("ward", "average", "complete", "single")
DENSE_EIGH_MAX = 500    # above this many points, classical MDS uses Lanczos (top-k only)
LANCZOS_SEED = 0        # fixed Lanczos start vector, so classical MDS is reproducible
ASSIGN_BLOCK = 100000   # vectors assigned to their nearest cluster mean at a time (memory)


def window_dissimilarity(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """
    Summarizes each window with two L2 measures: how much topological signal it carries, and
    how far it sits from a typical window of the same image. Use them as continuous heatmaps.

    Parameters
    ----------
    windows : np.ndarray of float - size (n, d)
        One flattened persistence image or silhouette per window.

    Returns
    -------
    l2_norms : np.ndarray of float - size (n,)
        L2 norm of each window vector.

    l2_to_mean : np.ndarray of float - size (n,)
        L2 distance from each window vector to the mean of all n windows.
    """

    l2_norms = np.linalg.norm(windows, axis=1)
    l2_to_mean = np.linalg.norm(windows - windows.mean(axis=0), axis=1)
    return l2_norms, l2_to_mean


def agglomerative_clusters(
        windows: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "ward"
        ) -> np.ndarray:

    """
    Groups windows whose persistence vectors are close in L2 (Euclidean) distance, so regions
    with similar local topology share a label. Labels are reordered so that cluster 0 has the
    smallest mean L2 norm, which lets a sequential colour map read as a heatmap.

    Parameters
    ----------
    windows : np.ndarray of float - size (n, d)
        One flattened persistence image or silhouette per window.

    n_clusters : int
        Number of clusters, clipped to n when there are fewer windows, default 4.

    linkage : str
        One of "ward", "average", "complete" or "single", default "ward".

    Returns
    -------
    labels : np.ndarray of int - size (n,)
        Cluster label in [0, min(n_clusters, n)) for each window, ordered by mean L2 norm.

    Raises
    ------
    ValueError
        If `linkage` is not a recognised option or `n_clusters` is below 1.
    """

    if linkage not in LINKAGES:
        raise ValueError(
            f"Unknown linkage {linkage!r}; expected 'ward', 'average', 'complete' or 'single'."
        )
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}.")

    n_clusters = min(n_clusters, len(windows))
    if n_clusters == 1:
        raw_labels = np.zeros(len(windows), dtype=int)  # sklearn needs at least 2 samples
    else:
        raw_labels = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean",
                                             linkage=linkage).fit_predict(windows)

    labels = _order_by_norm(windows, raw_labels, n_clusters)
    return labels


def _order_by_norm(windows: np.ndarray, raw_labels: np.ndarray, n_clusters: int) -> np.ndarray:

    """
    Renames cluster labels so that cluster 0 has the smallest mean L2 norm, which lets label
    order carry meaning (a sequential colour map then reads as a heatmap).

    Parameters
    ----------
    windows : np.ndarray of float - size (n, d)
        One vector per window or cell.

    raw_labels : np.ndarray of int - size (n,)
        Arbitrary cluster label in [0, n_clusters) for each vector; every cluster non-empty.

    n_clusters : int
        Number of clusters.

    Returns
    -------
    labels : np.ndarray of int - size (n,)
        The same partition, with clusters ranked by the mean L2 norm of their members.
    """

    norms = np.linalg.norm(windows, axis=1)
    order = np.argsort([norms[raw_labels == k].mean() for k in range(n_clusters)])
    rank = np.empty(n_clusters, dtype=int)
    rank[order] = np.arange(n_clusters)
    labels = rank[raw_labels]
    return labels


def agglomerative_clusters_capped(
        vectors: np.ndarray,
        n_clusters: int = 4,
        linkage: str = "ward",
        max_fitted: int = 10000,
        random_state: int = 0
        ) -> tuple[np.ndarray, bool, int]:

    """
    Agglomerative clustering that stays affordable for many vectors (e.g. one per cell):
    above `max_fitted` vectors it fits `agglomerative_clusters` on a seeded random subsample
    and gives every other vector the label of the nearest fitted cluster mean (L2). Labels
    are then ordered by mean L2 norm over all assigned vectors, as in
    `agglomerative_clusters`, which this reproduces exactly when n <= max_fitted.

    Parameters
    ----------
    vectors : np.ndarray of float - size (n, d)
        One flattened persistence image or silhouette per window or cell.

    n_clusters : int
        Number of clusters, clipped to the number of fitted vectors, default 4.

    linkage : str
        One of "ward", "average", "complete" or "single", default "ward".

    max_fitted : int
        Most vectors the agglomerative clustering is fitted on, default 10000.

    random_state : int
        Seed of numpy's default_rng used to draw the subsample, default 0.

    Returns
    -------
    clustering : tuple[np.ndarray, bool, int]
        (labels, subsampled, n_fitted): labels is an np.ndarray of int - size (n,) in
        [0, k) with k = min(n_clusters, n_fitted), cluster 0 having the lowest mean L2 norm;
        subsampled is True when a subsample was fitted; n_fitted = min(n, max_fitted).

    Raises
    ------
    ValueError
        If `linkage` is not a recognised option, `n_clusters` or `max_fitted` is below 1.
    """

    if max_fitted < 1:
        raise ValueError(f"max_fitted must be at least 1, got {max_fitted}.")
    vectors = np.asarray(vectors, dtype=float)
    n = len(vectors)
    subsampled = n > max_fitted
    if not subsampled:
        labels = agglomerative_clusters(vectors, n_clusters=n_clusters, linkage=linkage)
        n_fitted = n
    else:
        fitted = np.sort(np.random.default_rng(random_state).choice(n, max_fitted,
                                                                    replace=False))
        fitted_labels = agglomerative_clusters(vectors[fitted], n_clusters=n_clusters,
                                               linkage=linkage)
        n_found = int(fitted_labels.max()) + 1
        means = np.stack([vectors[fitted][fitted_labels == k].mean(axis=0)
                          for k in range(n_found)])

        ### Nearest cluster mean for every vector, in blocks to bound memory ###
        raw_labels = np.empty(n, dtype=int)
        for start in range(0, n, ASSIGN_BLOCK):
            block = vectors[start:start + ASSIGN_BLOCK]
            raw_labels[start:start + ASSIGN_BLOCK] = np.argmin(
                pairwise_distances(block, means, metric="euclidean"), axis=1)
        raw_labels[fitted] = fitted_labels  # fitted vectors keep their agglomerative label

        labels = _order_by_norm(vectors, raw_labels, n_found)
        n_fitted = max_fitted
    clustering = (labels, subsampled, n_fitted)
    return clustering


def l2_dissimilarity(windows: np.ndarray) -> np.ndarray:

    """
    Measures how different every pair of windows is, using the same L2 (Euclidean) distance
    the clustering uses, so the matrix can be embedded with MDS to show the cluster layout.

    Parameters
    ----------
    windows : np.ndarray of float - size (n, d)
        One flattened persistence image or silhouette per window.

    Returns
    -------
    dissimilarity : np.ndarray of float64 - size (n, n)
        Symmetric matrix of L2 distances between windows, with an exact zero diagonal.
    """

    dissimilarity = pairwise_distances(np.asarray(windows, dtype=np.float64),
                                       metric="euclidean")
    dissimilarity = (dissimilarity + dissimilarity.T) / 2  # remove rounding asymmetry
    np.fill_diagonal(dissimilarity, 0.0)
    return dissimilarity


def classical_mds(dissimilarity: np.ndarray, n_components: int = 2) -> np.ndarray:

    """
    Places points so their Euclidean distances best match `dissimilarity` in the
    least-squares sense of Torgerson scaling (double-centred squared distances, top
    eigenvectors, found with Lanczos for large n). Deterministic and fast; use it to start
    metric MDS.

    Parameters
    ----------
    dissimilarity : np.ndarray of float - size (n, n)
        Symmetric dissimilarity matrix with a zero diagonal.

    n_components : int
        Embedding dimension, default 2.

    Returns
    -------
    coords : np.ndarray of float64 - size (n, n_components)
        Classical MDS coordinates; axes with a non-positive eigenvalue (fewer than
        n_components real dimensions, e.g. n <= n_components) are set to 0.
    """

    n = len(dissimilarity)
    sq = np.asarray(dissimilarity, dtype=np.float64) ** 2

    # B = -1/2 J D^2 J with J = I - 11^T / n, without forming J
    gram = -0.5 * (sq - sq.mean(axis=0) - sq.mean(axis=1)[:, None] + sq.mean())
    if n > DENSE_EIGH_MAX and n_components < n - 1:
        start = np.random.default_rng(LANCZOS_SEED).standard_normal(n)
        eigvals, eigvecs = eigsh(gram, k=n_components, which="LA", v0=start)
    else:
        eigvals, eigvecs = np.linalg.eigh(gram)  # all n, fast for small n
    top = np.argsort(eigvals)[::-1][:n_components]
    scale = np.sqrt(np.clip(eigvals[top], 0.0, None))
    coords = np.zeros((n, n_components))
    coords[:, :len(top)] = eigvecs[:, top] * scale

    # Fix each axis's sign (largest-magnitude entry positive) so results are reproducible
    for axis in range(len(top)):
        if coords[np.argmax(np.abs(coords[:, axis])), axis] < 0:
            coords[:, axis] *= -1
    return coords


def kruskal_stress(dissimilarity: np.ndarray, coords: np.ndarray) -> float:

    """
    Scores how faithfully an embedding preserves the dissimilarities, independent of their
    scale: 0 is a perfect fit, below about 0.1 is good and above 0.2 is poor.
    stress-1 = sqrt( sum_{i<j} (d_ij - D_ij)^2 / sum_{i<j} D_ij^2 ), where D_ij is the input
    dissimilarity and d_ij the Euclidean distance between embedded points i and j.

    Parameters
    ----------
    dissimilarity : np.ndarray of float - size (n, n)
        Symmetric dissimilarity matrix D with a zero diagonal.

    coords : np.ndarray of float - size (n, k)
        Embedded points.

    Returns
    -------
    stress : float
        Kruskal stress-1; 0.0 when every dissimilarity is 0 (nothing to preserve).
    """

    # Both matrices are symmetric with zero diagonals, so full-matrix sums are twice the
    # i<j sums and the factor cancels; this avoids an (n^2 / 2)-long index array
    target = np.asarray(dissimilarity, dtype=np.float64)
    embedded = pairwise_distances(coords, metric="euclidean")
    np.fill_diagonal(embedded, 0.0)
    denominator = np.sum(target ** 2)
    stress = float(np.sqrt(np.sum((embedded - target) ** 2) / denominator)) \
        if denominator > 0 else 0.0
    return stress


def mds_embedding(
        dissimilarity: np.ndarray,
        n_components: int = 2,
        max_iter: int = 300,
        eps: float = 1e-6,
        random_state: int = 0
        ) -> tuple[np.ndarray, float]:

    """
    Projects windows to a few dimensions so that their L2 distances are kept as well as
    possible, for plotting how clusters relate. Runs metric MDS (SMACOF) once, started from
    the classical MDS solution, so the result is deterministic.

    Parameters
    ----------
    dissimilarity : np.ndarray of float - size (n, n)
        Symmetric dissimilarity matrix with a zero diagonal, e.g. from `l2_dissimilarity`.

    n_components : int
        Embedding dimension, default 2.

    max_iter : int
        Most SMACOF iterations, default 300.

    eps : float
        SMACOF convergence tolerance on the relative stress change, default 1e-6.

    random_state : int
        Seed passed to sklearn (unused with an explicit start, kept for reproducibility),
        default 0.

    Returns
    -------
    embedding : tuple[np.ndarray, float]
        (coords, stress): coords is an np.ndarray of float64 - size (n, n_components),
        centred at the origin; stress is its Kruskal stress-1 (see `kruskal_stress`).

    Raises
    ------
    ValueError
        If `dissimilarity` is not square or `n_components` is below 1.
    """

    dissimilarity = np.asarray(dissimilarity, dtype=np.float64)
    if dissimilarity.ndim != 2 or dissimilarity.shape[0] != dissimilarity.shape[1]:
        raise ValueError(f"dissimilarity must be a square matrix, got {dissimilarity.shape}.")
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}.")

    n = len(dissimilarity)
    init = classical_mds(dissimilarity, n_components)
    if n < 2 or not np.any(dissimilarity > 0):
        coords = init  # one point, or all windows identical: nothing for SMACOF to refine
    else:
        model = MDS(n_components=n_components, metric=True, n_init=1, max_iter=max_iter,
                    eps=eps, random_state=random_state, dissimilarity="precomputed",
                    normalized_stress=False)
        coords = model.fit_transform(dissimilarity, init=init)
        coords = coords - coords.mean(axis=0)
    embedding = (coords, kruskal_stress(dissimilarity, coords))
    return embedding
