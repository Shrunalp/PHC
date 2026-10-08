"""
Small helpers shared by the filtrations.

Contents
--------
remove_noisy_pts : function
    Drops the (0.25, 0.5) points that pixel-level holes add to a persistence diagram.
pointcloud2D : function
    Turns the foreground pixels of an image into a 2D point cloud for the alpha complex.
diagram_in_dimension : function
    Computes (ordinary or extended) persistence of a simplex tree and returns one dimension.
"""

import numpy as np

NOISE_POINT = np.array([0.25, 0.5])  # (birth, death) of the holes created at the pixel level
EXTENDED_MIN_PERSISTENCE = 1e-5     # extended intervals shorter than this are dropped


def remove_noisy_pts(pd: np.ndarray) -> np.ndarray:

    """
    Simplifies a persistence diagram by removing the holes created at the pixel level, which
    all appear at (0.25, 0.5) and carry no information about the tissue.

    Parameters
    ----------
    pd : np.ndarray of float - size (n, 2)
        Computed persistence diagram.

    Returns
    -------
    denoised_pd : np.ndarray of float - size (m, 2)
        The diagram without its (0.25, 0.5) points, where m <= n.
    """

    mask = np.any(pd != NOISE_POINT, axis=1)
    denoised_pd = pd[mask]
    return denoised_pd


def pointcloud2D(dilated_img: np.ndarray) -> np.ndarray:

    """
    Represents the cell boundaries of a preprocessed slide as a point cloud, one point per
    foreground pixel. Only needed when the alpha complex filtration is used.

    Parameters
    ----------
    dilated_img : np.ndarray of int - size (n, m)
        Preprocessed pathology slide (i.e., thresholded and dilated).

    Returns
    -------
    pointcloud : np.ndarray of int - size (k, 2)
        (x, y) = (column, row) of every pixel brighter than 0.
    """

    y_coords, x_coords = np.where(dilated_img > 0)
    pointcloud = np.column_stack((x_coords, y_coords))
    return pointcloud


def diagram_in_dimension(
        simplex_tree,
        dim: int = 1,
        persistence_type: str | None = None
        ) -> np.ndarray:

    """
    Runs persistence on a filtered complex and keeps the denoised diagram of one dimension,
    so every filtration shares the same ordinary / extended persistence handling.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        Filtered complex; extended persistence modifies it in place.

    dim : int
        Persistent homology dimension, default 1.

    persistence_type : str | None
        None (ordinary persistence) or "Extended" (extended persistence), default None.

    Returns
    -------
    denoised_pd : np.ndarray of float - size (n, 2)
        (birth, death) intervals in dimension `dim`, without the pixel-level noise points.
        In dimension 0 the last interval (the essential class of ordinary persistence) is
        dropped.

    Raises
    ------
    ValueError
        If `persistence_type` is not None or "Extended".
    """

    if persistence_type is None:
        simplex_tree.persistence()
    elif persistence_type == "Extended":
        simplex_tree.extend_filtration()
        simplex_tree.extended_persistence(min_persistence=EXTENDED_MIN_PERSISTENCE)
    else:
        raise ValueError(
            f"Unknown persistence type {persistence_type!r}; expected None or 'Extended'."
        )

    pd = simplex_tree.persistence_intervals_in_dimension(dim)
    if dim == 0:
        pd = pd[:-1]
    denoised_pd = remove_noisy_pts(pd)
    return denoised_pd
