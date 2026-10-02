"""
Generates PHC data by computing vectorized local persistence over sliding windows of an image.

Contents
--------
PHC : class
    Slides a window across an image and vectorizes the persistence of each window in parallel.
"""

from gudhi.representations import Silhouette
from gudhi.representations import PersistenceImage
from joblib import Parallel, delayed
import numpy as np

from .filtrations import adj_complex, alphacomplex, cubicalcomplex, lower_star


class PHC:

    """
    Computes localized persistence using Persistent Homology Convolutions, turning an image
    into a grid of vectorized persistence diagrams. Windows are independent, so `convolve`
    spreads them across worker processes.

    Parameters
    ----------
    persistence_type : str
        Filtration used on each window. One of "lower_star", "ext_lower_star", "alpha",
        "ext_alpha", "adj_complex", "ext_adj_complex" or "cubical_complex", default "alpha".

    window_size : int
        Side length of the square (n, n) subwindow persistence is computed on, default 32.

    stride : int
        Number of pixels the subwindow is translated by between windows, default 32.

    vectorization : str
        One of "PI" (persistence image) or "PL" (persistence silhouette), default "PI".

    vector_resolution : int
        Side length of the vectorized output for each window, default 20.

    dimension : int
        Persistent homology dimension, default 1.

    n_jobs : int
        Number of worker processes used by `convolve`; -1 uses every core and 1 runs
        serially in the calling process, default -1.
    """

    def __init__(
            self,
            persistence_type: str = "alpha",
            window_size: int = 32,
            stride: int = 32,
            vectorization: str = "PI",
            vector_resolution: int = 20,
            dimension: int = 1,
            n_jobs: int = -1
            ):
        self.persistence_type = persistence_type
        self.window_size = window_size
        self.stride = stride
        self.vectorization = vectorization
        self.vector_resolution = vector_resolution
        self.dim = dimension
        self.n_jobs = n_jobs

    def vectorize_window(self, subimg: np.ndarray) -> np.ndarray:

        """
        Computes and vectorizes the persistence of a single subwindow. It only sees its own
        window, which is what lets `convolve` run windows in separate processes.

        Parameters
        ----------
        subimg : np.ndarray of float - size (w, l)
            Subwindow of a greyscale pathology slide, where w, l <= self.window_size.

        Returns
        -------
        vectorized_pd : np.ndarray of float - size (self.vector_resolution ** 2,) or
                (1, self.vector_resolution)
            Persistence image (flattened) or silhouette of the window's persistence diagram.

        Raises
        ------
        ValueError
            If `persistence_type` or `vectorization` is not a recognised option.
        """

        if self.persistence_type == "lower_star":
            pd = lower_star(subimg, dim=self.dim)
        elif self.persistence_type == "ext_lower_star":
            pd = lower_star(subimg, dim=self.dim, persistence_type="Extended")
        elif self.persistence_type == "alpha":
            pd = alphacomplex(subimg, dim=self.dim)
        elif self.persistence_type == "ext_alpha":
            pd = alphacomplex(subimg, dim=self.dim, persistence_type="Extended")
        elif self.persistence_type == "adj_complex":
            pd = adj_complex(subimg, dim=self.dim)  # condition image before inputting
        elif self.persistence_type == "ext_adj_complex":
            pd = adj_complex(subimg, dim=self.dim, persistence_type="Extended")
        elif self.persistence_type == "cubical_complex":
            pd = cubicalcomplex(subimg, dim=self.dim)
        else:
            raise ValueError(
                f"Unknown persistence method {self.persistence_type!r}; expected 'lower_star', "
                "'ext_lower_star', 'alpha', 'ext_alpha', 'adj_complex', 'ext_adj_complex' "
                "or 'cubical_complex'."
            )

        if self.vectorization == "PL":
            SH = Silhouette(resolution=self.vector_resolution,
                            weight=lambda x: np.power(x[1]-x[0], 1))  # Initialize vectorization
            vectorized_pd = SH.fit_transform(pd)
        elif self.vectorization == "PI":
            PI = PersistenceImage(resolution=(self.vector_resolution, self.vector_resolution),
                                  bandwidth=1.0)  # Initialize vectorization
            PI.fit(pd)
            vectorized_pd = PI.transform(pd)[0]
        else:
            raise ValueError(
                f"Unknown vectorization {self.vectorization!r}; expected 'PI' or 'PL'."
            )

        return vectorized_pd

    def process_window(
            self,
            img: np.ndarray,
            x_cord: int,
            y_cord: int,
            window_width: int,
            window_length: int
            ) -> np.ndarray:

        """
        Cuts one subwindow out of the full image and vectorizes its persistence. Use it to
        inspect a single window; `convolve` handles the whole image.

        Parameters
        ----------
        img : np.ndarray of float - size (n, m)
            Greyscale pathology slide.

        x_cord : int
            Row index of the subwindow's top-left corner.

        y_cord : int
            Column index of the subwindow's top-left corner.

        window_width : int
            Number of rows in the subwindow, starting at x_cord.

        window_length : int
            Number of columns in the subwindow, starting at y_cord.

        Returns
        -------
        vectorized_pd : np.ndarray of float - size (self.vector_resolution ** 2,) or
                (1, self.vector_resolution)
            Vectorized representation of the subwindow's persistent homology.
        """

        subimg = img[x_cord:x_cord+window_width, y_cord:y_cord+window_length]
        vectorized_pd = self.vectorize_window(subimg)
        return vectorized_pd

    def convolve(self, img: np.ndarray) -> list[np.ndarray]:

        """
        Slides the window over the full image and vectorizes the persistence of every
        window, spreading the windows across `n_jobs` worker processes.

        Parameters
        ----------
        img : np.ndarray of float - size (n, m)
            Greyscale pathology slide.

        Returns
        -------
        windows : list of np.ndarray of float - length ceil(n / stride) * ceil(m / stride)
            Vectorized local persistence for each window, in row-major window order.
        """

        ### Store Dimensions ###
        width = img.shape[0]
        length = img.shape[1]
        window_length = self.window_size
        window_width = self.window_size

        # Ship only the small slices to workers rather than the full image for every task
        subimgs = [img[x_cord:x_cord+window_width, y_cord:y_cord+window_length]
                   for x_cord in range(0, width, self.stride)
                   for y_cord in range(0, length, self.stride)]

        # Parallel returns results in input order, so window ordering is preserved
        windows = Parallel(n_jobs=self.n_jobs)(
            delayed(self.vectorize_window)(subimg) for subimg in subimgs
        )

        return windows
