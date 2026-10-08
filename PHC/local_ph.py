"""
Generates PHC data by computing vectorized local persistence over sliding windows of an image.

Contents
--------
PHC : class
    Slides a window across an image, or a point cloud of cells, and vectorizes the persistence
    of each window in parallel; can also centre one window on every cell (per-cell mode).
"""

from collections.abc import Callable
from itertools import chain

from gudhi.representations import PersistenceImage, Silhouette
from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
from scipy.spatial import cKDTree

from .filtrations import adj_complex, alpha_pointcloud, alphacomplex, cubicalcomplex, lower_star

PI_BANDWIDTH = 1.0     # Gaussian bandwidth of per-window persistence images (pixels)
DEGENERATE_SPAN = 1.0  # range used when every diagram point shares one value (point units)
SEARCH_PAD = 1e-9      # relative radius padding, so KD-tree rounding never drops a neighbour
QUERY_BLOCK = 100000   # cells whose KD-tree neighbour lists are held in memory at once
CELLS_PER_TASK = 256   # most cells per joblib task in `convolve_cells` (amortizes overhead)
TASKS_PER_WORKER = 4   # aim for at least this many tasks per worker, for load balancing
VECTORIZE_CHUNK = 2000  # diagrams per worker task when vectorizing; fewer stay in-process


def _chunk_diagrams(chunk_pts: list[np.ndarray], dim: int) -> list[np.ndarray]:

    """
    Computes the alpha complex persistence of a batch of point clouds in one worker task, so
    many small windows (e.g. one per cell) do not each pay joblib's per-task overhead.

    Parameters
    ----------
    chunk_pts : list of np.ndarray of float - length c, each size (k, 2)
        Points of each window in the batch.

    dim : int
        Persistent homology dimension.

    Returns
    -------
    chunk_dgms : list of np.ndarray of float - length c, each size (m, 2)
        Finite (birth, death) diagram per window, as from `alpha_pointcloud`.
    """

    chunk_dgms = [alpha_pointcloud(pts, dim=dim)[0] for pts in chunk_pts]
    return chunk_dgms


class PHC:

    """
    Computes localized persistence using Persistent Homology Convolutions, turning an image
    into a grid of vectorized persistence diagrams. Windows are independent, so `convolve`
    spreads them across worker processes. `convolve_points` does the same for a point cloud
    (e.g. cell centroids) with the alpha complex, in float coordinates, and `convolve_cells`
    centres one window on every point instead of tiling.

    Parameters
    ----------
    persistence_type : str
        Filtration used on each window. One of "lower_star", "ext_lower_star", "alpha",
        "ext_alpha", "adj_complex", "ext_adj_complex" or "cubical_complex", default "alpha".

    window_size : int | float
        Side length of the square (n, n) subwindow persistence is computed on, default 32.
        `convolve_points` and `convolve_cells` also accept a float, in the units of the points.

    stride : int | float
        Number of pixels the subwindow is translated by between windows, default 32.
        `convolve_points` also accepts a float, in the units of the points; `convolve_cells`
        ignores it.

    vectorization : str
        One of "PI" (persistence image) or "PL" (persistence silhouette), default "PI".

    vector_resolution : int
        Side length of the vectorized output for each window, default 20.

    dimension : int
        Persistent homology dimension, default 1.

    n_jobs : int
        Number of worker processes used by `convolve`, `point_diagrams` and `convolve_cells`;
        -1 uses every core and 1 runs serially in the calling process, default -1.
    """

    def __init__(
            self,
            persistence_type: str = "alpha",
            window_size: int | float = 32,
            stride: int | float = 32,
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
            silhouette = Silhouette(resolution=self.vector_resolution,
                                    weight=lambda x: x[1] - x[0])  # persistence weighted
            vectorized_pd = silhouette.fit_transform(pd)
        elif self.vectorization == "PI":
            persistence_image = PersistenceImage(
                resolution=(self.vector_resolution, self.vector_resolution),
                bandwidth=PI_BANDWIDTH)
            vectorized_pd = persistence_image.fit_transform(pd)[0]
        else:
            raise ValueError(
                f"Unknown vectorization {self.vectorization!r}; expected 'PI' or 'PL'."
            )

        return vectorized_pd

    def window_grid(self, shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:

        """
        Lists where each window of `convolve` sits in the image, so callers can map a vectorized
        window back onto the pixels it came from (for example to draw a heatmap).

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the image passed to `convolve`; only the first two entries (n, m) are used.

        Returns
        -------
        grid : list of tuple[int, int, int, int] - length ceil(n / stride) * ceil(m / stride)
            (x_cord, y_cord, window_width, window_length) per window in `convolve` order, where
            x_cord is the row and y_cord the column of the top-left corner. Windows at the
            bottom and right edges are clipped to the image.
        """

        ### Store Dimensions ###
        width = shape[0]
        length = shape[1]

        grid = [(x_cord, y_cord,
                 min(self.window_size, width - x_cord), min(self.window_size, length - y_cord))
                for x_cord in range(0, width, self.stride)
                for y_cord in range(0, length, self.stride)]
        return grid

    def convolve(
            self,
            img: np.ndarray,
            progress: Callable[[int, int], None] | None = None
            ) -> list[np.ndarray]:

        """
        Slides the window over the full image and vectorizes the persistence of every
        window, spreading the windows across `n_jobs` worker processes.

        Parameters
        ----------
        img : np.ndarray of float - size (n, m)
            Greyscale pathology slide.

        progress : Callable[[int, int], None] | None
            Called as progress(n_done, n_total) after each finished window, for progress bars
            and time estimates, default None (no reporting).

        Returns
        -------
        windows : list of np.ndarray of float - length ceil(n / stride) * ceil(m / stride)
            Vectorized local persistence for each window, in row-major window order.
        """

        # Ship only the small slices to workers rather than the full image for every task
        subimgs = [img[x_cord:x_cord+window_width, y_cord:y_cord+window_length]
                   for x_cord, y_cord, window_width, window_length in self.window_grid(img.shape)]

        # The generator yields results in input order as they finish, so window ordering is
        # preserved and progress can be reported while workers are still running
        results = Parallel(n_jobs=self.n_jobs, return_as="generator")(
            delayed(self.vectorize_window)(subimg) for subimg in subimgs
        )
        windows = []
        for vectorized_pd in results:
            windows.append(vectorized_pd)
            if progress is not None:
                progress(len(windows), len(subimgs))

        return windows

    def point_window_grid(
            self,
            shape: tuple[float, float]
            ) -> list[tuple[float, float, float, float]]:

        """
        Lays out the windows of `convolve_points` over a box of the given size, the float
        counterpart of `window_grid`, so callers can map each vector back onto the region.

        Parameters
        ----------
        shape : tuple[float, float]
            (height, width) of the box, in the units of the points (e.g. slide pixels).

        Returns
        -------
        grid : list of tuple[float, float, float, float] - length ceil(height / stride) *
                ceil(width / stride)
            (row, col, window_height, window_width) per window in row-major order, relative
            to the box's top-left corner. Windows at the bottom and right edges are clipped.
        """

        ### Store Dimensions ###
        height = float(shape[0])
        width = float(shape[1])

        def starts(extent: float) -> list[float]:

            """
            Offsets of the windows along one axis, computed as multiples of the stride so
            float strides do not accumulate rounding error.

            Parameters
            ----------
            extent : float
                Length of the box along this axis.

            Returns
            -------
            offsets : list of float - length ceil(extent / stride)
                Window start positions in [0, extent).
            """

            n_steps = int(np.ceil(extent / self.stride))
            offsets = [step * float(self.stride) for step in range(n_steps)
                       if step * float(self.stride) < extent]
            return offsets

        grid = [(row, col, min(float(self.window_size), height - row),
                 min(float(self.window_size), width - col))
                for row in starts(height) for col in starts(width)]
        return grid

    def window_points(
            self,
            points: np.ndarray,
            origin: tuple[float, float],
            grid: list[tuple[float, float, float, float]]
            ) -> tuple[list[np.ndarray], np.ndarray]:

        """
        Assigns points to the windows of a grid, so each window's persistence can be computed
        on its own points. A point can belong to several windows when stride < window_size.

        Parameters
        ----------
        points : np.ndarray of float - size (n, 2)
            (x, y) coordinates, e.g. cell centroids in slide pixels.

        origin : tuple[float, float]
            (x, y) of the grid's top-left corner, in the units of `points`.

        grid : list of tuple[float, float, float, float] - length w
            (row, col, window_height, window_width) per window, as from `point_window_grid`.

        Returns
        -------
        window_pts : list of np.ndarray of float - length w, each size (k, 2)
            Points with x0 + col <= x < x0 + col + window_width and
            y0 + row <= y < y0 + row + window_height, shifted so the window corner is (0, 0).

        n_points : np.ndarray of int - size (w,)
            Number of points k in each window.
        """

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        origin_x, origin_y = float(origin[0]), float(origin[1])

        # Sort by y once, so each window row is a contiguous slice found by binary search
        by_y = points[np.argsort(points[:, 1], kind="stable")]
        window_pts = []
        band_cache = {}  # (row, height) -> band points sorted by x, shared by a grid row
        for row, col, window_height, window_width in grid:
            band_key = (row, window_height)
            if band_key not in band_cache:
                band_cache.clear()  # rows arrive in order; keep only the current band
                top, bottom = origin_y + row, origin_y + row + window_height
                lo, hi = np.searchsorted(by_y[:, 1], [top, bottom], side="left")
                band = by_y[lo:hi]
                band_cache[band_key] = band[np.argsort(band[:, 0], kind="stable")]
            band = band_cache[band_key]
            left, right = origin_x + col, origin_x + col + window_width
            lo, hi = np.searchsorted(band[:, 0], [left, right], side="left")
            window_pts.append(band[lo:hi] - (left, origin_y + row))
        n_points = np.array([len(pts) for pts in window_pts], dtype=int)
        return window_pts, n_points

    def point_diagrams(
            self,
            window_pts: list[np.ndarray],
            progress: Callable[[int, int], None] | None = None,
            chunk_size: int = 1
            ) -> list[np.ndarray]:

        """
        Computes the alpha complex persistence of each window's points across `n_jobs` worker
        processes, keeping the diagrams so they can be vectorized on a common range.

        Parameters
        ----------
        window_pts : list of np.ndarray of float - length w, each size (k, 2)
            Points of each window, as from `window_points` or `cell_window_points`.

        progress : Callable[[int, int], None] | None
            Called as progress(n_done, n_total) after each finished batch of windows (always
            ending with n_done == n_total = w), default None.

        chunk_size : int
            Windows sent to a worker in one task; larger batches cut joblib overhead when
            windows are small and many, default 1 (one task per window).

        Returns
        -------
        diagrams : list of np.ndarray of float - length w, each size (m, 2)
            Finite (birth, death) diagram in dimension `dimension` per window, in point units.

        Raises
        ------
        ValueError
            If `chunk_size` is below 1.
        """

        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")

        # Each task receives only the points of its own windows, never the whole point cloud
        chunks = [window_pts[start:start + chunk_size]
                  for start in range(0, len(window_pts), chunk_size)]

        # Generator output keeps input order while still reporting progress as batches finish
        results = Parallel(n_jobs=self.n_jobs, return_as="generator")(
            delayed(_chunk_diagrams)(chunk_pts, self.dim) for chunk_pts in chunks
        )
        diagrams = []
        for chunk_dgms in results:
            diagrams.extend(chunk_dgms)
            if progress is not None:
                progress(len(diagrams), len(window_pts))
        return diagrams

    def vectorize_diagrams(self, diagrams: list[np.ndarray]) -> np.ndarray:

        """
        Vectorizes many persistence diagrams on one shared range, so that vectors of different
        windows are comparable (for clustering or distances). `vectorize_window` instead fits
        each window on its own range, which is fine for images but not for comparing windows.

        The persistence image uses a persistence-weighted Gaussian on (birth, persistence)
        coordinates, with range [min, max] of all points on each axis padded by the bandwidth,
        and bandwidth = (larger axis span) / vector_resolution, i.e. about one image pixel.
        The silhouette is persistence weighted and sampled on [min birth, max death] of all
        diagrams. Above VECTORIZE_CHUNK diagrams the (per-diagram, hence identical) transform
        is spread over `n_jobs` worker processes.

        Parameters
        ----------
        diagrams : list of np.ndarray of float - length w, each size (m, 2)
            Finite (birth, death) diagrams; empty diagrams are allowed.

        Returns
        -------
        vectorized_pds : np.ndarray of float - size (w, vector_resolution ** 2) for "PI" or
                (w, vector_resolution) for "PL"
            One vector per diagram; empty diagrams give a zero vector. Contains no NaN or inf.

        Raises
        ------
        ValueError
            If `vectorization` is not a recognised option.
        """

        if self.vectorization == "PI":
            n_features = self.vector_resolution ** 2
        elif self.vectorization == "PL":
            n_features = self.vector_resolution
        else:
            raise ValueError(
                f"Unknown vectorization {self.vectorization!r}; expected 'PI' or 'PL'."
            )

        diagrams = [np.asarray(dgm, dtype=float).reshape(-1, 2) for dgm in diagrams]
        non_empty = [i for i, dgm in enumerate(diagrams) if len(dgm) > 0]
        vectorized_pds = np.zeros((len(diagrams), n_features))
        if non_empty:  # all-empty input keeps the zero vectors
            fitted = [diagrams[i] for i in non_empty]
            all_points = np.concatenate(fitted)

            if self.vectorization == "PI":
                births = all_points[:, 0]
                persistences = all_points[:, 1] - all_points[:, 0]
                span = max(np.ptp(births), np.ptp(persistences))
                bandwidth = (span if span > 0 else DEGENERATE_SPAN) / self.vector_resolution
                im_range = [births.min() - bandwidth, births.max() + bandwidth,
                            persistences.min() - bandwidth, persistences.max() + bandwidth]
                vectorizer = PersistenceImage(
                    bandwidth=bandwidth, weight=lambda x: x[1],  # x = (birth, persistence)
                    resolution=[self.vector_resolution, self.vector_resolution], im_range=im_range)
            else:
                low, high = all_points[:, 0].min(), all_points[:, 1].max()
                if high <= low:
                    high = low + DEGENERATE_SPAN
                vectorizer = Silhouette(resolution=self.vector_resolution,
                                        weight=lambda x: x[1] - x[0],  # persistence
                                        sample_range=[low, high])

            vectorizer.fit(fitted)  # sets the shared range; transform is then per diagram
            chunks = [fitted[start:start + VECTORIZE_CHUNK]
                      for start in range(0, len(fitted), VECTORIZE_CHUNK)]
            if len(chunks) > 1:
                parts = Parallel(n_jobs=self.n_jobs)(
                    delayed(vectorizer.transform)(chunk) for chunk in chunks)
            else:
                parts = [vectorizer.transform(fitted)]
            vectorized_pds[non_empty] = np.concatenate(parts)

        vectorized_pds = np.nan_to_num(vectorized_pds, nan=0.0, posinf=0.0, neginf=0.0)
        return vectorized_pds

    def _require_alpha(self) -> None:

        """
        Stops point cloud methods early when a pixel filtration was chosen, since the alpha
        complex is the only filtration defined on points.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `persistence_type` is not "alpha".
        """

        if self.persistence_type != "alpha":
            raise ValueError(
                f"Unknown point cloud persistence method {self.persistence_type!r}; "
                "expected 'alpha'."
            )

    def convolve_points(
            self,
            points: np.ndarray,
            origin: tuple[float, float],
            shape: tuple[float, float],
            progress: Callable[[int, int], None] | None = None
            ) -> tuple[np.ndarray, np.ndarray]:

        """
        Persistent Homology Convolution on a point cloud: slides the window over a box,
        computes the alpha complex persistence of the points in each window and vectorizes
        all windows on a common range. Use it on cell centroids to capture how cells are
        arranged (e.g. rings of cells around glands) rather than pixel intensities.

        Parameters
        ----------
        points : np.ndarray of float - size (n, 2)
            (x, y) coordinates, e.g. cell centroids in slide pixels.

        origin : tuple[float, float]
            (x, y) of the box's top-left corner, in the units of `points`.

        shape : tuple[float, float]
            (height, width) of the box, in the units of `points`.

        progress : Callable[[int, int], None] | None
            Called as progress(n_done, n_total) after each finished window, default None.

        Returns
        -------
        windows : np.ndarray of float - size (w, d)
            Vectorized persistence per window of `point_window_grid(shape)`, in its order,
            where d = vector_resolution ** 2 for "PI" and vector_resolution for "PL".

        n_points : np.ndarray of int - size (w,)
            Number of points in each window.

        Raises
        ------
        ValueError
            If `persistence_type` is not "alpha", the only filtration defined on points.
        """

        self._require_alpha()
        grid = self.point_window_grid(shape)
        window_pts, n_points = self.window_points(points, origin, grid)
        diagrams = self.point_diagrams(window_pts, progress=progress)
        windows = self.vectorize_diagrams(diagrams)
        return windows, n_points

    def cell_window_points(self, points: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:

        """
        Gathers, for every point (e.g. a cell centroid), the points inside a square window of
        side `window_size` centred on it, so each cell's local arrangement can be measured on
        its own. Neighbours are found with a Chebyshev (max-norm) KD-tree query and then kept
        by the exact half-open rule below, so points on the window edge are handled exactly.

        Parameters
        ----------
        points : np.ndarray of float - size (n, 2)
            (x, y) coordinates, e.g. cell centroids in slide pixels.

        Returns
        -------
        cell_pts : list of np.ndarray of float - length n, each size (k_i, 2)
            For cell i at (cx, cy) with h = window_size / 2, every point (x, y) (the cell
            itself included) with cx - h <= x < cx + h and cy - h <= y < cy + h, in input
            order and shifted so the window's top-left corner (cx - h, cy - h) is (0, 0).

        n_neighbours : np.ndarray of int - size (n,)
            Number of points k_i in each cell's window (at least 1, the cell itself).
        """

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        n = len(points)
        half = float(self.window_size) / 2
        lower = points - half   # window top-left corner per cell (inclusive bound)
        upper = points + half   # window bottom-right corner per cell (exclusive bound)
        workers = self.n_jobs if self.n_jobs >= 1 else -1  # cKDTree only knows -1 for "all"

        cell_pts, n_neighbours = [], np.zeros(n, dtype=int)
        if n > 0:
            tree = cKDTree(points)
            radius = half * (1 + SEARCH_PAD)  # closed ball, a superset of the half-open box
            for block_start in range(0, n, QUERY_BLOCK):  # bounds the size of the id lists
                block = np.arange(block_start, min(block_start + QUERY_BLOCK, n))
                neighbour_ids = tree.query_ball_point(points[block], r=radius, p=np.inf,
                                                      workers=workers, return_sorted=True)

                ### Flatten (cell, neighbour) pairs and apply the exact half-open rule ###
                lengths = np.fromiter((len(ids) for ids in neighbour_ids), dtype=int,
                                      count=len(block))
                flat_ids = np.fromiter(chain.from_iterable(neighbour_ids), dtype=int,
                                       count=int(lengths.sum()))
                owner = np.repeat(block, lengths)
                candidates = points[flat_ids]
                keep = np.all((candidates >= lower[owner]) & (candidates < upper[owner]),
                              axis=1)
                relative = candidates[keep] - lower[owner[keep]]
                block_counts = np.bincount(owner[keep] - block_start, minlength=len(block))

                n_neighbours[block] = block_counts
                cell_pts.extend(np.split(relative, np.cumsum(block_counts)[:-1]))
        return cell_pts, n_neighbours

    def convolve_cells(
            self,
            points: np.ndarray,
            progress: Callable[[int, int], None] | None = None,
            chunk_size: int | None = None
            ) -> tuple[np.ndarray, np.ndarray]:

        """
        Per-cell Persistent Homology Convolution: centres a `window_size` window on every
        point, computes the alpha complex persistence of the points in it (exactly one
        computation per point) and vectorizes all of them on one common range. Use it to give
        every cell a description of how its neighbours are arranged, instead of one per tile.

        Parameters
        ----------
        points : np.ndarray of float - size (n, 2)
            (x, y) coordinates, e.g. cell centroids in slide pixels.

        progress : Callable[[int, int], None] | None
            Called as progress(n_done, n) after each finished batch of cells, ending with
            n_done == n, default None.

        chunk_size : int | None
            Cells per worker task, default None (about TASKS_PER_WORKER tasks per worker,
            at most CELLS_PER_TASK cells each).

        Returns
        -------
        vectors : np.ndarray of float - size (n, d)
            Vectorized persistence of each cell's window, in input order, where
            d = vector_resolution ** 2 for "PI" and vector_resolution for "PL"; cells with
            fewer than 3 neighbours get a zero vector.

        n_neighbours : np.ndarray of int - size (n,)
            Number of points in each cell's window, the cell itself included.

        Raises
        ------
        ValueError
            If `persistence_type` is not "alpha", the only filtration defined on points.
        """

        self._require_alpha()
        cell_pts, n_neighbours = self.cell_window_points(points)
        if chunk_size is None:
            n_tasks = effective_n_jobs(self.n_jobs) * TASKS_PER_WORKER
            chunk_size = int(np.clip(np.ceil(len(cell_pts) / n_tasks), 1, CELLS_PER_TASK))
        diagrams = self.point_diagrams(cell_pts, progress=progress, chunk_size=chunk_size)
        vectors = self.vectorize_diagrams(diagrams)
        return vectors, n_neighbours
