"""
Runs PHC on the cells detected inside a QuPath annotation: alpha complex persistence of the
cell centroids, so regions are compared by how their cells are arranged. --mode windows
(default) tiles the annotation with sliding windows; --mode cells centres one window on
every cell and computes its local persistence. Writes per-window or per-cell results (L2
measures, agglomerative cluster labels and 2D / 3D metric MDS coordinates of the L2
dissimilarity matrix) for QuPath to draw.

Inputs
------
--cells : GeoJSON FeatureCollection
    Detections exported by QuPath, one Point feature per cell at its nucleus centroid (or its
    own centroid without a nucleus). Polygon features are also accepted: a cell then sits at
    the area centroid of its "nucleusGeometry" when present, else of its "geometry".
    Coordinates are full-resolution slide pixels.

--mask : PNG, 8-bit - size (ceil(H / D), ceil(W / D))
    255 inside the annotation and 0 outside; mask pixel (i, j) covers slide pixels
    x in [X + j*D, X + (j+1)*D) and y in [Y + i*D, Y + (i+1)*D), with D = --mask-downsample.

--origin-x, --origin-y, --width, --height : float
    Annotation bounding box (X, Y, W, H) in slide pixels; windows tile it from (X, Y).

--mode : str
    "windows" (tiled windows, default) or "cells" (one --window-size window centred on each
    cell; --stride is ignored).

--spatial : int
    1 to cluster cells (--mode cells only) with merges restricted to neighbours in the
    Delaunay triangulation of the clustered cells' centroids, so clusters are contiguous
    regions; all clustered cells are fitted (no subsample, --max-clustered does not apply).
    0 (default) keeps the unconstrained clustering.

--max-edge-length : float
    With --spatial 1, Delaunay edges longer than this (slide pixels) are dropped, default 0
    (no limit); the graph's components are then bridged by their nearest centroid pairs.

Outputs
-------
stdout : text
    Progress for QuPath, one line each: "PHC_STAGE <name> [<count>]" when a stage starts
    (centroids, persistence <n_windows or n_cells>, clustering <n_clustered>,
    mds <n_embedded>) and "PHC_PROGRESS <done> <total>" as windows or cells finish;
    "Warning: ..." lines when the optional plots cannot be written.

--output : JSON (--mode windows)
    Keys "mode" ("windows"), "windows" (list of dicts with keys "row", "col", "height",
    "width" in slide pixels relative to (X, Y), "coverage", "n_cells", "l2_norm",
    "l2_to_mean", "cluster"),
    "n_clusters", "n_windows_clustered", "n_cells" and "elapsed_s". "cluster" is -1 for
    windows with coverage below --min-coverage or fewer than --min-cells cells. With --mds 1
    each window also has "mds2" ([x, y]) and "mds3" ([x, y, z]), null for windows that were
    not embedded (excluded, or outside the --mds-max-windows subsample), and the top level
    has "mds" (dict with keys "n_embedded", "subsampled", "stress_2d", "stress_3d", the
    stresses being Kruskal stress-1), or null when MDS is off or fewer than 2 windows remain.

--output : JSON (--mode cells)
    Keys "mode" ("cells"), "cells" (one dict per GeoJSON feature, by feature index, with keys
    "index", "id", "x", "y" (centroid in slide pixels, null when the feature has none),
    "coverage", "n_cells" (neighbours in the cell's window, itself included), "l2_norm",
    "l2_to_mean", "cluster", "mds2", "mds3"), "n_clusters", "n_cells" (features with a
    centroid), "n_cells_clustered", "n_skipped" (features without a centroid), "clustering"
    (dict with keys "subsampled", "n_fitted", "spatial", "n_edges" (undirected adjacency
    edges used), "n_components" (graph components before bridging), "max_edge_length" and
    "backend" ("numba" or "sklearn": who built the constrained merge tree, "none" for one
    cluster), the last four null unless --spatial 1), "mds" (as above) and "elapsed_s". Without
    --spatial, above --max-clustered cells clustering is fitted on a seeded subsample and the
    other cells join the nearest cluster mean.

--plot-dir/<prefix>_mds_2d.png, <prefix>_mds_3d.png : PNG, 150 dpi
    Optional MDS scatter plots coloured by cluster (needs matplotlib), with
    <prefix> = --plot-prefix; "_cells_mds_2d.png" and "_cells_mds_3d.png" in cells mode.

--plot-dir/<prefix>_mds.csv : CSV
    Optional, one row per embedded window with columns window_index (index into "windows"),
    row, col, cluster, mds2_x, mds2_y, mds3_x, mds3_y, mds3_z. In cells mode
    "<prefix>_cells_mds.csv" with columns cell_index, id, x, y, cluster, mds2_x, mds2_y,
    mds3_x, mds3_y, mds3_z (embedded cells only).

--plot-dir/<prefix>_cells_delaunay.png : PNG, 150 dpi
    Optional, with --mode cells --spatial 1: clustered cell centroids coloured by cluster over
    the adjacency edges used (image coordinates, y down; needs matplotlib).
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time

# Module name -> pip package, checked before importing so a wrong Python gets a clear message
REQUIRED_MODULES = {"cv2": "opencv-python", "numpy": "numpy", "gudhi": "gudhi",
                    "sklearn": "scikit-learn", "joblib": "joblib"}


def check_environment() -> None:

    """
    Confirms this Python can run PHC before anything is imported. QuPath launched from Finder
    does not see the shell's PATH, so "python3" often resolves to a system Python without
    PHC's dependencies; this turns that into an actionable message instead of an ImportError.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If a dependency or the PHC package cannot be found; the message names the Python in
        use, every missing package and the QuPath preference to change.
    """

    missing = [package for module, package in REQUIRED_MODULES.items()
               if importlib.util.find_spec(module) is None]
    python = f"{sys.executable} (Python {sys.version.split()[0]})"
    if missing:
        sys.exit(f"The Python at {python} is missing: {', '.join(missing)}. In QuPath, set "
                 "Edit > Preferences > PHC > Python executable to an environment that has "
                 "PHC's requirements (e.g. ~/miniconda3/envs/phc/bin/python).")
    if importlib.util.find_spec("PHC") is None:
        sys.exit(f"The PHC package cannot be imported by {python}. In QuPath, set "
                 "Edit > Preferences > PHC > PHC library folder to the folder that contains "
                 "the PHC package (e.g. ~/PHC, the cloned repository).")


check_environment()

import cv2  # noqa: E402  (imported after the environment check above)
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from PHC import (PHC, agglomerative_clusters_capped, l2_dissimilarity,  # noqa: E402
                 mds_embedding, plot_embedding, plot_spatial_graph, read_cell_features,
                 spatial_agglomerative_clusters, window_dissimilarity)

VECTORIZATIONS = ("PI", "PL")
LINKAGES = ("ward", "average", "complete", "single")
MODES = ("windows", "cells")
MAX_CLUSTERED_WINDOWS = 10000   # agglomerative clustering needs O(n^2) memory
EXCLUDED = -1                   # cluster label for windows that fail the filters
PROGRESS_INTERVAL_S = 0.1       # most frequent progress update sent to QuPath
MIN_MDS_WINDOWS = 2             # MDS needs at least one pair of windows
MDS_SEED = 0                    # seed of the subsample and of sklearn's MDS
MDS_COMPONENTS = (2, 3)         # embedding dimensions, fitted side by side
CLUSTER_SEED = 0                # seed of the clustering subsample (cells mode, large n)
PLOT_DPI = 150
MDS_COLUMNS = ("mds2_x", "mds2_y", "mds3_x", "mds3_y", "mds3_z")
CSV_COLUMNS = ("window_index", "row", "col", "cluster") + MDS_COLUMNS
CELL_CSV_COLUMNS = ("cell_index", "id", "x", "y", "cluster") + MDS_COLUMNS
CELL_PLOT_TITLE = "MDS of PHC cell L2 distances"


def parse_args() -> argparse.Namespace:

    """
    Reads the region, PHC and clustering settings chosen in the QuPath dialog.

    Returns
    -------
    args : argparse.Namespace
        Parsed command line settings, with the defaults below for anything not passed.
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--mask-downsample", type=float, default=1.0)
    parser.add_argument("--origin-x", type=float, required=True)
    parser.add_argument("--origin-y", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=float, default=256.0)
    parser.add_argument("--stride", type=float, default=256.0)
    parser.add_argument("--dimension", type=int, default=1)
    parser.add_argument("--vectorization", choices=VECTORIZATIONS, default="PI")
    parser.add_argument("--vector-resolution", type=int, default=20)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--linkage", choices=LINKAGES, default="ward")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--mds", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mds-max-windows", type=int, default=5000)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--plot-prefix", default="phc")
    parser.add_argument("--mode", choices=MODES, default="windows")
    parser.add_argument("--max-clustered", type=int, default=MAX_CLUSTERED_WINDOWS)
    parser.add_argument("--spatial", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-edge-length", type=float, default=0.0)
    args = parser.parse_args()
    if args.mask_downsample < 1 or args.window_size <= 0 or args.stride <= 0:
        sys.exit("--mask-downsample must be at least 1, and --window-size and --stride "
                 "must be positive.")
    if args.mds_max_windows < 1:
        sys.exit(f"--mds-max-windows must be at least 1, got {args.mds_max_windows}.")
    if args.max_clustered < 1:
        sys.exit(f"--max-clustered must be at least 1, got {args.max_clustered}.")
    if not args.max_edge_length >= 0:  # also rejects NaN
        sys.exit(f"--max-edge-length must be at least 0, got {args.max_edge_length}.")
    if args.width <= 0 or args.height <= 0:
        sys.exit(f"The annotation box {args.width} x {args.height} px is empty.")
    return args


def read_mask(path: str) -> np.ndarray:

    """
    Loads the annotation mask written by QuPath as a boolean array.

    Parameters
    ----------
    path : str
        Path to an 8-bit PNG, non-zero inside the annotation.

    Returns
    -------
    mask : np.ndarray of bool - size (h, w)
        True for mask pixels inside the annotation.

    Raises
    ------
    SystemExit
        If the file is missing or cannot be decoded.
    """

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Could not read the annotation mask {path!r}.")
    mask = img > 0
    return mask


def window_coverage(
        mask: np.ndarray,
        grid: list[tuple[float, float, float, float]],
        downsample: float
        ) -> np.ndarray:

    """
    Estimates how much of each window lies inside the annotation, so windows that mostly
    show background can be left out of the clustering.

    Parameters
    ----------
    mask : np.ndarray of bool - size (h, w)
        Annotation mask; pixel (i, j) covers the slide pixels [j*D, (j+1)*D) x [i*D, (i+1)*D)
        relative to the box origin, with D = downsample.

    grid : list of tuple[float, float, float, float] - length n
        (row, col, window_height, window_width) per window, in slide pixels relative to the
        box origin.

    downsample : float
        Slide pixels per mask pixel (D >= 1).

    Returns
    -------
    coverage : np.ndarray of float - size (n,)
        Mean of the mask over the mask pixels each window touches, in [0, 1]; windows
        smaller than a mask pixel use the one pixel they fall in, and windows beyond the
        mask get 0.
    """

    n_rows, n_cols = mask.shape

    def mask_span(start: float, length: float, limit: int) -> tuple[int, int]:

        """
        Converts a window's extent along one axis to the range of mask pixels it touches.

        Parameters
        ----------
        start : float
            Window start in slide pixels, relative to the box origin.

        length : float
            Window length in slide pixels.

        limit : int
            Number of mask pixels along this axis.

        Returns
        -------
        span : tuple[int, int]
            (first, stop) mask indices, clipped to [0, limit]; first == stop when empty.
        """

        first = max(0, math.floor(start / downsample))
        stop = max(first + 1, math.ceil((start + length) / downsample))  # at least one pixel
        span = (min(first, limit), min(stop, limit))
        return span

    coverage = np.zeros(len(grid))
    for k, (row, col, window_height, window_width) in enumerate(grid):
        top, bottom = mask_span(row, window_height, n_rows)
        left, right = mask_span(col, window_width, n_cols)
        if bottom > top and right > left:
            coverage[k] = mask[top:bottom, left:right].mean()
    return coverage


def cell_coverage(
        mask: np.ndarray,
        corners: np.ndarray,
        window_size: float,
        downsample: float
        ) -> np.ndarray:

    """
    Estimates how much of each cell-centred window lies inside the annotation, so cells near
    its border can be left out of the clustering. Unlike tiles, these windows can reach past
    the box; the part outside the mask counts as outside the annotation. Vectorized with a
    summed-area table, so it is fast for many cells.

    Parameters
    ----------
    mask : np.ndarray of bool - size (h, w)
        Annotation mask; pixel (i, j) covers the slide pixels [j*D, (j+1)*D) x [i*D, (i+1)*D)
        relative to the box origin, with D = downsample.

    corners : np.ndarray of float - size (n, 2)
        (col, row) top-left corner of each window, in slide pixels relative to the box origin.

    window_size : float
        Side length of every window, in slide pixels.

    downsample : float
        Slide pixels per mask pixel (D >= 1).

    Returns
    -------
    coverage : np.ndarray of float - size (n,)
        Fraction of the mask pixels each window touches that are inside the annotation, in
        [0, 1]; windows smaller than a mask pixel use the one pixel they fall in.
    """

    n_rows, n_cols = mask.shape
    summed = np.zeros((n_rows + 1, n_cols + 1), dtype=np.int64)
    summed[1:, 1:] = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

    ### Mask pixels touched by each window, same rounding as window_coverage ###
    first = np.floor(corners / downsample).astype(np.int64)
    stop = np.maximum(first + 1, np.ceil((corners + window_size) / downsample).astype(np.int64))
    n_touched = np.prod(stop - first, axis=1)  # includes pixels beyond the mask (outside)
    left, right = np.clip(first[:, 0], 0, n_cols), np.clip(stop[:, 0], 0, n_cols)
    top, bottom = np.clip(first[:, 1], 0, n_rows), np.clip(stop[:, 1], 0, n_rows)
    n_inside = (summed[bottom, right] - summed[top, right] - summed[bottom, left]
                + summed[top, left])
    coverage = n_inside / n_touched
    return coverage


def report_stage(name: str, count: int | None = None) -> None:

    """
    Tells QuPath which stage is running, so its progress dialog can label the bar.

    Parameters
    ----------
    name : str
        One of "centroids", "persistence", "clustering" or "mds".

    count : int | None
        Number of items the stage works on (e.g. windows to cluster), default None.

    Returns
    -------
    None
    """

    suffix = "" if count is None else f" {count}"
    print(f"PHC_STAGE {name}{suffix}", flush=True)


class ProgressReporter:

    """
    Forwards PHC persistence progress to QuPath at most every PROGRESS_INTERVAL_S seconds,
    so a large annotation does not flood the pipe with one line per window. Holds the time
    of the last update as state.
    """

    def __init__(self):
        self.last_report = 0.0

    def __call__(self, n_done: int, n_total: int) -> None:

        """
        Prints a progress line if enough time has passed, and always for the last window.

        Parameters
        ----------
        n_done : int
            Windows finished so far.

        n_total : int
            Total number of windows.

        Returns
        -------
        None
        """

        now = time.perf_counter()
        if n_done == n_total or now - self.last_report >= PROGRESS_INTERVAL_S:
            print(f"PHC_PROGRESS {n_done} {n_total}", flush=True)
            self.last_report = now


def warn(message: str) -> None:

    """
    Reports a problem that does not stop the run (e.g. a plot that could not be written),
    as a line QuPath logs.

    Parameters
    ----------
    message : str
        What went wrong and what was skipped.

    Returns
    -------
    None
    """

    print(f"Warning: {message}", flush=True)


def embed_windows(windows: np.ndarray, max_windows: int, n_jobs: int = -1) -> dict | None:

    """
    Lays the clustered windows out in 2D and 3D with metric MDS on their L2 dissimilarity
    matrix, so QuPath can plot how the clusters relate. Above `max_windows` windows a seeded
    random subsample is embedded, which keeps the O(n^2) matrix and SMACOF affordable.

    Parameters
    ----------
    windows : np.ndarray of float - size (n, d)
        Persistence vectors of the clustered windows, in clustering order.

    max_windows : int
        Most windows to embed (m = min(n, max_windows)).

    n_jobs : int
        Worker processes allowed for the run; 1 fits the 2D and 3D embeddings one after the
        other, anything else fits them in two threads at once (same result), default -1.

    Returns
    -------
    embedding : dict | None
        Keys "index" (np.ndarray of int - size (m,), sorted rows of `windows` that were
        embedded), "mds2" (np.ndarray of float - size (m, 2)), "mds3" (np.ndarray of float -
        size (m, 3)), "subsampled" (bool), "stress_2d" and "stress_3d" (float, Kruskal
        stress-1); None when fewer than MIN_MDS_WINDOWS windows would be embedded.
    """

    n = len(windows)
    embedding = None
    if min(n, max_windows) >= MIN_MDS_WINDOWS:
        subsampled = n > max_windows
        if subsampled:
            index = np.sort(np.random.default_rng(MDS_SEED).choice(n, max_windows,
                                                                   replace=False))
        else:
            index = np.arange(n)

        report_stage("mds", len(index))
        dissimilarity = l2_dissimilarity(windows[index])
        # The two fits are independent and mostly in BLAS, so threads overlap them without
        # copying the matrix
        (mds2, stress_2d), (mds3, stress_3d) = Parallel(
            n_jobs=1 if n_jobs == 1 else len(MDS_COMPONENTS), backend="threading")(
            delayed(mds_embedding)(dissimilarity, n_components=n_components,
                                   random_state=MDS_SEED)
            for n_components in MDS_COMPONENTS)
        embedding = {"index": index, "mds2": mds2, "mds3": mds3, "subsampled": subsampled,
                     "stress_2d": stress_2d, "stress_3d": stress_3d}
    return embedding


def write_mds_outputs(
        embedding: dict,
        embedded_labels: np.ndarray,
        n_clusters: int,
        base: str,
        csv_columns: tuple[str, ...],
        csv_rows: list[list],
        title: str | None = None
        ) -> None:

    """
    Saves the MDS embedding as a CSV and as 2D / 3D publication plots coloured like the
    QuPath tiles. Never fails the run: each file that cannot be written (missing matplotlib,
    unwritable folder, ...) produces a warning line instead.

    Parameters
    ----------
    embedding : dict
        Output of `embed_windows` (keys "index", "mds2", "mds3", "stress_2d", "stress_3d").

    embedded_labels : np.ndarray of int - size (m,)
        Cluster label of each embedded window or cell, in embedding order.

    n_clusters : int
        Number of clusters, for the colour ramp.

    base : str
        Output path stem; files are "<base>.csv", "<base>_2d.png" and "<base>_3d.png".

    csv_columns : tuple[str, ...]
        CSV header, e.g. CSV_COLUMNS or CELL_CSV_COLUMNS.

    csv_rows : list of list - length m
        One CSV row per embedded item, matching `csv_columns`.

    title : str | None
        Plot title, default None (plot_embedding's window title).

    Returns
    -------
    None
    """

    plot_dir = os.path.dirname(base)

    ### CSV ###
    try:
        with open(f"{base}.csv", "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(csv_columns)
            writer.writerows(csv_rows)
    except Exception as error:  # noqa: BLE001  (optional output must not fail the run)
        warn(f"could not write the MDS CSV to {plot_dir!r} ({error}).")

    ### Plots ###
    if importlib.util.find_spec("matplotlib") is None:
        warn(f"matplotlib is not installed in {sys.executable}; MDS plots were skipped.")
    else:
        for n_dims in (2, 3):
            try:
                plot_embedding(embedding[f"mds{n_dims}"], embedded_labels,
                               f"{base}_{n_dims}d.png", n_clusters=n_clusters,
                               stress=embedding[f"stress_{n_dims}d"], title=title,
                               dpi=PLOT_DPI)
            except Exception as error:  # noqa: BLE001  (optional output must not fail run)
                warn(f"could not write the {n_dims}D MDS plot to {plot_dir!r} ({error}).")


def write_spatial_plot(
        points: np.ndarray,
        labels: np.ndarray,
        spatial: dict,
        n_clusters: int,
        path: str
        ) -> None:

    """
    Saves the Delaunay-constrained clusters as a figure (centroids coloured by cluster over
    the adjacency edges used). Never fails the run: a missing matplotlib or an unwritable
    folder produces a warning line instead.

    Parameters
    ----------
    points : np.ndarray of float - size (k, 2)
        (x, y) centroid of each clustered cell, in slide pixels.

    labels : np.ndarray of int - size (k,)
        Cluster label of each clustered cell.

    spatial : dict
        Clustering info from `spatial_agglomerative_clusters` (keys "adjacency", "n_edges",
        "n_components").

    n_clusters : int
        Number of clusters, for the colour ramp.

    path : str
        Output PNG path.

    Returns
    -------
    None
    """

    if importlib.util.find_spec("matplotlib") is None:
        warn(f"matplotlib is not installed in {sys.executable}; the Delaunay plot was skipped.")
    else:
        title = (f"Delaunay-constrained PHC cell clusters\n{len(points)} cells, "
                 f"{spatial['n_edges']} edges, {spatial['n_components']} component(s) "
                 "before bridging")
        try:
            plot_spatial_graph(points, labels, spatial["adjacency"], path,
                               n_clusters=n_clusters, title=title, dpi=PLOT_DPI)
        except Exception as error:  # noqa: BLE001  (optional output must not fail the run)
            warn(f"could not write the Delaunay plot to {os.path.dirname(path)!r} ({error}).")


def analyse_vectors(
        vectors: np.ndarray,
        kept: np.ndarray,
        n_items: int,
        args: argparse.Namespace,
        points: np.ndarray | None = None
        ) -> dict:

    """
    Shared second half of both modes: L2 measures, agglomerative clustering (subsample-fitted
    above --max-clustered, or Delaunay-constrained on all items when `points` is given) and
    MDS of the kept items' vectors, scattered back to all items so windows or cells that
    failed the filters get the "excluded" values.

    Parameters
    ----------
    vectors : np.ndarray of float - size (k, d)
        Persistence vectors of the kept windows or cells, in the order of `kept`.

    kept : np.ndarray of int - size (k,)
        Index of each kept item among all `n_items` items, increasing.

    n_items : int
        Number of windows or cells reported in the output (N).

    args : argparse.Namespace
        Parsed settings (n_clusters, linkage, max_clustered, max_edge_length, mds,
        mds_max_windows).

    points : np.ndarray of float - size (k, 2) | None
        (x, y) centroid of each kept item; when given, clustering is spatially constrained
        by their Delaunay triangulation, default None (unconstrained).

    Returns
    -------
    analysis : dict
        Keys "l2_norm", "l2_to_mean" (np.ndarray of float - size (N,), 0 when excluded),
        "cluster" (np.ndarray of int - size (N,), EXCLUDED when excluded), "n_clusters"
        (int), "subsampled" (bool), "n_fitted" (int), "spatial" (dict with keys "adjacency",
        "n_edges", "n_components" from `spatial_agglomerative_clusters`, or None when
        `points` is None), "mds2", "mds3" (lists of length N holding coordinate lists or
        None), "mds" (summary dict or None), "embedding" (output of `embed_windows` or None)
        and "embedded_ids" (np.ndarray of int - size (m,), item index of each embedded
        vector).
    """

    report_stage("clustering", len(kept))
    l2_norms, l2_to_mean = np.zeros(n_items), np.zeros(n_items)
    l2_norms[kept], l2_to_mean[kept] = window_dissimilarity(vectors)
    labels = np.full(n_items, EXCLUDED)
    spatial = None
    if points is None:
        labels[kept], subsampled, n_fitted = agglomerative_clusters_capped(
            vectors, n_clusters=args.n_clusters, linkage=args.linkage,
            max_fitted=args.max_clustered, random_state=CLUSTER_SEED)
    else:
        labels[kept], spatial = spatial_agglomerative_clusters(
            vectors, points, n_clusters=args.n_clusters, linkage=args.linkage,
            max_edge_length=args.max_edge_length)
        subsampled, n_fitted = False, len(kept)  # adjacency needs every cell

    ### MDS of the L2 dissimilarity matrix ###
    mds2, mds3 = [None] * n_items, [None] * n_items
    mds_summary = None
    embedded_ids = np.empty(0, dtype=int)
    embedding = (embed_windows(vectors, args.mds_max_windows, n_jobs=args.n_jobs)
                 if args.mds else None)
    if embedding is not None:
        embedded_ids = kept[embedding["index"]]
        for k, i in enumerate(embedded_ids):
            mds2[i] = embedding["mds2"][k].tolist()
            mds3[i] = embedding["mds3"][k].tolist()
        mds_summary = {"n_embedded": len(embedded_ids), "subsampled": embedding["subsampled"],
                       "stress_2d": embedding["stress_2d"], "stress_3d": embedding["stress_3d"]}

    analysis = {"l2_norm": l2_norms, "l2_to_mean": l2_to_mean, "cluster": labels,
                "n_clusters": int(labels.max()) + 1, "subsampled": subsampled,
                "n_fitted": n_fitted, "spatial": spatial, "mds2": mds2, "mds3": mds3,
                "mds": mds_summary, "embedding": embedding, "embedded_ids": embedded_ids}
    return analysis


def run_windows(args: argparse.Namespace, centroids: np.ndarray, mask: np.ndarray) -> dict:

    """
    Tiled mode: keeps the windows that lie inside the annotation and hold enough cells, runs
    alpha complex PHC on them and analyses them (clusters, L2 measures, MDS, plots).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line settings.

    centroids : np.ndarray of float - size (n, 2)
        (x, y) centroid of every usable cell, in slide pixels.

    mask : np.ndarray of bool - size (h, w)
        Annotation mask, as from `read_mask`.

    Returns
    -------
    results : dict
        The windows-mode output JSON without "elapsed_s" (keys described in the header).

    Raises
    ------
    SystemExit
        If no window passes the filters, or more than --max-clustered windows do.
    """

    ### Lay out windows and decide which ones to cluster ###
    localhom = PHC(persistence_type="alpha", window_size=args.window_size, stride=args.stride,
                   vectorization=args.vectorization, vector_resolution=args.vector_resolution,
                   dimension=args.dimension, n_jobs=args.n_jobs)
    grid = localhom.point_window_grid((args.height, args.width))
    window_pts, n_cells = localhom.window_points(centroids, (args.origin_x, args.origin_y),
                                                 grid)
    coverage = window_coverage(mask, grid, args.mask_downsample)
    inside = (coverage >= args.min_coverage) & (n_cells >= args.min_cells)
    n_inside = int(inside.sum())
    if n_inside == 0:
        sys.exit(f"No window has at least {args.min_coverage:.0%} of its area inside the "
                 f"annotation and at least {args.min_cells} cells (the most cells in a window "
                 f"is {int(n_cells.max())}); use a larger window size or a lower minimum.")
    if n_inside > args.max_clustered:
        sys.exit(f"{n_inside} windows exceed the clustering limit of {args.max_clustered}; "
                 "increase the window size or stride.")

    ### Local persistence of the kept windows, vectorized on one shared range ###
    kept = np.flatnonzero(inside)
    report_stage("persistence", n_inside)
    diagrams = localhom.point_diagrams([window_pts[k] for k in kept],
                                       progress=ProgressReporter())
    windows = localhom.vectorize_diagrams(diagrams)

    ### Measure, cluster and embed ###
    analysis = analyse_vectors(windows, kept, len(grid), args)
    labels = analysis["cluster"]
    if analysis["embedding"] is not None and args.plot_dir is not None:
        rows = [[int(i), grid[i][0], grid[i][1], int(labels[i]), *analysis["mds2"][i],
                 *analysis["mds3"][i]] for i in analysis["embedded_ids"]]
        write_mds_outputs(analysis["embedding"], labels[analysis["embedded_ids"]],
                          analysis["n_clusters"],
                          os.path.join(args.plot_dir, f"{args.plot_prefix}_mds"),
                          CSV_COLUMNS, rows)

    results = {
        "mode": "windows",
        "windows": [{"row": r, "col": c, "height": h, "width": w,
                     "coverage": float(coverage[i]), "n_cells": int(n_cells[i]),
                     "l2_norm": float(analysis["l2_norm"][i]),
                     "l2_to_mean": float(analysis["l2_to_mean"][i]),
                     "cluster": int(labels[i]), "mds2": analysis["mds2"][i],
                     "mds3": analysis["mds3"][i]}
                    for i, (r, c, h, w) in enumerate(grid)],
        "n_clusters": analysis["n_clusters"],
        "n_windows_clustered": n_inside,
        "n_cells": len(centroids),
        "mds": analysis["mds"],
    }
    return results


def run_cells(
        args: argparse.Namespace,
        centroids: np.ndarray,
        ids: list[str | None],
        mask: np.ndarray
        ) -> dict:

    """
    Per-cell mode: centres a --window-size window on every cell, computes the alpha complex
    persistence of the neighbouring centroids in it (one computation per cell), keeps the
    cells whose window lies inside the annotation and holds enough cells, and analyses them
    (clusters, L2 measures, MDS, plots).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line settings.

    centroids : np.ndarray of float - size (N, 2)
        (x, y) centroid of every GeoJSON feature in slide pixels, NaN rows for features
        without one.

    ids : list of str | None - length N
        Each feature's "id", or None.

    mask : np.ndarray of bool - size (h, w)
        Annotation mask, as from `read_mask`.

    Returns
    -------
    results : dict
        The cells-mode output JSON without "elapsed_s" (keys described in the header).

    Raises
    ------
    SystemExit
        If no cell passes the filters.
    """

    n_features = len(centroids)
    usable = np.flatnonzero(np.all(np.isfinite(centroids), axis=1))
    points = centroids[usable]

    ### One centred local persistence computation per cell ###
    localhom = PHC(persistence_type="alpha", window_size=args.window_size,
                   vectorization=args.vectorization, vector_resolution=args.vector_resolution,
                   dimension=args.dimension, n_jobs=args.n_jobs)
    report_stage("persistence", len(points))
    vectors, n_neighbours = localhom.convolve_cells(points, progress=ProgressReporter())

    ### Decide which cells to cluster ###
    corners = points - (args.origin_x, args.origin_y) - args.window_size / 2
    coverage = np.zeros(n_features)
    coverage[usable] = cell_coverage(mask, corners, args.window_size, args.mask_downsample)
    n_cells = np.zeros(n_features, dtype=int)
    n_cells[usable] = n_neighbours
    passes = (coverage[usable] >= args.min_coverage) & (n_neighbours >= args.min_cells)
    if not passes.any():
        sys.exit(f"No cell has at least {args.min_coverage:.0%} of its window inside the "
                 f"annotation and at least {args.min_cells} cells in its window (the most "
                 f"is {int(n_neighbours.max())}); use a larger window size or a lower minimum.")
    kept = usable[passes]

    ### Measure, cluster (optionally Delaunay-constrained) and embed ###
    analysis = analyse_vectors(vectors[passes], kept, n_features, args,
                               points=centroids[kept] if args.spatial else None)
    labels = analysis["cluster"]
    spatial = analysis["spatial"]
    if spatial is not None and args.plot_dir is not None:
        write_spatial_plot(centroids[kept], labels[kept], spatial, analysis["n_clusters"],
                           os.path.join(args.plot_dir,
                                        f"{args.plot_prefix}_cells_delaunay.png"))
    if analysis["embedding"] is not None and args.plot_dir is not None:
        rows = [[int(i), ids[i], float(centroids[i, 0]), float(centroids[i, 1]),
                 int(labels[i]), *analysis["mds2"][i], *analysis["mds3"][i]]
                for i in analysis["embedded_ids"]]
        write_mds_outputs(analysis["embedding"], labels[analysis["embedded_ids"]],
                          analysis["n_clusters"],
                          os.path.join(args.plot_dir, f"{args.plot_prefix}_cells_mds"),
                          CELL_CSV_COLUMNS, rows, title=CELL_PLOT_TITLE)

    has_centroid = np.zeros(n_features, dtype=bool)
    has_centroid[usable] = True
    results = {
        "mode": "cells",
        "cells": [{"index": i, "id": ids[i],
                   "x": float(centroids[i, 0]) if has_centroid[i] else None,
                   "y": float(centroids[i, 1]) if has_centroid[i] else None,
                   "coverage": float(coverage[i]), "n_cells": int(n_cells[i]),
                   "l2_norm": float(analysis["l2_norm"][i]),
                   "l2_to_mean": float(analysis["l2_to_mean"][i]),
                   "cluster": int(labels[i]), "mds2": analysis["mds2"][i],
                   "mds3": analysis["mds3"][i]}
                  for i in range(n_features)],
        "n_clusters": analysis["n_clusters"],
        "n_cells": len(points),
        "n_cells_clustered": len(kept),
        "n_skipped": n_features - len(points),
        "clustering": {"subsampled": analysis["subsampled"], "n_fitted": analysis["n_fitted"],
                       "spatial": spatial is not None,
                       "n_edges": None if spatial is None else spatial["n_edges"],
                       "n_components": None if spatial is None else spatial["n_components"],
                       "max_edge_length": None if spatial is None else args.max_edge_length,
                       "backend": None if spatial is None else spatial["backend"]},
        "mds": analysis["mds"],
    }
    return results


def main() -> None:

    """
    Reads the cell centroids and the annotation mask, runs the chosen mode (tiled windows
    or one window per cell) and writes the results as JSON (plus optional plots and CSV).

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the file holds no cells, or the chosen mode finds nothing to cluster.
    """

    args = parse_args()
    start = time.perf_counter()

    ### Cell centroids ###
    report_stage("centroids")
    centroids, ids = read_cell_features(args.cells)
    if not np.all(np.isfinite(centroids), axis=1).any():
        sys.exit("No cells were found in the annotation. Run cell detection first "
                 "(Analyze > Cell detection), then run PHC again.")
    mask = read_mask(args.mask)

    if args.mode == "windows":
        usable = np.all(np.isfinite(centroids), axis=1)
        results = run_windows(args, centroids[usable], mask)
    elif args.mode == "cells":
        results = run_cells(args, centroids, ids, mask)
    else:
        raise ValueError(f"Unknown mode {args.mode!r}; expected 'windows' or 'cells'.")

    results["elapsed_s"] = time.perf_counter() - start
    with open(args.output, "w") as out:
        json.dump(results, out)


if __name__ == "__main__":
    main()
