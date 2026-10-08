"""
Publication plots of PHC window embeddings, coloured like the QuPath cluster tiles.
matplotlib is imported only inside the plotting function, so PHC imports without it.

Contents
--------
VIRIDIS_ANCHORS : constant
    Five viridis anchor colours (dark purple to yellow) shared with the QuPath extension.
cluster_colors : function
    RGB colour of each cluster, at position k / (K - 1) along the viridis ramp.
plot_embedding : function
    Scatter plot of a 2D or 3D embedding, coloured by cluster, saved as an image file.
plot_spatial_graph : function
    Cell centroids coloured by cluster over their adjacency (e.g. Delaunay) edges, in image
    coordinates, saved as an image file.
"""

import numpy as np
from scipy.sparse import csr_matrix

# Same anchors and linear interpolation as PHCPipeline.RAMP in the QuPath extension
VIRIDIS_ANCHORS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98],
                            [253, 231, 37]], dtype=float)
RGB_MAX = 255.0
FIGURE_SIZE_IN = (6.0, 5.0)     # single-column figure, inches
MARKER_AREA = 14.0              # scatter marker area, points^2
MARKER_EDGE = "#4a4a4a"         # thin dark ring keeps yellow points visible on white
MARKER_EDGE_WIDTH = 0.3
MARKER_ALPHA = 0.85
INK = "#333333"                 # text and axis colour
GRID = "#e6e6e6"                # recessive grid colour
LEGEND_MARKER_SCALE = 2.0       # legend swatches larger than the data points
VIEW_ELEVATION, VIEW_AZIMUTH = 20.0, -60.0   # 3D camera angles, degrees
GRAPH_WIDTH_IN = 7.0            # spatial graph figure width, inches; height follows the data
GRAPH_HEIGHT_RANGE_IN = (3.0, 9.0)
GRAPH_TITLE_IN = 1.4            # room for the title, axis labels and legend, inches
GRAPH_TITLE_PAD_PT = 22.0       # title sits this far above the axes per legend row, points
GRAPH_LEGEND_COLUMNS = 6        # most legend entries per row above the axes
GRAPH_PAD_IN = 0.1              # margin kept when the saved figure is cropped, inches
GRAPH_EDGE = "#b0b0b0"          # recessive edge colour, under the points
GRAPH_EDGE_WIDTH = 0.3          # points
GRAPH_POINTS_AT_FULL_SIZE = 500  # above this many cells, markers shrink with 1 / n
GRAPH_MIN_MARKER_AREA = 0.6     # points^2, so 50k cells stay distinct dots
LEGEND_SWATCH_AREA = 30.0       # points^2, legend swatch size whatever the marker size


def cluster_colors(n_clusters: int) -> np.ndarray:

    """
    Gives each cluster the colour its tiles have in QuPath, so plots and the slide overlay
    can be read side by side: cluster k of K sits at k / (K - 1) on the viridis ramp
    (dark = lowest mean L2 norm), and a single cluster is yellow.

    Parameters
    ----------
    n_clusters : int
        Number of clusters K (at least 1).

    Returns
    -------
    colors : np.ndarray of float - size (n_clusters, 3)
        RGB colour of each cluster in [0, 1], rounded to 8-bit like the QuPath classes.
    """

    if n_clusters == 1:
        positions = np.ones(1)
    else:
        positions = np.arange(n_clusters) / (n_clusters - 1)
    scaled = positions * (len(VIRIDIS_ANCHORS) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(VIRIDIS_ANCHORS) - 1)
    t = (scaled - lower)[:, None]
    rgb = np.round(VIRIDIS_ANCHORS[lower] + t * (VIRIDIS_ANCHORS[upper] - VIRIDIS_ANCHORS[lower]))
    colors = rgb / RGB_MAX
    return colors


def plot_embedding(
        coords: np.ndarray,
        labels: np.ndarray,
        path: str,
        n_clusters: int | None = None,
        stress: float | None = None,
        title: str | None = None,
        dpi: int = 150
        ) -> None:

    """
    Draws windows as points in their MDS (or any 2D / 3D) embedding, coloured by PHC
    cluster, and saves the figure, so the separation between clusters can be shown in a
    paper. Uses the Agg backend through a standalone Figure, leaving pyplot state alone.

    Parameters
    ----------
    coords : np.ndarray of float - size (n, k)
        Embedded windows, with k = 2 or 3.

    labels : np.ndarray of int - size (n,)
        Cluster label in [0, n_clusters) for each window.

    path : str
        Output image path; the format follows its extension (e.g. ".png", ".pdf").

    n_clusters : int | None
        Number of clusters K used for the colour ramp, default None (labels.max() + 1).

    stress : float | None
        Kruskal stress-1 of the embedding, shown in the title, default None (not shown).

    title : str | None
        Plot title, default None ("MDS of PHC window L2 distances").

    dpi : int
        Resolution of raster outputs, default 150.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `coords` does not have 2 or 3 columns, or its length does not match `labels`.
    ImportError
        If matplotlib is not installed.
    """

    from matplotlib.figure import Figure  # lazy: matplotlib is optional for PHC
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_dims = coords.shape[1] if coords.ndim == 2 else 0
    if n_dims not in (2, 3):
        raise ValueError(f"coords must have 2 or 3 columns, got shape {coords.shape}.")
    if len(coords) != len(labels):
        raise ValueError(f"{len(coords)} points but {len(labels)} labels.")
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1 if len(labels) else 1
    colors = cluster_colors(n_clusters)

    ### Figure and axes ###
    fig = Figure(figsize=FIGURE_SIZE_IN)
    if n_dims == 3:
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    else:
        ax = fig.add_subplot()
        ax.set_aspect("equal", adjustable="datalim")  # MDS distances are comparable in x, y
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK)

    ### Points, one series per cluster so the legend names each ###
    for k in range(n_clusters):
        members = labels == k
        if not members.any():
            continue
        ax.scatter(*coords[members].T, s=MARKER_AREA, color=colors[k], alpha=MARKER_ALPHA,
                   edgecolors=MARKER_EDGE, linewidths=MARKER_EDGE_WIDTH,
                   label=f"PHC cluster {k + 1}")

    ### Labels ###
    ax.set_xlabel("MDS 1", color=INK)
    ax.set_ylabel("MDS 2", color=INK)
    if n_dims == 3:
        ax.set_zlabel("MDS 3", color=INK)
    ax.tick_params(colors=INK, labelsize=8)
    heading = "MDS of PHC window L2 distances" if title is None else title
    if stress is not None:
        heading += f"\n{n_dims}D, Kruskal stress-1 = {stress:.3f}"
    ax.set_title(heading, color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, markerscale=LEGEND_MARKER_SCALE, labelcolor=INK,
              loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)


def plot_spatial_graph(
        points: np.ndarray,
        labels: np.ndarray,
        adjacency: csr_matrix,
        path: str,
        n_clusters: int | None = None,
        title: str | None = None,
        dpi: int = 150
        ) -> None:

    """
    Draws cells at their centroids, coloured by PHC cluster like the QuPath overlay, over the
    adjacency edges that constrained the clustering, so the contiguous regions and the graph
    behind them can be checked or shown in a paper. Image coordinates (y grows downwards);
    edges are one LineCollection, so 10^5 edges draw quickly.

    Parameters
    ----------
    points : np.ndarray of float - size (n, 2)
        (x, y) centroid of each cell, in slide pixels.

    labels : np.ndarray of int - size (n,)
        Cluster label in [0, n_clusters) for each cell.

    adjacency : scipy.sparse.csr_matrix - size (n, n)
        Symmetric adjacency between cells; each nonzero (i, j) with i < j is drawn as an edge.

    path : str
        Output image path; the format follows its extension (e.g. ".png", ".pdf").

    n_clusters : int | None
        Number of clusters K used for the colour ramp, default None (labels.max() + 1).

    title : str | None
        Plot title, default None ("Spatially constrained PHC clusters" with the cell and
        edge counts).

    dpi : int
        Resolution of raster outputs, default 150.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `points` is not of size (n, 2), or `labels` or `adjacency` does not match it.
    ImportError
        If matplotlib is not installed.
    """

    from matplotlib.collections import LineCollection  # lazy: matplotlib is optional
    from matplotlib.figure import Figure

    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have size (n, 2), got {points.shape}.")
    if len(labels) != n or adjacency.shape != (n, n):
        raise ValueError(f"{n} points, {len(labels)} labels and adjacency of size "
                         f"{adjacency.shape}.")
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1 if n else 1
    colors = cluster_colors(n_clusters)
    rows, cols = adjacency.nonzero()
    upper = rows < cols
    segments = np.stack([points[rows[upper]], points[cols[upper]]], axis=1)

    ### Figure sized to the tissue's aspect ratio ###
    span = np.ptp(points, axis=0) if n else np.ones(2)
    aspect = span[1] / span[0] if span[0] > 0 else 1.0
    height = float(np.clip(GRAPH_WIDTH_IN * aspect + GRAPH_TITLE_IN, *GRAPH_HEIGHT_RANGE_IN))
    fig = Figure(figsize=(GRAPH_WIDTH_IN, height))
    ax = fig.add_subplot()
    ax.set_aspect("equal", adjustable="box")  # slide pixels are square; no padding added
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)

    ### Edges under the points, then one series per cluster so the legend names each ###
    ax.add_collection(LineCollection(segments, colors=GRAPH_EDGE,
                                     linewidths=GRAPH_EDGE_WIDTH, zorder=1))
    marker_area = max(GRAPH_MIN_MARKER_AREA,
                      MARKER_AREA * min(1.0, GRAPH_POINTS_AT_FULL_SIZE / max(n, 1)))
    for k in range(n_clusters):
        members = labels == k
        if not members.any():
            continue
        ax.scatter(*points[members].T, s=marker_area, color=colors[k], linewidths=0,
                   zorder=2, label=f"PHC cluster {k + 1}")
    ax.autoscale_view()
    ax.invert_yaxis()  # image coordinates: y grows downwards, as in QuPath

    ### Labels ###
    ax.set_xlabel("x (slide px)", color=INK)
    ax.set_ylabel("y (slide px)", color=INK)
    ax.tick_params(colors=INK, labelsize=8)
    heading = (f"Spatially constrained PHC clusters\n{n} cells, {int(upper.sum())} "
               "adjacency edges" if title is None else title)
    legend_rows = -(-n_clusters // GRAPH_LEGEND_COLUMNS)  # ceiling division
    ax.set_title(heading, color=INK, fontsize=10, pad=GRAPH_TITLE_PAD_PT * legend_rows)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=min(n_clusters, GRAPH_LEGEND_COLUMNS),
              borderaxespad=0.2, columnspacing=1.0, handletextpad=0.2,
              markerscale=float(np.sqrt(LEGEND_SWATCH_AREA / marker_area)))  # under title
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=GRAPH_PAD_IN)  # crop margins
