# QuPath PHC extension

Runs Persistent Homology Convolutions (PHC) on the cells detected inside an annotation. PHC
centres one square window on *each cell* and computes the alpha-complex persistence of the
centroids of the neighbouring cells in it (so rings of cells, such as glands, give strong
1-dimensional features), vectorizes it, and the cells are grouped by an agglomerative
clustering of their persistence vectors under L2 (Euclidean) distance. The results are stored
on the cells as measurements, so **Measure > Show measurement maps** colours the cells by
cluster. The same L2 distances are projected into 2D and 3D with metric MDS, shown in an
interactive plot linked to the cells (see [Per-cell windows](#per-cell-windows)). The
clustering can also be **spatially constrained** to the Delaunay triangulation of the cell
centroids, so each cluster is a contiguous region of tissue (see
[Spatially constrained clustering](#spatially-constrained-clustering)).

Since 0.6.1 the dialog only offers per-cell windows; the tiled-window mode of earlier versions
is no longer in the user interface.

The jar is the QuPath front end. The maths runs in your existing Python PHC library, so
QuPath shows exactly what the PHC library computes.

```
QuPath (this jar)                                 Python (PHC library)
Analyze > Cell detection (run first)
Extensions > PHC > Run PHC...
  parameter dialog (window size in um, ...)
  cell centroids in the ROI -> cells.geojson ---> phc_bridge.py --mode cells
  ROI mask (<= 4096 px a side) -> mask.png          cell centroids (nucleus, else cell)
                                                    one window centred on each cell
                                                    alpha persistence per window (parallel)
                                                    L2 measures + agglomerative clustering
                                                      (optionally on the Delaunay graph)
                                                    L2 distance matrix -> metric MDS (2D, 3D)
  cell measurements <------------------------------ results.json
  MDS viewer (2D / 3D tabs)                         <project>/PHC plots/*.png, *.csv
```

## Install (QuPath 0.7)

1. Download [`qupath-extension-phc-0.6.1.jar`](https://github.com/Shrunalp/PHC/releases/latest/download/qupath-extension-phc-0.6.1.jar)
   (also in [`dist/`](dist/qupath-extension-phc-0.6.1.jar)), drag it onto the QuPath window
   and restart QuPath. To build it yourself instead, see [Build](#build).
2. In **Edit > Preferences > PHC**, set:
   - **Python executable**: a Python with the packages of the repository's
     `requirements.txt` (`gudhi`, `scikit-learn`, `opencv-python`, `joblib`, and optionally
     `numba` for fast spatial clustering and `matplotlib` for plots), e.g.
     `~/miniconda3/envs/phc/bin/python`. Use the full path: QuPath started from Finder
     does not see your shell's `PATH`.
   - **PHC library folder**: the folder that *contains* the `PHC` package, i.e. the root
     of the cloned repository, e.g. `~/PHC`.

## Use

1. Draw or select an area annotation (rectangle, ellipse, polygon or brush).
2. Detect the cells in it first: **Analyze > Cell detection > Cell detection** (any detection
   command works, e.g. StarDist). PHC uses every detection whose centroid lies inside the
   annotation. Without detections PHC stops with a message asking you to run cell detection.
3. **Extensions > PHC > Run PHC on selected annotation...**, choose settings, press **Run**.
   A progress dialog shows the current stage; while the cells' windows are computed it shows a
   progress bar, the cell count and an estimated time to completion (from the average time per
   cell so far), plus the elapsed time. **Cancel** stops Python and its workers. A
   notification appears when the results are on the cells.
4. Read the results:
   - **Measure > Show measurement maps > `PHC: cluster`** colours the cells by cluster.
     **PHC cluster 1** has the lowest mean L2 norm (least topological signal), the highest
     cluster the most. Any other PHC measurement (e.g. `PHC: L2 norm`) can be mapped too.
   - The **MDS viewer** opens after the run (see below).
   - Running again on the same annotation replaces the cells' PHC results.
     **Extensions > PHC > Clear PHC heatmap** removes them (restoring classes PHC changed),
     and also removes heatmap tiles left by tiled runs of earlier versions.

## MDS embedding

After clustering, the clustered cells' persistence vectors are compared pairwise with the L2
(Euclidean) distance, and the distance matrix is projected with metric MDS
(`sklearn.manifold.MDS`, seeded from classical MDS, deterministic; the 2D and 3D fits run side
by side) into 2D and into 3D. Cells with similar local topology end up close together, so the
clusters appear as groups of points, and cells between clusters show up between the groups.

**The viewer** opens when a run with MDS finishes, and again from
**Extensions > PHC > Show MDS plot** for the selected annotation (it reads the cell
measurements, so it works after reopening a project).

- **2D** tab: scatter of MDS 1 against MDS 2, both axes on the same scale, points coloured by
  cluster (viridis), legend with the cells per cluster, stress in the subtitle. Hover a point
  for its cluster, the cells in its window and its L2 norm.
- **3D** tab: rotatable scatter with axes MDS 1, 2, 3. Drag to rotate, scroll to zoom, hover
  for details.
- **Click** a point (either tab) to select its cell in QuPath and centre the viewer on it.
  Selecting a cell in QuPath highlights its point.
- **Save PNG...** saves the tab's figure; the footer says where the Python plots were saved.

**Files**: when a QuPath project is open, Python also writes publication plots to
`<project folder>/PHC plots/` (the notification names the folder), named
`<image>_<annotation name, or id if unnamed>` with unsafe characters replaced by `_`:

| File | Content |
|---|---|
| `<prefix>_cells_mds_2d.png`, `<prefix>_cells_mds_3d.png` | matplotlib plots, 150 dpi, same colours and labels as the viewer |
| `<prefix>_cells_mds.csv` | `cell_index,id,x,y,cluster,mds2_x,mds2_y,mds3_x,mds3_y,mds3_z` |

Without a project no files are written (the viewer still works). The plots need `matplotlib`
in the PHC Python; without it (or if the folder cannot be written) the run still succeeds,
and the warning is logged and shown in the notification. The CSV only lists embedded cells.

## Per-cell windows

For every cell inside the annotation, Python takes the square window of side *Window size* centred on the cell's centroid, finds the
centroids of the neighbouring cells in it (half-open: `c.x - s/2 <= p.x < c.x + s/2`, same in
y, the cell itself included), and computes the alpha persistence of those points: one
computation per cell, all vectorized on one common range. The cells are then clustered,
measured (L2) and embedded with MDS. A cell is
clustered when its window has at least *Min cells per window* centroids (itself included) and
at least *Min ROI coverage* of it lies inside the annotation; other cells are still computed
and get coverage and neighbour count only.

No tiles are created: the results go onto the cells as measurements, so
**Measure > Show measurement maps > `PHC: cluster`** (or `PHC: L2 norm`) colours the cells.

| Cell measurement | Meaning |
|---|---|
| `PHC: cluster` | Agglomerative cluster of the cell's window, 1 = lowest mean L2 norm (clustered cells only) |
| `PHC: L2 norm` | L2 norm of the window's persistence vector (clustered cells only) |
| `PHC: L2 distance to ROI mean` | Distance to the mean vector of the clustered cells (clustered cells only) |
| `PHC: ROI coverage` | Fraction of the cell's window inside the annotation (every cell) |
| `PHC: cells in window` | Centroids in the cell's window, the cell included (every cell) |
| `PHC: MDS2 x/y`, `PHC: MDS3 x/y/z` | MDS coordinates (embedded cells only) |

The annotation gets `PHC cells: MDS2 stress` and `PHC cells: MDS3 stress` (Kruskal stress-1;
0 = distances kept exactly, below about 0.1 is a good fit). A new run removes stale cluster,
L2 and MDS values from cells that are not clustered (or not embedded) this time. Cell
measurements are never sent to Python, so earlier results do not affect a new run.

**Set cell classes to PHC clusters (replaces their current classification)** (off by default)
also gives each clustered cell the class `PHC cluster k`, in viridis colours. Before overwriting a class, PHC saves the original in the cell's object metadata
(key `phc.originalClass`, `""` for unclassified), which QuPath stores with the project, so
it survives saving and reopening. A later run keeps the first saved original; cells that are
not classified by the latest run (left out, or classes switched off) get their original class
back. **Clear PHC heatmap** restores every saved class and removes the per-cell measurements.

In the **MDS viewer**, points are coloured by cluster even when the cells keep their own
classes. On an annotation that still holds MDS results from a tiled run of an earlier version,
**Show MDS plot** asks which plot to show.

Results map back to cells by position: Java exports the annotation's cell centroids (one
GeoJSON Point per cell, with its UUID) in a fixed order, Python returns one entry per exported
cell in that order, and Java checks each entry's index
and id (the cell's UUID) before writing anything; a mismatch stops the run with an error.

Limits:

- Run time grows with the number of cells (one persistence computation each) and with the
  neighbours per window.
- Clustering above 10,000 cells is fitted on a random (seeded) subsample of 10,000 cells; every
  other clustered cell joins the cluster whose mean vector is nearest in L2. The notification
  says when this happened. This does not apply to spatially constrained clustering, which
  always uses every clustered cell (see below).
- MDS embeds at most *Max windows for MDS* cells (a random subsample beyond that).
- Neighbours only come from cells inside the annotation, so cells near its edge see fewer
  neighbours: their windows reach outside, and *Min ROI coverage* (default 0.5) leaves the
  worst of them out of the clustering.

## Spatially constrained clustering

Tick **Spatially constrained clustering (Delaunay adjacency of cell
centroids)** (off by default) and Python builds a Delaunay triangulation
(`scipy.spatial.Delaunay`) of the centroids of the *clustered* cells and passes its edges to
`AgglomerativeClustering` as the connectivity: two groups of cells can only merge when an
edge of the triangulation joins them. Clusters are therefore spatially contiguous regions
(e.g. "the glandular area" and "the stroma around it") instead of cells scattered over the
annotation that merely look alike. Persistence, L2 measures, MDS and the cell measurements are
the same as without the constraint; only the cluster labels change (still ordered by mean L2
norm, cluster 1 lowest).

- **Max Delaunay edge length** (µm, pixels on an uncalibrated image; default 0 = no limit):
  edges longer than this are dropped, so cells on either side of a gap (a lumen, a fold, an
  empty region) are not neighbours. Converted to slide pixels like the window size. Must be
  >= 0.
- **Contiguity guarantee**: when the Delaunay graph is connected, every cluster is one
  connected piece of it. The check verifies this independently with QuPath's own
  `DelaunayTools` triangulation.
- **Caveat, joined components**: if the graph falls apart (a short edge limit, or cell
  groups far apart), Python joins the parts at their closest pair of cells so clustering can
  run, and the notification warns: *"graph had N disconnected parts; they were joined at
  their closest cells"*. A cluster may then span two regions that only touch through such a
  joining edge.
- **No subsample**: the 10,000-cell subsample of plain per-cell clustering would break the
  adjacency, so spatial clustering always uses every clustered cell. Memory and time grow
  with the number of cells; very large annotations take longer.
- **Fast backend**: ward and average linkage run on a numba implementation of the same
  algorithm as scikit-learn (`PHC/constrained_linkage.py`) that gives identical clusters, e.g.
  ~49,000 cells cluster in about 2 s instead of about 75 s. Each Python process checks it
  against the installed scikit-learn first and falls back to scikit-learn if numba is missing
  or the check fails (e.g. after a scikit-learn upgrade); complete and single linkage always
  use scikit-learn. The first run compiles it (a few seconds) and caches the result in
  `PHC/__pycache__`. Set the environment variable `PHC_LINKAGE_BACKEND=sklearn` to force
  scikit-learn.
- The notification says e.g. *"3174 clustered into 4 spatially contiguous clusters (Delaunay
  graph: 9,412 edges)"*, and the progress dialog shows *"Clustering 3174 cells
  (Delaunay-constrained)"*.

The annotation gets two more measurements (removed again by a non-spatial per-cell run and
by **Clear PHC heatmap**):

| Annotation measurement | Meaning |
|---|---|
| `PHC cells: Delaunay edges` | Undirected edges of the adjacency used (after the edge limit and any joining edges) |
| `PHC cells: Delaunay components` | Connected parts of the Delaunay graph before they were joined (1 = connected) |

When plot files are written, Python may also save `<prefix>_cells_delaunay.png`: the cell
centroids coloured by cluster with the adjacency edges drawn thinly. The notification and the
MDS viewer's plot-folder note name it when it is present.

## Parameters

| Setting | Default | Notes |
|---|---|---|
| Window size | 100 µm | Side of the square centred on each cell. Converted to pixels with the image's pixel size; read as pixels if the image is uncalibrated (a warning is logged) |
| Min cells per window | 10 | Cells whose window holds fewer centroids (the centre cell included) are left out of the clustering |
| Homology dimension | 1 | 0 = connected components, 1 = loops; the alpha complex of points in the plane has nothing above 1 |
| Vectorization | PI | `PI` persistence image, `PL` persistence silhouette |
| Vector resolution | 20 | |
| Worker processes | -1 | All cores; 1 = serial |
| Number of clusters | 4 | |
| Linkage | ward | `ward`, `average`, `complete`, `single` |
| Min ROI coverage | 0.5 | Cells whose window lies mostly outside the annotation are left out of the clustering |
| Compute MDS embedding | on | 2D and 3D metric MDS of the L2 distance matrix, plus the viewer and plot files |
| Max windows for MDS | 3000 | At least 2. With more clustered cells a random (seeded) subsample of this many is embedded. Time and memory grow as n^2: about 3 s for 1000, 30 s for 3000, 90 s and 2.3 GB for 5000 |
| Set cell classes to PHC clusters | off | See above. Clear restores the original classes |
| Spatially constrained clustering | off | Cluster on the Delaunay graph of the cell centroids, so clusters are contiguous regions |
| Max Delaunay edge length | 0 µm | Spatial clustering only: longer edges are dropped; 0 = no limit, must be >= 0 |

The filtration is always the alpha complex of the window's cell centroids. A cell's centroid is
the centroid of its nucleus when it has one (cells from **Cell detection**), otherwise of its
outline. Any image type works: PHC reads detections, not pixels.

## Limits and caveats

- Clustering above 10,000 cells is fitted on a subsample (see above); agglomerative clustering
  needs O(n^2) memory.
- Run time grows with the number of cells and the cells per window; large windows on a dense
  annotation can take minutes.
- The ROI mask used for coverage is drawn at most 4096 px a side, so coverage of very large
  annotations is approximate at the edges.
- Cells are only counted where their centroid lies: a cell straddling a window edge belongs to
  the window holding its centroid.
- The time estimate covers the persistence stage, which dominates run time; clustering and MDS
  show an indeterminate bar.
- MDS needs the full n x n distance matrix and is fitted twice (2D and 3D), so it grows as
  O(n^2): about 3 s for 1000 cells, 30 s for 3000 (the default limit) and 90 s with 2.3 GB of
  memory for 5000. Lower *Max windows for MDS* for speed; raise it only if you need every
  cell in the plot.
- MDS axes have no units and no fixed orientation: only distances between points mean
  something. High stress means the plot distorts the distances; trust the 3D view more then.

## Build

Only needed after changing the Java code; a ready-built jar is in `dist/`. Requires JDK 25+
and an installed QuPath 0.7 (its jars are the compile classpath). After a rebuild, copy
`build/libs/qupath-extension-phc-<VERSION>.jar` into `dist/` so the two stay in step.

```bash
./build.sh          # -> build/libs/qupath-extension-phc-0.6.1.jar
./build.sh test     # also runs the headless end-to-end check on synthetic cells (PHCPipelineCheck),
                    # per-cell (and, through the scripting API, tiled) windows, plain and
                    # Delaunay-constrained clustering
./build.sh guitest  # also opens the progress dialog and MDS viewer; saves build/test/progress_dialog.png,
                    # build/test/mds_2d_view.png, mds_3d_view.png, mds_cells_2d_view.png and
                    # mds_cells_3d_view.png
```

`build.sh` documents the `JAVA_HOME`, `QUPATH_APP`, `PHC_PYTHON` and `PHC_LIBRARY` overrides.
For example, on macOS with an installed JDK 25 and the `phc` conda environment:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 25) \
QUPATH_APP=/Applications/QuPath-0.7.0-arm64.app \
PHC_PYTHON=~/miniconda3/envs/phc/bin/python ./build.sh test
```

`PHC_LIBRARY` defaults to the repository root (the folder above this one).

The tiled-window mode is still available to scripts and tests through `PHCParameters`
(`mode = "windows"`) and `phc_bridge.py --mode windows`; only the dialog no longer offers it.
