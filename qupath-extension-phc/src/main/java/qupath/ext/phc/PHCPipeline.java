/*
 * GUI-free PHC run: export an annotation's cells, run the Python bridge and put the per-cell
 * results on the cells; also clears them again.
 *
 * Contents
 * --------
 * PHCPipeline : class
 *     Runs PHC on one annotation and measures (optionally classifies) each cell.
 * PHCPipeline.Result : record
 *     The per-cell results of one run plus its MDS summary and plot files.
 * PHCPipeline.CellAssignment : record
 *     One exported cell paired with the bridge's result for the window centred on it.
 * PHCPipeline.ClearedResults : record
 *     How many legacy tiles were removed and cells lost their PHC results.
 */

package qupath.ext.phc;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

/**
 * Runs the whole PHC analysis for one annotation without touching the GUI, so it can be used
 * from the menu command, scripts or tests. PHC is the alpha persistence of the centroids of
 * the cells detected inside the annotation, in one window centred on each cell. Each cell
 * gets its window's neighbour count, coverage, agglomerative cluster, L2 measures and (when
 * MDS ran) its 2D and 3D MDS coordinates as measurements (see {@link #applyCellResults}).
 */
public final class PHCPipeline {

    private static final Logger logger = LoggerFactory.getLogger(PHCPipeline.class);

    /**
     * Prefix of the classes given to classified cells; also finds the heatmap tiles that
     * versions before 0.6.2 made, so Clear can remove them.
     */
    public static final String CLASS_PREFIX = "PHC cluster ";

    /** Cluster number, 1 = lowest mean L2 norm. */
    public static final String MEASUREMENT_CLUSTER = "PHC: cluster";
    /** L2 norm of the window's persistence vector (amount of topological signal). */
    public static final String MEASUREMENT_L2_NORM = "PHC: L2 norm";
    /** L2 distance from the window's vector to the ROI's mean vector (how atypical it is). */
    public static final String MEASUREMENT_L2_TO_MEAN = "PHC: L2 distance to ROI mean";
    /** Fraction of the cell's window that lies inside the annotation. */
    public static final String MEASUREMENT_COVERAGE = "PHC: ROI coverage";
    /** Centroids in the window centred on the cell, the cell included. */
    public static final String MEASUREMENT_CELLS_IN_WINDOW = "PHC: cells in window";
    /** 2D metric MDS coordinates of the window's persistence vector (cells that were embedded). */
    public static final String MEASUREMENT_MDS2_X = "PHC: MDS2 x";
    public static final String MEASUREMENT_MDS2_Y = "PHC: MDS2 y";
    /** 3D metric MDS coordinates of the window's persistence vector. */
    public static final String MEASUREMENT_MDS3_X = "PHC: MDS3 x";
    public static final String MEASUREMENT_MDS3_Y = "PHC: MDS3 y";
    public static final String MEASUREMENT_MDS3_Z = "PHC: MDS3 z";
    /** Kruskal stress-1 of the 2D / 3D embedding, stored on the annotation. */
    public static final String MEASUREMENT_CELLS_MDS2_STRESS = "PHC cells: MDS2 stress";
    public static final String MEASUREMENT_CELLS_MDS3_STRESS = "PHC cells: MDS3 stress";
    /** Annotation stress of tiled runs before 0.6.2; only removed, by Clear. */
    public static final List<String> LEGACY_TILE_STRESS = List.of("PHC: MDS2 stress",
            "PHC: MDS3 stress");
    /** Spatially constrained runs: undirected edges of the Delaunay adjacency used. */
    public static final String MEASUREMENT_CELLS_DELAUNAY_EDGES = "PHC cells: Delaunay edges";
    /** Connected parts of the Delaunay graph before Python joined them at their closest cells. */
    public static final String MEASUREMENT_CELLS_DELAUNAY_COMPONENTS =
            "PHC cells: Delaunay components";

    /**
     * Object metadata key holding a cell's class from before PHC classified it ("" for
     * unclassified), so Clear can restore it; saved with the project like other metadata.
     */
    public static final String ORIGINAL_CLASS_KEY = "phc.originalClass";

    /** Every measurement a run may put on a cell, removed by Clear. */
    public static final List<String> CELL_MEASUREMENTS = List.of(MEASUREMENT_CLUSTER,
            MEASUREMENT_L2_NORM, MEASUREMENT_L2_TO_MEAN, MEASUREMENT_COVERAGE,
            MEASUREMENT_CELLS_IN_WINDOW, MEASUREMENT_MDS2_X, MEASUREMENT_MDS2_Y,
            MEASUREMENT_MDS3_X, MEASUREMENT_MDS3_Y, MEASUREMENT_MDS3_Z);

    /** Measurements only clustered cells carry; removed from cells left out of a new run. */
    private static final List<String> CLUSTERED_CELL_MEASUREMENTS = List.of(MEASUREMENT_CLUSTER,
            MEASUREMENT_L2_NORM, MEASUREMENT_L2_TO_MEAN, MEASUREMENT_MDS2_X, MEASUREMENT_MDS2_Y,
            MEASUREMENT_MDS3_X, MEASUREMENT_MDS3_Y, MEASUREMENT_MDS3_Z);

    private static final String UNCLASSIFIED = "";  // ORIGINAL_CLASS_KEY value for no class

    /** Folder inside the QuPath project where the Python MDS plots and CSV are saved. */
    public static final String PLOT_FOLDER = "PHC plots";

    private static final Pattern UNSAFE_FILENAME_CHARS = Pattern.compile("[^A-Za-z0-9._-]+");
    private static final String FALLBACK_PREFIX = "phc";  // when nothing usable is left

    /** Viridis anchor colours (dark purple to yellow) used as a sequential cluster ramp. */
    private static final int[][] RAMP = {
            {68, 1, 84}, {59, 82, 139}, {33, 145, 140}, {94, 201, 98}, {253, 231, 37}};

    private static final int NOT_CLUSTERED = -1;  // bridge label for cells left out

    private PHCPipeline() {
    }

    /**
     * Exports the annotation's cells and runs PHC on them through the Python bridge, reporting
     * each stage to a listener (e.g. a progress bar). Pairs each cell with its result, keeps
     * the MDS summary and optionally asks Python to save its MDS plots and CSV. Does not modify
     * the hierarchy; use {@link #applyCellResults} for that.
     *
     * @param imageData (ImageData of BufferedImage) Image the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @param params (PHCParameters) Validated settings.
     * @param bridge (PythonBridge) Bridge configured with the user's Python.
     * @param listener (PythonBridge.ProgressListener) Receives stage and cell-count updates.
     * @param plots (PythonBridge.PlotTarget) Existing folder and prefix for the plot files, or
     *        null to write none.
     * @return (Result) Cell assignments, the MDS summary (null when MDS did not run) and plots.
     * @throws IOException When the cells cannot be written or Python fails.
     * @throws IllegalArgumentException When the annotation holds no detected cells (run cell
     *         detection first) or the settings are invalid.
     * @throws InterruptedException When interrupted while waiting for Python.
     * @throws java.util.concurrent.CancellationException When the bridge was cancelled.
     */
    public static Result analyse(ImageData<BufferedImage> imageData, PathObject annotation,
                                 PHCParameters params, PythonBridge bridge,
                                 PythonBridge.ProgressListener listener,
                                 PythonBridge.PlotTarget plots)
            throws IOException, InterruptedException {
        params.validate();
        listener.update(ProgressTracker.EXPORT, 0, 0);
        CellExporter.ExportedCells cells = CellExporter.export(imageData.getHierarchy(),
                annotation);
        double pixelSize = PHCParameters.pixelSizeMicrons(
                imageData.getServer().getPixelCalibration());
        if (!imageData.getServer().getPixelCalibration().hasPixelSizeMicrons()) {
            logger.warn("The image has no pixel size; PHC window size and edge length are read "
                    + "as pixels");
        }
        PythonBridge.BridgeResult bridgeResult = bridge.run(cells, params, pixelSize, listener,
                plots);
        listener.update(ProgressTracker.CELLS, 0, 0);
        List<CellAssignment> assignments = assignCells(bridgeResult, cells);
        boolean wrotePlots = plots != null && bridgeResult.mds() != null;
        Result result = new Result(assignments, bridgeResult.nClusters(),
                bridgeResult.nCellsClustered(), bridgeResult.nSkipped(),
                bridgeResult.clustering(), bridgeResult.mds(), wrotePlots ? plots : null,
                bridgeResult.warnings());
        return result;
    }

    /**
     * Pairs each exported cell with its per-cell result. The bridge has already checked that
     * entry i belongs to cell i (count, index and UUID).
     *
     * @param result (PythonBridge.BridgeResult) Per-cell results from the bridge.
     * @param cells (CellExporter.ExportedCells) What the bridge analysed, in export order.
     * @return (List of CellAssignment) One assignment per exported cell, in export order.
     */
    static List<CellAssignment> assignCells(PythonBridge.BridgeResult result,
                                            CellExporter.ExportedCells cells) {
        List<CellAssignment> assignments = new ArrayList<>();
        for (int i = 0; i < cells.cells().size(); i++) {
            assignments.add(new CellAssignment(cells.cells().get(i), result.cells().get(i)));
        }
        return assignments;
    }

    /**
     * Writes a run's results onto its cells as measurements, optionally sets their
     * classes to the PHC clusters, and fires the hierarchy events so QuPath redraws them.
     * Clustered cells get cluster, L2 norm, L2 distance to ROI mean, ROI coverage, cells in
     * window and (when embedded) the five MDS measurements; cells left out (cluster -1) get
     * coverage and cells in window only, and lose PHC values from earlier runs. Call it on the
     * JavaFX thread when QuPath shows the hierarchy.
     *
     * <p>Classes: with classifyCells, each clustered cell gets "PHC cluster k" and its earlier
     * class is saved under {@link #ORIGINAL_CLASS_KEY} (kept from the first PHC run, so a
     * re-run does not overwrite it). Cells PHC classified before but that are not classified
     * now (left out, or classifyCells off) get their original class back.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy holding the cells.
     * @param result (Result) Result from {@link #analyse}.
     * @param classifyCells (boolean) Whether to set cell classes to the PHC clusters.
     * @return (int) Number of cells whose class changed.
     */
    public static int applyCellResults(PathObjectHierarchy hierarchy, Result result,
                                       boolean classifyCells) {
        List<PathClass> classes = clusterClasses(result.nClusters());
        List<PathObject> measured = new ArrayList<>();
        List<PathObject> reclassified = new ArrayList<>();
        for (CellAssignment assignment : result.cells()) {
            PathObject cell = assignment.cell();
            PythonBridge.CellResult values = assignment.result();
            boolean clustered = values.cluster() != NOT_CLUSTERED;
            try (MeasurementList measurements = cell.getMeasurementList()) {
                measurements.removeAll(CLUSTERED_CELL_MEASUREMENTS.toArray(new String[0]));
                measurements.put(MEASUREMENT_COVERAGE, values.coverage());
                measurements.put(MEASUREMENT_CELLS_IN_WINDOW, values.nCells());
                if (clustered) {
                    measurements.put(MEASUREMENT_CLUSTER, values.cluster() + 1);
                    measurements.put(MEASUREMENT_L2_NORM, values.l2Norm());
                    measurements.put(MEASUREMENT_L2_TO_MEAN, values.l2ToMean());
                    if (values.mds2() != null && values.mds3() != null) {
                        measurements.put(MEASUREMENT_MDS2_X, values.mds2()[0]);
                        measurements.put(MEASUREMENT_MDS2_Y, values.mds2()[1]);
                        measurements.put(MEASUREMENT_MDS3_X, values.mds3()[0]);
                        measurements.put(MEASUREMENT_MDS3_Y, values.mds3()[1]);
                        measurements.put(MEASUREMENT_MDS3_Z, values.mds3()[2]);
                    }
                }
            }
            measured.add(cell);
            PathClass before = cell.getPathClass();
            if (classifyCells && clustered) {
                cell.getMetadata().putIfAbsent(ORIGINAL_CLASS_KEY,
                        before == null ? UNCLASSIFIED : before.toString());
                cell.setPathClass(classes.get(values.cluster()));
            } else {
                restoreClass(cell);
            }
            if (cell.getPathClass() != before) {
                reclassified.add(cell);
            }
        }
        hierarchy.fireObjectMeasurementsChangedEvent(PHCPipeline.class, measured);
        if (!reclassified.isEmpty()) {
            hierarchy.fireObjectClassificationsChangedEvent(PHCPipeline.class, reclassified);
        }
        int nReclassified = reclassified.size();
        return nReclassified;
    }

    /**
     * Removes every PHC result of an annotation: the PHC measurements of its cells (classes
     * set by PHC are restored), its MDS stress and Delaunay summary, and what versions before
     * 0.6.2 left (heatmap tiles and their stress measurements). Fires the hierarchy events;
     * call it on the JavaFX thread when QuPath shows the hierarchy.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation whose results are removed.
     * @return (ClearedResults) Legacy tiles removed, cells that had PHC measurements, and cells
     *         whose class came back.
     */
    public static ClearedResults clearResults(PathObjectHierarchy hierarchy,
                                              PathObject annotation) {
        List<PathObject> tiles = legacyTiles(annotation);
        hierarchy.removeObjects(tiles, true);
        List<PathObject> cleaned = new ArrayList<>();
        List<PathObject> restored = new ArrayList<>();
        for (PathObject cell : CellExporter.cellsInside(hierarchy, annotation)) {
            boolean hadMeasurements;
            try (MeasurementList measurements = cell.getMeasurementList()) {
                hadMeasurements = CELL_MEASUREMENTS.stream().anyMatch(measurements::containsKey);
                if (hadMeasurements) {
                    measurements.removeAll(CELL_MEASUREMENTS.toArray(new String[0]));
                }
            }
            if (hadMeasurements) {
                cleaned.add(cell);
            }
            if (restoreClass(cell)) {
                restored.add(cell);
            }
        }
        if (!cleaned.isEmpty()) {
            hierarchy.fireObjectMeasurementsChangedEvent(PHCPipeline.class, cleaned);
        }
        if (!restored.isEmpty()) {
            hierarchy.fireObjectClassificationsChangedEvent(PHCPipeline.class, restored);
        }
        storeMdsSummary(annotation, null);
        storeClusteringSummary(annotation, null);
        try (MeasurementList measurements = annotation.getMeasurementList()) {
            measurements.removeAll(LEGACY_TILE_STRESS.toArray(new String[0]));
        }
        hierarchy.fireObjectMeasurementsChangedEvent(PHCPipeline.class, List.of(annotation));
        ClearedResults cleared = new ClearedResults(tiles.size(), cleaned.size(),
                restored.size());
        return cleared;
    }

    /**
     * Gives a cell back the class it had before PHC classified it, and forgets it.
     *
     * @param cell (PathObject) Cell that may carry {@link #ORIGINAL_CLASS_KEY}.
     * @return (boolean) True when an original class was stored (and is now restored).
     */
    private static boolean restoreClass(PathObject cell) {
        Map<String, String> metadata = cell.getMetadata();
        String original = metadata.remove(ORIGINAL_CLASS_KEY);
        if (original != null) {
            cell.setPathClass(original.equals(UNCLASSIFIED) ? null
                    : PathClass.fromString(original));
        }
        boolean wasStored = original != null;
        return wasStored;
    }

    /**
     * Finds the cells of an annotation that carry a PHC cluster, e.g. to plot their MDS
     * embedding again after a project is reopened.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation PHC ran on.
     * @return (List of PathObject) Cells inside the annotation with
     *         {@link #MEASUREMENT_CLUSTER} and {@link #MEASUREMENT_CELLS_IN_WINDOW}.
     */
    public static List<PathObject> clusteredCells(PathObjectHierarchy hierarchy,
                                                  PathObject annotation) {
        List<PathObject> clustered = CellExporter.cellsInside(hierarchy, annotation).stream()
                .filter(cell -> cell.getMeasurementList().containsKey(MEASUREMENT_CLUSTER)
                        && cell.getMeasurementList().containsKey(MEASUREMENT_CELLS_IN_WINDOW))
                .toList();
        return clustered;
    }

    /**
     * Finds the heatmap tiles that tiled runs of versions before 0.6.2 put under an
     * annotation, so Clear can remove them.
     *
     * @param annotation (PathObject) Annotation that may hold old PHC tiles.
     * @return (List of PathObject) Child tiles whose class starts with {@link #CLASS_PREFIX}.
     */
    public static List<PathObject> legacyTiles(PathObject annotation) {
        Collection<PathObject> children = annotation.getChildObjects();
        List<PathObject> tiles = children.stream()
                .filter(PathObject::isTile)
                .filter(child -> child.getPathClass() != null
                        && child.getPathClass().getName().startsWith(CLASS_PREFIX))
                .toList();
        return tiles;
    }

    /**
     * Picks out the cells that carry MDS coordinates, e.g. to plot the embedding again after a
     * project is reopened.
     *
     * @param cells (Collection of PathObject) Cells, e.g. from {@link #clusteredCells}.
     * @return (List of PathObject) Cells with all five MDS measurements, in the same order.
     */
    public static List<PathObject> embeddedCells(Collection<PathObject> cells) {
        List<PathObject> embedded = cells.stream().filter(cell -> {
            MeasurementList m = cell.getMeasurementList();
            return m.containsKey(MEASUREMENT_MDS2_X) && m.containsKey(MEASUREMENT_MDS2_Y)
                    && m.containsKey(MEASUREMENT_MDS3_X) && m.containsKey(MEASUREMENT_MDS3_Y)
                    && m.containsKey(MEASUREMENT_MDS3_Z);
        }).toList();
        return embedded;
    }

    /**
     * Stores the MDS stress of a run on the annotation ("PHC cells: MDS2 stress" and
     * "PHC cells: MDS3 stress"); removes the old values when MDS did not run.
     *
     * @param annotation (PathObject) Annotation PHC ran on.
     * @param mds (PythonBridge.MdsResult) Summary of the embedding, or null.
     * @return (void)
     */
    public static void storeMdsSummary(PathObject annotation, PythonBridge.MdsResult mds) {
        try (MeasurementList measurements = annotation.getMeasurementList()) {
            measurements.remove(MEASUREMENT_CELLS_MDS2_STRESS);
            measurements.remove(MEASUREMENT_CELLS_MDS3_STRESS);
            if (mds != null && mds.stress2d() != null) {
                measurements.put(MEASUREMENT_CELLS_MDS2_STRESS, mds.stress2d());
            }
            if (mds != null && mds.stress3d() != null) {
                measurements.put(MEASUREMENT_CELLS_MDS3_STRESS, mds.stress3d());
            }
        }
    }

    /**
     * Stores how a run's clustering was constrained on the annotation: the Delaunay
     * edge and component counts of a spatially constrained run, so they are kept with the
     * project; removes earlier values when the run was not spatial (or clustering is null).
     *
     * @param annotation (PathObject) Annotation PHC ran on.
     * @param clustering (PythonBridge.ClusteringResult) Clustering summary of the run, or null.
     * @return (void)
     */
    public static void storeClusteringSummary(PathObject annotation,
                                              PythonBridge.ClusteringResult clustering) {
        try (MeasurementList measurements = annotation.getMeasurementList()) {
            measurements.remove(MEASUREMENT_CELLS_DELAUNAY_EDGES);
            measurements.remove(MEASUREMENT_CELLS_DELAUNAY_COMPONENTS);
            if (clustering != null && clustering.spatial()) {
                if (clustering.nEdges() != null) {
                    measurements.put(MEASUREMENT_CELLS_DELAUNAY_EDGES, clustering.nEdges());
                }
                if (clustering.nComponents() != null) {
                    measurements.put(MEASUREMENT_CELLS_DELAUNAY_COMPONENTS,
                            clustering.nComponents());
                }
            }
        }
    }

    /**
     * Gives where the MDS plots of an annotation are saved inside a project, with a file name
     * prefix that is safe on every operating system.
     *
     * @param projectDir (Path) Folder of the QuPath project.
     * @param imageName (String) Name of the image, an extension is dropped.
     * @param annotation (PathObject) Annotation PHC ran on; its name is used, or its id when it
     *        has none.
     * @return (PythonBridge.PlotTarget) "&lt;projectDir&gt;/PHC plots" and
     *         "&lt;image&gt;_&lt;annotation&gt;" (the folder is not created here).
     */
    public static PythonBridge.PlotTarget plotTarget(Path projectDir, String imageName,
                                                     PathObject annotation) {
        String annotationName = annotation.getName() == null || annotation.getName().isBlank()
                ? annotation.getID().toString() : annotation.getName();
        String image = imageName == null ? "" : GeneralTools.stripExtension(imageName);
        String prefix = sanitizeFileName(image + "_" + annotationName);
        PythonBridge.PlotTarget target = new PythonBridge.PlotTarget(
                projectDir.resolve(PLOT_FOLDER), prefix);
        return target;
    }

    /**
     * Replaces characters that are unsafe in file names, so image and annotation names can
     * name plot files.
     *
     * @param name (String) Raw name, may contain spaces, slashes or other symbols.
     * @return (String) Name made of letters, digits, '.', '_' and '-' only; "phc" if empty.
     */
    static String sanitizeFileName(String name) {
        String safe = UNSAFE_FILENAME_CHARS.matcher(name).replaceAll("_")
                .replaceAll("^[_.]+|_+$", "");
        String fileName = safe.isEmpty() ? FALLBACK_PREFIX : safe;
        return fileName;
    }

    /**
     * Creates one class per cluster, coloured along a sequential ramp so that cluster 1 (lowest
     * mean L2 norm) is dark and the highest cluster is bright, like a heatmap.
     *
     * @param nClusters (int) Number of clusters in this run.
     * @return (List of PathClass) Class for each cluster label, indexed from 0.
     */
    private static List<PathClass> clusterClasses(int nClusters) {
        List<PathClass> classes = new ArrayList<>();
        for (int k = 0; k < nClusters; k++) {
            int color = clusterColor(k, nClusters);
            PathClass pathClass = PathClass.fromString(CLASS_PREFIX + (k + 1), color);
            pathClass.setColor(color);  // classes are cached, so refresh the colour for this k
            classes.add(pathClass);
        }
        return classes;
    }

    /**
     * Gives the colour of a cluster on the viridis ramp, the same for classified cells, the MDS
     * viewer and Python's plots.
     *
     * @param cluster (int) Cluster label, 0-based (0 = lowest mean L2 norm).
     * @param nClusters (int) Number of clusters in the run, at least 1.
     * @return (int) Packed RGB colour: dark purple for cluster 0, yellow for the last.
     */
    public static int clusterColor(int cluster, int nClusters) {
        double position = nClusters == 1 ? 1.0 : (double) cluster / (nClusters - 1);
        int color = rampColor(position);
        return color;
    }

    /**
     * Interpolates the viridis ramp.
     *
     * @param position (double) Position along the ramp, 0 = dark purple, 1 = yellow.
     * @return (int) Packed RGB colour.
     */
    private static int rampColor(double position) {
        double scaled = position * (RAMP.length - 1);
        int lower = (int) Math.floor(scaled);
        int upper = Math.min(lower + 1, RAMP.length - 1);
        double t = scaled - lower;
        int[] rgb = new int[3];
        for (int c = 0; c < 3; c++) {
            rgb[c] = (int) Math.round(RAMP[lower][c] + t * (RAMP[upper][c] - RAMP[lower][c]));
        }
        int color = ColorTools.packRGB(rgb[0], rgb[1], rgb[2]);
        return color;
    }

    /**
     * The outcome of one PHC run, for callers that show the MDS embedding as well as the cells.
     *
     * @param cells (List of CellAssignment) Each exported cell with its result, not yet applied
     *        (see {@link #applyCellResults}).
     * @param nClusters (int) Number of clusters actually used.
     * @param nClustered (int) Cells that were clustered.
     * @param nSkipped (int) Cells without a usable centroid.
     * @param clustering (PythonBridge.ClusteringResult) Whether the clustering was fitted on a
     *        subsample or constrained to the Delaunay graph.
     * @param mds (PythonBridge.MdsResult) Summary of the MDS embedding, or null when MDS was
     *        switched off or skipped.
     * @param plots (PythonBridge.PlotTarget) Where Python saved its plots and CSV, or null
     *        when none were requested or MDS did not run.
     * @param warnings (List of String) Non-fatal problems Python reported, e.g. plots that
     *        could not be written; empty when there were none.
     */
    public record Result(List<CellAssignment> cells, int nClusters, int nClustered,
                         int nSkipped,
                         PythonBridge.ClusteringResult clustering, PythonBridge.MdsResult mds,
                         PythonBridge.PlotTarget plots, List<String> warnings) {
    }

    /**
     * One exported cell and the bridge's result for the window centred on it.
     *
     * @param cell (PathObject) The cell, as exported (same object as in the hierarchy).
     * @param result (PythonBridge.CellResult) Its coverage, neighbour count, L2 measures,
     *        cluster and MDS coordinates.
     */
    public record CellAssignment(PathObject cell, PythonBridge.CellResult result) {
    }

    /**
     * What {@link #clearResults} changed.
     *
     * @param nTiles (int) Heatmap tiles of versions before 0.6.2 that were removed.
     * @param nCleaned (int) Cells that had PHC measurements removed.
     * @param nRestored (int) Cells whose class from before PHC was restored.
     */
    public record ClearedResults(int nTiles, int nCleaned, int nRestored) {
    }
}
