/*
 * Headless end-to-end check of the PHC extension on synthetic cell detections and the real bridge.
 *
 * Contents
 * --------
 * PHCPipelineCheck : class
 *     Runs cell export, Python bridge, tiling, per-cell windows (plain and Delaunay-constrained
 *     clustering), MDS and hierarchy updates and asserts on them.
 */

package qupath.ext.phc;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

import qupath.lib.analysis.DelaunayTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/**
 * Checks the GUI-free half of the extension end to end, so a broken cell export, bridge call
 * or tile layout shows up without opening QuPath. Uses {@link SyntheticSlide}: ring-arranged
 * cells on the left of the annotation and random cells on the right.
 */
public final class PHCPipelineCheck {

    private static final float OVERLAY_ALPHA = 0.55f;
    /** macOS system Python, which lacks PHC's dependencies; what "python3" means from Finder. */
    private static final String SYSTEM_PYTHON = "/usr/bin/python3";
    private static final double MIN_PURITY = 0.9;    // ring / random split of a 2-cluster run
    private static final double TOLERANCE = 1e-6;
    private static final double MIN_MDS_SEPARATION = 0.8;  // nearest-centroid accuracy in MDS2
    private static final int SUBSAMPLE_WINDOWS = 20;       // fewer than the 50 default windows
    private static final String PLOT_PREFIX = "synthetic_ring_random";
    private static final double MIN_CELL_ACCURACY = 0.8;   // ring / random split of cells
    private static final int CELL_MDS_MAX = 500;            // per-cell MDS subsample, for speed
    private static final double CENTROID_TOLERANCE = 0.05;  // px; GeoJSON rounds to 0.01 px
    private static final int N_PRECLASSIFIED = 25;          // cells given a class before PHC
    private static final String PRECLASS = "Tumor";
    private static final double MAX_EDGE_MICRONS = 15;      // 30 px: drops the longest edges
    private static final int SAMPLE_EDGES = 9412;           // notification formatting check
    private static final int SAMPLE_COMPONENTS = 3;

    /** Default settings: 100 um windows and stride, PI, 10 cells, 4 clusters. */
    private static final PHCParameters DEFAULTS = tiledParams(1, 100, 100, "PI", 4, "ward",
            true, PHCParameters.DEFAULT_MDS_MAX_WINDOWS);
    /** 100 um = 200 px windows at 0.5 um per pixel. */
    private static final double DEFAULT_WINDOW_PX = 200;
    /** Listener for runs whose progress is not checked. */
    private static final PythonBridge.ProgressListener NO_PROGRESS = (stage, done, total) -> { };

    private static int failures = 0;

    private PHCPipelineCheck() {
    }

    /**
     * Runs every check and exits non-zero if any fail.
     *
     * @param args (String[]) Python executable, PHC library folder, and the folder to write
     *        the test image and rendered heatmaps to.
     * @return (void)
     * @throws Exception When the test image cannot be written or opened.
     */
    public static void main(String[] args) throws Exception {
        PythonBridge bridge = new PythonBridge(args[0], args[1]);
        File outDir = new File(args[2]);
        ImageData<BufferedImage> imageData = SyntheticSlide.createImageData(outDir);
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        PathObject annotation = SyntheticSlide.addAnnotationWithCells(imageData);
        PathObject empty = SyntheticSlide.addEmptyAnnotation(imageData);
        ROI roi = annotation.getROI();

        checkSettings(imageData);
        List<PathObject> cells = checkExport(hierarchy, annotation, empty);

        List<PathObject> tiles = timedRun(imageData, annotation, DEFAULTS, bridge,
                "100 um windows, PI, 4 clusters");
        checkTiles(tiles, roi, DEFAULTS, DEFAULT_WINDOW_PX, cells);
        render(cells, tiles, new File(outDir, "heatmap_alpha_4clusters.png"));

        PHCParameters twoClusters = tiledParams(1, 100, 100, "PI", 2, "ward", true,
                PHCParameters.DEFAULT_MDS_MAX_WINDOWS);
        List<PathObject> splitTiles = timedRun(imageData, annotation, twoClusters, bridge,
                "100 um windows, PI, 2 clusters");
        checkRingsVersusRandom(splitTiles);
        render(cells, splitTiles, new File(outDir, "heatmap_alpha_2clusters.png"));

        PHCParameters overlap = tiledParams(1, 100, 50, "PL", 3, "average", true,
                PHCParameters.DEFAULT_MDS_MAX_WINDOWS);
        List<PathObject> overlapTiles = timedRun(imageData, annotation, overlap, bridge,
                "100 um windows, stride 50 um, PL, 3 clusters, average");
        checkTiles(overlapTiles, roi, overlap, DEFAULT_WINDOW_PX, cells);
        render(cells, overlapTiles, new File(outDir, "heatmap_alpha_overlap_pl.png"));

        checkBridgeTotals(hierarchy, annotation, bridge);
        checkMds(imageData, annotation, bridge, outDir);
        checkPlotNames(annotation);
        checkSecondRun(imageData, annotation, tiles, bridge);
        PHCPipeline.Result perCell = checkPerCell(imageData, annotation, bridge, outDir, tiles);
        checkSpatial(imageData, annotation, bridge, outDir, perCell);
        checkCellClassification(imageData, annotation, bridge);
        checkErrors(imageData, annotation, empty);
        checkProgressTracker();
        checkProgressEvents(imageData, annotation, bridge);
        checkCancel(imageData, annotation, args[0], args[1]);
        check(new PythonBridge("~/envs/py", "").pythonExecutable()
                        .equals(System.getProperty("user.home") + "/envs/py"),
                "a ~/ Python path expands to the home folder");

        System.out.println(failures == 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED");
        System.exit(failures == 0 ? 0 : 1);
    }

    /**
     * Checks micrometre settings become slide pixels, and the dialog header names the unit.
     *
     * @param imageData (ImageData of BufferedImage) Calibrated test image.
     * @return (void)
     */
    private static void checkSettings(ImageData<BufferedImage> imageData) {
        PixelCalibration calibration = imageData.getServer().getPixelCalibration();
        double pixelSize = PHCParameters.pixelSizeMicrons(calibration);
        check(pixelSize == SyntheticSlide.PIXEL_SIZE_MICRONS, "pixel size is read from the "
                + "image calibration (" + pixelSize + " um)");
        List<String> args = DEFAULTS.toBridgeArgs(pixelSize);
        check(args.get(args.indexOf("--window-size") + 1).equals("200.0")
                        && args.get(args.indexOf("--stride") + 1).equals("200.0")
                        && args.get(args.indexOf("--min-cells") + 1).equals("10"),
                "100 um window and stride become 200 px at 0.5 um/px: " + args);
        check(args.get(args.indexOf("--mds") + 1).equals("1")
                        && args.get(args.indexOf("--mds-max-windows") + 1)
                                .equals(Integer.toString(PHCParameters.DEFAULT_MDS_MAX_WINDOWS)),
                "MDS is on by default for at most " + PHCParameters.DEFAULT_MDS_MAX_WINDOWS
                        + " windows");
        PHCParameters mdsOff = tiledParams(1, 100, 100, "PI", 4, "ward", false, 2);
        check(mdsOff.toBridgeArgs(pixelSize).get(mdsOff.toBridgeArgs(pixelSize)
                .indexOf("--mds") + 1).equals("0"), "switching MDS off passes --mds 0");
        check(PHCParameters.pixelSizeMicrons(PixelCalibration.getDefaultInstance())
                        == PHCParameters.UNCALIBRATED_PIXEL_SIZE,
                "an uncalibrated image reads window lengths as pixels");
        String header = PHCCommand.progressHeader(DEFAULTS, 1234, calibration);
        check(header.equals("Alpha PHC on 1234 cells, 100 " + GeneralTools.micrometerSymbol()
                + " windows (stride 100 " + GeneralTools.micrometerSymbol() + "), PI, 4 "
                + "clusters"), "progress header: " + header);
        check(PHCCommand.progressHeader(DEFAULTS, 1, PixelCalibration.getDefaultInstance())
                .contains("100 px windows"), "uncalibrated header uses px");

        check(args.get(args.indexOf("--mode") + 1).equals(PHCParameters.MODE_WINDOWS)
                        && !DEFAULTS.isCellMode() && !DEFAULTS.classifyCells(),
                "tiled settings pass --mode windows and do not classify cells");
        PHCParameters perCell = cellParams(4, true, false);
        List<String> cellArgs = perCell.toBridgeArgs(pixelSize);
        check(cellArgs.get(cellArgs.indexOf("--mode") + 1).equals(PHCParameters.MODE_CELLS)
                && perCell.isCellMode(), "per-cell settings pass --mode cells: " + cellArgs);
        String cellHeader = PHCCommand.progressHeader(perCell, 3200, calibration);
        check(cellHeader.equals("Alpha PHC per cell on 3200 cells, 100 "
                + GeneralTools.micrometerSymbol() + " windows centred on each cell, PI, 4 "
                + "clusters"), "per-cell progress header: " + cellHeader);
        PHCParameters fromDialog = PHCParameters.fromParameterList(
                PHCParameters.createParameterList());
        check(!fromDialog.isCellMode() && !fromDialog.classifyCells(),
                "the dialog defaults to tiled windows without classifying cells");
        check(PHCParameters.modeForLabel("Per-cell windows").equals(PHCParameters.MODE_CELLS),
                "'Per-cell windows' maps to --mode cells");
        check(!perCell.spatialClustering() && perCell.maxEdgeLength() == 0
                        && cellArgs.get(cellArgs.indexOf("--spatial") + 1).equals("0")
                        && cellArgs.get(cellArgs.indexOf("--max-edge-length") + 1).equals("0.0")
                        && args.get(args.indexOf("--spatial") + 1).equals("0"),
                "unconstrained settings pass --spatial 0 --max-edge-length 0.0");
        PHCParameters spatial = spatialParams(4, true, MAX_EDGE_MICRONS);
        spatial.validate();
        List<String> spatialArgs = spatial.toBridgeArgs(pixelSize);
        check(spatialArgs.get(spatialArgs.indexOf("--spatial") + 1).equals("1")
                        && spatialArgs.get(spatialArgs.indexOf("--max-edge-length") + 1)
                                .equals("30.0"),
                "spatial settings pass --spatial 1 and a 15 um edge limit as 30 px: "
                        + spatialArgs);
        check(PHCCommand.progressHeader(spatial, 3200, calibration).endsWith(
                ", 4 clusters, Delaunay-constrained"), "spatial progress header: "
                + PHCCommand.progressHeader(spatial, 3200, calibration));
        check(!fromDialog.spatialClustering() && fromDialog.maxEdgeLength() == 0,
                "the dialog defaults to unconstrained clustering with no edge limit");
        try {
            new PHCParameters(1, 100, 100, "PI", 20, 10, 4, "ward", 0.5, -1, true, 2,
                    PHCParameters.MODE_WINDOWS, false, true, 0).validate();
            check(false, "spatial clustering is rejected for tiled windows");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("Per-cell windows"), "spatial clustering is rejected "
                    + "for tiled windows: " + e.getMessage());
        }
        try {
            spatialParams(4, true, -1).validate();
            check(false, "a negative max edge length is rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains(">= 0"), "a negative max edge length is rejected: "
                    + e.getMessage());
        }
        try {
            new PHCParameters(1, 100, 100, "PI", 20, 10, 4, "ward", 0.5, -1, true, 2, "tiles",
                    false, false, 0).validate();
            check(false, "an unknown mode is rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("cells"), "an unknown mode lists the modes: "
                    + e.getMessage());
        }
    }

    /**
     * Builds tiled-window settings from the values the checks vary, with the rest fixed
     * (vector resolution 20, min 10 cells, min coverage 0.5, all cores, cell classes left
     * alone, no spatial clustering). Package-private so the dialog check can share it.
     *
     * @param dimension (int) Persistent homology dimension.
     * @param windowSize (double) Side of each square window, in um.
     * @param stride (double) Step between windows, in um.
     * @param vectorization (String) "PI" or "PL", or an unknown value to test validation.
     * @param nClusters (int) Number of clusters.
     * @param linkage (String) Agglomerative linkage, e.g. "ward".
     * @param computeMds (boolean) Whether to compute the MDS embedding.
     * @param mdsMaxWindows (int) Most windows embedded by MDS.
     * @return (PHCParameters) Tiled-window settings, not yet validated.
     */
    static PHCParameters tiledParams(int dimension, double windowSize, double stride,
                                     String vectorization, int nClusters, String linkage,
                                     boolean computeMds, int mdsMaxWindows) {
        PHCParameters params = new PHCParameters(dimension, windowSize, stride, vectorization,
                20, 10, nClusters, linkage, 0.5, -1, computeMds, mdsMaxWindows,
                PHCParameters.MODE_WINDOWS, PHCParameters.DEFAULT_CLASSIFY_CELLS,
                PHCParameters.DEFAULT_SPATIAL_CLUSTERING, PHCParameters.DEFAULT_MAX_EDGE_LENGTH);
        return params;
    }

    /**
     * Builds per-cell settings: 100 um (200 px) windows centred on each cell, PI, 10 cells.
     *
     * @param nClusters (int) Number of clusters.
     * @param computeMds (boolean) Whether to compute MDS (at most {@link #CELL_MDS_MAX} cells).
     * @param classify (boolean) Whether to set cell classes to the PHC clusters.
     * @return (PHCParameters) Per-cell settings.
     */
    private static PHCParameters cellParams(int nClusters, boolean computeMds, boolean classify) {
        PHCParameters params = new PHCParameters(1, 100, 100, "PI", 20, 10, nClusters, "ward",
                0.5, -1, computeMds, CELL_MDS_MAX, PHCParameters.MODE_CELLS, classify, false,
                PHCParameters.DEFAULT_MAX_EDGE_LENGTH);
        return params;
    }

    /**
     * Checks which detections are exported: the annotation's cells, but not cells outside it,
     * and nothing for an annotation without detections.
     *
     * @param hierarchy (PathObjectHierarchy) Test hierarchy.
     * @param annotation (PathObject) Annotation holding the synthetic cells.
     * @param empty (PathObject) Annotation holding no detections.
     * @return (List of PathObject) The exported cells.
     */
    private static List<PathObject> checkExport(PathObjectHierarchy hierarchy,
                                                PathObject annotation, PathObject empty) {
        CellExporter.ExportedCells exported = CellExporter.export(hierarchy, annotation);
        check(exported.cells().size() == SyntheticSlide.nCellsInside(), "exports the "
                + SyntheticSlide.nCellsInside() + " cells inside the annotation, not the ones "
                + "outside (" + exported.cells().size() + ")");
        check(exported.maskDownsample() == 1.0 && exported.mask().getWidth()
                        == (int) SyntheticSlide.ANNOTATION_WIDTH
                        && exported.mask().getHeight() == (int) SyntheticSlide.ANNOTATION_HEIGHT,
                "a 2000 x 1000 px annotation gets a full-resolution mask");
        check(exported.originX() == SyntheticSlide.ANNOTATION_X
                && exported.originY() == SyntheticSlide.ANNOTATION_Y
                && exported.width() == SyntheticSlide.ANNOTATION_WIDTH
                && exported.height() == SyntheticSlide.ANNOTATION_HEIGHT,
                "export carries the annotation's bounding box");
        check(CellExporter.cellsInside(hierarchy, empty).isEmpty(),
                "an annotation without detections has no cells");
        List<PathObject> cells = exported.cells();
        return cells;
    }

    /**
     * Runs the pipeline and prints how long it took.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param params (PHCParameters) Settings for this run.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param label (String) Name printed with the timing.
     * @return (List of PathObject) Heatmap tiles.
     * @throws Exception When the pipeline fails.
     */
    private static List<PathObject> timedRun(ImageData<BufferedImage> imageData,
                                             PathObject annotation, PHCParameters params,
                                             PythonBridge bridge, String label)
            throws Exception {
        long start = System.nanoTime();
        List<PathObject> tiles = PHCPipeline.analyse(imageData, annotation, params, bridge,
                NO_PROGRESS, null).tiles();
        System.out.printf("  %s: %d tiles in %.1f s%n", label, tiles.size(),
                (System.nanoTime() - start) / 1e9);
        return tiles;
    }

    /**
     * Checks tile geometry, classes and measurements for one run, including that each tile's
     * cell count matches a recount of the exported centroids.
     *
     * @param tiles (List of PathObject) Tiles from the pipeline.
     * @param roi (ROI) Annotation ROI.
     * @param params (PHCParameters) Settings of the run.
     * @param windowPx (double) Expected window side in slide pixels.
     * @param cells (List of PathObject) Exported cells.
     * @return (void)
     */
    private static void checkTiles(List<PathObject> tiles, ROI roi, PHCParameters params,
                                   double windowPx, List<PathObject> cells) {
        check(!tiles.isEmpty(), "pipeline produced tiles");
        boolean insideBounds = tiles.stream().allMatch(t ->
                t.getROI().getBoundsX() >= roi.getBoundsX() - TOLERANCE
                        && t.getROI().getBoundsY() >= roi.getBoundsY() - TOLERANCE
                        && t.getROI().getBoundsX() + t.getROI().getBoundsWidth()
                                <= roi.getBoundsX() + roi.getBoundsWidth() + TOLERANCE
                        && t.getROI().getBoundsY() + t.getROI().getBoundsHeight()
                                <= roi.getBoundsY() + roi.getBoundsHeight() + TOLERANCE);
        check(insideBounds, "every tile lies within the annotation's bounding box");
        check(tiles.stream().allMatch(t -> t.getROI().getBoundsWidth() <= windowPx + TOLERANCE)
                        && tiles.stream().anyMatch(t -> Math.abs(t.getROI().getBoundsWidth()
                                - windowPx) < TOLERANCE),
                "tiles are " + windowPx + " px, the window size converted from um");
        Set<String> classes = tiles.stream().map(t -> t.getPathClass().getName())
                .collect(Collectors.toSet());
        check(classes.size() == params.nClusters(), "tiles use " + params.nClusters()
                + " cluster classes " + classes);
        boolean measured = tiles.stream().allMatch(t -> {
            var m = t.getMeasurementList();
            return m.containsKey(PHCPipeline.MEASUREMENT_L2_NORM)
                    && m.containsKey(PHCPipeline.MEASUREMENT_L2_TO_MEAN)
                    && m.containsKey(PHCPipeline.MEASUREMENT_CLUSTER)
                    && m.get(PHCPipeline.MEASUREMENT_COVERAGE) >= params.minCoverage()
                    && m.get(PHCPipeline.MEASUREMENT_CELL_COUNT) >= params.minCells();
        });
        check(measured, "every tile has L2 norm, L2-to-mean, cluster, coverage >= "
                + params.minCoverage() + " and cell count >= " + params.minCells());

        int mismatches = 0;
        for (PathObject tile : tiles) {
            ROI box = tile.getROI();
            long recount = cells.stream().map(SyntheticSlide::centroid).filter(p ->
                    p[0] >= box.getBoundsX() && p[0] < box.getBoundsX() + box.getBoundsWidth()
                            && p[1] >= box.getBoundsY()
                            && p[1] < box.getBoundsY() + box.getBoundsHeight()).count();
            if (recount != (long) tile.getMeasurementList()
                    .get(PHCPipeline.MEASUREMENT_CELL_COUNT)) {
                mismatches++;
            }
        }
        check(mismatches == 0, "'" + PHCPipeline.MEASUREMENT_CELL_COUNT + "' matches the "
                + "centroids inside each tile (" + mismatches + " mismatches)");
    }

    /**
     * Checks that a 2-cluster run separates the ring half from the random half: each cluster
     * should hold windows from mostly one half.
     *
     * @param tiles (List of PathObject) Tiles of a 2-cluster run with windows aligned to the
     *        halves.
     * @return (void)
     */
    private static void checkRingsVersusRandom(List<PathObject> tiles) {
        Map<String, int[]> sides = new HashMap<>();  // class -> {ring windows, random windows}
        for (PathObject tile : tiles) {
            int[] counts = sides.computeIfAbsent(tile.getPathClass().getName(), k -> new int[2]);
            counts[SyntheticSlide.isRingSide(tile.getROI().getCentroidX()) ? 0 : 1]++;
        }
        int majority = sides.values().stream().mapToInt(c -> Math.max(c[0], c[1])).sum();
        double purity = tiles.isEmpty() ? 0 : (double) majority / tiles.size();
        String summary = sides.entrySet().stream().map(e -> e.getKey() + "=" + e.getValue()[0]
                + " ring/" + e.getValue()[1] + " random").collect(Collectors.joining(", "));
        check(sides.size() == 2 && purity >= MIN_PURITY, String.format("ring and random "
                + "windows fall in different clusters (purity %.2f: %s)", purity, summary));
    }

    /**
     * Checks the bridge's own totals: every exported cell is read, and windows are counted.
     *
     * @param hierarchy (PathObjectHierarchy) Test hierarchy.
     * @param annotation (PathObject) Annotation with cells.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When the bridge fails.
     */
    private static void checkBridgeTotals(PathObjectHierarchy hierarchy, PathObject annotation,
                                          PythonBridge bridge) throws Exception {
        CellExporter.ExportedCells exported = CellExporter.export(hierarchy, annotation);
        PythonBridge.BridgeResult result = bridge.run(exported, DEFAULTS,
                SyntheticSlide.PIXEL_SIZE_MICRONS, NO_PROGRESS, null);
        long expectedWindows = (long) (Math.ceil(SyntheticSlide.ANNOTATION_HEIGHT
                / DEFAULT_WINDOW_PX) * Math.ceil(SyntheticSlide.ANNOTATION_WIDTH
                / DEFAULT_WINDOW_PX));
        check(result.nCells() == exported.cells().size(), "bridge reads all "
                + exported.cells().size() + " centroids (n_cells = " + result.nCells() + ")");
        check(result.windows().size() == expectedWindows
                        && result.nWindowsClustered() == expectedWindows,
                "bridge reports " + expectedWindows + " windows, all clustered ("
                        + result.windows().size() + ", " + result.nWindowsClustered() + ")");
    }

    /**
     * Checks the MDS embedding end to end: coordinates on every clustered tile, the stress
     * summary, the plot files, the subsample limit, switching MDS off, and that ring and
     * random windows land in different parts of the 2D plane.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param outDir (File) Folder the plot files are written under.
     * @return (void)
     * @throws Exception When a run fails or the plot folder cannot be created.
     */
    private static void checkMds(ImageData<BufferedImage> imageData, PathObject annotation,
                                 PythonBridge bridge, File outDir) throws Exception {
        Path plotDir = new File(outDir, PHCPipeline.PLOT_FOLDER).toPath();
        Files.createDirectories(plotDir);
        PythonBridge.PlotTarget plots = new PythonBridge.PlotTarget(plotDir, PLOT_PREFIX);
        for (Path file : plots.files(PHCParameters.MODE_WINDOWS)) {
            Files.deleteIfExists(file);
        }
        PHCPipeline.Result full = PHCPipeline.analyse(imageData, annotation, DEFAULTS, bridge,
                (stage, done, total) -> { }, plots);
        List<PathObject> embedded = PHCPipeline.embeddedTiles(full.tiles());
        PythonBridge.MdsResult mds = full.mds();
        check(!full.tiles().isEmpty() && embedded.size() == full.tiles().size(),
                "every clustered tile gets MDS2 and MDS3 coordinates (" + embedded.size()
                        + " of " + full.tiles().size() + ")");
        check(mds != null && mds.nEmbedded() == full.tiles().size() && !mds.subsampled(),
                "MDS summary embeds all windows without subsampling: " + mds);
        check(mds != null && mds.stress2d() != null && mds.stress3d() != null
                        && mds.stress2d() >= 0 && mds.stress3d() >= 0,
                "stress is reported for 2D and 3D: " + mds);
        List<Path> plotFiles = plots.files(PHCParameters.MODE_WINDOWS);
        boolean plotsWritten = full.plots() != null && plotFiles.stream().allMatch(f -> {
            try {
                return Files.size(f) > 0;
            } catch (IOException e) {
                return false;
            }
        });
        check(plotsWritten && full.warnings().isEmpty(), "Python wrote the MDS plots and CSV "
                + plotFiles + " without warnings " + full.warnings());

        Path notADir = plotDir.resolve("not_a_folder.txt");
        Files.writeString(notADir, "plots cannot go inside a file");
        PHCPipeline.Result unwritable = PHCPipeline.analyse(imageData, annotation, DEFAULTS,
                bridge, (stage, done, total) -> { },
                new PythonBridge.PlotTarget(notADir, PLOT_PREFIX));
        check(!unwritable.tiles().isEmpty() && !unwritable.warnings().isEmpty(),
                "an unwritable plot folder is a warning, not a failure: "
                        + unwritable.warnings());
        checkMdsSeparation(embedded);

        PHCParameters subsample = tiledParams(1, 100, 100, "PI", 4, "ward", true,
                SUBSAMPLE_WINDOWS);
        PHCPipeline.Result sub = PHCPipeline.analyse(imageData, annotation, subsample, bridge,
                (stage, done, total) -> { }, null);
        int nSub = PHCPipeline.embeddedTiles(sub.tiles()).size();
        check(nSub == SUBSAMPLE_WINDOWS && sub.mds() != null && sub.mds().subsampled()
                        && sub.mds().nEmbedded() == SUBSAMPLE_WINDOWS,
                "a limit of " + SUBSAMPLE_WINDOWS + " MDS windows embeds exactly that many of "
                        + sub.tiles().size() + " tiles (" + nSub + ", " + sub.mds() + ")");

        PHCParameters off = tiledParams(1, 100, 100, "PI", 4, "ward", false,
                PHCParameters.DEFAULT_MDS_MAX_WINDOWS);
        PHCPipeline.Result noMds = PHCPipeline.analyse(imageData, annotation, off, bridge,
                (stage, done, total) -> { }, null);
        boolean anyMds = noMds.tiles().stream().anyMatch(t -> t.getMeasurementList().keySet()
                .stream().anyMatch(k -> k.startsWith("PHC: MDS")));
        check(!noMds.tiles().isEmpty() && !anyMds && noMds.mds() == null
                        && noMds.plots() == null,
                "with MDS off no tile has MDS measurements and there is no MDS summary");
    }

    /**
     * Checks ring and random windows fall in separate regions of the 2D MDS plane: each tile
     * is assigned to the nearer of the two group centroids, and nearly all must land in their
     * own group.
     *
     * @param embedded (List of PathObject) Tiles with MDS coordinates.
     * @return (void)
     */
    private static void checkMdsSeparation(List<PathObject> embedded) {
        double[][] sums = new double[2][3];  // group -> {sum x, sum y, count}
        for (PathObject tile : embedded) {
            int group = SyntheticSlide.isRingSide(tile.getROI().getCentroidX()) ? 0 : 1;
            sums[group][0] += tile.getMeasurementList().get(PHCPipeline.MEASUREMENT_MDS2_X);
            sums[group][1] += tile.getMeasurementList().get(PHCPipeline.MEASUREMENT_MDS2_Y);
            sums[group][2]++;
        }
        int correct = 0;
        for (PathObject tile : embedded) {
            int group = SyntheticSlide.isRingSide(tile.getROI().getCentroidX()) ? 0 : 1;
            double x = tile.getMeasurementList().get(PHCPipeline.MEASUREMENT_MDS2_X);
            double y = tile.getMeasurementList().get(PHCPipeline.MEASUREMENT_MDS2_Y);
            double[] distance = new double[2];
            for (int g = 0; g < 2; g++) {
                distance[g] = Math.hypot(x - sums[g][0] / sums[g][2], y - sums[g][1] / sums[g][2]);
            }
            correct += (distance[group] <= distance[1 - group]) ? 1 : 0;
        }
        double accuracy = embedded.isEmpty() ? 0 : (double) correct / embedded.size();
        check(sums[0][2] > 0 && sums[1][2] > 0 && accuracy >= MIN_MDS_SEPARATION,
                String.format("ring and random windows separate in the MDS2 plane "
                        + "(nearest-centroid accuracy %.2f over %d tiles)", accuracy,
                        embedded.size()));
    }

    /**
     * Checks plot files get safe names built from the image and annotation names.
     *
     * @param annotation (PathObject) Annotation used for the unnamed case.
     * @return (void)
     */
    private static void checkPlotNames(PathObject annotation) {
        PathObject named = PathObjects.createAnnotationObject(
                annotation.getROI());
        named.setName("Tumour #1 / left");
        PythonBridge.PlotTarget target = PHCPipeline.plotTarget(Path.of("/projects/p1"),
                "slide: A.svs", named);
        check(target.dir().equals(Path.of("/projects/p1", PHCPipeline.PLOT_FOLDER))
                        && target.prefix().equals("slide_A_Tumour_1_left"),
                "plot files go to <project>/PHC plots with a sanitized prefix: " + target);
        String unnamed = PHCPipeline.plotTarget(Path.of("/p"), "img.tif", annotation).prefix();
        check(unnamed.equals("img_" + annotation.getID()), "an unnamed annotation uses its id: "
                + unnamed);
    }

    /**
     * Adds a run's tiles to the hierarchy as the menu command does, then checks the tiles are
     * not exported as cells and a second run gives the same windows and counts.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells.
     * @param tiles (List of PathObject) Tiles of the default run.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When the second run fails.
     */
    private static void checkSecondRun(ImageData<BufferedImage> imageData, PathObject annotation,
                                       List<PathObject> tiles, PythonBridge bridge)
            throws Exception {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        annotation.addChildObjects(tiles);
        hierarchy.fireHierarchyChangedEvent(annotation);
        check(PHCPipeline.existingTiles(annotation).size() == tiles.size(),
                "tiles are found as existing PHC tiles after adding");
        List<PathObject> cells = CellExporter.cellsInside(hierarchy, annotation);
        check(cells.size() == SyntheticSlide.nCellsInside()
                        && cells.stream().noneMatch(PathObject::isTile),
                "PHC tiles are not exported as cells on a second run (" + cells.size() + ")");

        List<PathObject> again = PHCPipeline.analyse(imageData, annotation, DEFAULTS, bridge,
                NO_PROGRESS, null).tiles();
        List<Double> firstCounts = cellCounts(tiles);
        List<Double> secondCounts = cellCounts(again);
        check(firstCounts.equals(secondCounts), "a second run gives the same tiles and cell "
                + "counts");
        hierarchy.removeObjects(PHCPipeline.existingTiles(annotation), true);
        annotation.addChildObjects(again);
        hierarchy.fireHierarchyChangedEvent(annotation);
        check(PHCPipeline.existingTiles(annotation).size() == again.size(),
                "a second run replaces the first run's tiles");
    }

    /**
     * Runs per-cell windows end to end and checks the results land on the right cells: one
     * result per exported cell, "PHC: cells in window" equal to a brute-force neighbour count
     * under the half-open rule, ring and random cells in different clusters, MDS on clustered
     * cells, the plot files, no tiles touched, and tiled results unchanged afterwards.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells (and tiles from earlier runs).
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param outDir (File) Folder the plot files are written under.
     * @param firstTiles (List of PathObject) Tiles of the first default tiled run.
     * @return (PHCPipeline.Result) The 2-cluster per-cell result, to compare later runs with.
     * @throws Exception When a run fails.
     */
    private static PHCPipeline.Result checkPerCell(ImageData<BufferedImage> imageData, PathObject annotation,
                                     PythonBridge bridge, File outDir,
                                     List<PathObject> firstTiles) throws Exception {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        List<PathObject> tilesBefore = PHCPipeline.existingTiles(annotation);
        PythonBridge.MdsResult tiledMds = PHCPipeline.analyse(imageData, annotation, DEFAULTS,
                bridge, (stage, done, total) -> { }, null).mds();
        PHCPipeline.storeMdsSummary(annotation, tiledMds, PHCParameters.MODE_WINDOWS);

        Path plotDir = new File(outDir, PHCPipeline.PLOT_FOLDER).toPath();
        Files.createDirectories(plotDir);
        PythonBridge.PlotTarget plots = new PythonBridge.PlotTarget(plotDir, PLOT_PREFIX);
        for (Path file : plots.files(PHCParameters.MODE_CELLS)) {
            Files.deleteIfExists(file);
        }
        PHCParameters params = cellParams(2, true, false);
        List<String> stages = new ArrayList<>();
        long[] persistence = {0, 0};
        long start = System.nanoTime();
        PHCPipeline.Result result = PHCPipeline.analyse(imageData, annotation, params, bridge,
                (stage, done, total) -> {
                    if (stages.isEmpty() || !stages.get(stages.size() - 1).equals(stage)) {
                        stages.add(stage);
                    }
                    if (stage.equals(ProgressTracker.PERSISTENCE)) {
                        persistence[0] = done;
                        persistence[1] = total;
                    }
                }, plots);
        System.out.printf("  per-cell 100 um windows, PI, 2 clusters: %d cells in %.1f s%n",
                result.cells().size(), (System.nanoTime() - start) / 1e9);
        int nCells = SyntheticSlide.nCellsInside();
        check(result.mode().equals(PHCParameters.MODE_CELLS) && result.tiles().isEmpty()
                        && result.cells().size() == nCells,
                "per-cell mode returns one result per exported cell (" + result.cells().size()
                        + " of " + nCells + ") and no tiles");
        check(stages.contains(ProgressTracker.CELLS) && !stages.contains(ProgressTracker.TILES)
                        && persistence[0] == nCells && persistence[1] == nCells,
                "per-cell progress counts " + persistence[0] + " / " + persistence[1]
                        + " cells, stages " + stages);
        check(result.nSkipped() == 0 && result.clustering() != null
                        && !result.clustering().subsampled()
                        && result.clustering().nFitted() == result.nClustered(),
                "no cell skipped, clustering fitted on all " + result.nClustered()
                        + " clustered cells: " + result.clustering());

        int nReclassified = PHCPipeline.applyCellResults(hierarchy, result, false);
        PHCPipeline.storeMdsSummary(annotation, result.mds(), PHCParameters.MODE_CELLS);
        List<PathObject> tilesAfter = PHCPipeline.existingTiles(annotation);
        check(tilesAfter.size() == tilesBefore.size() && tilesAfter.containsAll(tilesBefore)
                        && nReclassified == 0,
                "per-cell mode neither creates nor removes tiles (" + tilesAfter.size()
                        + ") and leaves classes alone");
        List<PathObject> exported = CellExporter.cellsInside(hierarchy, annotation);
        check(exported.stream().allMatch(c -> c.getMeasurementList()
                        .containsKey(PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW)
                        && c.getMeasurementList().containsKey(PHCPipeline.MEASUREMENT_COVERAGE)),
                "every exported cell gets '" + PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW
                        + "' and coverage");
        checkNeighbourCounts(result, params.windowSize() / SyntheticSlide.PIXEL_SIZE_MICRONS);
        checkClusteredCells(result);
        checkCellSeparation(result);

        List<PathObject> clustered = PHCPipeline.clusteredCells(hierarchy, annotation);
        List<PathObject> embedded = PHCPipeline.embeddedTiles(clustered);
        int expectedEmbedded = Math.min(result.nClustered(), CELL_MDS_MAX);
        check(clustered.size() == result.nClustered() && result.mds() != null
                        && embedded.size() == expectedEmbedded
                        && result.mds().nEmbedded() == expectedEmbedded
                        && result.mds().subsampled() == (result.nClustered() > CELL_MDS_MAX),
                "MDS coordinates on " + embedded.size() + " of " + clustered.size()
                        + " clustered cells: " + result.mds());
        var annotationMeasurements = annotation.getMeasurementList();
        check(annotationMeasurements.containsKey(PHCPipeline.MEASUREMENT_CELLS_MDS2_STRESS)
                        && annotationMeasurements.containsKey(PHCPipeline.MEASUREMENT_MDS2_STRESS)
                        && tiledMds != null && annotationMeasurements.get(
                                PHCPipeline.MEASUREMENT_MDS2_STRESS) == tiledMds.stress2d(),
                "tiled and per-cell MDS stress coexist on the annotation");
        boolean cellPlots = result.plots() != null && plots.files(PHCParameters.MODE_CELLS)
                .stream().allMatch(Files::exists);
        check(cellPlots && plots.files(PHCParameters.MODE_CELLS).get(0).getFileName()
                        .toString().equals(PLOT_PREFIX + "_cells_mds_2d.png"),
                "Python wrote the per-cell plots " + plots.files(PHCParameters.MODE_CELLS)
                        + " " + result.warnings());

        // Tiled mode after per-cell: PHC cell measurements must not reach Python
        Path geojson = Files.createTempFile("phc-cells", ".geojson");
        PythonBridge.writeCells(geojson, exported);
        String written = Files.readString(geojson);
        Files.delete(geojson);
        check(!written.contains("PHC:"), "cells with per-cell PHC measurements are exported "
                + "without them");
        List<PathObject> tiledAgain = PHCPipeline.analyse(imageData, annotation, DEFAULTS,
                bridge, NO_PROGRESS, null).tiles();
        boolean sameL2 = tiledAgain.size() == firstTiles.size();
        for (int i = 0; sameL2 && i < tiledAgain.size(); i++) {
            sameL2 = Math.abs(tiledAgain.get(i).getMeasurementList()
                    .get(PHCPipeline.MEASUREMENT_L2_NORM) - firstTiles.get(i)
                    .getMeasurementList().get(PHCPipeline.MEASUREMENT_L2_NORM)) < TOLERANCE;
        }
        check(cellCounts(tiledAgain).equals(cellCounts(firstTiles)) && sameL2,
                "a tiled run after a per-cell run gives the same tiles, counts and L2 norms");

        try {
            List<PathObject> swapped = new ArrayList<>(exported);
            Collections.swap(swapped, 0, 1);
            PythonBridge.checkCellOrder(result.cells().stream()
                    .map(PHCPipeline.CellAssignment::result).toList(), swapped);
            check(false, "results whose ids do not match the exported cells are rejected");
        } catch (IOException e) {
            check(e.getMessage().contains("id"), "results whose ids do not match the exported "
                    + "cells are rejected: " + e.getMessage());
        }
        return result;
    }

    /**
     * Runs per-cell windows with spatially constrained (Delaunay) clustering and checks: one
     * result per cell, the clustering summary (spatial, edges, no subsample), that every
     * cluster is one connected region of a Delaunay triangulation QuPath builds independently
     * from the same centroids, ring / random separation, the annotation summary, a run with a
     * maximum edge length, and that switching it off gives exactly the earlier per-cell result.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param outDir (File) Folder the plot files are written under.
     * @param plain (PHCPipeline.Result) 2-cluster per-cell result without the constraint.
     * @return (void)
     * @throws Exception When a run fails.
     */
    private static void checkSpatial(ImageData<BufferedImage> imageData, PathObject annotation,
                                     PythonBridge bridge, File outDir, PHCPipeline.Result plain)
            throws Exception {
        Path plotDir = new File(outDir, PHCPipeline.PLOT_FOLDER).toPath();
        PythonBridge.PlotTarget plots = new PythonBridge.PlotTarget(plotDir, PLOT_PREFIX);
        Files.deleteIfExists(plots.delaunayPlot());
        PHCParameters spatial = spatialParams(2, true, PHCParameters.DEFAULT_MAX_EDGE_LENGTH);
        List<String> clusteringMessages = new ArrayList<>();
        ProgressTracker tracker = new ProgressTracker(0, ProgressTracker.CELLS_NOUN,
                spatial.isSpatial());
        long start = System.nanoTime();
        PHCPipeline.Result result = PHCPipeline.analyse(imageData, annotation, spatial, bridge,
                (stage, done, total) -> {
                    tracker.update(stage, done, total, 0);
                    if (stage.equals(ProgressTracker.CLUSTERING)) {
                        clusteringMessages.add(tracker.message());
                    }
                }, plots);
        System.out.printf("  per-cell 100 um windows, PI, 2 clusters, Delaunay-constrained: "
                + "%d cells in %.1f s%n", result.cells().size(), (System.nanoTime() - start) / 1e9);
        int nCells = SyntheticSlide.nCellsInside();
        check(result.mode().equals(PHCParameters.MODE_CELLS) && result.tiles().isEmpty()
                        && result.cells().size() == nCells
                        && result.nClustered() == plain.nClustered(),
                "a spatial per-cell run returns one result per cell (" + result.cells().size()
                        + " of " + nCells + ") and clusters the same " + plain.nClustered()
                        + " cells");
        PythonBridge.ClusteringResult clustering = result.clustering();
        check(clustering != null && clustering.spatial() && clustering.nEdges() != null
                        && clustering.nEdges() > 0 && !clustering.subsampled()
                        && clustering.nFitted() == result.nClustered()
                        && clustering.nComponents() != null && clustering.nComponents() >= 1,
                "spatial clustering is reported with Delaunay edges, no subsample: "
                        + clustering);
        check(clusteringMessages.stream().anyMatch(m -> m.startsWith("Clustering "
                        + result.nClustered() + " cells (Delaunay-constrained)")),
                "the clustering stage says it is Delaunay-constrained: " + clusteringMessages);
        if (clustering != null && clustering.nComponents() != null
                && clustering.nComponents() == 1) {
            checkContiguity(result);
        } else {
            System.out.println("  skipped the strict contiguity check: the Delaunay graph had "
                    + (clustering == null ? null : clustering.nComponents()) + " parts");
        }

        PHCPipeline.applyCellResults(imageData.getHierarchy(), result, false);
        checkCellSeparation(result);
        PHCPipeline.storeClusteringSummary(annotation, clustering);
        var measurements = annotation.getMeasurementList();
        check(clustering != null && clustering.nEdges() != null
                        && measurements.get(PHCPipeline.MEASUREMENT_CELLS_DELAUNAY_EDGES)
                                == clustering.nEdges()
                        && measurements.containsKey(
                                PHCPipeline.MEASUREMENT_CELLS_DELAUNAY_COMPONENTS),
                "the annotation stores '" + PHCPipeline.MEASUREMENT_CELLS_DELAUNAY_EDGES
                        + "' and '" + PHCPipeline.MEASUREMENT_CELLS_DELAUNAY_COMPONENTS + "'");
        String summary = PHCCommand.cellSummary(result, 1.0);
        check(summary.contains("spatially contiguous clusters (Delaunay graph: ")
                        && summary.contains(" edges)"), "notification: " + summary);
        String notes = PHCCommand.notes(result);
        if (Files.exists(plots.delaunayPlot())) {
            check(notes.contains(plots.delaunayPlot().getFileName().toString()),
                    "the plot folder note names the Delaunay plot: " + notes);
        } else {
            System.out.println("  Python wrote no " + plots.delaunayPlot().getFileName()
                    + " (optional): " + result.warnings());
        }
        checkSpatialSummaryText();

        PHCParameters limited = spatialParams(2, false, MAX_EDGE_MICRONS);
        PHCPipeline.Result limitedResult = PHCPipeline.analyse(imageData, annotation, limited, bridge,
                (stage, done, total) -> { }, null);
        PythonBridge.ClusteringResult limitedClustering = limitedResult.clustering();
        double limitPx = MAX_EDGE_MICRONS / SyntheticSlide.PIXEL_SIZE_MICRONS;
        check(limitedClustering != null && limitedClustering.spatial()
                        && limitedClustering.maxEdgeLength() != null
                        && Math.abs(limitedClustering.maxEdgeLength() - limitPx) < TOLERANCE
                        && limitedClustering.nComponents() != null
                        && clustering != null && clustering.nComponents() != null
                        && limitedClustering.nComponents() >= clustering.nComponents()
                        && limitedResult.cells().size() == nCells,
                "a " + MAX_EDGE_MICRONS + " um edge limit reaches Python as " + limitPx
                        + " px and never joins more of the graph: " + limitedClustering);

        PHCParameters off = new PHCParameters(1, 100, 100, "PI", 20, 10, 2, "ward", 0.5, -1,
                true, CELL_MDS_MAX, PHCParameters.MODE_CELLS, false, false, 0);
        PHCPipeline.Result offResult = PHCPipeline.analyse(imageData, annotation, off, bridge,
                (stage, done, total) -> { }, null);
        int differences = 0;
        for (int i = 0; i < plain.cells().size(); i++) {
            PythonBridge.CellResult a = plain.cells().get(i).result();
            PythonBridge.CellResult b = offResult.cells().get(i).result();
            differences += a.cluster() == b.cluster()
                    && Math.abs(a.l2Norm() - b.l2Norm()) < TOLERANCE ? 0 : 1;
        }
        check(offResult.cells().size() == plain.cells().size() && differences == 0
                        && offResult.clustering() != null && !offResult.clustering().spatial()
                        && offResult.clustering().nEdges() == null,
                "with spatial clustering off, clusters and L2 norms equal the earlier per-cell "
                        + "run (" + differences + " differences): " + offResult.clustering());
        PHCPipeline.storeClusteringSummary(annotation, offResult.clustering());
        check(!annotation.getMeasurementList().containsKey(
                        PHCPipeline.MEASUREMENT_CELLS_DELAUNAY_EDGES),
                "a non-spatial run removes the annotation's Delaunay summary");
    }

    /**
     * Builds per-cell settings with Delaunay-constrained clustering: 100 um windows, PI,
     * 10 cells, ward.
     *
     * @param nClusters (int) Number of clusters.
     * @param computeMds (boolean) Whether to compute MDS (at most {@link #CELL_MDS_MAX} cells).
     * @param maxEdgeMicrons (double) Longest Delaunay edge kept, in um, 0 = no limit.
     * @return (PHCParameters) Spatial per-cell settings.
     */
    private static PHCParameters spatialParams(int nClusters, boolean computeMds,
                                               double maxEdgeMicrons) {
        PHCParameters params = new PHCParameters(1, 100, 100, "PI", 20, 10, nClusters, "ward",
                0.5, -1, computeMds, CELL_MDS_MAX, PHCParameters.MODE_CELLS, false, true,
                maxEdgeMicrons);
        return params;
    }

    /**
     * Checks every cluster of a spatial run is one connected region: QuPath's own Delaunay
     * triangulation is built on the clustered cells' centroids (as Python reports them), and a
     * breadth-first search over neighbours within the same cluster must reach the whole
     * cluster from any one of its cells.
     *
     * @param result (PHCPipeline.Result) Spatial per-cell result with a connected graph and no
     *        edge limit.
     * @return (void)
     */
    private static void checkContiguity(PHCPipeline.Result result) {
        Map<PathObject, Integer> clusterOf = new IdentityHashMap<>();  // centroid point -> label
        for (PHCPipeline.CellAssignment assignment : result.cells()) {
            PythonBridge.CellResult values = assignment.result();
            if (values.cluster() >= 0 && values.x() != null && values.y() != null) {
                clusterOf.put(PathObjects.createDetectionObject(ROIs.createPointsROI(values.x(),
                        values.y(), ImagePlane.getDefaultPlane())), values.cluster());
            }
        }
        DelaunayTools.Subdivision subdivision = DelaunayTools.createFromCentroids(
                clusterOf.keySet(), false);
        Map<PathObject, List<PathObject>> neighbours = subdivision.getAllNeighbors();
        Map<Integer, Integer> components = new HashMap<>();  // cluster -> connected parts
        Set<PathObject> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (PathObject seed : clusterOf.keySet()) {
            if (!seen.add(seed)) {
                continue;
            }
            int cluster = clusterOf.get(seed);
            components.merge(cluster, 1, Integer::sum);
            Deque<PathObject> queue = new ArrayDeque<>(List.of(seed));
            while (!queue.isEmpty()) {
                for (PathObject next : neighbours.getOrDefault(queue.poll(), List.of())) {
                    Integer label = clusterOf.get(next);
                    if (label != null && label == cluster && seen.add(next)) {
                        queue.add(next);
                    }
                }
            }
        }
        boolean contiguous = components.size() == result.nClusters()
                && components.values().stream().allMatch(parts -> parts == 1);
        check(contiguous && subdivision.size() == clusterOf.size(), "each of the "
                + result.nClusters() + " spatial clusters is one connected region of QuPath's "
                + "Delaunay graph of " + subdivision.size() + " centroids (parts per cluster: "
                + components + ")");
    }

    /**
     * Checks the notification of a spatial run formats the edge count with separators and
     * warns when the Delaunay graph had several parts that were joined.
     *
     * @return (void)
     */
    private static void checkSpatialSummaryText() {
        PHCPipeline.Result sample = new PHCPipeline.Result(PHCParameters.MODE_CELLS, List.of(),
                List.of(), 4, 0, 0, new PythonBridge.ClusteringResult(false, 0, true,
                        SAMPLE_EDGES, SAMPLE_COMPONENTS, 0.0), null, null, List.of());
        String summary = PHCCommand.cellSummary(sample, 0);
        check(summary.contains("4 spatially contiguous clusters (Delaunay graph: 9,412 edges)")
                        && summary.contains("graph had " + SAMPLE_COMPONENTS
                                + " disconnected parts; they were joined at their closest "
                                + "cells"), "a disconnected graph is reported: " + summary);
    }

    /**
     * Recounts each cell's neighbours in Java, with the contract's half-open rule
     * (c.x - WS/2 <= p.x < c.x + WS/2, same in y, the cell itself included), on the
     * centroids Python reports, and checks they also match the Java centroids.
     *
     * @param result (PHCPipeline.Result) Per-cell result.
     * @param windowPx (double) Window side WS in slide pixels.
     * @return (void)
     */
    private static void checkNeighbourCounts(PHCPipeline.Result result, double windowPx) {
        List<PHCPipeline.CellAssignment> cells = result.cells();
        int n = cells.size();
        double[] xs = new double[n];
        double[] ys = new double[n];
        int offCentre = 0;
        for (int i = 0; i < n; i++) {
            PythonBridge.CellResult values = cells.get(i).result();
            double[] javaCentroid = SyntheticSlide.centroid(cells.get(i).cell());
            xs[i] = values.x() == null ? Double.NaN : values.x();
            ys[i] = values.y() == null ? Double.NaN : values.y();
            if (!(Math.abs(xs[i] - javaCentroid[0]) <= CENTROID_TOLERANCE
                    && Math.abs(ys[i] - javaCentroid[1]) <= CENTROID_TOLERANCE)) {
                offCentre++;
            }
        }
        check(offCentre == 0, "Python's centroid of every cell matches the Java nucleus "
                + "centroid (" + offCentre + " off), so results map to the right cells");
        double half = windowPx / 2;
        int mismatches = 0;
        for (int i = 0; i < n; i++) {
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (xs[i] - half <= xs[j] && xs[j] < xs[i] + half
                        && ys[i] - half <= ys[j] && ys[j] < ys[i] + half) {
                    count++;
                }
            }
            double measured = cells.get(i).cell().getMeasurementList()
                    .get(PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW);
            mismatches += count == (int) measured ? 0 : 1;
        }
        check(mismatches == 0, "'" + PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW + "' equals a "
                + "brute-force half-open neighbour count for all " + n + " cells (" + mismatches
                + " mismatches)");
    }

    /**
     * Checks clustered cells carry cluster, L2 and coverage values and left-out cells carry
     * only coverage and cells in window.
     *
     * @param result (PHCPipeline.Result) Applied per-cell result.
     * @return (void)
     */
    private static void checkClusteredCells(PHCPipeline.Result result) {
        int nClustered = 0;
        int bad = 0;
        for (PHCPipeline.CellAssignment assignment : result.cells()) {
            var m = assignment.cell().getMeasurementList();
            if (assignment.result().cluster() >= 0) {
                nClustered++;
                bad += m.get(PHCPipeline.MEASUREMENT_CLUSTER) == assignment.result().cluster() + 1
                        && m.containsKey(PHCPipeline.MEASUREMENT_L2_NORM)
                        && m.containsKey(PHCPipeline.MEASUREMENT_L2_TO_MEAN) ? 0 : 1;
            } else {
                bad += m.keySet().stream().filter(k -> k.startsWith("PHC")).count() == 2 ? 0 : 1;
            }
        }
        check(bad == 0 && nClustered == result.nClustered() && nClustered > 0
                        && nClustered < result.cells().size(),
                nClustered + " clustered cells carry cluster and L2 values; the "
                        + (result.cells().size() - nClustered) + " left out (edge coverage) "
                        + "carry only coverage and cells in window (" + bad + " wrong)");
    }

    /**
     * Checks a 2-cluster per-cell run separates cells of the ring half from cells of the
     * random half.
     *
     * @param result (PHCPipeline.Result) Applied per-cell result.
     * @return (void)
     */
    private static void checkCellSeparation(PHCPipeline.Result result) {
        Map<Integer, int[]> sides = new HashMap<>();  // cluster -> {ring cells, random cells}
        int n = 0;
        for (PHCPipeline.CellAssignment assignment : result.cells()) {
            if (assignment.result().cluster() < 0) {
                continue;
            }
            int[] counts = sides.computeIfAbsent(assignment.result().cluster(), k -> new int[2]);
            counts[SyntheticSlide.isRingSide(assignment.cell().getROI().getCentroidX()) ? 0
                    : 1]++;
            n++;
        }
        int majority = sides.values().stream().mapToInt(c -> Math.max(c[0], c[1])).sum();
        double accuracy = n == 0 ? 0 : (double) majority / n;
        String summary = sides.entrySet().stream().map(e -> "cluster " + e.getKey() + "="
                + e.getValue()[0] + " ring/" + e.getValue()[1] + " random")
                .collect(Collectors.joining(", "));
        check(sides.size() == 2 && accuracy >= MIN_CELL_ACCURACY, String.format("ring and "
                + "random cells fall in different clusters (accuracy %.2f: %s)", accuracy,
                summary));
    }

    /**
     * Checks "Set cell classes to PHC clusters": clustered cells get "PHC cluster k" matching
     * their cluster, earlier classes are remembered (also across a second run), and Clear
     * restores them and removes every per-cell PHC measurement.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When a run fails.
     */
    private static void checkCellClassification(ImageData<BufferedImage> imageData,
                                                PathObject annotation, PythonBridge bridge)
            throws Exception {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        List<PathObject> cells = CellExporter.cellsInside(hierarchy, annotation);
        for (int i = 0; i < N_PRECLASSIFIED; i++) {
            cells.get(i).setPathClass(PathClass.fromString(PRECLASS));
        }
        Map<PathObject, PathClass> originals = new HashMap<>();
        cells.forEach(c -> originals.put(c, c.getPathClass()));
        int nTiles = PHCPipeline.existingTiles(annotation).size();

        PHCParameters classify = cellParams(3, false, true);
        PHCPipeline.Result first = PHCPipeline.analyse(imageData, annotation, classify, bridge,
                (stage, done, total) -> { }, null);
        PHCPipeline.applyCellResults(hierarchy, first, true);
        int wrong = 0;
        for (PHCPipeline.CellAssignment assignment : first.cells()) {
            PathObject cell = assignment.cell();
            int cluster = assignment.result().cluster();
            boolean ok = cluster >= 0
                    ? cell.getPathClass() != null && cell.getPathClass().getName()
                            .equals(PHCPipeline.CLASS_PREFIX + (cluster + 1))
                            && cell.getPathClass().getColor()
                                    == PHCPipeline.clusterColor(cluster, first.nClusters())
                            && cell.getMetadata().containsKey(PHCPipeline.ORIGINAL_CLASS_KEY)
                    : cell.getPathClass() == originals.get(cell);
            wrong += ok ? 0 : 1;
        }
        check(wrong == 0 && first.nClusters() == 3, "classify sets clustered cells to 'PHC "
                + "cluster k' in the viridis colours and keeps left-out cells' classes ("
                + wrong + " wrong)");
        check(cells.stream().noneMatch(c -> c.getMeasurementList().keySet().stream()
                        .anyMatch(k -> k.startsWith("PHC: MDS"))),
                "a per-cell run with MDS off removes MDS values from an earlier run");

        PHCPipeline.Result second = PHCPipeline.analyse(imageData, annotation, classify, bridge,
                (stage, done, total) -> { }, null);
        PHCPipeline.applyCellResults(hierarchy, second, true);
        PHCPipeline.ClearedCells cleared = PHCPipeline.clearCellResults(hierarchy, annotation);
        long restored = cells.stream().filter(c -> c.getPathClass() == originals.get(c)).count();
        long leftover = cells.stream().filter(c -> c.getMeasurementList().keySet().stream()
                .anyMatch(k -> k.startsWith("PHC")) || c.getMetadata()
                .containsKey(PHCPipeline.ORIGINAL_CLASS_KEY)).count();
        long tumour = cells.stream().filter(c -> c.getPathClass() != null
                && c.getPathClass().getName().equals(PRECLASS)).count();
        check(restored == cells.size() && tumour == N_PRECLASSIFIED,
                "Clear restores every original class after two classifying runs (" + restored
                        + " of " + cells.size() + ", " + tumour + " '" + PRECLASS + "')");
        check(leftover == 0 && cleared.nCleaned() == cells.size()
                        && cleared.nRestored() == second.nClustered()
                        && PHCPipeline.existingTiles(annotation).size() == nTiles,
                "Clear removes every per-cell PHC measurement (" + cleared + ", " + leftover
                        + " cells left with PHC values) and per-cell runs leave the tiles");
        cells.forEach(c -> c.setPathClass(null));
    }

    /**
     * Lists the cell count of each tile, in order, to compare two runs.
     *
     * @param tiles (List of PathObject) Heatmap tiles.
     * @return (List of Double) Cell count per tile.
     */
    private static List<Double> cellCounts(List<PathObject> tiles) {
        List<Double> counts = tiles.stream().map(t -> t.getMeasurementList()
                .get(PHCPipeline.MEASUREMENT_CELL_COUNT)).toList();
        return counts;
    }

    /**
     * Confirms bad settings, a missing cell detection and a missing Python give clear errors
     * instead of silent failure.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation with cells.
     * @param empty (PathObject) Annotation without detections.
     * @return (void)
     */
    private static void checkErrors(ImageData<BufferedImage> imageData, PathObject annotation,
                                    PathObject empty) {
        try {
            tiledParams(1, 100, 100, "bogus", 4, "ward", true,
                    PHCParameters.DEFAULT_MDS_MAX_WINDOWS).validate();
            check(false, "unknown vectorization is rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("PI"), "unknown vectorization error lists options: "
                    + e.getMessage());
        }
        try {
            tiledParams(1, 100, 100, "PI", 4, "ward", true, 1).validate();
            check(false, "fewer than 2 MDS windows is rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("MDS"), "fewer than 2 MDS windows is rejected: "
                    + e.getMessage());
        }
        try {
            tiledParams(2, 100, 100, "PI", 4, "ward", true,
                    PHCParameters.DEFAULT_MDS_MAX_WINDOWS).validate();
            check(false, "homology dimension 2 is rejected for alpha in the plane");
        } catch (IllegalArgumentException e) {
            check(true, "homology dimension 2 is rejected: " + e.getMessage());
        }
        try {
            PHCPipeline.analyse(imageData, empty, DEFAULTS,
                    new PythonBridge("/no/such/python", ""), NO_PROGRESS, null);
            check(false, "an annotation without detections is reported");
        } catch (Exception e) {
            check(e instanceof IllegalArgumentException
                            && e.getMessage().contains("Run cell detection first"),
                    "an annotation without detections says to run cell detection: "
                            + e.getMessage());
        }
        try {
            PHCPipeline.analyse(imageData, annotation, DEFAULTS,
                    new PythonBridge("/no/such/python", ""), NO_PROGRESS, null);
            check(false, "missing Python is reported");
        } catch (Exception e) {
            check(e.getMessage().contains("Could not start Python"),
                    "missing Python gives a preferences hint: " + e.getMessage());
        }
        if (new File(SYSTEM_PYTHON).canExecute()) {
            try {
                PHCPipeline.analyse(imageData, annotation, DEFAULTS,
                        new PythonBridge(SYSTEM_PYTHON, ""), NO_PROGRESS, null);
                check(false, "a Python without PHC's dependencies is reported");
            } catch (Exception e) {
                check(e.getMessage().contains("is missing:"),
                        "a Python without PHC's dependencies names the missing packages: "
                                + e.getMessage());
            }
        }
    }

    /**
     * Checks the ETA arithmetic and message text with a fake clock.
     *
     * @return (void)
     */
    private static void checkProgressTracker() {
        long second = 1_000_000_000L;
        check(ProgressTracker.formatDuration(0).equals("0:00")
                        && ProgressTracker.formatDuration(65).equals("1:05")
                        && ProgressTracker.formatDuration(3725).equals("1:02:05"),
                "durations format as m:ss and h:mm:ss");

        ProgressTracker tracker = new ProgressTracker(0, ProgressTracker.WINDOWS_NOUN, false);
        tracker.update(ProgressTracker.CENTROIDS, 0, 0, 2 * second);
        check(tracker.fraction() == ProgressTracker.INDETERMINATE
                        && tracker.message().startsWith("Reading cell centroids"),
                "non-window stages show an indeterminate bar: " + tracker.message());
        tracker.update(ProgressTracker.MDS, 0, 50, 2 * second);
        check(tracker.fraction() == ProgressTracker.INDETERMINATE
                        && tracker.message().startsWith("Projecting 50 windows with MDS (2D and "
                                + "3D)"), "the MDS stage is indeterminate: " + tracker.message());
        ProgressTracker cellTracker = new ProgressTracker(0, ProgressTracker.CELLS_NOUN, false);
        cellTracker.update(ProgressTracker.PERSISTENCE, 120, 3200, second);
        check(cellTracker.message().startsWith("Computing persistence: 120 / 3200 cells (4%)"),
                "per-cell progress counts cells: " + cellTracker.message());
        cellTracker.update(ProgressTracker.CLUSTERING, 0, 3150, second);
        check(cellTracker.message().startsWith("Clustering 3150 cells (elapsed"),
                "per-cell clustering counts cells: " + cellTracker.message());
        ProgressTracker spatialTracker = new ProgressTracker(0, ProgressTracker.CELLS_NOUN,
                true);
        spatialTracker.update(ProgressTracker.CLUSTERING, 0, 3200, second);
        check(spatialTracker.message().startsWith("Clustering 3200 cells (Delaunay-constrained)"),
                "spatial clustering is labelled: " + spatialTracker.message());
        tracker.update(ProgressTracker.PERSISTENCE, 0, 256, 5 * second);
        tracker.update(ProgressTracker.PERSISTENCE, 10, 256, 5 * second + second / 2);
        check(tracker.message().contains("estimating time left"),
                "no ETA in the first second of the stage: " + tracker.message());
        tracker.update(ProgressTracker.PERSISTENCE, 64, 256, 15 * second);
        check(Math.abs(tracker.secondsRemaining() - 30.0) < 1e-9,
                "ETA = stage time / done * remaining (10 s / 64 * 192 = 30 s)");
        check(tracker.fraction() == 0.25, "bar shows 64 / 256 = 25%");
        check(tracker.message().equals("Computing persistence: 64 / 256 windows (25%), about "
                + "0:30 left (elapsed 0:15)"), "message: " + tracker.message());
        tracker.tick(20 * second);
        check(tracker.message().contains("about 0:45 left (elapsed 0:20)"),
                "clock ticks move elapsed time and ETA without new windows: "
                        + tracker.message());
        try {
            tracker.update("conditioning", 0, 0, 0);
            check(false, "unknown stage is rejected");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("centroids"), "the old conditioning stage is "
                    + "rejected and the error lists stages");
        }
    }

    /**
     * Checks a real run reports its stages in order and window counts that only go up, to
     * the number of windows that pass the coverage and min-cells filters (one tile each).
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When the pipeline fails.
     */
    private static void checkProgressEvents(ImageData<BufferedImage> imageData,
                                            PathObject annotation, PythonBridge bridge)
            throws Exception {
        PHCParameters params = tiledParams(1, 100, 25, "PI", 4, "ward", true,
                PHCParameters.DEFAULT_MDS_MAX_WINDOWS);
        double stridePx = params.stride() / SyntheticSlide.PIXEL_SIZE_MICRONS;
        long gridWindows = (long) Math.ceil(SyntheticSlide.ANNOTATION_HEIGHT / stridePx)
                * (long) Math.ceil(SyntheticSlide.ANNOTATION_WIDTH / stridePx);  // box grid
        List<String> stages = new ArrayList<>();
        List<long[]> windows = new ArrayList<>();
        List<PathObject> tiles = PHCPipeline.analyse(imageData, annotation, params, bridge,
                (stage, done, total) -> {
                    if (stages.isEmpty() || !stages.get(stages.size() - 1).equals(stage)) {
                        stages.add(stage);
                    }
                    if (stage.equals(ProgressTracker.PERSISTENCE)) {
                        windows.add(new long[] {done, total});
                    }
                }, null).tiles();
        check(stages.equals(List.of(ProgressTracker.EXPORT, ProgressTracker.STARTING,
                ProgressTracker.CENTROIDS, ProgressTracker.PERSISTENCE,
                ProgressTracker.CLUSTERING, ProgressTracker.MDS, ProgressTracker.TILES)),
                "stages arrive in order "
                + stages);
        boolean increasing = true;
        for (int i = 1; i < windows.size(); i++) {
            increasing &= windows.get(i)[0] > windows.get(i - 1)[0];
        }
        long[] last = windows.isEmpty() ? new long[] {-1, -1} : windows.get(windows.size() - 1);
        long expected = tiles.size();  // corner windows with < min cells are skipped
        check(increasing && windows.size() >= 2 && last[0] == last[1] && last[1] == expected
                        && expected <= gridWindows,
                "window counts rise to " + last[0] + " / " + last[1] + " (expected "
                        + expected + " filtered windows of a " + gridWindows + "-window grid) "
                        + "over " + windows.size() + " updates");
    }

    /**
     * Cancels a long run once windows start finishing and checks Python and its joblib
     * workers are gone.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param python (String) Python executable.
     * @param libraryDir (String) Folder holding the PHC package.
     * @return (void)
     * @throws Exception When the run fails for a reason other than cancellation.
     */
    private static void checkCancel(ImageData<BufferedImage> imageData, PathObject annotation,
                                    String python, String libraryDir) throws Exception {
        PythonBridge bridge = new PythonBridge(python, libraryDir);
        PHCParameters slow = tiledParams(1, 200, 8, "PI", 4, "ward", true,
                PHCParameters.DEFAULT_MDS_MAX_WINDOWS);  // 400 px every 16 px: ~7,900 windows
        List<Integer> descendantsWhileRunning = Collections.synchronizedList(new ArrayList<>());
        long start = System.nanoTime();
        try {
            PHCPipeline.analyse(imageData, annotation, slow, bridge, (stage, done, total) -> {
                if (stage.equals(ProgressTracker.PERSISTENCE) && done > 0
                        && descendantsWhileRunning.isEmpty()) {
                    descendantsWhileRunning.add((int) ProcessHandle.current().descendants()
                            .count());
                    new Thread(bridge::cancel).start();
                }
            }, null);
            check(false, "a cancelled run stops with a CancellationException");
        } catch (CancellationException e) {
            double seconds = (System.nanoTime() - start) / 1e9;
            check(true, String.format("a cancelled run stops with a CancellationException "
                    + "(%.1f s)", seconds));
        }
        Thread.sleep(500);  // let the OS reap the killed processes
        long left = ProcessHandle.current().descendants().filter(ProcessHandle::isAlive).count();
        check(!descendantsWhileRunning.isEmpty() && descendantsWhileRunning.get(0) > 1
                        && left == 0, "cancel stops Python and its "
                        + (descendantsWhileRunning.isEmpty() ? 0 : descendantsWhileRunning.get(0) - 1)
                        + " joblib workers (" + left + " processes left)");
    }

    /**
     * Draws the tiles over the cells in their class colours, to eyeball the heatmap.
     *
     * @param cells (List of PathObject) Cells, drawn as dots.
     * @param tiles (List of PathObject) Heatmap tiles.
     * @param out (File) PNG to write.
     * @return (void)
     * @throws IOException When the PNG cannot be written.
     */
    private static void render(List<PathObject> cells, List<PathObject> tiles, File out)
            throws IOException {
        BufferedImage canvas = SyntheticSlide.drawCells(cells);
        Graphics2D g = canvas.createGraphics();
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, OVERLAY_ALPHA));
        for (PathObject tile : tiles) {
            g.setColor(new Color(tile.getPathClass().getColor()));
            g.fill(tile.getROI().getShape());
        }
        g.dispose();
        ImageIO.write(canvas, "png", out);
    }

    /**
     * Records and prints one check result.
     *
     * @param passed (boolean) Whether the check passed.
     * @param description (String) What was checked.
     * @return (void)
     */
    private static void check(boolean passed, String description) {
        System.out.println((passed ? "PASS " : "FAIL ") + description);
        if (!passed) {
            failures++;
        }
    }
}
