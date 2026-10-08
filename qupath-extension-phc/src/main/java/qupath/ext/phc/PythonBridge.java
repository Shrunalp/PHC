/*
 * Runs the bundled phc_bridge.py with the user's Python and reads back its results.
 *
 * Contents
 * --------
 * PythonBridge : class
 *     Launches the PHC Python process for an annotation's cells and parses its JSON output.
 * PythonBridge.BridgeResult : record
 *     Everything the bridge reports for one run.
 * PythonBridge.WindowResult : record
 *     Position, cell count, L2 measures, cluster label and MDS coordinates of one PHC window.
 * PythonBridge.CellResult : record
 *     Centroid, neighbour count, L2 measures, cluster label and MDS coordinates of one cell's
 *     window (per-cell mode).
 * PythonBridge.ClusteringResult : record
 *     How per-cell clustering was fitted: on a subsample or not, and whether it was
 *     constrained to the Delaunay graph of the cell centroids.
 * PythonBridge.MdsResult : record
 *     Summary of the MDS embedding of the windows' L2 dissimilarity matrix.
 * PythonBridge.PlotTarget : record
 *     Folder and file name prefix for the matplotlib MDS plots and CSV Python writes.
 * PythonBridge.ProgressListener : interface
 *     Receives stage and window-count updates while a run is in progress.
 */

package qupath.ext.phc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.io.PathIO;
import qupath.lib.io.PathIO.GeoJsonExportOptions;
import qupath.lib.objects.PathObject;

/**
 * Hands an annotation's detected cells to the user's Python environment, where the PHC library
 * does the centroid, alpha persistence, L2 and clustering work, and returns the parsed
 * results. Keeping the maths in Python means QuPath shows exactly what the PHC library
 * computes.
 */
public final class PythonBridge {

    private static final Logger logger = LoggerFactory.getLogger(PythonBridge.class);

    private static final String BRIDGE_RESOURCE = "/qupath/ext/phc/phc_bridge.py";
    private static final int ERROR_TAIL_LINES = 15;  // lines of Python output shown on failure
    private static final String STAGE_PREFIX = "PHC_STAGE ";
    private static final String PROGRESS_PREFIX = "PHC_PROGRESS ";
    private static final String WARNING_PREFIX = "Warning: ";  // non-fatal, e.g. no matplotlib
    private static final long TERMINATE_GRACE_SECONDS = 2;  // before a cancelled run is killed

    private final String pythonExecutable;
    private final String phcLibraryDir;
    private volatile Process process;
    private volatile boolean cancelled = false;

    /**
     * Creates a bridge for one Python environment.
     *
     * @param pythonExecutable (String) Path to a Python that has gudhi, scikit-learn, opencv
     *        and joblib installed.
     * @param phcLibraryDir (String) Folder containing the PHC package, added to PYTHONPATH;
     *        blank when PHC is already installed in that environment.
     */
    public PythonBridge(String pythonExecutable, String phcLibraryDir) {
        this.pythonExecutable = expandHome(pythonExecutable);
        this.phcLibraryDir = expandHome(phcLibraryDir);
    }

    /**
     * Gives the Python this bridge will run, after "~" expansion, for messages and logs.
     *
     * @return (String) Python executable path.
     */
    public String pythonExecutable() {
        return pythonExecutable;
    }

    /**
     * Expands a leading "~" so paths can be typed home-relative, as in a shell.
     *
     * @param path (String) Path typed by the user, may be null or blank.
     * @return (String) The path with "~" replaced by the home folder, or "" when blank.
     */
    static String expandHome(String path) {
        String expanded = path == null ? "" : path.strip();
        if (expanded.startsWith("~")) {
            expanded = System.getProperty("user.home") + expanded.substring(1);
        }
        return expanded;
    }

    /**
     * Stops a running bridge: Python and its joblib worker processes are asked to exit, then
     * killed if they have not after a short grace period. Safe to call from any thread, and
     * before or after a run.
     *
     * @return (void)
     */
    public void cancel() {
        cancelled = true;
        Process running = process;
        if (running == null) {
            return;
        }
        running.descendants().forEach(ProcessHandle::destroy);  // joblib workers
        running.destroy();
        try {
            if (!running.waitFor(TERMINATE_GRACE_SECONDS, TimeUnit.SECONDS)) {
                running.descendants().forEach(ProcessHandle::destroyForcibly);
                running.destroyForcibly();
            }
        } catch (InterruptedException e) {
            running.destroyForcibly();
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Runs alpha PHC on an annotation's cell centroids and waits for the results, optionally
     * asking Python to save its MDS plots and CSV. Call it off the JavaFX thread; large
     * annotations with small strides can take minutes.
     *
     * @param cells (CellExporter.ExportedCells) Cells, ROI mask and bounding box to analyse.
     * @param params (PHCParameters) Validated PHC, clustering and MDS settings.
     * @param pixelSizeMicrons (double) Micrometres per slide pixel, used to convert window size
     *        and stride to pixels; {@link PHCParameters#UNCALIBRATED_PIXEL_SIZE} for pixels.
     * @param listener (ProgressListener) Receives stage and window-count updates as they
     *        arrive, on the calling thread.
     * @param plots (PlotTarget) Existing folder and prefix for the plot files, or null to
     *        write none.
     * @return (BridgeResult) Per-window positions, L2 measures, cluster labels and MDS
     *         coordinates.
     * @throws IOException When Python cannot be started, fails, or writes no results.
     * @throws InterruptedException When the calling thread is interrupted while waiting.
     * @throws CancellationException When {@link #cancel()} stopped the run.
     */
    public BridgeResult run(CellExporter.ExportedCells cells, PHCParameters params,
                            double pixelSizeMicrons, ProgressListener listener,
                            PlotTarget plots)
            throws IOException, InterruptedException {
        Path workDir = Files.createTempDirectory("qupath-phc");
        try {
            Path script = workDir.resolve("phc_bridge.py");
            Path cellsPath = workDir.resolve("cells.geojson");
            Path maskPath = workDir.resolve("mask.png");
            Path outputPath = workDir.resolve("results.json");
            extractBridgeScript(script);
            writeCells(cellsPath, cells.cells());
            ImageIO.write(cells.mask(), "png", maskPath.toFile());

            List<String> command = new ArrayList<>(List.of(pythonExecutable, script.toString(),
                    "--cells", cellsPath.toString(), "--mask", maskPath.toString(),
                    "--mask-downsample", Double.toString(cells.maskDownsample()),
                    "--origin-x", Double.toString(cells.originX()),
                    "--origin-y", Double.toString(cells.originY()),
                    "--width", Double.toString(cells.width()),
                    "--height", Double.toString(cells.height()),
                    "--output", outputPath.toString()));
            command.addAll(params.toBridgeArgs(pixelSizeMicrons));
            if (plots != null) {
                command.addAll(List.of("--plot-dir", plots.dir().toString(),
                        "--plot-prefix", plots.prefix()));
            }
            logger.info("Running PHC: {}", String.join(" ", command));

            Deque<String> outputTail = new ArrayDeque<>();
            List<String> warnings = new ArrayList<>();
            listener.update(ProgressTracker.STARTING, 0, 0);
            int exitCode = startAndWait(command, outputTail, warnings, listener);
            if (cancelled) {
                throw new CancellationException("PHC was cancelled.");
            }
            if (exitCode != 0 || !Files.exists(outputPath)) {
                throw new IOException("PHC Python process failed (exit code " + exitCode + "):\n"
                        + String.join("\n", outputTail));
            }
            BridgeResult parsed;
            try (Reader reader = Files.newBufferedReader(outputPath, StandardCharsets.UTF_8)) {
                parsed = new Gson().fromJson(reader, BridgeResult.class);
            }
            String mode = parsed.mode() == null ? PHCParameters.MODE_WINDOWS : parsed.mode();
            if (!mode.equals(params.mode())) {
                throw new IOException("PHC Python returned '" + mode + "' results for a '"
                        + params.mode() + "' run; is phc_bridge.py up to date?");
            }
            List<CellResult> cellResults = parsed.cells() == null ? List.of() : parsed.cells();
            if (params.isCellMode()) {
                checkCellOrder(cellResults, cells.cells());
            }
            BridgeResult result = new BridgeResult(mode,
                    parsed.windows() == null ? List.of() : parsed.windows(), cellResults,
                    parsed.nClusters(), parsed.nWindowsClustered(), parsed.nCells(),
                    parsed.nCellsClustered(), parsed.nSkipped(), parsed.clustering(),
                    parsed.elapsedSeconds(), parsed.mds(),
                    List.copyOf(warnings));  // warnings are not in the JSON
            return result;
        } finally {
            deleteRecursively(workDir);
        }
    }

    /**
     * Writes the cells Python reads, in list order, without measurements, so the bridge's
     * feature index i is cells.get(i) and earlier PHC measurements never reach Python.
     *
     * @param path (Path) GeoJSON file to write.
     * @param cells (List of PathObject) Exported cells, in the order results map back to.
     * @return (void)
     * @throws IOException When the file cannot be written.
     */
    static void writeCells(Path path, List<PathObject> cells) throws IOException {
        PathIO.exportObjectsAsGeoJSON(path, cells, GeoJsonExportOptions.FEATURE_COLLECTION,
                GeoJsonExportOptions.EXCLUDE_MEASUREMENTS);
    }

    /**
     * Confirms per-cell results line up with the exported cells, so entry i really belongs
     * to cell i: one entry per cell, indices 0..n-1 in order, and the GeoJSON "id" (when
     * Python reports one) equal to the cell's UUID.
     *
     * @param results (List of CellResult) Per-cell entries from the bridge.
     * @param cells (List of PathObject) Cells exported for this run, in export order.
     * @return (void)
     * @throws IOException When the count, an index or an id does not match.
     */
    static void checkCellOrder(List<CellResult> results, List<PathObject> cells)
            throws IOException {
        if (results.size() != cells.size()) {
            throw new IOException("PHC Python returned " + results.size() + " cell results for "
                    + cells.size() + " exported cells.");
        }
        for (int i = 0; i < cells.size(); i++) {
            CellResult entry = results.get(i);
            UUID expected = cells.get(i).getID();
            if (entry.index() != i) {
                throw new IOException("PHC cell result " + i + " has index " + entry.index()
                        + "; results must follow the exported cell order.");
            }
            if (entry.id() != null && expected != null && !entry.id().equals(expected.toString())) {
                throw new IOException("PHC cell result " + i + " has id " + entry.id()
                        + " but the exported cell is " + expected + ".");
            }
        }
    }

    /**
     * Starts Python, forwards its progress lines to the listener, collects the bridge's own
     * "Warning: " lines and keeps the last other lines for error messages. gudhi prints a
     * warning for every empty diagram, so the remaining output is only logged at debug level.
     *
     * @param command (List of String) Full command line to run.
     * @param outputTail (Deque of String) Filled with the last {@link #ERROR_TAIL_LINES} lines.
     * @param warnings (List of String) Filled with the bridge's warnings, prefix removed.
     * @param listener (ProgressListener) Receives parsed PHC_STAGE / PHC_PROGRESS lines.
     * @return (int) The process exit code.
     * @throws IOException When the process cannot be started.
     * @throws InterruptedException When interrupted while waiting for the process.
     */
    private int startAndWait(List<String> command, Deque<String> outputTail,
                             List<String> warnings, ProgressListener listener)
            throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        if (phcLibraryDir != null && !phcLibraryDir.isBlank()) {
            Map<String, String> env = builder.environment();
            String existing = env.getOrDefault("PYTHONPATH", "");
            env.put("PYTHONPATH", existing.isEmpty()
                    ? phcLibraryDir : phcLibraryDir + java.io.File.pathSeparator + existing);
        }

        if (cancelled) {
            return -1;
        }
        Process started;
        try {
            started = builder.start();
            process = started;
        } catch (IOException e) {
            throw new IOException("Could not start Python at '" + pythonExecutable + "'. Set the "
                    + "full path under Edit > Preferences > PHC.", e);
        }
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(started.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(WARNING_PREFIX)) {
                    logger.warn("[phc_bridge] {}", line);
                    warnings.add(line.substring(WARNING_PREFIX.length()).strip());
                } else if (!forwardProgress(line, listener)) {
                    logger.debug("[phc_bridge] {}", line);
                    outputTail.addLast(line);
                    if (outputTail.size() > ERROR_TAIL_LINES) {
                        outputTail.removeFirst();
                    }
                }
            }
        } catch (IOException e) {
            if (!cancelled) {  // a cancelled run closes the stream on purpose
                started.destroyForcibly();
                throw e;
            }
        }
        int exitCode = started.waitFor();
        process = null;
        return exitCode;
    }

    /**
     * Passes a bridge progress line to the listener.
     *
     * @param line (String) One line of Python output.
     * @param listener (ProgressListener) Receives the parsed update.
     * @return (boolean) True if the line was a progress line, false for ordinary output.
     */
    private static boolean forwardProgress(String line, ProgressListener listener) {
        boolean isProgress = true;
        if (line.startsWith(PROGRESS_PREFIX)) {
            String[] parts = line.substring(PROGRESS_PREFIX.length()).strip().split(" ");
            listener.update(ProgressTracker.PERSISTENCE, Long.parseLong(parts[0]),
                    Long.parseLong(parts[1]));
        } else if (line.startsWith(STAGE_PREFIX)) {
            String[] parts = line.substring(STAGE_PREFIX.length()).strip().split(" ");
            long count = parts.length > 1 ? Long.parseLong(parts[1]) : 0;
            listener.update(parts[0], 0, count);  // a new stage starts with nothing done
        } else {
            isProgress = false;
        }
        return isProgress;
    }

    /**
     * Copies phc_bridge.py out of the jar so Python can run it.
     *
     * @param target (Path) Where to write the script.
     * @throws IOException When the resource is missing from the jar or cannot be written.
     */
    private static void extractBridgeScript(Path target) throws IOException {
        try (InputStream stream = PythonBridge.class.getResourceAsStream(BRIDGE_RESOURCE)) {
            if (stream == null) {
                throw new IOException("phc_bridge.py is missing from the extension jar.");
            }
            Files.copy(stream, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Removes the temporary folder holding the exported cells, mask and results.
     *
     * @param dir (Path) Folder to delete, with everything in it.
     */
    private static void deleteRecursively(Path dir) {
        try (Stream<Path> paths = Files.walk(dir)) {
            paths.sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
        } catch (IOException e) {
            logger.warn("Could not delete temporary folder {}", dir, e);
        }
    }

    /**
     * Receives progress while a run is in progress, e.g. to drive a progress bar.
     */
    @FunctionalInterface
    public interface ProgressListener {

        /**
         * Reports the current stage and how far it has got.
         *
         * @param stage (String) One of the stage names in {@link ProgressTracker}.
         * @param done (long) Items finished in this stage (windows for persistence).
         * @param total (long) Items in this stage, 0 when unknown.
         * @return (void)
         */
        void update(String stage, long done, long total);
    }

    /**
     * Everything phc_bridge.py reports for one run, in either mode.
     *
     * @param mode (String) {@link PHCParameters#MODE_WINDOWS} or {@link PHCParameters#MODE_CELLS}.
     * @param windows (List of WindowResult) One entry per PHC window, rows outer, columns
     *        inner; empty in per-cell mode.
     * @param cells (List of CellResult) One entry per exported cell, in export order; empty in
     *        tiled-window mode.
     * @param nClusters (int) Number of clusters actually used (at most the number requested).
     * @param nWindowsClustered (int) Windows that passed the coverage and min-cells filters and
     *        were clustered (tiled-window mode).
     * @param nCells (int) Cell centroids read from the exported cells.
     * @param nCellsClustered (int) Cells whose window passed the filters and were clustered
     *        (per-cell mode).
     * @param nSkipped (int) Exported cells without a usable centroid (per-cell mode).
     * @param clustering (ClusteringResult) How per-cell clustering was fitted, or null in
     *        tiled-window mode.
     * @param elapsedSeconds (double) Python-side run time, in seconds.
     * @param mds (MdsResult) Summary of the MDS embedding, or null when MDS was switched off
     *        or skipped (fewer than 2 clustered windows).
     * @param warnings (List of String) Non-fatal problems Python reported on "Warning: " lines
     *        (e.g. plots not written); filled in by {@link PythonBridge}, not read from JSON.
     */
    public record BridgeResult(
            String mode,
            List<WindowResult> windows,
            List<CellResult> cells,
            @SerializedName("n_clusters") int nClusters,
            @SerializedName("n_windows_clustered") int nWindowsClustered,
            @SerializedName("n_cells") int nCells,
            @SerializedName("n_cells_clustered") int nCellsClustered,
            @SerializedName("n_skipped") int nSkipped,
            ClusteringResult clustering,
            @SerializedName("elapsed_s") double elapsedSeconds,
            MdsResult mds,
            List<String> warnings) {
    }

    /**
     * Summary of the metric MDS embedding of the clustered windows' L2 dissimilarity matrix.
     *
     * @param nEmbedded (int) Windows that received MDS coordinates.
     * @param subsampled (boolean) True when more windows were clustered than the MDS limit,
     *        so only a random subsample was embedded.
     * @param stress2d (Double) Kruskal stress-1 of the 2D embedding (0 = perfect), or null.
     * @param stress3d (Double) Kruskal stress-1 of the 3D embedding, or null.
     */
    public record MdsResult(
            @SerializedName("n_embedded") int nEmbedded,
            boolean subsampled,
            @SerializedName("stress_2d") Double stress2d,
            @SerializedName("stress_3d") Double stress3d) {
    }

    /**
     * Where Python saves its publication plots of the MDS embedding, and the names it uses.
     *
     * @param dir (Path) Existing folder the files are written to.
     * @param prefix (String) File name prefix, already safe for a file name.
     */
    public record PlotTarget(Path dir, String prefix) {

        /** Suffixes Python appends to the prefix, per the bridge contract. */
        public static final String MDS_2D_SUFFIX = "_mds_2d.png";
        public static final String MDS_3D_SUFFIX = "_mds_3d.png";
        public static final String CSV_SUFFIX = "_mds.csv";
        /** Extra suffix before the above in per-cell mode, e.g. "_cells_mds_2d.png". */
        public static final String CELLS_SUFFIX = "_cells";
        /** Optional per-cell plot of the Delaunay adjacency, e.g. "_cells_delaunay.png". */
        public static final String DELAUNAY_SUFFIX = "_delaunay.png";

        /**
         * Gives where Python puts its optional plot of the Delaunay adjacency of a spatially
         * constrained per-cell run, so callers can mention it when it exists.
         *
         * @return (Path) "&lt;dir&gt;/&lt;prefix&gt;_cells_delaunay.png" (may not exist).
         */
        public Path delaunayPlot() {
            Path plot = dir.resolve(prefix + CELLS_SUFFIX + DELAUNAY_SUFFIX);
            return plot;
        }

        /**
         * Lists the files a run with MDS writes here in either mode, so callers can check or
         * report them.
         *
         * @param mode (String) {@link PHCParameters#MODE_WINDOWS} or
         *        {@link PHCParameters#MODE_CELLS}.
         * @return (List of Path) The 2D plot, the 3D plot and the coordinate CSV.
         * @throws IllegalArgumentException When mode is not a known mode.
         */
        public List<Path> files(String mode) {
            String base;
            if (PHCParameters.MODE_WINDOWS.equals(mode)) {
                base = prefix;
            } else if (PHCParameters.MODE_CELLS.equals(mode)) {
                base = prefix + CELLS_SUFFIX;
            } else {
                throw new IllegalArgumentException("Unknown mode '" + mode + "'; expected one of "
                        + PHCParameters.MODES + ".");
            }
            List<Path> files = List.of(dir.resolve(base + MDS_2D_SUFFIX),
                    dir.resolve(base + MDS_3D_SUFFIX), dir.resolve(base + CSV_SUFFIX));
            return files;
        }
    }

    /**
     * Position, cell count, L2 measures, cluster label and MDS coordinates of one PHC window,
     * in full-resolution slide pixels relative to the annotation's bounding box origin.
     *
     * @param row (double) Top edge of the window.
     * @param col (double) Left edge of the window.
     * @param height (double) Window height, smaller than the window size at the bottom edge.
     * @param width (double) Window width, smaller than the window size at the right edge.
     * @param coverage (double) Fraction of the window inside the ROI.
     * @param nCells (int) Cell centroids inside the window.
     * @param l2Norm (double) L2 norm of the window's persistence vector.
     * @param l2ToMean (double) L2 distance from the window's vector to the ROI's mean vector.
     * @param cluster (int) Agglomerative cluster, 0 = lowest mean L2 norm; -1 = left out (too
     *        little ROI coverage or too few cells).
     * @param mds2 (double[] - size (2)) 2D MDS coordinates, or null when the window was not
     *        embedded (left out, not in the subsample, or MDS off).
     * @param mds3 (double[] - size (3)) 3D MDS coordinates, or null as for mds2.
     */
    public record WindowResult(
            double row,
            double col,
            double height,
            double width,
            double coverage,
            @SerializedName("n_cells") int nCells,
            @SerializedName("l2_norm") double l2Norm,
            @SerializedName("l2_to_mean") double l2ToMean,
            int cluster,
            double[] mds2,
            double[] mds3) {
    }

    /**
     * Local persistence results of one cell's window in per-cell mode: the square of side
     * window size centred on the cell's centroid.
     *
     * @param index (int) GeoJSON feature index, i.e. position in the exported cell list.
     * @param id (String) The feature's "id" (the cell's UUID), or null when absent.
     * @param x (Double) Centroid x used by Python, in slide pixels, or null when the cell had
     *        no usable centroid.
     * @param y (Double) Centroid y used by Python, or null as for x.
     * @param coverage (double) Fraction of the cell's window inside the ROI.
     * @param nCells (int) Neighbouring centroids in the window, including the cell itself.
     * @param l2Norm (double) L2 norm of the window's persistence vector, 0 when not clustered.
     * @param l2ToMean (double) L2 distance to the mean vector of the clustered cells, 0 when
     *        not clustered.
     * @param cluster (int) Cluster, 0 = lowest mean L2 norm; -1 = not clustered (too little
     *        coverage, too few neighbours, or no centroid).
     * @param mds2 (double[] - size (2)) 2D MDS coordinates, or null when not embedded.
     * @param mds3 (double[] - size (3)) 3D MDS coordinates, or null when not embedded.
     */
    public record CellResult(
            int index,
            String id,
            Double x,
            Double y,
            double coverage,
            @SerializedName("n_cells") int nCells,
            @SerializedName("l2_norm") double l2Norm,
            @SerializedName("l2_to_mean") double l2ToMean,
            int cluster,
            double[] mds2,
            double[] mds3) {
    }

    /**
     * How the per-cell clustering was fitted: on every clustered cell, or on a random
     * subsample with the rest assigned to the nearest cluster mean; and whether it was
     * constrained to the Delaunay graph of the clustered cells' centroids (spatial mode, which
     * never subsamples).
     *
     * @param subsampled (boolean) True when more cells were clustered than the cap (10,000);
     *        always false in spatial mode.
     * @param nFitted (int) Cells agglomerative clustering was fitted on.
     * @param spatial (boolean) True when the clustering used the Delaunay adjacency as its
     *        connectivity, so clusters are spatially contiguous.
     * @param nEdges (Integer) Undirected edges of the adjacency actually used (after dropping
     *        long edges and joining components), or null when not spatial.
     * @param nComponents (Integer) Connected parts of the Delaunay graph before Python joined
     *        them at their closest cells, or null when not spatial.
     * @param maxEdgeLength (Double) Longest Delaunay edge kept, in slide pixels (0 = no
     *        limit), or null when not spatial.
     */
    public record ClusteringResult(
            boolean subsampled,
            @SerializedName("n_fitted") int nFitted,
            boolean spatial,
            @SerializedName("n_edges") Integer nEdges,
            @SerializedName("n_components") Integer nComponents,
            @SerializedName("max_edge_length") Double maxEdgeLength) {
    }
}
