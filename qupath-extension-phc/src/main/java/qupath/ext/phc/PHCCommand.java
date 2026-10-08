/*
 * Menu commands that run PHC on the selected annotation, clear its heatmap and show its MDS plot.
 *
 * Contents
 * --------
 * PHCCommand : class
 *     Checks for detected cells, shows the parameter dialog, runs PHC, adds the heatmap tiles
 *     or per-cell measurements and opens the MDS viewer.
 */

package qupath.ext.phc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;

import javafx.beans.property.StringProperty;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.objects.PathObject;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.plugins.parameters.StringParameter;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Connects the PHC pipeline to QuPath's GUI: it checks the selection and its detected cells,
 * asks for settings, runs the analysis off the JavaFX thread, puts the heatmap tiles under
 * the annotation (tiled windows) or the measurements on its cells (per-cell windows) and
 * opens the MDS viewer. Remembers the open viewer, so a new run replaces it.
 */
public final class PHCCommand {

    private static final Logger logger = LoggerFactory.getLogger(PHCCommand.class);
    private static final String TITLE = "PHC";
    private static final String LENGTH_FORMAT = "0.##";  // e.g. 100 or 12.5 in the header
    private static final String SPATIAL_HEADER = ", Delaunay-constrained";  // header suffix

    private final QuPathGUI qupath;
    private final StringProperty pythonPath;
    private final StringProperty phcLibraryDir;
    private final ParameterList params = PHCParameters.createParameterList();  // kept per session
    private MDSViewer mdsViewer;          // open MDS window, or null
    private PathObject mdsAnnotation;     // annotation whose tiles or cells mdsViewer shows

    /**
     * Creates the command for one QuPath window.
     *
     * @param qupath (QuPathGUI) Running QuPath instance.
     * @param pythonPath (StringProperty) Persistent preference for the Python executable;
     *        shown in the dialog and updated with what the user enters there.
     * @param phcLibraryDir (StringProperty) Persistent preference for the PHC library folder,
     *        handled the same way.
     */
    public PHCCommand(QuPathGUI qupath, StringProperty pythonPath, StringProperty phcLibraryDir) {
        this.qupath = qupath;
        this.pythonPath = pythonPath;
        this.phcLibraryDir = phcLibraryDir;
    }

    /**
     * Runs PHC on the selected annotation: validates the selection, checks it holds detected
     * cells, shows the dialog, then computes in the background. Call it on the JavaFX thread
     * (from a menu item).
     *
     * @return (void)
     */
    public void run() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        PathObject annotation = selectedAnnotation(imageData);
        if (annotation == null) {
            return;
        }
        int nCells = CellExporter.cellsInside(imageData.getHierarchy(), annotation).size();
        if (nCells == 0) {
            Dialogs.showErrorMessage(TITLE, CellExporter.NO_CELLS_MESSAGE);
            return;
        }
        // Show the saved environment, which may have been edited under Edit > Preferences
        ((StringParameter) params.getParameters().get(PHCParameters.PYTHON_PATH_KEY))
                .setValue(pythonPath.get());
        ((StringParameter) params.getParameters().get(PHCParameters.PHC_LIBRARY_DIR_KEY))
                .setValue(phcLibraryDir.get());
        if (!GuiTools.showParameterDialog("Run PHC", params)) {
            return;
        }
        PHCParameters settings;
        try {
            settings = PHCParameters.fromParameterList(params);
        } catch (IllegalArgumentException e) {
            Dialogs.showErrorMessage(TITLE, e.getMessage());
            return;
        }

        pythonPath.set(params.getStringParameterValue(PHCParameters.PYTHON_PATH_KEY).strip());
        phcLibraryDir.set(params.getStringParameterValue(PHCParameters.PHC_LIBRARY_DIR_KEY)
                .strip());
        PythonBridge bridge = new PythonBridge(pythonPath.get(), phcLibraryDir.get());
        File python = new File(bridge.pythonExecutable());
        if (python.isAbsolute() && !python.canExecute()) {
            Dialogs.showErrorMessage(TITLE, "No Python executable at " + python + ". Set the "
                    + "full path, e.g. ~/miniconda3/envs/phc/bin/python.");
            return;
        }
        logger.info("PHC will use Python {}", bridge.pythonExecutable());

        // MDS plots and the Delaunay plot each need the folder
        PythonBridge.PlotTarget plots = settings.computeMds() || settings.isSpatial()
                ? createPlotTarget(imageData, annotation) : null;
        long start = System.nanoTime();
        PHCTask task = new PHCTask(imageData, annotation, settings, bridge, plots);
        task.setOnSucceeded(event -> showResult(imageData, annotation, settings, task.getValue(),
                (System.nanoTime() - start) / 1e9));
        task.setOnCancelled(event -> Dialogs.showInfoNotification(TITLE, "PHC was cancelled."));
        task.setOnFailed(event -> reportFailure(task.getException()));
        task.createProgressDialog(qupath.getStage(), progressHeader(settings, nCells,
                imageData.getServer().getPixelCalibration()));

        Thread worker = new Thread(task, "phc-runner");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Describes a run in one line for the progress dialog.
     *
     * @param settings (PHCParameters) Settings of the run.
     * @param nCells (int) Detected cells inside the annotation.
     * @param calibration (PixelCalibration) Image calibration, which decides whether lengths
     *        are micrometres or pixels.
     * @return (String) Header text, e.g. "Alpha PHC on 1234 cells, 100 µm windows (stride
     *         100 µm), PI, 4 clusters", or in per-cell mode "Alpha PHC per cell on 1234
     *         cells, 100 µm windows centred on each cell, PI, 4 clusters" (followed by
     *         ", Delaunay-constrained" for spatially constrained clustering).
     */
    static String progressHeader(PHCParameters settings, int nCells,
                                 PixelCalibration calibration) {
        String unit = calibration.hasPixelSizeMicrons() ? GeneralTools.micrometerSymbol() : "px";
        DecimalFormat lengths = new DecimalFormat(LENGTH_FORMAT);
        String header;
        if (settings.isCellMode()) {
            header = String.format("Alpha PHC per cell on %d cells, %s %s windows centred on "
                            + "each cell, %s, %d clusters", nCells,
                    lengths.format(settings.windowSize()), unit, settings.vectorization(),
                    settings.nClusters()) + (settings.isSpatial() ? SPATIAL_HEADER : "");
        } else {
            header = String.format("Alpha PHC on %d cells, %s %s windows (stride %s %s), "
                            + "%s, %d clusters", nCells, lengths.format(settings.windowSize()),
                    unit, lengths.format(settings.stride()), unit, settings.vectorization(),
                    settings.nClusters());
        }
        return header;
    }

    /**
     * Removes the PHC results of the selected annotation: its heatmap tiles, the per-cell PHC
     * measurements on its cells, and classes PHC gave its cells (the originals come back).
     *
     * @return (void)
     */
    public void clear() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        PathObject annotation = selectedAnnotation(imageData);
        if (annotation == null) {
            return;
        }
        List<PathObject> oldTiles = PHCPipeline.existingTiles(annotation);
        imageData.getHierarchy().removeObjects(oldTiles, true);
        PHCPipeline.ClearedCells cleared = PHCPipeline.clearCellResults(
                imageData.getHierarchy(), annotation);
        PHCPipeline.storeMdsSummary(annotation, null, PHCParameters.MODE_WINDOWS);
        PHCPipeline.storeMdsSummary(annotation, null, PHCParameters.MODE_CELLS);
        PHCPipeline.storeClusteringSummary(annotation, null);
        imageData.getHierarchy().fireObjectMeasurementsChangedEvent(this, List.of(annotation));
        if (annotation == mdsAnnotation) {
            closeViewer();
        }
        String message = "Removed " + oldTiles.size() + " PHC tiles";
        if (cleared.nCleaned() > 0) {
            message += " and the PHC measurements of " + cleared.nCleaned() + " cells";
        }
        if (cleared.nRestored() > 0) {
            message += "; restored the original class of " + cleared.nRestored() + " cells";
        }
        Dialogs.showInfoNotification(TITLE, message + ".");
    }

    /**
     * Tells the user why a run failed; a cancellation that surfaces as an error is reported
     * as a cancellation instead.
     *
     * @param error (Throwable) Exception thrown by the task.
     * @return (void)
     */
    private static void reportFailure(Throwable error) {
        if (error instanceof CancellationException) {
            Dialogs.showInfoNotification(TITLE, "PHC was cancelled.");
            return;
        }
        logger.error("PHC failed", error);
        Dialogs.showErrorMessage(TITLE, error.getMessage());
    }

    /**
     * Opens the MDS plot of the selected annotation again, from its tiles' or cells'
     * measurements, e.g. after the project was closed and reopened. When both tiled and
     * per-cell results have MDS coordinates, asks which to show.
     *
     * @return (void)
     */
    public void showMds() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        PathObject annotation = selectedAnnotation(imageData);
        if (annotation == null) {
            return;
        }
        List<PathObject> tiles = PHCPipeline.existingTiles(annotation);
        List<PathObject> cells = PHCPipeline.clusteredCells(imageData.getHierarchy(), annotation);
        if (tiles.isEmpty() && cells.isEmpty()) {
            Dialogs.showErrorMessage(TITLE, "This annotation has no PHC results. Run PHC on it "
                    + "first (Extensions > PHC > Run PHC on selected annotation).");
            return;
        }
        List<String> choices = new ArrayList<>();
        if (!PHCPipeline.embeddedTiles(tiles).isEmpty()) {
            choices.add(PHCParameters.MODE_LABELS.get(0));
        }
        if (!PHCPipeline.embeddedTiles(cells).isEmpty()) {
            choices.add(PHCParameters.MODE_LABELS.get(1));
        }
        if (choices.isEmpty()) {
            Dialogs.showErrorMessage(TITLE, "This annotation's PHC results have no MDS "
                    + "measurements. Run PHC again with 'Compute MDS embedding' switched on.");
            return;
        }
        String choice = choices.size() == 1 ? choices.get(0)
                : Dialogs.showChoiceDialog(TITLE, "This annotation has MDS results for tiled "
                        + "and per-cell windows. Show which?", choices, choices.get(0));
        if (choice == null) {
            return;
        }
        String mode = PHCParameters.modeForLabel(choice);
        PythonBridge.PlotTarget plots = plotTarget(imageData, annotation);
        boolean plotsExist = plots != null
                && plots.files(mode).stream().allMatch(Files::exists);
        MeasurementList measurements = annotation.getMeasurementList();
        String[] stressNames = PHCPipeline.stressNames(mode);
        openViewer(imageData, annotation,
                PHCParameters.MODE_CELLS.equals(mode) ? cells : tiles,
                measuredOrNull(measurements, stressNames[0]),
                measuredOrNull(measurements, stressNames[1]),
                plotsExist ? plots : null, mode);
    }

    /**
     * Applies a finished run: tiles for tiled windows, cell measurements (and classes) for
     * per-cell windows. Runs on the JavaFX thread (the task's success handler).
     *
     * @param imageData (ImageData of BufferedImage) Image whose hierarchy is updated.
     * @param annotation (PathObject) Parent annotation.
     * @param settings (PHCParameters) Settings of the run.
     * @param result (PHCPipeline.Result) New tiles or cell results, MDS summary and plots.
     * @param seconds (double) Total run time, reported to the user.
     * @return (void)
     */
    private void showResult(ImageData<BufferedImage> imageData, PathObject annotation,
                            PHCParameters settings, PHCPipeline.Result result, double seconds) {
        if (settings.isCellMode()) {
            showCellResult(imageData, annotation, settings, result, seconds);
        } else {
            showTileResult(imageData, annotation, result, seconds);
        }
    }

    /**
     * Puts per-cell results on the annotation's cells (no tiles are added or removed), stores
     * the per-cell MDS stress and Delaunay summary on the annotation and opens the MDS viewer
     * when MDS ran.
     *
     * @param imageData (ImageData of BufferedImage) Image whose hierarchy is updated.
     * @param annotation (PathObject) Annotation PHC ran on.
     * @param settings (PHCParameters) Settings of the run, for classifyCells.
     * @param result (PHCPipeline.Result) Per-cell results, MDS summary and plot files.
     * @param seconds (double) Total run time, reported to the user.
     * @return (void)
     */
    private void showCellResult(ImageData<BufferedImage> imageData, PathObject annotation,
                                PHCParameters settings, PHCPipeline.Result result,
                                double seconds) {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        int nReclassified = PHCPipeline.applyCellResults(hierarchy, result,
                settings.classifyCells());
        PHCPipeline.storeMdsSummary(annotation, result.mds(), PHCParameters.MODE_CELLS);
        PHCPipeline.storeClusteringSummary(annotation, result.clustering());
        hierarchy.fireObjectMeasurementsChangedEvent(this, List.of(annotation));
        String message = cellSummary(result, seconds);
        if (settings.classifyCells()) {
            message += " Cell classes set to the PHC clusters (" + nReclassified + " changed); "
                    + "Clear PHC heatmap restores the originals.";
        }
        message += notes(result);
        Dialogs.showInfoNotification(TITLE, message);
        List<PathObject> clustered = result.cells().stream()
                .map(PHCPipeline.CellAssignment::cell)
                .filter(cell -> cell.getMeasurementList()
                        .containsKey(PHCPipeline.MEASUREMENT_CLUSTER))
                .toList();
        if (result.mds() != null && !PHCPipeline.embeddedTiles(clustered).isEmpty()) {
            openViewer(imageData, annotation, clustered, result.mds().stress2d(),
                    result.mds().stress3d(), result.plots(), PHCParameters.MODE_CELLS);
        } else if (annotation == mdsAnnotation) {
            closeViewer();  // it may show values that were just replaced
        }
    }

    /**
     * Summarises a per-cell run for the notification.
     *
     * @param result (PHCPipeline.Result) Per-cell result.
     * @param seconds (double) Total run time.
     * @return (String) e.g. "Computed local PH for 3200 cells; 3150 clustered into 4 clusters
     *         in 12.3 s. Use Measure > Show measurement maps > 'PHC: cluster'.", or for
     *         spatially constrained clustering "... clustered into 4 spatially contiguous
     *         clusters (Delaunay graph: 9,412 edges) in 12.3 s. ...", plus notes on a
     *         disconnected Delaunay graph, subsampled clustering and skipped cells.
     */
    static String cellSummary(PHCPipeline.Result result, double seconds) {
        PythonBridge.ClusteringResult clustering = result.clustering();
        boolean spatial = clustering != null && clustering.spatial();
        String clusters = "clusters";
        if (spatial) {
            clusters = "spatially contiguous clusters" + (clustering.nEdges() == null ? ""
                    : String.format(Locale.ROOT, " (Delaunay graph: %,d edges)",
                            clustering.nEdges()));
        }
        String summary = String.format("Computed local PH for %d cells; %d clustered into %d "
                        + "%s in %.1f s. Use Measure > Show measurement maps > '%s'.",
                result.cells().size() - result.nSkipped(), result.nClustered(),
                result.nClusters(), clusters, seconds, PHCPipeline.MEASUREMENT_CLUSTER);
        if (spatial && clustering.nComponents() != null && clustering.nComponents() > 1) {
            summary += String.format(" Warning: the Delaunay graph had %d disconnected parts; "
                    + "they were joined at their closest cells, so a cluster may span more "
                    + "than one region.", clustering.nComponents());
        }
        if (clustering != null && clustering.subsampled()) {
            summary += String.format(" Clustering was fitted on a random subsample of %d cells; "
                    + "the rest were assigned to the nearest cluster mean.", clustering.nFitted());
        }
        if (result.nSkipped() > 0) {
            summary += " " + result.nSkipped() + " cells had no usable centroid and were "
                    + "skipped.";
        }
        return summary;
    }

    /**
     * Adds where the plots went (naming the Delaunay graph plot when Python wrote one) and any
     * Python warnings to a notification.
     *
     * @param result (PHCPipeline.Result) Result of the run.
     * @return (String) Text to append, starting with a space, or "" when there is nothing.
     */
    static String notes(PHCPipeline.Result result) {
        String plotsNote = "";
        if (result.plots() != null) {
            plotsNote = " MDS plots saved in " + result.plots().dir();
            Path delaunay = result.plots().delaunayPlot();
            if (PHCParameters.MODE_CELLS.equals(result.mode()) && Files.exists(delaunay)) {
                plotsNote += ", with the Delaunay graph in " + delaunay.getFileName();
            }
            plotsNote += ".";
        }
        if (!result.warnings().isEmpty()) {
            plotsNote += " Warning: " + String.join("; ", result.warnings());
        }
        return plotsNote;
    }

    /**
     * Replaces any earlier PHC tiles under the annotation with the new ones, stores the MDS
     * stress on the annotation and opens the MDS viewer when MDS ran.
     *
     * @param imageData (ImageData of BufferedImage) Image whose hierarchy is updated.
     * @param annotation (PathObject) Parent annotation.
     * @param result (PHCPipeline.Result) New heatmap tiles, MDS summary and plot files.
     * @param seconds (double) Total run time, reported to the user.
     * @return (void)
     */
    private void showTileResult(ImageData<BufferedImage> imageData, PathObject annotation,
                                PHCPipeline.Result result, double seconds) {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        hierarchy.removeObjects(PHCPipeline.existingTiles(annotation), true);
        annotation.addChildObjects(result.tiles());
        PHCPipeline.storeMdsSummary(annotation, result.mds(), PHCParameters.MODE_WINDOWS);
        hierarchy.fireHierarchyChangedEvent(annotation);
        String plotsNote = notes(result);
        Dialogs.showInfoNotification(TITLE, String.format("Added %d tiles in %.1f s. For a "
                + "continuous heatmap use Measure > Show measurement maps > '%s'.%s",
                result.tiles().size(), seconds, PHCPipeline.MEASUREMENT_L2_NORM, plotsNote));
        if (result.mds() != null && !PHCPipeline.embeddedTiles(result.tiles()).isEmpty()) {
            openViewer(imageData, annotation, result.tiles(), result.mds().stress2d(),
                    result.mds().stress3d(), result.plots(), PHCParameters.MODE_WINDOWS);
        } else if (annotation == mdsAnnotation) {
            closeViewer();  // it shows tiles that were just replaced
        }
    }

    /**
     * Opens the MDS viewer for an annotation's tiles or clustered cells, closing any earlier
     * one. Clicking a point selects the tile or cell and centres the QuPath viewer on it.
     *
     * @param imageData (ImageData of BufferedImage) Image holding the tiles or cells.
     * @param annotation (PathObject) Annotation PHC ran on.
     * @param tiles (List of PathObject) PHC tiles of the annotation, or its clustered cells.
     * @param stress2d (Double) Stress of the 2D embedding, or null when unknown.
     * @param stress3d (Double) Stress of the 3D embedding, or null when unknown.
     * @param plots (PythonBridge.PlotTarget) Where Python saved its plots, or null.
     * @param mode (String) {@link PHCParameters#MODE_WINDOWS} or {@link PHCParameters#MODE_CELLS}.
     * @return (void)
     */
    private void openViewer(ImageData<BufferedImage> imageData, PathObject annotation,
                            List<PathObject> tiles, Double stress2d, Double stress3d,
                            PythonBridge.PlotTarget plots, String mode) {
        closeViewer();
        String prefix = plots != null ? plots.prefix()
                : PHCPipeline.plotTarget(Path.of(""), imageName(imageData), annotation).prefix();
        mdsViewer = new MDSViewer(tiles, imageData.getHierarchy(), stress2d, stress3d,
                plots == null ? null : plots.dir(), prefix, tile -> {
                    QuPathViewer viewer = qupath.getViewer();
                    if (viewer != null && viewer.getImageData() == imageData) {
                        viewer.setCenterPixelLocation(tile.getROI().getCentroidX(),
                                tile.getROI().getCentroidY());
                    }
                }, PHCParameters.MODE_CELLS.equals(mode) ? ProgressTracker.CELLS_NOUN
                        : ProgressTracker.WINDOWS_NOUN);
        mdsAnnotation = annotation;
        mdsViewer.show(qupath.getStage());
    }

    /**
     * Closes the MDS viewer if one is open.
     *
     * @return (void)
     */
    private void closeViewer() {
        if (mdsViewer != null) {
            mdsViewer.close();
        }
        mdsViewer = null;
        mdsAnnotation = null;
    }

    /**
     * Prepares the folder for Python's MDS plots inside the open project.
     *
     * @param imageData (ImageData of BufferedImage) Image PHC runs on.
     * @param annotation (PathObject) Annotation PHC runs on.
     * @return (PythonBridge.PlotTarget) Existing "PHC plots" folder and file prefix, or null
     *         when no project is open or the folder cannot be created.
     */
    private PythonBridge.PlotTarget createPlotTarget(ImageData<BufferedImage> imageData,
                                                     PathObject annotation) {
        PythonBridge.PlotTarget plots = plotTarget(imageData, annotation);
        if (plots != null) {
            try {
                Files.createDirectories(plots.dir());
            } catch (IOException e) {
                logger.warn("Could not create {}; MDS plots will not be saved", plots.dir(), e);
                plots = null;
            }
        }
        return plots;
    }

    /**
     * Works out where an annotation's MDS plots belong in the open project.
     *
     * @param imageData (ImageData of BufferedImage) Image the annotation belongs to.
     * @param annotation (PathObject) Annotation PHC ran on.
     * @return (PythonBridge.PlotTarget) Folder and prefix (not created), or null when no
     *         project is open.
     */
    private PythonBridge.PlotTarget plotTarget(ImageData<BufferedImage> imageData,
                                               PathObject annotation) {
        Project<BufferedImage> project = qupath.getProject();
        PythonBridge.PlotTarget plots = null;
        if (project != null && project.getPath() != null) {
            Path projectPath = project.getPath();  // the .qpproj file of a default project
            Path projectDir = Files.isDirectory(projectPath) ? projectPath
                    : projectPath.getParent();
            plots = PHCPipeline.plotTarget(projectDir, imageName(imageData), annotation);
        }
        return plots;
    }

    /**
     * Names the image as the project shows it, falling back to the image server's name.
     *
     * @param imageData (ImageData of BufferedImage) Current image.
     * @return (String) Image name, may be empty.
     */
    private String imageName(ImageData<BufferedImage> imageData) {
        Project<BufferedImage> project = qupath.getProject();
        ProjectImageEntry<BufferedImage> entry = project == null ? null
                : project.getEntry(imageData);
        String name = entry != null ? entry.getImageName()
                : imageData.getServer().getMetadata().getName();
        String imageName = name == null ? "" : name;
        return imageName;
    }

    /**
     * Reads an optional measurement.
     *
     * @param measurements (MeasurementList) Measurements of an object.
     * @param name (String) Measurement name.
     * @return (Double) The value, or null when it is missing or NaN.
     */
    private static Double measuredOrNull(MeasurementList measurements, String name) {
        double value = measurements.containsKey(name) ? measurements.get(name) : Double.NaN;
        Double measured = Double.isNaN(value) ? null : value;
        return measured;
    }

    /**
     * Returns the selected annotation, or explains to the user why there is none.
     *
     * @param imageData (ImageData of BufferedImage) Current image, may be null.
     * @return (PathObject) The selected annotation with an area ROI, or null.
     */
    private static PathObject selectedAnnotation(ImageData<BufferedImage> imageData) {
        if (imageData == null) {
            Dialogs.showErrorMessage(TITLE, "Open an image first.");
            return null;
        }
        PathObject selected = imageData.getHierarchy().getSelectionModel().getSelectedObject();
        if (selected == null || !selected.isAnnotation() || selected.getROI() == null
                || !selected.getROI().isArea()) {
            Dialogs.showErrorMessage(TITLE, "Select an area annotation (rectangle, ellipse, "
                    + "polygon or brush) to run PHC on.");
            return null;
        }
        return selected;
    }
}
