/*
 * On-screen check of the PHC progress dialog (progress, ETA, auto-close, Cancel) and MDS viewer.
 *
 * Contents
 * --------
 * PHCTaskDialogCheck : class
 *     Runs PHCTask with its real ProgressDialog (plain and Delaunay-constrained clustering),
 *     opens the MDS viewer on the cells and asserts on both.
 */

package qupath.ext.phc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.Event;
import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;

import javax.imageio.ImageIO;

import org.controlsfx.dialog.ProgressDialog;

import qupath.lib.common.ColorTools;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

/**
 * Exercises the progress dialog as a user sees it, which the headless check cannot: the
 * dialog must open, show the cell count and time left, close itself when PHC finishes, and
 * stop Python when Cancel is pressed; the MDS viewer must draw both plots in the cluster
 * colours and link its points to the cell selection. Runs on the synthetic cells of
 * {@link SyntheticSlide}.
 */
public final class PHCTaskDialogCheck {

    private static final long POLL_MS = 100;
    private static final long TIMEOUT_S = 300;
    private static final double SNAPSHOT_AT = 0.4;   // progress at which to screenshot
    private static final double CANCEL_AT = 0.15;    // progress at which to press Cancel
    /** A message with a real estimate, e.g. "..., about 0:12 left (elapsed 0:04)". */
    private static final String ETA_PATTERN = ".*, about [0-9:]+ left \\(elapsed.*";
    private static final long RENDER_WAIT_MS = 1500;  // let the viewer lay out and render
    private static final int COLOUR_TOLERANCE = 12;   // per channel, for antialiased dots
    private static final int MIN_PIXELS_PER_CLUSTER = 20;
    private static final int MIN_3D_COLOURS = 200;    // shaded spheres give many tones
    private static final int CELL_MDS_MAX = 1000;     // per-cell MDS subsample, for speed
    /** e.g. "Computing persistence: 120 / 3200 cells (4%), ...". */
    private static final String CELL_PROGRESS_PATTERN =
            "Computing persistence: [0-9]+ / [0-9]+ cells \\([0-9]+%\\).*";

    private static int failures = 0;

    private PHCTaskDialogCheck() {
    }

    /**
     * Runs the dialog checks and exits non-zero if any fail.
     *
     * @param args (String[]) Python executable, PHC library folder, and the folder to write the
     *        test image and dialog screenshot to.
     * @return (void)
     * @throws Exception When the test image cannot be opened or JavaFX fails to start.
     */
    public static void main(String[] args) throws Exception {
        Platform.startup(() -> { });
        Platform.setImplicitExit(false);
        ImageData<BufferedImage> imageData = SyntheticSlide.createImageData(new File(args[2]));
        PathObject annotation = SyntheticSlide.addAnnotationWithCells(imageData);
        PHCParameters slow = PHCPipelineCheck.slowParams();  // time to see the ETA

        checkCompletedRun(imageData, annotation, slow, new PythonBridge(args[0], args[1]),
                new File(args[2], "progress_dialog.png"));
        checkCellRun(imageData, annotation, new PythonBridge(args[0], args[1]),
                new File(args[2]));
        checkSpatialCellRun(imageData, annotation, new PythonBridge(args[0], args[1]));
        checkCancelledRun(imageData, annotation, slow, new PythonBridge(args[0], args[1]));

        System.out.println(failures == 0 ? "ALL DIALOG CHECKS PASSED"
                : failures + " DIALOG CHECK(S) FAILED");
        System.exit(failures == 0 ? 0 : 1);
    }

    /**
     * Runs a task to completion and checks the dialog opens, reports an ETA and closes.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param settings (PHCParameters) Settings for the run.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param screenshot (File) PNG the mid-run dialog is saved to.
     * @return (void)
     * @throws Exception When waiting is interrupted or the screenshot cannot be written.
     */
    private static void checkCompletedRun(ImageData<BufferedImage> imageData,
                                          PathObject annotation, PHCParameters settings,
                                          PythonBridge bridge, File screenshot)
            throws Exception {
        PHCTask task = new PHCTask(imageData, annotation, settings, bridge, null);
        ProgressDialog dialog = openDialog(task, PHCCommand.progressHeader(settings,
                SyntheticSlide.nCellsInside(), imageData.getServer().getPixelCalibration()));
        List<String> messages = new CopyOnWriteArrayList<>();
        task.messageProperty().addListener((obs, old, message) -> messages.add(message));
        AtomicBoolean shownWhileRunning = new AtomicBoolean(false);
        AtomicBoolean snapshotTaken = new AtomicBoolean(false);

        startAndWait(task, () -> {
            if (dialog.isShowing() && task.isRunning()) {
                shownWhileRunning.set(true);
            }
            if (!snapshotTaken.get() && task.getProgress() >= SNAPSHOT_AT
                    && dialog.isShowing()) {
                snapshotTaken.set(true);
                WritableImage image = dialog.getDialogPane().snapshot(null, null);
                try {
                    ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", screenshot);
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            }
        }, null);

        check(shownWhileRunning.get(), "dialog is open while PHC runs");
        // Task state may only be read on the JavaFX thread
        javafx.concurrent.Worker.State state = onFx(task::getState);
        PHCPipeline.Result result = onFx(task::getValue);
        int nResults = result == null ? 0 : result.cells().size();
        check(state == javafx.concurrent.Worker.State.SUCCEEDED
                && nResults == SyntheticSlide.nCellsInside(), "task succeeds with " + nResults
                + " cell results");
        check(messages.stream().anyMatch(m -> m.matches(ETA_PATTERN)),
                "dialog showed a time-left estimate, e.g. '" + messages.stream()
                        .filter(m -> m.matches(ETA_PATTERN))
                        .findFirst().orElse("none")
                        + "'");
        check(messages.stream().anyMatch(m -> m.startsWith("Clustering")),
                "dialog showed the clustering stage");
        check(!onFx(dialog::isShowing), "dialog closes itself when PHC finishes");
        check(snapshotTaken.get(), "mid-run screenshot saved to " + screenshot);
        check(messages.stream().anyMatch(m -> m.startsWith("Projecting")),
                "dialog showed the MDS stage");
    }

    /**
     * Runs per-cell windows through PHCTask and its dialog, applies the results to the cells
     * on the JavaFX thread as the menu command does, and checks the MDS viewer on the cells:
     * snapshots, colours, and that a click selects a cell.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @param outDir (File) Folder the snapshots are written to.
     * @return (void)
     * @throws Exception When waiting is interrupted or the viewer fails.
     */
    private static void checkCellRun(ImageData<BufferedImage> imageData, PathObject annotation,
                                     PythonBridge bridge, File outDir) throws Exception {
        PHCParameters settings = new PHCParameters(1, 100, "PI", 20, 10, 4, "ward", 0.5, -1,
                true, CELL_MDS_MAX, false, false, 0);
        PHCTask task = new PHCTask(imageData, annotation, settings, bridge, null);
        int nCells = SyntheticSlide.nCellsInside();
        ProgressDialog dialog = openDialog(task, PHCCommand.progressHeader(settings, nCells,
                imageData.getServer().getPixelCalibration()));
        check(onFx(dialog::getHeaderText).startsWith("Alpha PHC per cell on " + nCells),
                "per-cell dialog header: " + onFx(dialog::getHeaderText));
        List<String> messages = new CopyOnWriteArrayList<>();
        task.messageProperty().addListener((obs, old, message) -> messages.add(message));
        startAndWait(task, () -> { }, null);

        PHCPipeline.Result result = onFx(task::getValue);
        check(onFx(task::getState) == javafx.concurrent.Worker.State.SUCCEEDED && result != null
                        && result.cells().size() == nCells,
                "per-cell task succeeds with " + (result == null ? 0 : result.cells().size())
                        + " cell results");
        check(messages.stream().anyMatch(m -> m.matches(CELL_PROGRESS_PATTERN)),
                "per-cell dialog counts cells, e.g. '" + messages.stream()
                        .filter(m -> m.matches(CELL_PROGRESS_PATTERN)).findFirst()
                        .orElse("none") + "'");
        check(messages.stream().anyMatch(m -> m.startsWith("Clustering")
                        && m.contains(" cells")),
                "per-cell dialog showed the clustering stage in cells");
        check(!onFx(dialog::isShowing), "per-cell dialog closes itself when PHC finishes");
        if (result == null) {
            return;
        }
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        onFx(() -> PHCPipeline.applyCellResults(hierarchy, result, false));
        List<PathObject> clustered = PHCPipeline.clusteredCells(hierarchy, annotation);
        checkViewer(hierarchy, clustered, result.mds(), new File(outDir, "mds_cells_2d_view.png"),
                new File(outDir, "mds_cells_3d_view.png"));
        System.out.println("  " + PHCCommand.cellSummary(result, 0));
    }

    /**
     * Runs a per-cell task with Delaunay-constrained clustering (MDS off, for speed) and checks
     * the dialog header and clustering stage say so and the task succeeds.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When waiting is interrupted.
     */
    private static void checkSpatialCellRun(ImageData<BufferedImage> imageData,
                                            PathObject annotation, PythonBridge bridge)
            throws Exception {
        PHCParameters settings = new PHCParameters(1, 100, "PI", 20, 10, 4, "ward", 0.5, -1,
                false, CELL_MDS_MAX, false, true, 0);
        PHCTask task = new PHCTask(imageData, annotation, settings, bridge, null);
        int nCells = SyntheticSlide.nCellsInside();
        ProgressDialog dialog = openDialog(task, PHCCommand.progressHeader(settings, nCells,
                imageData.getServer().getPixelCalibration()));
        check(onFx(dialog::getHeaderText).endsWith("Delaunay-constrained"),
                "spatial dialog header: " + onFx(dialog::getHeaderText));
        List<String> messages = new CopyOnWriteArrayList<>();
        task.messageProperty().addListener((obs, old, message) -> messages.add(message));
        startAndWait(task, () -> { }, null);
        PHCPipeline.Result result = onFx(task::getValue);
        check(onFx(task::getState) == javafx.concurrent.Worker.State.SUCCEEDED && result != null
                        && result.cells().size() == nCells && result.clustering() != null
                        && result.clustering().spatial(),
                "spatial per-cell task succeeds: " + (result == null ? null
                        : result.clustering()));
        check(messages.stream().anyMatch(m -> m.startsWith("Clustering ")
                        && m.contains(" cells (Delaunay-constrained)")),
                "spatial dialog showed '" + messages.stream()
                        .filter(m -> m.startsWith("Clustering")).findFirst().orElse("none")
                        + "'");
        if (result != null) {
            System.out.println("  " + PHCCommand.cellSummary(result, 0));
        }
    }

    /**
     * Opens the MDS viewer on the clustered cells, saves both tabs, and checks the plots use
     * the cluster colours, a click selects and centres on the cell, a QuPath selection
     * highlights the point, and closing the window stops listening.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy holding the cells.
     * @param objects (List of PathObject) Clustered cells.
     * @param mds (PythonBridge.MdsResult) MDS summary of the run, may be null.
     * @param view2d (File) PNG the 2D tab is saved to.
     * @param view3d (File) PNG the 3D tab is saved to.
     * @return (void)
     * @throws Exception When the viewer fails or a snapshot cannot be written.
     */
    private static void checkViewer(PathObjectHierarchy hierarchy, List<PathObject> objects,
                                    PythonBridge.MdsResult mds, File view2d, File view3d)
            throws Exception {
        List<PathObject> embedded = PHCPipeline.embeddedCells(objects);
        check(mds != null && !embedded.isEmpty(), "the run has an MDS embedding of "
                + embedded.size() + " cells: " + mds);
        if (embedded.isEmpty() || mds == null) {
            return;
        }
        AtomicReference<PathObject> centred = new AtomicReference<>();
        MDSViewer viewer = onFx(() -> {
            MDSViewer created = new MDSViewer(objects, hierarchy, mds.stress2d(),
                    mds.stress3d(), null, "synthetic", centred::set);
            created.show(null);
            return created;
        });
        Thread.sleep(RENDER_WAIT_MS);
        check(onFx(() -> viewer.getStage().getTitle()).contains("cells"),
                "viewer title names the cells: " + onFx(() -> viewer.getStage().getTitle()));

        onFx(() -> {
            viewer.saveSnapshot(MDSViewer.TAB_2D, view2d);
            return null;
        });
        onFx(() -> {
            viewer.selectTab(MDSViewer.TAB_3D);
            return null;
        });
        Thread.sleep(RENDER_WAIT_MS);
        onFx(() -> {
            viewer.saveSnapshot(MDSViewer.TAB_3D, view3d);
            return null;
        });
        BufferedImage image2d = ImageIO.read(view2d);
        BufferedImage image3d = ImageIO.read(view3d);
        int nClusters = (int) objects.stream().mapToDouble(t -> t.getMeasurementList()
                .get(PHCPipeline.MEASUREMENT_CLUSTER)).max().orElse(1);
        Set<Integer> clusterColours = new HashSet<>();
        embedded.forEach(t -> clusterColours.add(PHCPipeline.clusterColor((int) t
                .getMeasurementList().get(PHCPipeline.MEASUREMENT_CLUSTER) - 1, nClusters)));
        int missing = 0;
        for (int colour : clusterColours) {
            missing += countNear(image2d, colour) >= MIN_PIXELS_PER_CLUSTER ? 0 : 1;
        }
        check(missing == 0, "2D plot saved to " + view2d + " shows all "
                + clusterColours.size() + " cluster colours (" + missing + " missing)");
        int tones = distinctColours(image3d);
        check(tones >= MIN_3D_COLOURS, "3D plot saved to " + view3d + " is shaded ("
                + tones + " distinct colours)");

        // Click on a point in the 2D plot
        onFx(() -> {
            viewer.selectTab(MDSViewer.TAB_2D);
            return null;
        });
        Thread.sleep(RENDER_WAIT_MS);
        PathObject target = embedded.get(embedded.size() / 2);
        PathObject clicked = onFx(() -> {
            Canvas canvas = viewer.canvas2d();
            Point2D local = viewer.canvasPosition(target);
            Point2D scene = canvas.localToScene(local);
            Point2D screen = canvas.localToScreen(local);
            Event.fireEvent(canvas, new MouseEvent(MouseEvent.MOUSE_CLICKED, scene.getX(),
                    scene.getY(), screen.getX(), screen.getY(), MouseButton.PRIMARY, 1, false,
                    false, false, false, false, false, false, false, false, true, null));
            return hierarchy.getSelectionModel().getSelectedObject();
        });
        boolean samePoint = clicked != null && onFx(() -> viewer.canvasPosition(clicked)
                .distance(viewer.canvasPosition(target)) < 1.0);
        check(samePoint && clicked.isCell(), "clicking a 2D point selects its cell in QuPath");
        check(centred.get() == clicked && clicked != null, "the click centres the viewer on "
                + "the cell");
        check(onFx(viewer::highlightedCell) == clicked, "the clicked point is highlighted");

        PathObject other = embedded.get(0);
        onFx(() -> {
            hierarchy.getSelectionModel().setSelectedObject(other);
            return null;
        });
        check(onFx(viewer::highlightedCell) == other, "selecting a cell in QuPath highlights "
                + "its point");
        onFx(() -> {
            viewer.close();
            hierarchy.getSelectionModel().setSelectedObject(target);
            return null;
        });
        check(onFx(viewer::highlightedCell) == other && !onFx(viewer.getStage()::isShowing),
                "closing the viewer stops it following the selection");
    }

    /**
     * Counts pixels close to a colour, to find a cluster's dots in a snapshot.
     *
     * @param image (BufferedImage) Snapshot.
     * @param rgb (int) Packed RGB colour.
     * @return (int) Pixels within {@link #COLOUR_TOLERANCE} of the colour in every channel.
     */
    private static int countNear(BufferedImage image, int rgb) {
        int count = 0;
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int pixel = image.getRGB(x, y);
                boolean near = Math.abs(ColorTools.red(pixel) - ColorTools.red(rgb))
                        <= COLOUR_TOLERANCE
                        && Math.abs(ColorTools.green(pixel) - ColorTools.green(rgb))
                        <= COLOUR_TOLERANCE
                        && Math.abs(ColorTools.blue(pixel) - ColorTools.blue(rgb))
                        <= COLOUR_TOLERANCE;
                count += near ? 1 : 0;
            }
        }
        return count;
    }

    /**
     * Counts the distinct colours in an image, a rough test that 3D shading was rendered.
     *
     * @param image (BufferedImage) Snapshot.
     * @return (int) Number of distinct RGB values.
     */
    private static int distinctColours(BufferedImage image) {
        Set<Integer> colours = new HashSet<>();
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                colours.add(image.getRGB(x, y) & 0xFFFFFF);
            }
        }
        int count = colours.size();
        return count;
    }

    /**
     * Presses Cancel part-way through and checks the task, dialog and Python all stop.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @param annotation (PathObject) Annotation to analyse.
     * @param settings (PHCParameters) Settings for the run.
     * @param bridge (PythonBridge) Bridge to the PHC library.
     * @return (void)
     * @throws Exception When waiting is interrupted.
     */
    private static void checkCancelledRun(ImageData<BufferedImage> imageData,
                                          PathObject annotation, PHCParameters settings,
                                          PythonBridge bridge) throws Exception {
        PHCTask task = new PHCTask(imageData, annotation, settings, bridge, null);
        ProgressDialog dialog = openDialog(task, PHCCommand.progressHeader(settings,
                SyntheticSlide.nCellsInside(), imageData.getServer().getPixelCalibration()));
        AtomicBoolean pressed = new AtomicBoolean(false);
        startAndWait(task, () -> {
            if (!pressed.get() && task.getProgress() >= CANCEL_AT) {
                pressed.set(true);
                ((Button) dialog.getDialogPane().lookupButton(ButtonType.CANCEL)).fire();
            }
        }, pressed);

        check(onFx(task::getState) == javafx.concurrent.Worker.State.CANCELLED,
                "pressing Cancel cancels the task");
        check(!onFx(dialog::isShowing), "dialog closes after Cancel");
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        long alive = Long.MAX_VALUE;
        while (System.nanoTime() < deadline && alive > 0) {
            Thread.sleep(POLL_MS);
            alive = ProcessHandle.current().descendants().filter(ProcessHandle::isAlive).count();
        }
        check(alive == 0, "Python and its workers stop after Cancel (" + alive + " left)");
    }

    /**
     * Creates the task's dialog on the JavaFX thread.
     *
     * @param task (PHCTask) Task the dialog follows.
     * @param header (String) One-line description of the run, as the menu command builds it.
     * @return (ProgressDialog) Dialog bound to the task.
     * @throws Exception When the JavaFX thread fails.
     */
    private static ProgressDialog openDialog(PHCTask task, String header) throws Exception {
        ProgressDialog dialog = onFx(() -> task.createProgressDialog(null,
header));
        return dialog;
    }

    /**
     * Runs the task on a worker thread and polls on the JavaFX thread until it finishes.
     *
     * @param task (PHCTask) Task to run.
     * @param poll (Runnable) Called on the JavaFX thread every {@link #POLL_MS} ms.
     * @param untilCancelled (AtomicBoolean) When not null, stop waiting once it is true and
     *        the task is no longer running.
     * @return (void)
     * @throws Exception When waiting is interrupted.
     */
    private static void startAndWait(PHCTask task, Runnable poll, AtomicBoolean untilCancelled)
            throws Exception {
        Thread worker = new Thread(task, "phc-runner");
        worker.setDaemon(true);
        worker.start();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_S);
        while (!task.isDone() && System.nanoTime() < deadline) {
            onFx(() -> {
                poll.run();
                return null;
            });
            Thread.sleep(POLL_MS);
        }
        Thread.sleep(2 * POLL_MS);  // let the dialog react to the final state
    }

    /**
     * Evaluates something on the JavaFX thread and waits for the answer.
     *
     * @param action (java.util.concurrent.Callable of T) Work to run on the JavaFX thread.
     * @param <T> Result type.
     * @return (T) The action's result.
     * @throws Exception When the action throws or the wait is interrupted.
     */
    private static <T> T onFx(java.util.concurrent.Callable<T> action) throws Exception {
        AtomicReference<T> result = new AtomicReference<>();
        AtomicReference<Exception> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        Platform.runLater(() -> {
            try {
                result.set(action.call());
            } catch (Exception e) {
                error.set(e);
            } finally {
                latch.countDown();
            }
        });
        latch.await();
        if (error.get() != null) {
            throw error.get();
        }
        return result.get();
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
