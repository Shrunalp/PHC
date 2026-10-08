/*
 * Background JavaFX task for one PHC run, with its progress dialog.
 *
 * Contents
 * --------
 * PHCTask : class
 *     Runs the PHC pipeline off the JavaFX thread, reporting progress, ETA and cancellation.
 */

package qupath.ext.phc;

import java.awt.image.BufferedImage;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.stage.Window;

import org.controlsfx.dialog.ProgressDialog;

import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;

/**
 * Runs PHC on one annotation as a JavaFX task, so the progress bar, the estimated time to
 * completion and the Cancel button stay responsive while Python works. Cancelling stops the
 * Python process and its workers.
 */
public final class PHCTask extends Task<PHCPipeline.Result> {

    private static final long CLOCK_PERIOD_MS = 1000;  // refresh elapsed time and ETA
    private static final double DIALOG_WIDTH = 640;     // fits the longest stage / ETA message

    private final ImageData<BufferedImage> imageData;
    private final PathObject annotation;
    private final PHCParameters settings;
    private final PythonBridge bridge;
    private final PythonBridge.PlotTarget plots;

    /**
     * Prepares a run; nothing starts until the task is run on a thread.
     *
     * @param imageData (ImageData of BufferedImage) Image the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @param settings (PHCParameters) Validated settings.
     * @param bridge (PythonBridge) Bridge configured with the user's Python.
     * @param plots (PythonBridge.PlotTarget) Existing folder and prefix for Python's MDS plots
     *        and CSV, or null to write none.
     */
    public PHCTask(ImageData<BufferedImage> imageData, PathObject annotation,
                   PHCParameters settings, PythonBridge bridge, PythonBridge.PlotTarget plots) {
        this.imageData = imageData;
        this.annotation = annotation;
        this.settings = settings;
        this.bridge = bridge;
        this.plots = plots;
        updateTitle("PHC");
        updateMessage("Starting");
        updateProgress(ProgressTracker.INDETERMINATE, 1);
    }

    /**
     * Runs the pipeline, forwarding each stage and window (or cell) count to the progress bar
     * and a once-a-second clock to the ETA message.
     *
     * @return (PHCPipeline.Result) Heatmap tiles or per-cell results, not yet added to the
     *         hierarchy, with the MDS summary and plot files.
     * @throws Exception When there are no detected cells, Python fails, or the run is cancelled.
     */
    @Override
    protected PHCPipeline.Result call() throws Exception {
        ProgressTracker tracker = new ProgressTracker(System.nanoTime(), settings.isCellMode()
                ? ProgressTracker.CELLS_NOUN : ProgressTracker.WINDOWS_NOUN,
                settings.isSpatial());
        ScheduledExecutorService clock = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "phc-progress-clock");
            thread.setDaemon(true);
            return thread;
        });
        clock.scheduleAtFixedRate(() -> {
            tracker.tick(System.nanoTime());
            updateMessage(tracker.message());
        }, CLOCK_PERIOD_MS, CLOCK_PERIOD_MS, TimeUnit.MILLISECONDS);

        try {
            PHCPipeline.Result result = PHCPipeline.analyse(imageData, annotation, settings,
                    bridge, (stage, done, total) -> {
                        tracker.update(stage, done, total, System.nanoTime());
                        updateMessage(tracker.message());
                        // INDETERMINATE (negative) makes the bar indeterminate (animated)
                        updateProgress(tracker.fraction(), 1);
                    }, plots);
            return result;
        } finally {
            clock.shutdownNow();
        }
    }

    /**
     * Stops Python when the task is cancelled; JavaFX calls this on the application thread.
     *
     * @return (void)
     */
    @Override
    protected void cancelled() {
        Thread stopper = new Thread(bridge::cancel, "phc-cancel");  // may wait up to 2 s
        stopper.setDaemon(true);
        stopper.start();
    }

    /**
     * Creates the progress dialog for this task: a progress bar, the stage / ETA message and a
     * Cancel button. It opens when the task starts and closes when it finishes.
     *
     * @param owner (Window) Window the dialog belongs to, may be null.
     * @param header (String) One-line description of the run, shown above the bar.
     * @return (ProgressDialog) The dialog, already bound to this task.
     */
    public ProgressDialog createProgressDialog(Window owner, String header) {
        ProgressDialog dialog = new ProgressDialog(this);
        if (owner != null) {
            dialog.initOwner(owner);
        }
        dialog.setTitle("PHC");
        dialog.setHeaderText(header);
        // sized for the first message ("Starting") by default, which cuts off the ETA later
        dialog.getDialogPane().setPrefWidth(DIALOG_WIDTH);
        dialog.setResizable(true);
        dialog.getDialogPane().getButtonTypes().add(ButtonType.CANCEL);
        Button cancelButton = (Button) dialog.getDialogPane().lookupButton(ButtonType.CANCEL);
        cancelButton.addEventFilter(ActionEvent.ACTION, event -> cancel());
        dialog.setOnCloseRequest(event -> {
            if (isRunning()) {
                cancel();  // closing the window also stops the run
            }
        });
        return dialog;
    }
}
