/*
 * Turns PHC stage and window- or cell-count updates into progress-bar values and ETA messages.
 *
 * Contents
 * --------
 * ProgressTracker : class
 *     Tracks the current PHC stage and estimates the time left from the window (or cell) rate.
 */

package qupath.ext.phc;

/**
 * Keeps track of which PHC stage is running and how fast windows (or, in per-cell mode,
 * cells) are finishing, so the progress dialog can show a fraction and an estimated time to
 * completion. Holds the current
 * stage and timing as state; times are passed in, which keeps the estimates testable. Methods
 * are synchronized because updates arrive on the worker thread while a clock ticks on another.
 */
public final class ProgressTracker {

    /** Stage names, in the order a run goes through them. */
    public static final String EXPORT = "export";
    public static final String STARTING = "starting";
    public static final String CENTROIDS = "centroids";
    public static final String PERSISTENCE = "persistence";
    public static final String CLUSTERING = "clustering";
    public static final String MDS = "mds";
    public static final String TILES = "tiles";
    /** Per-cell mode's last stage, instead of {@link #TILES}: matching results to the cells. */
    public static final String CELLS = "cells";

    /** What the persistence stage counts in each mode. */
    public static final String WINDOWS_NOUN = "windows";
    public static final String CELLS_NOUN = "cells";

    /** Value of {@link #fraction()} when the stage has no measurable progress. */
    public static final double INDETERMINATE = -1;

    private static final double NANOS_PER_SECOND = 1e9;
    private static final double MIN_SECONDS_FOR_ETA = 1.0;  // too early to estimate before this
    private static final int SECONDS_PER_MINUTE = 60;
    private static final int SECONDS_PER_HOUR = 3600;
    private static final String SPATIAL_NOTE = " (Delaunay-constrained)";  // clustering label

    private final long runStartNanos;
    private final String noun;
    private final boolean spatial;
    private String stage = EXPORT;
    private long stageStartNanos;
    private long done = 0;
    private long total = 0;
    private long nowNanos;

    /**
     * Starts tracking a run, recording what its persistence stage counts and whether its
     * clustering is constrained to the Delaunay graph of the cell centroids, so the stage
     * messages say so.
     *
     * @param runStartNanos (long) {@link System#nanoTime()} when the run began.
     * @param noun (String) {@link #WINDOWS_NOUN} for tiled windows, {@link #CELLS_NOUN} for
     *        per-cell windows; used in the stage messages.
     * @param spatial (boolean) True for spatially constrained (Delaunay) clustering, e.g.
     *        "Clustering 3200 cells (Delaunay-constrained)".
     */
    public ProgressTracker(long runStartNanos, String noun, boolean spatial) {
        this.runStartNanos = runStartNanos;
        this.noun = noun;
        this.spatial = spatial;
        this.stageStartNanos = runStartNanos;
        this.nowNanos = runStartNanos;
    }

    /**
     * Records a progress event; a new stage name restarts the stage clock used for the ETA.
     *
     * @param newStage (String) One of the stage constants above.
     * @param newDone (long) Items finished in this stage (windows for persistence).
     * @param newTotal (long) Items in this stage, 0 or less when unknown.
     * @param now (long) {@link System#nanoTime()} of the event.
     * @return (void)
     * @throws IllegalArgumentException When newStage is not a known stage.
     */
    public synchronized void update(String newStage, long newDone, long newTotal, long now) {
        stageLabel(newStage, 0, 0, noun, spatial);  // validates the stage name
        if (!newStage.equals(stage)) {
            stage = newStage;
            stageStartNanos = now;
        }
        done = newDone;
        total = newTotal;
        nowNanos = now;
    }

    /**
     * Advances the clock without new progress, so the elapsed time and the estimate keep
     * moving while a slow window is being computed.
     *
     * @param now (long) {@link System#nanoTime()} of the tick.
     * @return (void)
     */
    public synchronized void tick(long now) {
        nowNanos = now;
    }

    /**
     * Gives the progress-bar value for the current stage.
     *
     * @return (double) Fraction in [0, 1] while windows are computed, else
     *         {@link #INDETERMINATE}.
     */
    public synchronized double fraction() {
        double fraction = INDETERMINATE;
        if (stage.equals(PERSISTENCE) && total > 0) {
            fraction = Math.min(1.0, (double) done / total);
        }
        return fraction;
    }

    /**
     * Estimates the seconds left in the persistence stage from the average time per window
     * so far.
     *
     * @return (double) Seconds remaining, or -1 when there is not enough data yet.
     */
    public synchronized double secondsRemaining() {
        double stageSeconds = (nowNanos - stageStartNanos) / NANOS_PER_SECOND;
        double remaining = -1;
        if (stage.equals(PERSISTENCE) && done > 0 && total > 0
                && stageSeconds >= MIN_SECONDS_FOR_ETA) {
            remaining = stageSeconds / done * (total - done);
        }
        return remaining;
    }

    /**
     * Builds the text shown above the progress bar.
     *
     * @return (String) Stage description, the ETA while windows are computed, and the elapsed
     *         time, e.g. "Computing persistence: 120 / 256 windows (47%), about 0:42 left
     *         (elapsed 0:38)".
     */
    public synchronized String message() {
        String text = stageLabel(stage, done, total, noun, spatial);
        if (stage.equals(PERSISTENCE) && total > 0) {
            double remaining = secondsRemaining();
            text += remaining < 0 ? ", estimating time left"
                    : ", about " + formatDuration(remaining) + " left";
        }
        double elapsed = (nowNanos - runStartNanos) / NANOS_PER_SECOND;
        String message = text + " (elapsed " + formatDuration(elapsed) + ")";
        return message;
    }

    /**
     * Formats a duration as m:ss, or h:mm:ss from one hour.
     *
     * @param seconds (double) Duration in seconds; negative values are treated as 0.
     * @return (String) Formatted duration, rounded up to the next second.
     */
    public static String formatDuration(double seconds) {
        long total = (long) Math.ceil(Math.max(0, seconds));
        long hours = total / SECONDS_PER_HOUR;
        long minutes = (total % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
        long secs = total % SECONDS_PER_MINUTE;
        String formatted = hours > 0 ? String.format("%d:%02d:%02d", hours, minutes, secs)
                : String.format("%d:%02d", minutes, secs);
        return formatted;
    }

    /**
     * Describes a stage in words.
     *
     * @param name (String) Stage name.
     * @param stageDone (long) Items finished in the stage.
     * @param stageTotal (long) Items in the stage, 0 or less when unknown.
     * @param items (String) What is counted, "windows" or "cells".
     * @param spatialClustering (boolean) Whether clustering is Delaunay-constrained.
     * @return (String) Human-readable description of the stage.
     * @throws IllegalArgumentException When name is not a known stage.
     */
    private static String stageLabel(String name, long stageDone, long stageTotal,
                                     String items, boolean spatialClustering) {
        String label;
        switch (name) {
            case EXPORT -> label = "Exporting detected cells";
            case STARTING -> label = "Starting Python";
            case CENTROIDS -> label = "Reading cell centroids";
            case PERSISTENCE -> label = stageTotal > 0
                    ? String.format("Computing persistence: %d / %d %s (%d%%)", stageDone,
                            stageTotal, items, Math.round(100.0 * stageDone / stageTotal))
                    : "Computing persistence";
            case CLUSTERING -> label = (stageTotal > 0
                    ? "Clustering " + stageTotal + " " + items : "Clustering " + items)
                    + (spatialClustering ? SPATIAL_NOTE : "");
            case MDS -> label = stageTotal > 0
                    ? "Projecting " + stageTotal + " " + items + " with MDS (2D and 3D)"
                    : "Projecting " + items + " with MDS (2D and 3D)";
            case TILES -> label = "Building heatmap tiles";
            case CELLS -> label = "Matching results to cells";
            default -> throw new IllegalArgumentException("Unknown PHC stage '" + name
                    + "'; expected export, starting, centroids, persistence, clustering, "
                    + "mds, tiles or cells.");
        }
        return label;
    }
}
