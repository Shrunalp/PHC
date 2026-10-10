/*
 * User-facing settings for a PHC run, and the QuPath dialog that edits them.
 *
 * Contents
 * --------
 * PHCParameters : record
 *     Validated alpha PHC, clustering (optionally Delaunay-constrained) and MDS settings,
 *     convertible to bridge arguments.
 */

package qupath.ext.phc;

import java.util.List;

import qupath.lib.common.GeneralTools;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.plugins.parameters.ParameterList;

/**
 * Holds every setting the PHC bridge needs, so one validated object travels from the dialog to
 * the Python process. PHC is computed on the alpha complex of the cell centroids in one
 * window centred on each cell. Build it with {@link #fromParameterList(ParameterList)} after
 * showing {@link #createParameterList()} to the user.
 *
 * @param dimension (int) Persistent homology dimension, 0 or 1 (the alpha complex of points in
 *        the plane has no higher homology).
 * @param windowSize (double) Side length of the square window centred on each cell, in
 *        micrometres; in pixels when the image has no pixel size.
 * @param vectorization (String) One of {@link #VECTORIZATIONS}.
 * @param vectorResolution (int) Side length of each persistence image, or silhouette length.
 * @param minCells (int) Cells whose window holds fewer cell centroids than this (the centre
 *        cell included) are not clustered.
 * @param nClusters (int) Number of agglomerative clusters.
 * @param linkage (String) One of {@link #LINKAGES}.
 * @param minCoverage (double) Fraction of a cell's window that must lie inside the ROI for
 *        the cell to be clustered.
 * @param nJobs (int) Worker processes for the persistence stage, -1 = all cores.
 * @param computeMds (boolean) Whether to project the cells' L2 dissimilarity matrix into 2D
 *        and 3D with metric MDS, default true.
 * @param mdsMaxWindows (int) Most cells embedded by MDS, at least 2, default 3000; larger
 *        runs embed a random subsample of this many clustered cells.
 * @param classifyCells (boolean) Set each clustered cell's class to its "PHC cluster k"
 *        class, remembering the original so Clear can restore it; default false.
 * @param spatialClustering (boolean) Constrain the agglomerative clustering to the Delaunay
 *        triangulation of the clustered cells' centroids, so each cluster is a spatially
 *        contiguous region; default false.
 * @param maxEdgeLength (double) Delaunay edges longer than this are dropped from the
 *        adjacency, in the same unit as windowSize; default 0 = no limit. Must be >= 0.
 */
public record PHCParameters(
        int dimension,
        double windowSize,
        String vectorization,
        int vectorResolution,
        int minCells,
        int nClusters,
        String linkage,
        double minCoverage,
        int nJobs,
        boolean computeMds,
        int mdsMaxWindows,
        boolean classifyCells,
        boolean spatialClustering,
        double maxEdgeLength) {

    /** Vectorizations understood by PHC: persistence image or persistence silhouette. */
    public static final List<String> VECTORIZATIONS = List.of("PI", "PL");

    /** Default for {@link #classifyCells()}: cell classes are left alone. */
    public static final boolean DEFAULT_CLASSIFY_CELLS = false;

    /** Default for {@link #spatialClustering()}: clusters are not constrained in space. */
    public static final boolean DEFAULT_SPATIAL_CLUSTERING = false;

    /** Default for {@link #maxEdgeLength()}: every Delaunay edge is kept (no limit). */
    public static final double DEFAULT_MAX_EDGE_LENGTH = 0;

    /** Linkages supported by sklearn's AgglomerativeClustering with a Euclidean metric. */
    public static final List<String> LINKAGES = List.of("ward", "average", "complete", "single");

    /** Dialog key for the Python executable; not part of the record, stored as a preference. */
    public static final String PYTHON_PATH_KEY = "pythonPath";

    /** Dialog key for the folder holding the PHC package; stored as a preference. */
    public static final String PHC_LIBRARY_DIR_KEY = "phcLibraryDir";

    /** Pixel size assumed for an uncalibrated image, so lengths are read as pixels. */
    public static final double UNCALIBRATED_PIXEL_SIZE = 1.0;

    /** Default for {@link #computeMds()}: MDS runs unless switched off. */
    public static final boolean DEFAULT_COMPUTE_MDS = true;

    /**
     * Default for {@link #mdsMaxWindows()}. MDS needs O(n^2) memory and time: about 3 s for
     * 1000 windows, 30 s for 3000 and 90 s (2.3 GB) for 5000.
     */
    public static final int DEFAULT_MDS_MAX_WINDOWS = 3000;

    private static final int MAX_ALPHA_DIMENSION = 1;  // points in the plane: H0 and H1 only
    private static final int MIN_MDS_WINDOWS = 2;      // MDS of one point is meaningless

    /**
     * Tells whether the clustering of this run is constrained to the Delaunay graph of the
     * cell centroids, so callers can label progress and results accordingly.
     *
     * @return (boolean) True when {@link #spatialClustering()} is on.
     */
    public boolean isSpatial() {
        boolean spatial = spatialClustering;
        return spatial;
    }

    /**
     * Builds the dialog contents with the default settings. The Python environment comes
     * first, so users can see which Python a run will use.
     *
     * @return (ParameterList) Editable parameters, keyed by the record component names plus
     *         {@link #PYTHON_PATH_KEY} and {@link #PHC_LIBRARY_DIR_KEY}.
     */
    public static ParameterList createParameterList() {
        String micrometres = GeneralTools.micrometerSymbol();
        ParameterList params = new ParameterList()
                .addTitleParameter("Python environment")
                .addStringParameter(PYTHON_PATH_KEY, "Python executable", "",
                        "Full path to a Python with gudhi, scikit-learn, opencv and joblib, "
                                + "e.g. ~/miniconda3/envs/phc/bin/python")
                .addStringParameter(PHC_LIBRARY_DIR_KEY, "PHC library folder", "",
                        "Folder that contains the PHC package, e.g. ~/PHC")
                .addTitleParameter("Alpha persistence of cell centroids (per-cell windows)")
                .addDoubleParameter("windowSize", "Window size", 100, micrometres,
                        "Side length of the square window centred on each cell, in "
                                + micrometres + " (in pixels if the image has no pixel size)")
                .addIntParameter("minCells", "Min cells per window", 10, null,
                        "Cells whose window holds fewer cell centroids (the centre cell "
                                + "included) are left out of the clustering")
                .addIntParameter("dimension", "Homology dimension", 1, null,
                        "0 = connected components, 1 = loops (e.g. glands)")
                .addChoiceParameter("vectorization", "Vectorization", "PI", VECTORIZATIONS,
                        "PI = persistence image, PL = persistence silhouette")
                .addIntParameter("vectorResolution", "Vector resolution", 20, null,
                        "Persistence image side length (or silhouette length)")
                .addIntParameter("nJobs", "Worker processes", -1, null,
                        "-1 uses every core, 1 runs serially")
                .addTitleParameter("Clustering (L2 distance)")
                .addIntParameter("nClusters", "Number of clusters", 4, null,
                        "Agglomerative clusters the cells are grouped into")
                .addChoiceParameter("linkage", "Linkage", "ward", LINKAGES,
                        "How the distance between two clusters is measured")
                .addDoubleParameter("minCoverage", "Min ROI coverage", 0.5, null,
                        "Cells whose window has less of its area inside the ROI are left out "
                                + "of the clustering")
                .addTitleParameter("MDS embedding")
                .addBooleanParameter("computeMds", "Compute MDS embedding", DEFAULT_COMPUTE_MDS,
                        "Project the windows' L2 distances into 2D and 3D (metric MDS) and "
                                + "show them as interactive plots")
                .addIntParameter("mdsMaxWindows", "Max windows for MDS",
                        DEFAULT_MDS_MAX_WINDOWS, null,
                        "Larger runs embed a random subsample of this many windows. MDS time "
                                + "and memory grow as n^2: about 3 s for 1000 windows, 30 s "
                                + "for 3000, 90 s and 2.3 GB for 5000")
                .addTitleParameter("Cell classes and spatial clustering")
                .addBooleanParameter("classifyCells", "Set cell classes to PHC clusters "
                                + "(replaces their current classification)",
                        DEFAULT_CLASSIFY_CELLS,
                        "Give each clustered cell the class 'PHC cluster k'. Clear PHC "
                                + "heatmap restores the original classes")
                .addBooleanParameter("spatialClustering", "Spatially constrained clustering "
                                + "(Delaunay adjacency of cell centroids)",
                        DEFAULT_SPATIAL_CLUSTERING,
                        "Only cells joined by an edge of the Delaunay triangulation of their "
                                + "centroids can merge, so every cluster is a spatially "
                                + "contiguous region. All clustered cells are used (no "
                                + "10,000-cell subsample)")
                .addDoubleParameter("maxEdgeLength", "Max Delaunay edge length",
                        DEFAULT_MAX_EDGE_LENGTH, micrometres,
                        "Spatially constrained clustering only: Delaunay edges longer than "
                                + "this, in " + micrometres + " (in pixels if the image has "
                                + "no pixel size), are dropped, so distant cells across gaps "
                                + "are not neighbours. 0 = no limit");
        return params;
    }

    /**
     * Reads the values the user entered in the dialog and checks them.
     *
     * @param params (ParameterList) List created by {@link #createParameterList()}.
     * @return (PHCParameters) Validated settings.
     * @throws IllegalArgumentException When a value is out of range.
     */
    public static PHCParameters fromParameterList(ParameterList params) {
        PHCParameters settings = new PHCParameters(
                params.getIntParameterValue("dimension"),
                params.getDoubleParameterValue("windowSize"),
                (String) params.getChoiceParameterValue("vectorization"),
                params.getIntParameterValue("vectorResolution"),
                params.getIntParameterValue("minCells"),
                params.getIntParameterValue("nClusters"),
                (String) params.getChoiceParameterValue("linkage"),
                params.getDoubleParameterValue("minCoverage"),
                params.getIntParameterValue("nJobs"),
                params.getBooleanParameterValue("computeMds"),
                params.getIntParameterValue("mdsMaxWindows"),
                params.getBooleanParameterValue("classifyCells"),
                params.getBooleanParameterValue("spatialClustering"),
                params.getDoubleParameterValue("maxEdgeLength"));
        settings.validate();
        return settings;
    }

    /**
     * Gives the pixel size window lengths are converted with, so micrometre settings become
     * slide pixels.
     *
     * @param calibration (PixelCalibration) Calibration of the image PHC runs on.
     * @return (double) Averaged pixel size in micrometres, or
     *         {@link #UNCALIBRATED_PIXEL_SIZE} when the image has none (lengths are then pixels).
     */
    public static double pixelSizeMicrons(PixelCalibration calibration) {
        double pixelSize = UNCALIBRATED_PIXEL_SIZE;
        if (calibration.hasPixelSizeMicrons()
                && calibration.getAveragedPixelSizeMicrons() > 0) {
            pixelSize = calibration.getAveragedPixelSizeMicrons();
        }
        return pixelSize;
    }

    /**
     * Rejects settings PHC or the clustering step cannot run with, before any work starts.
     *
     * @return (void)
     * @throws IllegalArgumentException When a value is out of range or an option is unknown.
     */
    public void validate() {
        requireOption("vectorization", vectorization, VECTORIZATIONS);
        requireOption("linkage", linkage, LINKAGES);
        if (dimension < 0 || dimension > MAX_ALPHA_DIMENSION) {
            throw new IllegalArgumentException("Homology dimension must be 0 or 1 for the alpha "
                    + "complex of cell centroids, got " + dimension);
        }
        if (!(windowSize > 0)) {
            throw new IllegalArgumentException("Window size must be greater than 0.");
        }
        if (vectorResolution < 1 || minCells < 0 || nClusters < 1 || nJobs == 0) {
            throw new IllegalArgumentException("Vector resolution and clusters must be >= 1; "
                    + "min cells must be >= 0; worker processes must be -1 or >= 1.");
        }
        if (minCoverage <= 0 || minCoverage > 1) {
            throw new IllegalArgumentException("Min ROI coverage must be in (0, 1], got "
                    + minCoverage);
        }
        if (mdsMaxWindows < MIN_MDS_WINDOWS) {
            throw new IllegalArgumentException("Max windows for MDS must be >= "
                    + MIN_MDS_WINDOWS + ", got " + mdsMaxWindows);
        }
        if (!(maxEdgeLength >= 0) || Double.isInfinite(maxEdgeLength)) {
            throw new IllegalArgumentException("Max Delaunay edge length must be >= 0 (0 = no "
                    + "limit), got " + maxEdgeLength);
        }
    }

    /**
     * Converts the settings into phc_bridge.py command line flags, with window size and max
     * edge length converted to full-resolution slide pixels. --spatial and --max-edge-length
     * are always passed; classifyCells stays on the Java side.
     *
     * @param pixelSizeMicrons (double) Micrometres per slide pixel, from
     *        {@link #pixelSizeMicrons(PixelCalibration)}.
     * @return (List of String) Flag and value pairs, ready to append to the Python command.
     */
    public List<String> toBridgeArgs(double pixelSizeMicrons) {
        List<String> args = List.of(
                "--window-size", Double.toString(windowSize / pixelSizeMicrons),
                "--dimension", Integer.toString(dimension),
                "--vectorization", vectorization,
                "--vector-resolution", Integer.toString(vectorResolution),
                "--min-cells", Integer.toString(minCells),
                "--n-clusters", Integer.toString(nClusters),
                "--linkage", linkage,
                "--min-coverage", Double.toString(minCoverage),
                "--n-jobs", Integer.toString(nJobs),
                "--mds", computeMds ? "1" : "0",
                "--mds-max-windows", Integer.toString(mdsMaxWindows),
                "--spatial", isSpatial() ? "1" : "0",
                "--max-edge-length", Double.toString(maxEdgeLength / pixelSizeMicrons));
        return args;
    }

    /**
     * Fails with a message naming the allowed values when a string option is unknown.
     *
     * @param label (String) Human-readable name of the option, used in the message.
     * @param value (String) Value chosen by the user.
     * @param allowed (List of String) Values the option accepts.
     * @return (void)
     * @throws IllegalArgumentException When value is not in allowed.
     */
    private static void requireOption(String label, String value, List<String> allowed) {
        if (!allowed.contains(value)) {
            throw new IllegalArgumentException("Unknown " + label + " '" + value
                    + "'; expected one of " + allowed + ".");
        }
    }
}
