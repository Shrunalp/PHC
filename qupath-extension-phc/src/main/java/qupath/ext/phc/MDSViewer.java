/*
 * Interactive 2D and 3D scatter plots of the MDS embedding of a PHC run, linked to its tiles or
 * cells.
 *
 * Contents
 * --------
 * MDSViewer : class
 *     Window with a 2D canvas scatter and a rotatable 3D scatter of the MDS coordinates stored
 *     on PHC tiles (tiled windows) or cells (per-cell windows).
 */

package qupath.ext.phc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Consumer;

import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.geometry.Pos;
import javafx.scene.AmbientLight;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.PerspectiveCamera;
import javafx.scene.PointLight;
import javafx.scene.Scene;
import javafx.scene.SceneAntialiasing;
import javafx.scene.SubScene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.PickResult;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Background;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.Sphere;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.scene.transform.Rotate;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.stage.Window;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.common.ColorTools;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;

/**
 * Shows where each PHC window sits in the metric MDS embedding of the windows' L2
 * dissimilarity matrix, so windows with similar topology can be found as clusters of points.
 * A window is a tile (tiled-window mode) or the window centred on a cell (per-cell mode); the
 * viewer handles both the same way. Points share the viridis cluster colours of the tile
 * classes. Clicking a point selects its tile or cell in QuPath and centres the viewer on it;
 * selecting it in QuPath highlights its point. Reads everything from the objects'
 * measurements, so it also works after a project is reopened. Create and use it on the JavaFX
 * thread.
 */
public final class MDSViewer {

    private static final Logger logger = LoggerFactory.getLogger(MDSViewer.class);

    /** Index of the 2D and 3D tabs, for {@link #saveSnapshot(int, File)}. */
    public static final int TAB_2D = 0;
    public static final int TAB_3D = 1;

    private static final double WINDOW_WIDTH = 900;
    private static final double WINDOW_HEIGHT = 760;
    private static final double FIGURE_PADDING = 16;
    private static final double SECTION_GAP = 6;
    private static final double LEGEND_GAP = 16;
    private static final double LEGEND_SWATCH_RADIUS = 5;
    private static final double TITLE_FONT_SIZE = 16;
    private static final double TEXT_FONT_SIZE = 12;
    private static final double TICK_FONT_SIZE = 11;
    private static final double AXIS_FONT_SIZE = 12;

    /** Text and chart-chrome colours on the white figure surface (text never wears data colour). */
    private static final Color SURFACE = Color.WHITE;
    private static final Color INK_PRIMARY = Color.web("#1f1f1f");
    private static final Color INK_SECONDARY = Color.web("#5f5f5f");
    private static final Color GRID = Color.web("#e8e8e8");
    private static final Color AXIS = Color.web("#b5b5b5");
    private static final Color HIGHLIGHT_RING = Color.web("#111111");

    /** Plot margins of the 2D canvas, in pixels, leaving room for ticks and axis titles. */
    private static final double MARGIN_LEFT = 62;
    private static final double MARGIN_RIGHT = 18;
    private static final double MARGIN_TOP = 12;
    private static final double MARGIN_BOTTOM = 48;
    private static final double DATA_PADDING = 0.05;     // fraction of the range around the data
    private static final int TARGET_TICKS = 6;
    private static final double TICK_LENGTH = 4;
    private static final double TICK_LABEL_GAP = 6;
    private static final double AXIS_TITLE_OFFSET = 34;   // from the plot edge

    /** 2D point marks: >= 8 px dots with a surface ring; smaller for dense plots. */
    private static final double POINT_RADIUS = 4;
    private static final double DENSE_POINT_RADIUS = 3;
    private static final int DENSE_POINTS = 1500;
    private static final double POINT_RING_WIDTH = 1.2;
    private static final double SELECTED_RADIUS = 7;
    private static final double SELECTED_RING_WIDTH = 2;
    private static final double HOVER_RING_WIDTH = 1.5;
    private static final double HIT_RADIUS = 8;           // pointer distance that still hits
    private static final long DRAW_ORDER_SEED = 0;        // shuffles points so no cluster hides

    /** 3D scene: data fills a cube of side 2 * CUBE_HALF, viewed from CAMERA_DISTANCE. */
    private static final double CUBE_HALF = 150;
    private static final double SPHERE_RADIUS = 3.2;
    private static final double DENSE_SPHERE_RADIUS = 2.2;
    private static final int SPHERE_DIVISIONS = 8;
    private static final double SELECTED_SPHERE_SCALE = 2.6;
    private static final double AXIS_RADIUS = 0.7;
    private static final double AXIS_EXTENT = 1.12;       // axes reach a little past the data
    private static final double AXIS_LABEL_OFFSET = 14;
    private static final double AXIS_LABEL_FONT_SIZE = 14;
    private static final double CAMERA_DISTANCE = 560;
    private static final double MIN_CAMERA_DISTANCE = 200;
    private static final double MAX_CAMERA_DISTANCE = 2400;
    private static final double FIELD_OF_VIEW = 35;
    private static final double CAMERA_FAR_CLIP = 10000;
    private static final double ZOOM_PER_SCROLL_PIXEL = 1.5;
    private static final double DEGREES_PER_DRAG_PIXEL = 0.4;
    private static final double INITIAL_ANGLE_X = -22;
    private static final double INITIAL_ANGLE_Y = 38;
    private static final double AMBIENT_LEVEL = 0.55;
    private static final double LIGHT_LEVEL = 0.6;        // ambient + light ~ true colour
    private static final double MIN_AXIS_FRACTION = 0.25; // shortest axis, of CUBE_HALF
    private static final double LIGHT_X = -300;
    private static final double LIGHT_Y = -400;
    private static final double LIGHT_Z = -900;
    private static final Color AXIS_3D = Color.web("#8a8a8a");

    private static final double TOOLTIP_OFFSET = 14;
    private static final String TOOLTIP_STYLE = "-fx-background-color: white; "
            + "-fx-border-color: #c8c8c8; -fx-border-radius: 4; -fx-background-radius: 4; "
            + "-fx-padding: 6 8 6 8; -fx-font-size: 12px; -fx-text-fill: #1f1f1f;";
    private static final String STRESS_FORMAT = "%.3f";
    private static final String L2_FORMAT = "%.4g";

    private final List<EmbeddedPoint> points;
    private final int nTiles;
    private final String noun;            // "windows" or "cells"
    private final PathObjectHierarchy hierarchy;
    private final Consumer<PathObject> centreOn;
    private final Path plotDir;
    private final String filePrefix;
    private final Stage stage;
    private final TabPane tabPane;
    private final Scatter2D scatter2d;
    private final Scatter3D scatter3d;
    private final List<Region> figures = new ArrayList<>();  // snapshot target per tab
    private final PathObjectSelectionListener selectionListener;

    /**
     * Builds the viewer window for the MDS coordinates stored on a run's tiles or cells;
     * nothing is shown until {@link #show(Window)}.
     *
     * @param tiles (Collection of PathObject) PHC tiles of one annotation, or its clustered
     *        cells in per-cell mode; those without MDS measurements are counted in the
     *        subtitle but not plotted.
     * @param hierarchy (PathObjectHierarchy) Hierarchy holding the tiles, whose selection
     *        model links points and tiles.
     * @param stress2d (Double) Kruskal stress-1 of the 2D embedding, or null when unknown.
     * @param stress3d (Double) Kruskal stress-1 of the 3D embedding, or null when unknown.
     * @param plotDir (Path) Folder Python saved its plots and CSV to, or null when none.
     * @param filePrefix (String) Name suggested for saved snapshots, e.g. the plot prefix.
     * @param centreOn (Consumer of PathObject) Centres the QuPath viewer on a clicked tile or
     *        cell; may do nothing (e.g. in tests).
     * @param noun (String) {@link ProgressTracker#WINDOWS_NOUN} for tiles,
     *        {@link ProgressTracker#CELLS_NOUN} for cells; used in titles and hover text.
     * @throws IllegalArgumentException When no object carries MDS coordinates.
     */
    public MDSViewer(Collection<PathObject> tiles, PathObjectHierarchy hierarchy, Double stress2d,
                     Double stress3d, Path plotDir, String filePrefix,
                     Consumer<PathObject> centreOn, String noun) {
        this.noun = noun;
        this.points = embeddedPoints(tiles);
        if (points.isEmpty()) {
            throw new IllegalArgumentException("These PHC " + noun + " have no MDS coordinates. "
                    + "Run PHC again with 'Compute MDS embedding' switched on.");
        }
        this.nTiles = tiles.size();
        this.hierarchy = hierarchy;
        this.centreOn = centreOn;
        this.plotDir = plotDir;
        this.filePrefix = filePrefix;

        scatter2d = new Scatter2D();
        scatter3d = new Scatter3D();
        String figureTitle = isCells() ? "MDS of per-cell PHC persistence vectors"
                : "MDS of PHC persistence vectors";
        tabPane = new TabPane(
                createTab("2D", buildFigure(figureTitle + " (2D)",
                        subtitle(stress2d), scatter2d), TAB_2D),
                createTab("3D", buildFigure(figureTitle + " (3D)",
                        subtitle(stress3d) + "  ·  drag to rotate, scroll to zoom",
                        scatter3d), TAB_3D));
        tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

        stage = new Stage();
        stage.setTitle("PHC MDS embedding (" + noun + ")");
        stage.setScene(new Scene(tabPane, WINDOW_WIDTH, WINDOW_HEIGHT));

        selectionListener = (selected, previous, all) -> {
            if (Platform.isFxApplicationThread()) {
                highlight(selected);
            } else {
                Platform.runLater(() -> highlight(selected));
            }
        };
        hierarchy.getSelectionModel().addPathObjectSelectionListener(selectionListener);
        stage.setOnHidden(event -> hierarchy.getSelectionModel()
                .removePathObjectSelectionListener(selectionListener));
        highlight(hierarchy.getSelectionModel().getSelectedObject());
    }

    /**
     * Opens the window next to QuPath.
     *
     * @param owner (Window) Window the viewer belongs to, may be null.
     * @return (void)
     */
    public void show(Window owner) {
        if (owner != null && stage.getOwner() == null) {
            stage.initOwner(owner);
        }
        stage.show();
        stage.toFront();
    }

    /**
     * Closes the window and stops listening to the tile selection.
     *
     * @return (void)
     */
    public void close() {
        stage.close();
        hierarchy.getSelectionModel().removePathObjectSelectionListener(selectionListener);
    }

    /**
     * Tells whether the points are cells (per-cell windows) rather than tiles.
     *
     * @return (boolean) True in per-cell mode.
     */
    private boolean isCells() {
        boolean cells = ProgressTracker.CELLS_NOUN.equals(noun);
        return cells;
    }

    /**
     * Gives the window, e.g. to check whether it is still open.
     *
     * @return (Stage) The viewer's stage.
     */
    public Stage getStage() {
        return stage;
    }

    /**
     * Shows one of the tabs.
     *
     * @param tabIndex (int) {@link #TAB_2D} or {@link #TAB_3D}.
     * @return (void)
     */
    public void selectTab(int tabIndex) {
        tabPane.getSelectionModel().select(tabIndex);
    }

    /**
     * Saves the figure of a tab (title, legend and plot) as a PNG, as the Save PNG button
     * does. The tab is shown first, since hidden tabs are not drawn.
     *
     * @param tabIndex (int) {@link #TAB_2D} or {@link #TAB_3D}.
     * @param file (File) PNG to write.
     * @return (void)
     * @throws IOException When the PNG cannot be written.
     */
    public void saveSnapshot(int tabIndex, File file) throws IOException {
        selectTab(tabIndex);
        Region figure = figures.get(tabIndex);
        figure.applyCss();
        figure.layout();
        WritableImage image = figure.snapshot(null, null);
        ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
    }

    /**
     * Gives the 2D canvas, so tests can send it mouse events.
     *
     * @return (Canvas) Canvas the 2D scatter is drawn on.
     */
    Canvas canvas2d() {
        return scatter2d.canvas;
    }

    /**
     * Finds where a tile's (or cell's) point is drawn on the 2D canvas.
     *
     * @param tile (PathObject) Tile or cell with MDS coordinates.
     * @return (Point2D) Canvas coordinates of the point's centre, or null when it is not
     *         plotted.
     */
    Point2D canvasPosition(PathObject tile) {
        Integer index = scatter2d.indexOf.get(tile);
        Point2D position = index == null ? null
                : new Point2D(scatter2d.screenX[index], scatter2d.screenY[index]);
        return position;
    }

    /**
     * Gives the tile or cell whose point is highlighted, to check the link with QuPath's
     * selection.
     *
     * @return (PathObject) Highlighted tile or cell, or null.
     */
    PathObject highlightedTile() {
        PathObject tile = scatter2d.selected < 0 ? null : points.get(scatter2d.selected).object();
        return tile;
    }

    /**
     * Selects a tile or cell in QuPath and centres the viewer on it, after a click on its
     * point.
     *
     * @param index (int) Index of the clicked point in {@link #points}.
     * @return (void)
     */
    private void selectTile(int index) {
        PathObject tile = points.get(index).object();
        hierarchy.getSelectionModel().setSelectedObject(tile);
        highlight(tile);
        centreOn.accept(tile);
    }

    /**
     * Highlights the point of the tile or cell selected in QuPath in both plots, or clears the
     * highlight when the selection is not plotted.
     *
     * @param tile (PathObject) Newly selected object, may be null.
     * @return (void)
     */
    private void highlight(PathObject tile) {
        Integer index = tile == null ? null : scatter2d.indexOf.get(tile);
        int selected = index == null ? -1 : index;
        scatter2d.setSelected(selected);
        scatter3d.setSelected(selected);
    }

    /**
     * Describes the embedding below the title: windows (or cells) embedded and stress.
     *
     * @param stress (Double) Kruskal stress-1 of this embedding, or null.
     * @return (String) e.g. "Metric MDS of the L2 distances between 50 windows  ·  stress 0.123".
     */
    private String subtitle(Double stress) {
        String windows = points.size() == nTiles
                ? points.size() + " " + noun
                : points.size() + " of " + nTiles + " " + noun + " (random subsample)";
        String stressText = stress == null ? "stress n/a"
                : "Kruskal stress " + String.format(STRESS_FORMAT, stress);
        String text = "Metric MDS of the L2 distances between " + windows + "  ·  "
                + stressText;
        return text;
    }

    /**
     * Puts a figure and its Save PNG button / plot-folder note (naming the Delaunay graph
     * plot of a spatial per-cell run when Python wrote one) into a tab.
     *
     * @param name (String) Tab title.
     * @param figure (Region) Title, legend and plot.
     * @param tabIndex (int) Position of the tab, used by the save button.
     * @return (Tab) Tab ready for the tab pane.
     */
    private Tab createTab(String name, Region figure, int tabIndex) {
        figures.add(figure);
        Button save = new Button("Save PNG...");
        save.setOnAction(event -> promptSave(tabIndex, name));
        String note = plotDir == null
                ? "Open a QuPath project to also save matplotlib plots and a CSV of the "
                        + "coordinates."
                : "Matplotlib plots and CSV saved in " + plotDir;
        Path delaunay = plotDir == null || !isCells() ? null
                : new PythonBridge.PlotTarget(plotDir, filePrefix).delaunayPlot();
        if (delaunay != null && delaunay.toFile().isFile()) {
            note += " (Delaunay graph: " + delaunay.getFileName() + ")";
        }
        Label noteLabel = new Label(note);
        noteLabel.setWrapText(true);
        noteLabel.setTextFill(INK_SECONDARY);
        HBox footer = new HBox(LEGEND_GAP, save, noteLabel);
        footer.setAlignment(Pos.CENTER_LEFT);
        footer.setPadding(new Insets(SECTION_GAP, FIGURE_PADDING, FIGURE_PADDING / 2,
                FIGURE_PADDING));
        BorderPane content = new BorderPane(figure);
        content.setBottom(footer);
        Tab tab = new Tab(name, content);
        return tab;
    }

    /**
     * Asks where to save a tab's figure and writes it.
     *
     * @param tabIndex (int) {@link #TAB_2D} or {@link #TAB_3D}.
     * @param name (String) Tab title, used in the suggested file name.
     * @return (void)
     */
    private void promptSave(int tabIndex, String name) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save MDS plot");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG image",
                "*.png"));
        String kind = isCells() ? PythonBridge.PlotTarget.CELLS_SUFFIX : "";
        chooser.setInitialFileName(filePrefix + kind + "_mds_" + name.toLowerCase()
                + "_view.png");
        if (plotDir != null && plotDir.toFile().isDirectory()) {
            chooser.setInitialDirectory(plotDir.toFile());
        }
        File file = chooser.showSaveDialog(stage);
        if (file == null) {
            return;
        }
        try {
            saveSnapshot(tabIndex, file);
        } catch (IOException e) {
            logger.error("Could not save {}", file, e);
        }
    }

    /**
     * Stacks a title, subtitle and cluster legend above a plot, on a white surface.
     *
     * @param title (String) Figure title.
     * @param subtitle (String) Line under the title (windows, stress, hints).
     * @param plot (Region) The 2D or 3D plot, which grows to fill the window.
     * @return (Region) The figure, which is also what Save PNG captures.
     */
    private Region buildFigure(String title, String subtitle, Region plot) {
        Label titleLabel = new Label(title);
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, TITLE_FONT_SIZE));
        titleLabel.setTextFill(INK_PRIMARY);
        Label subtitleLabel = new Label(subtitle);
        subtitleLabel.setFont(Font.font(TEXT_FONT_SIZE));
        subtitleLabel.setTextFill(INK_SECONDARY);
        VBox figure = new VBox(SECTION_GAP, titleLabel, subtitleLabel, buildLegend(), plot);
        VBox.setVgrow(plot, Priority.ALWAYS);
        figure.setPadding(new Insets(FIGURE_PADDING, FIGURE_PADDING, 0, FIGURE_PADDING));
        figure.setBackground(Background.fill(SURFACE));
        return figure;
    }

    /**
     * Builds the cluster legend: a coloured dot and "PHC cluster k (n windows)" per cluster.
     *
     * @return (HBox) Legend row, clusters in increasing order.
     */
    private HBox buildLegend() {
        Map<Integer, int[]> counts = new TreeMap<>();   // cluster -> {count}
        Map<Integer, Color> colours = new HashMap<>();
        for (EmbeddedPoint point : points) {
            counts.computeIfAbsent(point.cluster(), k -> new int[1])[0]++;
            colours.putIfAbsent(point.cluster(), point.colour());
        }
        HBox legend = new HBox(LEGEND_GAP);
        legend.setAlignment(Pos.CENTER_LEFT);
        for (Map.Entry<Integer, int[]> entry : counts.entrySet()) {
            Circle swatch = new Circle(LEGEND_SWATCH_RADIUS, colours.get(entry.getKey()));
            Label label = new Label(PHCPipeline.CLASS_PREFIX + entry.getKey() + " ("
                    + entry.getValue()[0] + ")");
            label.setFont(Font.font(TEXT_FONT_SIZE));
            label.setTextFill(INK_PRIMARY);
            HBox item = new HBox(SECTION_GAP, swatch, label);
            item.setAlignment(Pos.CENTER_LEFT);
            legend.getChildren().add(item);
        }
        return legend;
    }

    /**
     * Creates the hover tooltip shared by both plots: cluster, cells in the window and L2
     * norm.
     *
     * @return (Label) Hidden, unmanaged label that follows the pointer.
     */
    private static Label createTooltip() {
        Label tooltip = new Label();
        tooltip.setStyle(TOOLTIP_STYLE);
        tooltip.setManaged(false);
        tooltip.setMouseTransparent(true);
        tooltip.setVisible(false);
        return tooltip;
    }

    /**
     * Shows the tooltip for a point near the pointer, kept inside the plot.
     *
     * @param tooltip (Label) Tooltip from {@link #createTooltip()}.
     * @param point (EmbeddedPoint) Point under the pointer.
     * @param x (double) Pointer x in the plot's coordinates.
     * @param y (double) Pointer y in the plot's coordinates.
     * @param bounds (Region) Plot the tooltip must stay inside.
     * @return (void)
     */
    private void showTooltip(Label tooltip, EmbeddedPoint point, double x, double y,
                             Region bounds) {
        String count = (long) point.nCells() + (isCells() ? " cells in window" : " cells");
        tooltip.setText(PHCPipeline.CLASS_PREFIX + point.cluster() + "\n" + count
                + "\nL2 norm " + String.format(L2_FORMAT, point.l2Norm()));
        tooltip.autosize();
        double left = x + TOOLTIP_OFFSET;
        double top = y + TOOLTIP_OFFSET;
        if (left + tooltip.getWidth() > bounds.getWidth()) {
            left = x - TOOLTIP_OFFSET - tooltip.getWidth();
        }
        if (top + tooltip.getHeight() > bounds.getHeight()) {
            top = y - TOOLTIP_OFFSET - tooltip.getHeight();
        }
        tooltip.relocate(Math.max(0, left), Math.max(0, top));
        tooltip.setVisible(true);
        tooltip.toFront();
    }

    /**
     * Reads the MDS coordinates and hover details off the tiles or cells. Colours come from
     * the cluster number on the viridis ramp (as the tile classes), since cells keep their
     * own class unless PHC was asked to classify them.
     *
     * @param tiles (Collection of PathObject) PHC tiles or clustered cells.
     * @return (List of EmbeddedPoint) One entry per object with MDS coordinates, in the same
     *         order.
     */
    private static List<EmbeddedPoint> embeddedPoints(Collection<PathObject> tiles) {
        int nClusters = (int) tiles.stream()
                .filter(t -> t.getMeasurementList().containsKey(PHCPipeline.MEASUREMENT_CLUSTER))
                .mapToDouble(t -> t.getMeasurementList().get(PHCPipeline.MEASUREMENT_CLUSTER))
                .max().orElse(1);
        List<EmbeddedPoint> embedded = new ArrayList<>();
        for (PathObject tile : PHCPipeline.embeddedTiles(tiles)) {
            MeasurementList m = tile.getMeasurementList();
            int cluster = (int) m.get(PHCPipeline.MEASUREMENT_CLUSTER);
            int rgb = PHCPipeline.clusterColor(cluster - 1, nClusters);
            String countKey = m.containsKey(PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW)
                    ? PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW : PHCPipeline.MEASUREMENT_CELL_COUNT;
            embedded.add(new EmbeddedPoint(tile, cluster,
                    Color.rgb(ColorTools.red(rgb), ColorTools.green(rgb), ColorTools.blue(rgb)),
                    m.get(countKey),
                    m.get(PHCPipeline.MEASUREMENT_L2_NORM),
                    new double[] {m.get(PHCPipeline.MEASUREMENT_MDS2_X),
                            m.get(PHCPipeline.MEASUREMENT_MDS2_Y)},
                    new double[] {m.get(PHCPipeline.MEASUREMENT_MDS3_X),
                            m.get(PHCPipeline.MEASUREMENT_MDS3_Y),
                            m.get(PHCPipeline.MEASUREMENT_MDS3_Z)}));
        }
        return embedded;
    }

    /**
     * Gives the centre and half-width of the coordinates along one axis.
     *
     * @param coordinates (List of double[]) Points, each of size (d).
     * @param axis (int) Axis to measure, in [0, d).
     * @return (double[] - size (2)) Centre and half of the range (1 when all are equal).
     */
    private static double[] centreAndHalfRange(List<double[]> coordinates, int axis) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double[] point : coordinates) {
            min = Math.min(min, point[axis]);
            max = Math.max(max, point[axis]);
        }
        double half = max > min ? (max - min) / 2 : 1.0;
        double[] centreHalf = {(min + max) / 2, half};
        return centreHalf;
    }

    /**
     * Chooses round tick values (1, 2 or 5 times a power of ten) covering a range.
     *
     * @param min (double) Lowest visible value.
     * @param max (double) Highest visible value.
     * @return (double[]) Ticks in increasing order, about {@link #TARGET_TICKS} of them; the
     *         step between them is ticks[1] - ticks[0].
     */
    private static double[] niceTicks(double min, double max) {
        double rough = (max - min) / TARGET_TICKS;
        double magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
        double normalised = rough / magnitude;
        double step = magnitude * (normalised < 1.5 ? 1 : normalised < 3 ? 2
                : normalised < 7 ? 5 : 10);
        List<Double> values = new ArrayList<>();
        for (double tick = Math.ceil(min / step) * step; tick <= max + step * 1e-9;
             tick += step) {
            values.add(Math.abs(tick) < step * 1e-9 ? 0.0 : tick);  // avoid "-0.0"
        }
        double[] ticks = values.stream().mapToDouble(Double::doubleValue).toArray();
        return ticks;
    }

    /**
     * One plotted window: its tile or cell, cluster, colour, hover details and MDS coordinates.
     *
     * @param object (PathObject) The PHC tile, or the cell the window is centred on.
     * @param cluster (int) Cluster number, 1 = lowest mean L2 norm.
     * @param colour (Color) Viridis colour of the cluster.
     * @param nCells (double) Cell centroids in the window.
     * @param l2Norm (double) L2 norm of the window's persistence vector.
     * @param mds2 (double[] - size (2)) 2D MDS coordinates.
     * @param mds3 (double[] - size (3)) 3D MDS coordinates.
     */
    private record EmbeddedPoint(PathObject object, int cluster, Color colour, double nCells,
                                double l2Norm, double[] mds2, double[] mds3) {
    }

    /**
     * Canvas scatter of the 2D embedding with axes, grid, hover tooltip and click-to-select.
     * A canvas draws thousands of points much faster than one chart node per point. Holds
     * the current hover and selection as state.
     */
    private final class Scatter2D extends Pane {

        private final Canvas canvas = new Canvas();
        private final Label tooltip = createTooltip();
        private final Map<PathObject, Integer> indexOf = new HashMap<>();
        private final int[] drawOrder;
        private final double[] screenX;
        private final double[] screenY;
        private final double pointRadius;
        private int selected = -1;
        private int hovered = -1;

        /**
         * Sets up the canvas and its mouse handlers; drawing happens on every resize.
         */
        Scatter2D() {
            int n = points.size();
            screenX = new double[n];
            screenY = new double[n];
            pointRadius = n > DENSE_POINTS ? DENSE_POINT_RADIUS : POINT_RADIUS;
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                indexOf.put(points.get(i).object(), i);
                order.add(i);
            }
            // tiles / cells come in slide order; shuffle so one cluster never paints over all
            Collections.shuffle(order, new Random(DRAW_ORDER_SEED));
            drawOrder = order.stream().mapToInt(Integer::intValue).toArray();

            canvas.setManaged(false);
            getChildren().addAll(canvas, tooltip);
            setMinSize(0, 0);
            widthProperty().addListener((obs, old, value) -> redraw());
            heightProperty().addListener((obs, old, value) -> redraw());
            canvas.setOnMouseMoved(this::onMove);
            canvas.setOnMouseExited(event -> {
                hovered = -1;
                tooltip.setVisible(false);
                redraw();
            });
            canvas.setOnMouseClicked(event -> {
                int hit = pointAt(event.getX(), event.getY());
                if (hit >= 0) {
                    selectTile(hit);
                }
            });
        }

        /**
         * Highlights one point, or none.
         *
         * @param index (int) Point index, -1 for none.
         * @return (void)
         */
        void setSelected(int index) {
            selected = index;
            redraw();
        }

        /**
         * Updates the hover highlight and tooltip as the pointer moves.
         *
         * @param event (MouseEvent) Pointer movement over the canvas.
         * @return (void)
         */
        private void onMove(MouseEvent event) {
            int hit = pointAt(event.getX(), event.getY());
            if (hit != hovered) {
                hovered = hit;
                redraw();
            }
            if (hit >= 0) {
                showTooltip(tooltip, points.get(hit), event.getX(), event.getY(), this);
            } else {
                tooltip.setVisible(false);
            }
        }

        /**
         * Finds the point under the pointer.
         *
         * @param x (double) Canvas x.
         * @param y (double) Canvas y.
         * @return (int) Index of the nearest point within {@link #HIT_RADIUS}, or -1.
         */
        private int pointAt(double x, double y) {
            int nearest = -1;
            double best = HIT_RADIUS * HIT_RADIUS;
            for (int i = 0; i < screenX.length; i++) {
                double dx = screenX[i] - x;
                double dy = screenY[i] - y;
                double distance = dx * dx + dy * dy;
                if (distance <= best) {
                    best = distance;
                    nearest = i;
                }
            }
            return nearest;
        }

        /**
         * Lays the canvas over the whole pane.
         *
         * @return (void)
         */
        @Override
        protected void layoutChildren() {
            canvas.setWidth(getWidth());
            canvas.setHeight(getHeight());
            super.layoutChildren();
        }

        /**
         * Redraws grid, axes and points. Both axes share one scale, so distances on screen are
         * proportional to MDS distances, and the data is centred in the plot area.
         *
         * @return (void)
         */
        private void redraw() {
            double width = getWidth();
            double height = getHeight();
            double plotW = width - MARGIN_LEFT - MARGIN_RIGHT;
            double plotH = height - MARGIN_TOP - MARGIN_BOTTOM;
            if (plotW <= 0 || plotH <= 0) {
                return;
            }
            canvas.setWidth(width);
            canvas.setHeight(height);
            GraphicsContext g = canvas.getGraphicsContext2D();
            g.setFill(SURFACE);
            g.fillRect(0, 0, width, height);

            // --- Equal-aspect scale: data units -> pixels ---
            List<double[]> coordinates = points.stream().map(EmbeddedPoint::mds2).toList();
            double[] xs = centreAndHalfRange(coordinates, 0);
            double[] ys = centreAndHalfRange(coordinates, 1);
            double scale = Math.min(plotW / (2 * xs[1]), plotH / (2 * ys[1]))
                    / (1 + 2 * DATA_PADDING);
            double centreX = MARGIN_LEFT + plotW / 2;
            double centreY = MARGIN_TOP + plotH / 2;
            double xMin = xs[0] - plotW / 2 / scale;
            double xMax = xs[0] + plotW / 2 / scale;
            double yMin = ys[0] - plotH / 2 / scale;
            double yMax = ys[0] + plotH / 2 / scale;

            drawAxes(g, niceTicks(xMin, xMax), niceTicks(yMin, yMax), xs[0], ys[0], scale,
                    centreX, centreY, plotW, plotH);

            // --- Points: surface ring then fill, in shuffled order ---
            for (int i = 0; i < points.size(); i++) {
                screenX[i] = centreX + (points.get(i).mds2()[0] - xs[0]) * scale;
                screenY[i] = centreY - (points.get(i).mds2()[1] - ys[0]) * scale;  // y up
            }
            g.setLineWidth(POINT_RING_WIDTH);
            g.setStroke(SURFACE);
            for (int i : drawOrder) {
                double d = 2 * pointRadius;
                g.setFill(points.get(i).colour());
                g.fillOval(screenX[i] - pointRadius, screenY[i] - pointRadius, d, d);
                g.strokeOval(screenX[i] - pointRadius, screenY[i] - pointRadius, d, d);
            }
            if (hovered >= 0 && hovered != selected) {
                drawRinged(g, hovered, pointRadius + 1, HOVER_RING_WIDTH);
            }
            if (selected >= 0) {
                drawRinged(g, selected, SELECTED_RADIUS, SELECTED_RING_WIDTH);
            }
        }

        /**
         * Draws one emphasised point: its colour, a surface ring and a dark outer ring.
         *
         * @param g (GraphicsContext) Canvas graphics.
         * @param index (int) Point index.
         * @param radius (double) Radius of the coloured disc, in pixels.
         * @param ringWidth (double) Width of the dark outer ring, in pixels.
         * @return (void)
         */
        private void drawRinged(GraphicsContext g, int index, double radius, double ringWidth) {
            double x = screenX[index];
            double y = screenY[index];
            double outer = radius + POINT_RING_WIDTH + ringWidth / 2;
            g.setFill(points.get(index).colour());
            g.fillOval(x - radius, y - radius, 2 * radius, 2 * radius);
            g.setStroke(SURFACE);
            g.setLineWidth(2 * POINT_RING_WIDTH);
            g.strokeOval(x - radius, y - radius, 2 * radius, 2 * radius);
            g.setStroke(HIGHLIGHT_RING);
            g.setLineWidth(ringWidth);
            g.strokeOval(x - outer, y - outer, 2 * outer, 2 * outer);
        }

        /**
         * Draws hairline grid lines, the left and bottom axes, tick labels and axis titles.
         *
         * @param g (GraphicsContext) Canvas graphics.
         * @param xTicks (double[]) Tick values along MDS 1.
         * @param yTicks (double[]) Tick values along MDS 2.
         * @param dataX (double) MDS 1 value at the plot centre.
         * @param dataY (double) MDS 2 value at the plot centre.
         * @param scale (double) Pixels per MDS unit, the same on both axes.
         * @param centreX (double) Canvas x of the plot centre.
         * @param centreY (double) Canvas y of the plot centre.
         * @param plotW (double) Width of the plot area.
         * @param plotH (double) Height of the plot area.
         * @return (void)
         */
        private void drawAxes(GraphicsContext g, double[] xTicks, double[] yTicks, double dataX,
                              double dataY, double scale, double centreX, double centreY,
                              double plotW, double plotH) {
            double left = MARGIN_LEFT;
            double right = MARGIN_LEFT + plotW;
            double top = MARGIN_TOP;
            double bottom = MARGIN_TOP + plotH;
            String xFormat = tickFormat(xTicks);
            String yFormat = tickFormat(yTicks);
            g.setLineWidth(1);
            g.setFont(Font.font(TICK_FONT_SIZE));
            for (double tick : xTicks) {
                double x = Math.round(centreX + (tick - dataX) * scale) + 0.5;  // crisp hairline
                g.setStroke(GRID);
                g.strokeLine(x, top, x, bottom);
                g.setStroke(AXIS);
                g.strokeLine(x, bottom, x, bottom + TICK_LENGTH);
                g.setFill(INK_SECONDARY);
                g.setTextAlign(TextAlignment.CENTER);
                g.fillText(String.format(xFormat, tick), x,
                        bottom + TICK_LENGTH + TICK_LABEL_GAP + TICK_FONT_SIZE);
            }
            for (double tick : yTicks) {
                double y = Math.round(centreY - (tick - dataY) * scale) + 0.5;
                g.setStroke(GRID);
                g.strokeLine(left, y, right, y);
                g.setStroke(AXIS);
                g.strokeLine(left - TICK_LENGTH, y, left, y);
                g.setFill(INK_SECONDARY);
                g.setTextAlign(TextAlignment.RIGHT);
                g.fillText(String.format(yFormat, tick), left - TICK_LENGTH - TICK_LABEL_GAP,
                        y + TICK_FONT_SIZE / 3);
            }
            g.setStroke(AXIS);
            g.strokeLine(left + 0.5, top, left + 0.5, bottom);
            g.strokeLine(left, bottom + 0.5, right, bottom + 0.5);

            g.setFill(INK_PRIMARY);
            g.setFont(Font.font(AXIS_FONT_SIZE));
            g.setTextAlign(TextAlignment.CENTER);
            g.fillText("MDS 1", (left + right) / 2, bottom + AXIS_TITLE_OFFSET + AXIS_FONT_SIZE);
            g.save();
            g.translate(left - AXIS_TITLE_OFFSET - AXIS_FONT_SIZE / 2, (top + bottom) / 2);
            g.rotate(-90);
            g.fillText("MDS 2", 0, 0);
            g.restore();
        }

        /**
         * Picks enough decimals to tell neighbouring ticks apart.
         *
         * @param ticks (double[]) Evenly spaced tick values.
         * @return (String) Format string such as "%.1f".
         */
        private static String tickFormat(double[] ticks) {
            double step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
            int decimals = (int) Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
            String format = "%." + decimals + "f";
            return format;
        }
    }

    /**
     * Rotatable 3D scatter of the 3D embedding: one shaded sphere per window (tile or cell),
     * three labelled
     * axes, perspective camera; drag rotates, scroll zooms, click selects. Holds the rotation,
     * zoom and selection as state.
     */
    private final class Scatter3D extends Pane {

        private final SubScene subScene;
        private final Label tooltip = createTooltip();
        private final List<Sphere> spheres = new ArrayList<>();
        private final Map<Node, Integer> indexOfSphere = new HashMap<>();
        private final DoubleProperty angleX = new SimpleDoubleProperty(INITIAL_ANGLE_X);
        private final DoubleProperty angleY = new SimpleDoubleProperty(INITIAL_ANGLE_Y);
        private final PerspectiveCamera camera = new PerspectiveCamera(true);
        private int selected = -1;
        private double pressX;
        private double pressY;

        /**
         * Builds the spheres, axes, lights and camera, and wires the mouse handlers.
         */
        Scatter3D() {
            Group world = new Group();
            Rotate rotateX = new Rotate(0, Rotate.X_AXIS);
            Rotate rotateY = new Rotate(0, Rotate.Y_AXIS);
            rotateX.angleProperty().bind(angleX);
            rotateY.angleProperty().bind(angleY);
            world.getTransforms().addAll(rotateX, rotateY);

            // --- Spheres, isotropically scaled into the cube ---
            List<double[]> coordinates = points.stream().map(EmbeddedPoint::mds3).toList();
            double[][] axes = {centreAndHalfRange(coordinates, 0),
                    centreAndHalfRange(coordinates, 1), centreAndHalfRange(coordinates, 2)};
            double half = Math.max(axes[0][1], Math.max(axes[1][1], axes[2][1]));
            double scale = CUBE_HALF / half;
            double radius = points.size() > DENSE_POINTS ? DENSE_SPHERE_RADIUS : SPHERE_RADIUS;
            Map<Color, PhongMaterial> materials = new HashMap<>();
            for (int i = 0; i < points.size(); i++) {
                EmbeddedPoint point = points.get(i);
                Sphere sphere = new Sphere(radius, SPHERE_DIVISIONS);
                sphere.setMaterial(materials.computeIfAbsent(point.colour(), PhongMaterial::new));
                // JavaFX y points down; flip MDS 2 so it points up as in the 2D plot
                sphere.setTranslateX((point.mds3()[0] - axes[0][0]) * scale);
                sphere.setTranslateY(-(point.mds3()[1] - axes[1][0]) * scale);
                sphere.setTranslateZ((point.mds3()[2] - axes[2][0]) * scale);
                spheres.add(sphere);
                indexOfSphere.put(sphere, i);
            }
            world.getChildren().addAll(spheres);
            double[] axisHalf = new double[axes.length];
            for (int a = 0; a < axes.length; a++) {
                axisHalf[a] = Math.max(axes[a][1] * scale, MIN_AXIS_FRACTION * CUBE_HALF)
                        * AXIS_EXTENT;
            }
            addAxes(world, axisHalf);

            PointLight light = new PointLight(Color.gray(LIGHT_LEVEL));
            light.setTranslateX(LIGHT_X);
            light.setTranslateY(LIGHT_Y);
            light.setTranslateZ(LIGHT_Z);
            Group root = new Group(world, new AmbientLight(Color.gray(AMBIENT_LEVEL)), light);
            camera.setFieldOfView(FIELD_OF_VIEW);
            camera.setFarClip(CAMERA_FAR_CLIP);
            camera.setTranslateZ(-CAMERA_DISTANCE);

            subScene = new SubScene(root, 1, 1, true, SceneAntialiasing.BALANCED);
            subScene.setFill(SURFACE);
            subScene.setCamera(camera);
            subScene.setManaged(false);
            getChildren().addAll(subScene, tooltip);
            setMinSize(0, 0);

            subScene.setOnMousePressed(event -> {
                pressX = event.getSceneX();
                pressY = event.getSceneY();
            });
            subScene.setOnMouseDragged(event -> {
                angleY.set(angleY.get() + (event.getSceneX() - pressX) * DEGREES_PER_DRAG_PIXEL);
                angleX.set(angleX.get() - (event.getSceneY() - pressY) * DEGREES_PER_DRAG_PIXEL);
                pressX = event.getSceneX();
                pressY = event.getSceneY();
                tooltip.setVisible(false);
            });
            subScene.setOnScroll(this::onScroll);
            subScene.setOnMouseMoved(event -> {
                int hit = pickedPoint(event.getPickResult());
                if (hit >= 0) {
                    showTooltip(tooltip, points.get(hit), event.getX(), event.getY(), this);
                } else {
                    tooltip.setVisible(false);
                }
            });
            subScene.setOnMouseExited(event -> tooltip.setVisible(false));
            subScene.setOnMouseClicked(event -> {
                int hit = pickedPoint(event.getPickResult());
                if (event.isStillSincePress() && hit >= 0) {
                    selectTile(hit);
                }
            });
        }

        /**
         * Enlarges the selected window's sphere so it stands out, restoring the previous one.
         *
         * @param index (int) Point index, -1 for none.
         * @return (void)
         */
        void setSelected(int index) {
            if (selected >= 0) {
                spheres.get(selected).setScaleX(1);
                spheres.get(selected).setScaleY(1);
                spheres.get(selected).setScaleZ(1);
            }
            selected = index;
            if (selected >= 0) {
                spheres.get(selected).setScaleX(SELECTED_SPHERE_SCALE);
                spheres.get(selected).setScaleY(SELECTED_SPHERE_SCALE);
                spheres.get(selected).setScaleZ(SELECTED_SPHERE_SCALE);
            }
        }

        /**
         * Moves the camera towards or away from the data.
         *
         * @param event (ScrollEvent) Scroll over the scene; scrolling up zooms in.
         * @return (void)
         */
        private void onScroll(ScrollEvent event) {
            double distance = -camera.getTranslateZ() - event.getDeltaY() * ZOOM_PER_SCROLL_PIXEL;
            distance = Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, distance));
            camera.setTranslateZ(-distance);
        }

        /**
         * Maps a pick result to the point whose sphere was hit.
         *
         * @param pick (PickResult) Result of a mouse event on the scene.
         * @return (int) Point index, or -1 when no sphere was hit.
         */
        private int pickedPoint(PickResult pick) {
            Integer index = pick == null ? null : indexOfSphere.get(pick.getIntersectedNode());
            int hit = index == null ? -1 : index;
            return hit;
        }

        /**
         * Adds three thin axes through the data centre, each reaching a little past the data
         * along it, with "MDS 1/2/3" at their positive ends. Labels counter-rotate so they
         * always face the camera.
         *
         * @param world (Group) Rotating group the axes join.
         * @param axisHalf (double[] - size (3)) Half-length of the MDS 1, 2 and 3 axes, in
         *        scene units.
         * @return (void)
         */
        private void addAxes(Group world, double[] axisHalf) {
            PhongMaterial material = new PhongMaterial(AXIS_3D);
            // Cylinders stand along y; rotate them onto x and z
            Cylinder xAxis = new Cylinder(AXIS_RADIUS, 2 * axisHalf[0]);
            xAxis.setRotationAxis(Rotate.Z_AXIS);
            xAxis.setRotate(90);
            Cylinder yAxis = new Cylinder(AXIS_RADIUS, 2 * axisHalf[1]);
            Cylinder zAxis = new Cylinder(AXIS_RADIUS, 2 * axisHalf[2]);
            zAxis.setRotationAxis(Rotate.X_AXIS);
            zAxis.setRotate(90);
            for (Cylinder axis : List.of(xAxis, yAxis, zAxis)) {
                axis.setMaterial(material);
                axis.setMouseTransparent(true);
                world.getChildren().add(axis);
            }
            world.getChildren().addAll(axisLabel("MDS 1", axisHalf[0] + AXIS_LABEL_OFFSET, 0, 0),
                    axisLabel("MDS 2", 0, -axisHalf[1] - AXIS_LABEL_OFFSET, 0),
                    axisLabel("MDS 3", 0, 0, axisHalf[2] + AXIS_LABEL_OFFSET));
        }

        /**
         * Creates a camera-facing axis label.
         *
         * @param text (String) Label text.
         * @param x (double) Position in the rotating group.
         * @param y (double) Position in the rotating group (down is positive).
         * @param z (double) Position in the rotating group.
         * @return (Text) Label that undoes the group's rotation, so it stays readable.
         */
        private Text axisLabel(String text, double x, double y, double z) {
            Text label = new Text(text);
            label.setFont(Font.font(null, FontWeight.BOLD, AXIS_LABEL_FONT_SIZE));
            label.setFill(INK_PRIMARY);
            label.setMouseTransparent(true);
            label.setTranslateX(x - label.getLayoutBounds().getWidth() / 2);
            label.setTranslateY(y + label.getLayoutBounds().getHeight() / 4);
            label.setTranslateZ(z);
            // world = Rx * Ry, so the label applies Ry^-1 then Rx^-1 to cancel it
            Rotate undoY = new Rotate(0, Rotate.Y_AXIS);
            Rotate undoX = new Rotate(0, Rotate.X_AXIS);
            undoY.angleProperty().bind(angleY.negate());
            undoX.angleProperty().bind(angleX.negate());
            label.getTransforms().addAll(undoY, undoX);
            return label;
        }

        /**
         * Sizes the 3D scene to the pane.
         *
         * @return (void)
         */
        @Override
        protected void layoutChildren() {
            subScene.setWidth(getWidth());
            subScene.setHeight(getHeight());
            super.layoutChildren();
        }
    }
}
