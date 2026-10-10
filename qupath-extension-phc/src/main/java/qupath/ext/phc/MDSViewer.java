/*
 * Window with interactive 2D and 3D scatter plots of the MDS embedding of a PHC run, linked to
 * its cells.
 *
 * Contents
 * --------
 * MDSViewer : class
 *     Window with a 2D and a 3D tab plotting the MDS coordinates stored on PHC cells.
 */

package qupath.ext.phc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Consumer;

import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.Background;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
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
 * Shows where each cell's window sits in the metric MDS embedding of the cells' L2
 * dissimilarity matrix, so cells with similar local topology can be found as clusters of
 * points. Points share the viridis cluster colours of the PHC classes. Clicking a point
 * selects its cell in QuPath and centres the viewer on it; selecting it in QuPath highlights
 * its point. Reads everything from the cells' measurements, so it also works after a project
 * is reopened. Create and use it on the JavaFX thread.
 */
public final class MDSViewer {

    private static final Logger logger = LoggerFactory.getLogger(MDSViewer.class);

    /** Index of the 2D and 3D tabs, for {@link #saveSnapshot(int, File)}. */
    public static final int TAB_2D = 0;
    public static final int TAB_3D = 1;

    private static final String NOUN = "cells";  // what the points are, in titles and text
    private static final double WINDOW_WIDTH = 900;
    private static final double WINDOW_HEIGHT = 760;
    private static final double FIGURE_PADDING = 16;
    private static final double SECTION_GAP = 6;
    private static final double LEGEND_GAP = 16;
    private static final double LEGEND_SWATCH_RADIUS = 5;
    private static final double TITLE_FONT_SIZE = 16;
    private static final double TEXT_FONT_SIZE = 12;
    private static final String STRESS_FORMAT = "%.3f";

    private final List<MDSScatter.Point> points;
    private final int nCells;
    private final PathObjectHierarchy hierarchy;
    private final Consumer<PathObject> centreOn;
    private final Path plotDir;
    private final String filePrefix;
    private final Stage stage;
    private final TabPane tabPane;
    private final MDSScatter2D scatter2d;
    private final MDSScatter3D scatter3d;
    private final List<Region> figures = new ArrayList<>();  // snapshot target per tab
    private final PathObjectSelectionListener selectionListener;

    /**
     * Builds the viewer window for the MDS coordinates stored on a run's cells; nothing is
     * shown until {@link #show(Window)}.
     *
     * @param cells (Collection of PathObject) Clustered cells of one annotation; those
     *        without MDS measurements are counted in the subtitle but not plotted.
     * @param hierarchy (PathObjectHierarchy) Hierarchy holding the cells, whose selection
     *        model links points and cells.
     * @param stress2d (Double) Kruskal stress-1 of the 2D embedding, or null when unknown.
     * @param stress3d (Double) Kruskal stress-1 of the 3D embedding, or null when unknown.
     * @param plotDir (Path) Folder Python saved its plots and CSV to, or null when none.
     * @param filePrefix (String) Name suggested for saved snapshots, e.g. the plot prefix.
     * @param centreOn (Consumer of PathObject) Centres the QuPath viewer on a clicked cell;
     *        may do nothing (e.g. in tests).
     * @throws IllegalArgumentException When no cell carries MDS coordinates.
     */
    public MDSViewer(Collection<PathObject> cells, PathObjectHierarchy hierarchy,
                     Double stress2d, Double stress3d, Path plotDir, String filePrefix,
                     Consumer<PathObject> centreOn) {
        this.points = embeddedPoints(cells);
        if (points.isEmpty()) {
            throw new IllegalArgumentException("These PHC " + NOUN + " have no MDS coordinates. "
                    + "Run PHC again with 'Compute MDS embedding' switched on.");
        }
        this.nCells = cells.size();
        this.hierarchy = hierarchy;
        this.centreOn = centreOn;
        this.plotDir = plotDir;
        this.filePrefix = filePrefix;

        scatter2d = new MDSScatter2D(points, this::select);
        scatter3d = new MDSScatter3D(points, this::select);
        String figureTitle = "MDS of per-cell PHC persistence vectors";
        tabPane = new TabPane(
                createTab("2D", buildFigure(figureTitle + " (2D)",
                        subtitle(stress2d), scatter2d), TAB_2D),
                createTab("3D", buildFigure(figureTitle + " (3D)",
                        subtitle(stress3d) + "  ·  drag to rotate, scroll to zoom",
                        scatter3d), TAB_3D));
        tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);

        stage = new Stage();
        stage.setTitle("PHC MDS embedding (" + NOUN + ")");
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
     * Closes the window and stops listening to the cell selection.
     *
     * @return (void)
     */
    public void close() {
        stage.close();
        hierarchy.getSelectionModel().removePathObjectSelectionListener(selectionListener);
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
        return scatter2d.canvas();
    }

    /**
     * Finds where a cell's point is drawn on the 2D canvas.
     *
     * @param cell (PathObject) Cell with MDS coordinates.
     * @return (Point2D) Canvas coordinates of the point's centre, or null when it is not
     *         plotted.
     */
    Point2D canvasPosition(PathObject cell) {
        int index = scatter2d.indexOf(cell);
        Point2D position = index < 0 ? null : scatter2d.position(index);
        return position;
    }

    /**
     * Gives the cell whose point is highlighted, to check the link with QuPath's selection.
     *
     * @return (PathObject) Highlighted cell, or null.
     */
    PathObject highlightedCell() {
        int index = scatter2d.selected();
        PathObject cell = index < 0 ? null : points.get(index).object();
        return cell;
    }

    /**
     * Selects a cell in QuPath and centres the viewer on it, after a click on its point.
     *
     * @param index (int) Index of the clicked point in {@link #points}.
     * @return (void)
     */
    private void select(int index) {
        PathObject cell = points.get(index).object();
        hierarchy.getSelectionModel().setSelectedObject(cell);
        highlight(cell);
        centreOn.accept(cell);
    }

    /**
     * Highlights the point of the cell selected in QuPath in both plots, or clears the
     * highlight when the selection is not plotted.
     *
     * @param cell (PathObject) Newly selected object, may be null.
     * @return (void)
     */
    private void highlight(PathObject cell) {
        int index = scatter2d.indexOf(cell);
        scatter2d.setSelected(index);
        scatter3d.setSelected(index);
    }

    /**
     * Describes the embedding below the title: cells embedded and stress.
     *
     * @param stress (Double) Kruskal stress-1 of this embedding, or null.
     * @return (String) e.g. "Metric MDS of the L2 distances between 50 cells  ·  stress 0.123".
     */
    private String subtitle(Double stress) {
        String embedded = points.size() == nCells
                ? points.size() + " " + NOUN
                : points.size() + " of " + nCells + " " + NOUN + " (random subsample)";
        String stressText = stress == null ? "stress n/a"
                : "Kruskal stress " + String.format(STRESS_FORMAT, stress);
        String text = "Metric MDS of the L2 distances between " + embedded + "  ·  "
                + stressText;
        return text;
    }

    /**
     * Puts a figure and its Save PNG button / plot-folder note (naming the Delaunay graph
     * plot of a spatially constrained run when Python wrote one) into a tab.
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
        Path delaunay = plotDir == null ? null
                : new PythonBridge.PlotTarget(plotDir, filePrefix).delaunayPlot();
        if (delaunay != null && delaunay.toFile().isFile()) {
            note += " (Delaunay graph: " + delaunay.getFileName() + ")";
        }
        Label noteLabel = new Label(note);
        noteLabel.setWrapText(true);
        noteLabel.setTextFill(MDSScatter.INK_SECONDARY);
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
        chooser.setInitialFileName(filePrefix + PythonBridge.PlotTarget.CELLS_SUFFIX + "_mds_"
                + name.toLowerCase() + "_view.png");
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
     * @param subtitle (String) Line under the title (cells, stress, hints).
     * @param plot (Region) The 2D or 3D plot, which grows to fill the window.
     * @return (Region) The figure, which is also what Save PNG captures.
     */
    private Region buildFigure(String title, String subtitle, Region plot) {
        Label titleLabel = new Label(title);
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, TITLE_FONT_SIZE));
        titleLabel.setTextFill(MDSScatter.INK_PRIMARY);
        Label subtitleLabel = new Label(subtitle);
        subtitleLabel.setFont(Font.font(TEXT_FONT_SIZE));
        subtitleLabel.setTextFill(MDSScatter.INK_SECONDARY);
        VBox figure = new VBox(SECTION_GAP, titleLabel, subtitleLabel, buildLegend(), plot);
        VBox.setVgrow(plot, Priority.ALWAYS);
        figure.setPadding(new Insets(FIGURE_PADDING, FIGURE_PADDING, 0, FIGURE_PADDING));
        figure.setBackground(Background.fill(MDSScatter.SURFACE));
        return figure;
    }

    /**
     * Builds the cluster legend: a coloured dot and "PHC cluster k (n)" per cluster.
     *
     * @return (HBox) Legend row, clusters in increasing order.
     */
    private HBox buildLegend() {
        Map<Integer, int[]> counts = new TreeMap<>();   // cluster -> {count}
        Map<Integer, Color> colours = new HashMap<>();
        for (MDSScatter.Point point : points) {
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
            label.setTextFill(MDSScatter.INK_PRIMARY);
            HBox item = new HBox(SECTION_GAP, swatch, label);
            item.setAlignment(Pos.CENTER_LEFT);
            legend.getChildren().add(item);
        }
        return legend;
    }

    /**
     * Reads the MDS coordinates and hover details off the cells. Colours come from the
     * cluster number on the viridis ramp, since cells keep their own class unless PHC was
     * asked to classify them.
     *
     * @param cells (Collection of PathObject) Clustered cells.
     * @return (List of MDSScatter.Point) One entry per cell with MDS coordinates, in the same
     *         order.
     */
    private static List<MDSScatter.Point> embeddedPoints(Collection<PathObject> cells) {
        int nClusters = (int) cells.stream()
                .filter(c -> c.getMeasurementList().containsKey(PHCPipeline.MEASUREMENT_CLUSTER))
                .mapToDouble(c -> c.getMeasurementList().get(PHCPipeline.MEASUREMENT_CLUSTER))
                .max().orElse(1);
        List<MDSScatter.Point> embedded = new ArrayList<>();
        for (PathObject cell : PHCPipeline.embeddedCells(cells)) {
            MeasurementList m = cell.getMeasurementList();
            int cluster = (int) m.get(PHCPipeline.MEASUREMENT_CLUSTER);
            int rgb = PHCPipeline.clusterColor(cluster - 1, nClusters);
            embedded.add(new MDSScatter.Point(cell, cluster,
                    Color.rgb(ColorTools.red(rgb), ColorTools.green(rgb), ColorTools.blue(rgb)),
                    m.get(PHCPipeline.MEASUREMENT_CELLS_IN_WINDOW),
                    m.get(PHCPipeline.MEASUREMENT_L2_NORM),
                    new double[] {m.get(PHCPipeline.MEASUREMENT_MDS2_X),
                            m.get(PHCPipeline.MEASUREMENT_MDS2_Y)},
                    new double[] {m.get(PHCPipeline.MEASUREMENT_MDS3_X),
                            m.get(PHCPipeline.MEASUREMENT_MDS3_Y),
                            m.get(PHCPipeline.MEASUREMENT_MDS3_Z)}));
        }
        return embedded;
    }
}
