/*
 * Canvas scatter plot of the 2D MDS embedding of a PHC run's cells.
 *
 * Contents
 * --------
 * MDSScatter2D : class
 *     2D scatter with axes, grid, hover tooltip and click-to-select, drawn on one canvas.
 */

package qupath.ext.phc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.IntConsumer;

import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import qupath.lib.objects.PathObject;

/**
 * Canvas scatter of the 2D embedding with axes, grid, hover tooltip and click-to-select. A
 * canvas draws thousands of points much faster than one chart node per point. Holds the
 * current hover and selection as state.
 */
final class MDSScatter2D extends Pane {

    private static final Color GRID = Color.web("#e8e8e8");
    private static final Color AXIS = Color.web("#b5b5b5");
    private static final Color HIGHLIGHT_RING = Color.web("#111111");
    private static final double TICK_FONT_SIZE = 11;
    private static final double AXIS_FONT_SIZE = 12;

    /** Plot margins of the canvas, in pixels, leaving room for ticks and axis titles. */
    private static final double MARGIN_LEFT = 62;
    private static final double MARGIN_RIGHT = 18;
    private static final double MARGIN_TOP = 12;
    private static final double MARGIN_BOTTOM = 48;
    private static final double DATA_PADDING = 0.05;     // fraction of the range around the data
    private static final int TARGET_TICKS = 6;
    private static final double TICK_LENGTH = 4;
    private static final double TICK_LABEL_GAP = 6;
    private static final double AXIS_TITLE_OFFSET = 34;   // from the plot edge

    /** Point marks: >= 8 px dots with a surface ring; smaller for dense plots. */
    private static final double POINT_RADIUS = 4;
    private static final double DENSE_POINT_RADIUS = 3;
    private static final double POINT_RING_WIDTH = 1.2;
    private static final double SELECTED_RADIUS = 7;
    private static final double SELECTED_RING_WIDTH = 2;
    private static final double HOVER_RING_WIDTH = 1.5;
    private static final double HIT_RADIUS = 8;           // pointer distance that still hits
    private static final long DRAW_ORDER_SEED = 0;        // shuffles points so no cluster hides

    private final List<MDSScatter.Point> points;
    private final Canvas canvas = new Canvas();
    private final Label tooltip = MDSScatter.createTooltip();
    private final Map<PathObject, Integer> indexOf = new HashMap<>();
    private final int[] drawOrder;
    private final double[] screenX;
    private final double[] screenY;
    private final double pointRadius;
    private int selected = -1;
    private int hovered = -1;

    /**
     * Sets up the canvas and its mouse handlers; drawing happens on every resize.
     *
     * @param points (List of MDSScatter.Point) Embedded cells to plot.
     * @param onClick (IntConsumer) Called with the index of a clicked point in points.
     */
    MDSScatter2D(List<MDSScatter.Point> points, IntConsumer onClick) {
        this.points = points;
        int n = points.size();
        screenX = new double[n];
        screenY = new double[n];
        pointRadius = n > MDSScatter.DENSE_POINTS ? DENSE_POINT_RADIUS : POINT_RADIUS;
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            indexOf.put(points.get(i).object(), i);
            order.add(i);
        }
        // cells come in slide order; shuffle so one cluster never paints over all
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
                onClick.accept(hit);
            }
        });
    }

    /**
     * Gives the canvas, so tests can send it mouse events.
     *
     * @return (Canvas) Canvas the scatter is drawn on.
     */
    Canvas canvas() {
        return canvas;
    }

    /**
     * Finds the index of a cell's point, e.g. to highlight the cell selected in QuPath.
     *
     * @param cell (PathObject) Cell, may be null.
     * @return (int) Index in the plotted points, or -1 when the cell is not plotted.
     */
    int indexOf(PathObject cell) {
        Integer index = cell == null ? null : indexOf.get(cell);
        int found = index == null ? -1 : index;
        return found;
    }

    /**
     * Finds where a point is drawn on the canvas.
     *
     * @param index (int) Index of a plotted point.
     * @return (Point2D) Canvas coordinates of the point's centre.
     */
    Point2D position(int index) {
        Point2D position = new Point2D(screenX[index], screenY[index]);
        return position;
    }

    /**
     * Gives the highlighted point.
     *
     * @return (int) Index of the selected point, or -1 for none.
     */
    int selected() {
        return selected;
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
            MDSScatter.showTooltip(tooltip, points.get(hit), event.getX(), event.getY(), this);
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
        g.setFill(MDSScatter.SURFACE);
        g.fillRect(0, 0, width, height);

        // --- Equal-aspect scale: data units -> pixels ---
        List<double[]> coordinates = points.stream().map(MDSScatter.Point::mds2).toList();
        double[] xs = MDSScatter.centreAndHalfRange(coordinates, 0);
        double[] ys = MDSScatter.centreAndHalfRange(coordinates, 1);
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
        g.setStroke(MDSScatter.SURFACE);
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
        g.setStroke(MDSScatter.SURFACE);
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
            g.setFill(MDSScatter.INK_SECONDARY);
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
            g.setFill(MDSScatter.INK_SECONDARY);
            g.setTextAlign(TextAlignment.RIGHT);
            g.fillText(String.format(yFormat, tick), left - TICK_LENGTH - TICK_LABEL_GAP,
                    y + TICK_FONT_SIZE / 3);
        }
        g.setStroke(AXIS);
        g.strokeLine(left + 0.5, top, left + 0.5, bottom);
        g.strokeLine(left, bottom + 0.5, right, bottom + 0.5);

        g.setFill(MDSScatter.INK_PRIMARY);
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
