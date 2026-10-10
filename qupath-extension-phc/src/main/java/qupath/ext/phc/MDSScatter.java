/*
 * Shared pieces of the 2D and 3D MDS scatter plots: plotted points, colours and the tooltip.
 *
 * Contents
 * --------
 * MDSScatter : class
 *     Colours, hover tooltip and range helper used by both MDS scatter plots.
 * MDSScatter.Point : record
 *     One plotted cell with its cluster, colour, hover details and MDS coordinates.
 */

package qupath.ext.phc;

import java.util.List;

import javafx.scene.control.Label;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;

import qupath.lib.objects.PathObject;

/**
 * Holds what {@link MDSScatter2D} and {@link MDSScatter3D} share, so both plots look and
 * behave alike: the chart colours on the white figure surface, the hover tooltip and the
 * data range of the coordinates.
 */
final class MDSScatter {

    /** Text and chart-chrome colours on the white figure surface (text never in data colour). */
    static final Color SURFACE = Color.WHITE;
    static final Color INK_PRIMARY = Color.web("#1f1f1f");
    static final Color INK_SECONDARY = Color.web("#5f5f5f");

    /** Above this many points both plots draw smaller marks. */
    static final int DENSE_POINTS = 1500;

    private static final double TOOLTIP_OFFSET = 14;
    private static final String TOOLTIP_STYLE = "-fx-background-color: white; "
            + "-fx-border-color: #c8c8c8; -fx-border-radius: 4; -fx-background-radius: 4; "
            + "-fx-padding: 6 8 6 8; -fx-font-size: 12px; -fx-text-fill: #1f1f1f;";
    private static final String L2_FORMAT = "%.4g";

    private MDSScatter() {
    }

    /**
     * Creates the hover tooltip shared by both plots: cluster, cells in the window and L2
     * norm.
     *
     * @return (Label) Hidden, unmanaged label that follows the pointer.
     */
    static Label createTooltip() {
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
     * @param point (Point) Point under the pointer.
     * @param x (double) Pointer x in the plot's coordinates.
     * @param y (double) Pointer y in the plot's coordinates.
     * @param bounds (Region) Plot the tooltip must stay inside.
     * @return (void)
     */
    static void showTooltip(Label tooltip, Point point, double x, double y, Region bounds) {
        tooltip.setText(PHCPipeline.CLASS_PREFIX + point.cluster() + "\n"
                + (long) point.nCells() + " cells in window"
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
     * Gives the centre and half-width of the coordinates along one axis.
     *
     * @param coordinates (List of double[]) Points, each of size (d).
     * @param axis (int) Axis to measure, in [0, d).
     * @return (double[] - size (2)) Centre and half of the range (1 when all are equal).
     */
    static double[] centreAndHalfRange(List<double[]> coordinates, int axis) {
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
     * One plotted cell: the cell, its cluster, colour, hover details and MDS coordinates.
     *
     * @param object (PathObject) The cell the window is centred on.
     * @param cluster (int) Cluster number, 1 = lowest mean L2 norm.
     * @param colour (Color) Viridis colour of the cluster.
     * @param nCells (double) Cell centroids in the window.
     * @param l2Norm (double) L2 norm of the window's persistence vector.
     * @param mds2 (double[] - size (2)) 2D MDS coordinates.
     * @param mds3 (double[] - size (3)) 3D MDS coordinates.
     */
    record Point(PathObject object, int cluster, Color colour, double nCells, double l2Norm,
                 double[] mds2, double[] mds3) {
    }
}
