/*
 * Collects the detected cells inside a QuPath annotation, plus the ROI mask the PHC bridge reads.
 *
 * Contents
 * --------
 * CellExporter : class
 *     Finds the detections whose centroid lies inside an annotation and rasterizes its ROI.
 * CellExporter.ExportedCells : record
 *     The cells, the ROI mask and the bounding box that maps bridge windows back onto the slide.
 */

package qupath.ext.phc;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.roi.interfaces.ROI;

/**
 * Gathers the cells PHC is computed on: every detection (cells from Analyze > Cell detection,
 * or any other detections) whose ROI centroid lies inside the annotation. PHC heatmap tiles,
 * and tiles in general, are detections too and are always left out, so a second run does not
 * mistake the first run's tiles for cells.
 */
public final class CellExporter {

    /** Longest side of the ROI mask; larger annotations get a coarser mask. */
    public static final int MAX_MASK_SIDE = 4096;

    /** Shown when an annotation holds no detections, here and by the menu command. */
    public static final String NO_CELLS_MESSAGE = "There are no detected cells inside this "
            + "annotation. Run cell detection first (Analyze > Cell detection > Cell detection, "
            + "or any detection command) with the annotation selected, then run PHC again.";

    private CellExporter() {
    }

    /**
     * Finds the detections PHC will use for an annotation, so callers can check there are any
     * before asking for settings.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @return (List of PathObject) Non-tile detections whose ROI centroid lies inside the
     *         annotation's ROI, possibly empty.
     */
    public static List<PathObject> cellsInside(PathObjectHierarchy hierarchy,
                                               PathObject annotation) {
        ROI roi = annotation.getROI();
        // Detections may be children of the annotation or sit elsewhere in the hierarchy
        Set<PathObject> candidates = new LinkedHashSet<>(hierarchy.getAllDetectionsForROI(roi));
        candidates.addAll(annotation.getDescendantObjects(new ArrayList<>()));
        List<PathObject> cells = candidates.stream()
                .filter(PathObject::isDetection)
                .filter(object -> !object.isTile())
                .filter(object -> object.getROI() != null
                        && object.getROI().getImagePlane().equals(roi.getImagePlane())
                        && roi.contains(object.getROI().getCentroidX(),
                                object.getROI().getCentroidY()))
                .toList();
        return cells;
    }

    /**
     * Collects the annotation's cells and draws its ROI mask, ready for the PHC bridge.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @return (ExportedCells) Cells, mask of size (h, w) with h, w <= {@link #MAX_MASK_SIDE},
     *         its downsample and the annotation's bounding box in slide pixels.
     * @throws IllegalArgumentException When the ROI has no area or holds no detected cells.
     */
    public static ExportedCells export(PathObjectHierarchy hierarchy, PathObject annotation) {
        ROI roi = annotation.getROI();
        if (roi == null || !roi.isArea()) {
            throw new IllegalArgumentException("PHC needs an area annotation (rectangle, ellipse, "
                    + "polygon or brush), not a line or points.");
        }
        List<PathObject> cells = cellsInside(hierarchy, annotation);
        if (cells.isEmpty()) {
            throw new IllegalArgumentException(NO_CELLS_MESSAGE);
        }
        double longestSide = Math.max(roi.getBoundsWidth(), roi.getBoundsHeight());
        double maskDownsample = Math.max(1.0, longestSide / MAX_MASK_SIDE);
        BufferedImage mask = drawMask(roi, maskDownsample);
        ExportedCells exported = new ExportedCells(cells, mask, maskDownsample,
                roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight());
        return exported;
    }

    /**
     * Rasterizes the ROI onto a grid over its bounding box, so the bridge can tell how much of
     * each window lies inside the annotation. Mask pixel (i, j) covers slide pixels
     * x in [X + j*D, X + (j+1)*D) and y in [Y + i*D, Y + (i+1)*D).
     *
     * @param roi (ROI) Area ROI in full-resolution pixels.
     * @param maskDownsample (double) Slide pixels per mask pixel (D), at least 1.
     * @return (BufferedImage) 8-bit mask of size (ceil(H / D), ceil(W / D)): 255 inside the
     *         ROI, 0 outside.
     */
    private static BufferedImage drawMask(ROI roi, double maskDownsample) {
        int width = Math.max(1, (int) Math.ceil(roi.getBoundsWidth() / maskDownsample));
        int height = Math.max(1, (int) Math.ceil(roi.getBoundsHeight() / maskDownsample));
        BufferedImage mask = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = mask.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_OFF);  // keep the mask strictly 0 / 255
        graphics.scale(1.0 / maskDownsample, 1.0 / maskDownsample);
        graphics.translate(-roi.getBoundsX(), -roi.getBoundsY());
        graphics.setColor(Color.WHITE);
        graphics.fill(roi.getShape());
        graphics.dispose();
        return mask;
    }

    /**
     * The cells of one annotation, ready for the PHC bridge, plus the bounding box that maps
     * the bridge's windows back onto the slide.
     *
     * @param cells (List of PathObject) Detections whose centroid lies inside the annotation.
     * @param mask (BufferedImage) 8-bit ROI mask over the bounding box, 255 inside.
     * @param maskDownsample (double) Slide pixels per mask pixel, at least 1.
     * @param originX (double) Left edge of the annotation's bounding box, in slide pixels.
     * @param originY (double) Top edge of the annotation's bounding box, in slide pixels.
     * @param width (double) Width of the bounding box, in slide pixels.
     * @param height (double) Height of the bounding box, in slide pixels.
     */
    public record ExportedCells(List<PathObject> cells, BufferedImage mask,
                                double maskDownsample, double originX, double originY,
                                double width, double height) {
    }
}
