/*
 * Collects the detected cells inside a QuPath annotation, their centroids and the ROI mask the
 * PHC bridge reads, and writes the centroids for Python.
 *
 * Contents
 * --------
 * CellExporter : class
 *     Finds the detections whose centroid lies inside an annotation, rasterizes its ROI and
 *     writes the cell centroids as GeoJSON.
 * CellExporter.ExportedCells : record
 *     The cells, their centroids, the ROI mask and the annotation's bounding box.
 */

package qupath.ext.phc;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import qupath.lib.geom.Point2;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImageRegion;
import qupath.lib.roi.PolygonROI;
import qupath.lib.roi.interfaces.ROI;

/**
 * Gathers the cells PHC is computed on: every detection (cells from Analyze > Cell detection,
 * or any other detections) whose centroid lies inside the annotation. A cell's centroid is
 * computed once, from its nucleus when it has one, and the same point decides whether the cell
 * is inside and is what Python receives. Tiles (including PHC heatmap tiles of versions before
 * 0.6.2) are detections too and are always left out.
 */
public final class CellExporter {

    /** Longest side of the ROI mask; larger annotations get a coarser mask. */
    public static final int MAX_MASK_SIDE = 4096;

    /** Shown when an annotation holds no detections, here and by the menu command. */
    public static final String NO_CELLS_MESSAGE = "There are no detected cells inside this "
            + "annotation. Run cell detection first (Analyze > Cell detection > Cell detection, "
            + "or any detection command) with the annotation selected, then run PHC again.";

    /** Export order: by centroid y, then x, then UUID; the same on every run. */
    private static final Comparator<Map.Entry<PathObject, double[]>> EXPORT_ORDER =
            Comparator.<Map.Entry<PathObject, double[]>>comparingDouble(
                            entry -> entry.getValue()[1])
                    .thenComparingDouble(entry -> entry.getValue()[0])
                    .thenComparing(entry -> entry.getKey().getID());

    private CellExporter() {
    }

    /**
     * Finds the detections PHC will use for an annotation, so callers can check there are any
     * before asking for settings, or clean up their PHC results.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @return (List of PathObject) Non-tile detections whose centroid (see
     *         {@link #centroidRoi(PathObject)}) lies inside the annotation's ROI, possibly empty.
     */
    public static List<PathObject> cellsInside(PathObjectHierarchy hierarchy,
                                               PathObject annotation) {
        List<PathObject> cells = collectInside(hierarchy, annotation, new ArrayList<>());
        return cells;
    }

    /**
     * Collects the annotation's cells with their centroids and draws its ROI mask, ready for
     * the PHC bridge.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @return (ExportedCells) Cells, centroids, mask of size (h, w) with h, w <=
     *         {@link #MAX_MASK_SIDE}, its downsample and the annotation's bounding box in slide
     *         pixels.
     * @throws IllegalArgumentException When the ROI has no area or holds no detected cells.
     */
    public static ExportedCells export(PathObjectHierarchy hierarchy, PathObject annotation) {
        ROI roi = annotation.getROI();
        if (roi == null || !roi.isArea()) {
            throw new IllegalArgumentException("PHC needs an area annotation (rectangle, ellipse, "
                    + "polygon or brush), not a line or points.");
        }
        List<double[]> centroids = new ArrayList<>();
        List<PathObject> cells = collectInside(hierarchy, annotation, centroids);
        if (cells.isEmpty()) {
            throw new IllegalArgumentException(NO_CELLS_MESSAGE);
        }
        double longestSide = Math.max(roi.getBoundsWidth(), roi.getBoundsHeight());
        double maskDownsample = Math.max(1.0, longestSide / MAX_MASK_SIDE);
        BufferedImage mask = drawMask(roi, maskDownsample);
        ExportedCells exported = new ExportedCells(cells, centroids.toArray(new double[0][]),
                mask, maskDownsample, roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(),
                roi.getBoundsHeight());
        return exported;
    }

    /**
     * Finds the non-tile detections whose centroid lies inside the annotation, computing each
     * centroid once so the inside test and the export use the same point.
     *
     * @param hierarchy (PathObjectHierarchy) Hierarchy the annotation belongs to.
     * @param annotation (PathObject) Annotation with an area ROI.
     * @param centroids (List of double[]) Filled with the centroid (x, y) of each returned cell,
     *        in the same order.
     * @return (List of PathObject) Cells inside the annotation, possibly empty.
     */
    private static List<PathObject> collectInside(PathObjectHierarchy hierarchy,
                                                  PathObject annotation,
                                                  List<double[]> centroids) {
        ROI roi = annotation.getROI();
        // Detections may be children of the annotation or sit elsewhere in the hierarchy; the
        // region query (by bounds) also finds cells whose nucleus is inside but whose outline
        // centroid is not
        Set<PathObject> candidates = new LinkedHashSet<>(
                hierarchy.getAllDetectionsForRegion(ImageRegion.createInstance(roi)));
        candidates.addAll(annotation.getDescendantObjects(new ArrayList<>()));
        List<Map.Entry<PathObject, double[]>> inside = new ArrayList<>();
        for (PathObject object : candidates) {
            if (!object.isDetection() || object.isTile() || object.getROI() == null
                    || !object.getROI().getImagePlane().equals(roi.getImagePlane())) {
                continue;
            }
            double[] centroid = centroid(centroidRoi(object));
            if (roi.contains(centroid[0], centroid[1])) {
                inside.add(Map.entry(object, centroid));
            }
        }
        // The hierarchy returns detections in a different order on every run; a fixed order
        // keeps seeded subsamples (MDS, capped clustering) and tie-breaking reproducible
        inside.sort(EXPORT_ORDER);
        List<PathObject> cells = new ArrayList<>();
        for (Map.Entry<PathObject, double[]> entry : inside) {
            cells.add(entry.getKey());
            centroids.add(entry.getValue());
        }
        return cells;
    }

    /**
     * Picks the ROI whose centroid stands for a cell: the nucleus, which marks where the cell
     * sits more precisely than its estimated boundary, or the object's own ROI when there is
     * no nucleus (plain detections, or cells without one).
     *
     * @param cell (PathObject) Detection or cell.
     * @return (ROI) Nucleus ROI when present, else the object's ROI.
     */
    static ROI centroidRoi(PathObject cell) {
        ROI nucleus = cell instanceof PathCellObject cellObject ? cellObject.getNucleusROI()
                : null;
        ROI centroidRoi = nucleus != null ? nucleus : cell.getROI();
        return centroidRoi;
    }

    /**
     * Gives the area centroid of a ROI in double precision. QuPath's own centroid of a polygon
     * ROI carries float rounding (~1e-4 px), so polygons use the shoelace formula on the
     * stored vertices; other ROIs (ellipses, rectangles, points) have an exact centroid already.
     *
     * @param roi (ROI) Nucleus or cell ROI.
     * @return (double[] - size (2)) Centroid (x, y) in slide pixels.
     */
    static double[] centroid(ROI roi) {
        double[] centroid = {roi.getCentroidX(), roi.getCentroidY()};
        if (roi instanceof PolygonROI) {
            List<Point2> vertices = roi.getAllPoints();
            Point2 origin = vertices.get(0);  // shift to keep slide-scale coordinates precise
            double area2 = 0;
            double sumX = 0;
            double sumY = 0;
            for (int i = 0; i < vertices.size(); i++) {
                Point2 p = vertices.get(i);
                Point2 q = vertices.get((i + 1) % vertices.size());
                double px = p.getX() - origin.getX();
                double py = p.getY() - origin.getY();
                double qx = q.getX() - origin.getX();
                double qy = q.getY() - origin.getY();
                double cross = px * qy - qx * py;
                area2 += cross;
                sumX += (px + qx) * cross;
                sumY += (py + qy) * cross;
            }
            if (area2 != 0) {  // degenerate polygons keep QuPath's centroid
                centroid[0] = origin.getX() + sumX / (3 * area2);
                centroid[1] = origin.getY() + sumY / (3 * area2);
            }
        }
        return centroid;
    }

    /**
     * Writes the cells Python reads as a GeoJSON FeatureCollection of Point features at each
     * cell's exported centroid, in export order, so the bridge's feature index i is cell i.
     * Only the centroids and UUIDs are written: outlines made the file ~10x larger and slow to
     * write and parse, and measurements (including earlier PHC results) never reach Python.
     *
     * @param path (Path) GeoJSON file to write.
     * @param cells (ExportedCells) Exported cells and their centroids.
     * @return (void)
     * @throws IOException When the file cannot be written.
     */
    static void writeCells(Path path, ExportedCells cells) throws IOException {
        try (Writer out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("{\"type\":\"FeatureCollection\",\"features\":[");
            StringBuilder feature = new StringBuilder();
            for (int i = 0; i < cells.cells().size(); i++) {
                double[] centroid = cells.centroids()[i];
                feature.setLength(0);
                feature.append(i == 0 ? "" : ",")
                        .append("{\"type\":\"Feature\",\"id\":\"")
                        .append(cells.cells().get(i).getID())
                        .append("\",\"geometry\":{\"type\":\"Point\",\"coordinates\":[")
                        .append(centroid[0]).append(',').append(centroid[1]).append("]}}");
                out.append(feature);
            }
            out.write("]}");
        }
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
     * The cells of one annotation, ready for the PHC bridge, plus the bounding box the bridge
     * measures window positions from.
     *
     * @param cells (List of PathObject) Detections whose centroid lies inside the annotation.
     * @param centroids (double[][] - size (n, 2)) Centroid (x, y) of each cell in slide
     *        pixels, in the order of cells; the point Python computes PHC on.
     * @param mask (BufferedImage) 8-bit ROI mask over the bounding box, 255 inside.
     * @param maskDownsample (double) Slide pixels per mask pixel, at least 1.
     * @param originX (double) Left edge of the annotation's bounding box, in slide pixels.
     * @param originY (double) Top edge of the annotation's bounding box, in slide pixels.
     * @param width (double) Width of the bounding box, in slide pixels.
     * @param height (double) Height of the bounding box, in slide pixels.
     */
    public record ExportedCells(List<PathObject> cells, double[][] centroids,
                                BufferedImage mask, double maskDownsample, double originX,
                                double originY, double width, double height) {
    }
}
