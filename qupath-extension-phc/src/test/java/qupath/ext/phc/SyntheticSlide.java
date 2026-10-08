/*
 * Builds a blank calibrated test image with synthetic cell detections for the PHC checks.
 *
 * Contents
 * --------
 * SyntheticSlide : class
 *     Creates the test image, an annotation full of ring-arranged and random cells, and helpers.
 */

package qupath.ext.phc;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.ImageServers;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/**
 * Gives the checks a reproducible slide without a dataset: a blank image calibrated at
 * {@link #PIXEL_SIZE_MICRONS}, and an annotation whose left half holds cells arranged on rings
 * (gland-like loops, strong H1 alpha persistence) and whose right half holds the same number
 * of uniformly random cells.
 */
final class SyntheticSlide {

    static final int IMAGE_WIDTH = 2400;
    static final int IMAGE_HEIGHT = 1200;
    static final double PIXEL_SIZE_MICRONS = 0.5;

    /** Annotation bounding box, in slide pixels: 2000 x 1000 px = 1000 x 500 um. */
    static final double ANNOTATION_X = 200;
    static final double ANNOTATION_Y = 100;
    static final double ANNOTATION_WIDTH = 2000;
    static final double ANNOTATION_HEIGHT = 1000;

    /** An area with no cells, to check the "run cell detection first" error. */
    static final double EMPTY_X = 2250;
    static final double EMPTY_Y = 200;
    static final double EMPTY_WIDTH = 100;
    static final double EMPTY_HEIGHT = 800;

    private static final double RING_SPACING = 100;   // ring centres on a 100 px lattice
    private static final double RING_RADIUS = 35;
    private static final int CELLS_PER_RING = 16;
    private static final double CELL_RADIUS = 5;
    private static final double NUCLEUS_RADIUS = 3;
    private static final int N_OUTSIDE_CELLS = 5;     // cells left of the annotation
    private static final double OUTSIDE_X = 100;
    private static final double OUTSIDE_Y_STEP = 150;
    private static final long SEED = 42;
    private static final int CELL_DOT_RADIUS = 2;     // rendering only

    private SyntheticSlide() {
    }

    /**
     * Writes a blank PNG and opens it as a calibrated image.
     *
     * @param dir (File) Folder the PNG is written to.
     * @return (ImageData of BufferedImage) Image with an empty hierarchy and a pixel size of
     *         {@link #PIXEL_SIZE_MICRONS} um.
     * @throws Exception When the PNG cannot be written or opened.
     */
    static ImageData<BufferedImage> createImageData(File dir) throws Exception {
        File png = new File(dir, "blank_slide.png");
        BufferedImage blank = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = blank.createGraphics();
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
        graphics.dispose();
        ImageIO.write(blank, "png", png);

        ImageServer<BufferedImage> server = ImageServers.buildServer(png.getAbsolutePath());
        server.setMetadata(new ImageServerMetadata.Builder(server.getMetadata())
                .pixelSizeMicrons(PIXEL_SIZE_MICRONS, PIXEL_SIZE_MICRONS).build());
        ImageData<BufferedImage> imageData = new ImageData<>(server);
        return imageData;
    }

    /**
     * Adds the test annotation with its ring and random cells as children, plus a few cells
     * outside it at the hierarchy root.
     *
     * @param imageData (ImageData of BufferedImage) Image from {@link #createImageData(File)}.
     * @return (PathObject) The annotation, already in the hierarchy.
     */
    static PathObject addAnnotationWithCells(ImageData<BufferedImage> imageData) {
        PathObjectHierarchy hierarchy = imageData.getHierarchy();
        PathObject annotation = PathObjects.createAnnotationObject(ROIs.createRectangleROI(
                ANNOTATION_X, ANNOTATION_Y, ANNOTATION_WIDTH, ANNOTATION_HEIGHT, null));
        hierarchy.addObject(annotation);
        annotation.addChildObjects(ringCells());
        annotation.addChildObjects(randomCells(ringCells().size()));
        hierarchy.fireHierarchyChangedEvent(annotation);

        List<PathObject> outside = new ArrayList<>();
        for (int i = 0; i < N_OUTSIDE_CELLS; i++) {
            outside.add(cell(OUTSIDE_X, ANNOTATION_Y + (i + 1) * OUTSIDE_Y_STEP));
        }
        hierarchy.addObjects(outside);
        return annotation;
    }

    /**
     * Adds an annotation that holds no detections.
     *
     * @param imageData (ImageData of BufferedImage) Test image.
     * @return (PathObject) The empty annotation, already in the hierarchy.
     */
    static PathObject addEmptyAnnotation(ImageData<BufferedImage> imageData) {
        PathObject annotation = PathObjects.createAnnotationObject(ROIs.createRectangleROI(
                EMPTY_X, EMPTY_Y, EMPTY_WIDTH, EMPTY_HEIGHT, null));
        imageData.getHierarchy().addObject(annotation);
        return annotation;
    }

    /**
     * Number of cells inside the test annotation.
     *
     * @return (int) Ring cells plus random cells.
     */
    static int nCellsInside() {
        int nCells = 2 * ringCells().size();
        return nCells;
    }

    /**
     * Tells whether a slide x coordinate lies in the ring (left) half of the annotation.
     *
     * @param x (double) Slide x coordinate, in pixels.
     * @return (boolean) True for the ring half, false for the random half.
     */
    static boolean isRingSide(double x) {
        boolean ringSide = x < ANNOTATION_X + ANNOTATION_WIDTH / 2;
        return ringSide;
    }

    /**
     * Gives the point PHC uses for a cell: its nucleus centroid, else its ROI centroid.
     *
     * @param cell (PathObject) Cell or other detection.
     * @return (double[]) Centroid {x, y} in slide pixels.
     */
    static double[] centroid(PathObject cell) {
        ROI roi = cell.getROI();
        if (cell instanceof PathCellObject cellObject && cellObject.getNucleusROI() != null) {
            roi = cellObject.getNucleusROI();
        }
        double[] point = {roi.getCentroidX(), roi.getCentroidY()};
        return point;
    }

    /**
     * Draws cells as dots on a white canvas the size of the image, for heatmap renders.
     *
     * @param cells (List of PathObject) Cells to draw.
     * @return (BufferedImage) RGB canvas of size (IMAGE_HEIGHT, IMAGE_WIDTH).
     */
    static BufferedImage drawCells(List<PathObject> cells) {
        BufferedImage canvas = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = canvas.createGraphics();
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
        graphics.setColor(Color.BLACK);
        for (PathObject cell : cells) {
            double[] point = centroid(cell);
            graphics.fillOval((int) point[0] - CELL_DOT_RADIUS, (int) point[1] - CELL_DOT_RADIUS,
                    2 * CELL_DOT_RADIUS, 2 * CELL_DOT_RADIUS);
        }
        graphics.dispose();
        return canvas;
    }

    /**
     * Places rings of cells on a lattice over the left half of the annotation; every
     * 200 x 200 px window aligned with the annotation holds exactly four whole rings.
     *
     * @return (List of PathObject) Ring cells, CELLS_PER_RING per ring.
     */
    private static List<PathObject> ringCells() {
        List<PathObject> cells = new ArrayList<>();
        Random random = new Random(SEED);
        double halfSpacing = RING_SPACING / 2;
        for (double cy = ANNOTATION_Y + halfSpacing; cy < ANNOTATION_Y + ANNOTATION_HEIGHT;
             cy += RING_SPACING) {
            for (double cx = ANNOTATION_X + halfSpacing; isRingSide(cx); cx += RING_SPACING) {
                double phase = random.nextDouble() * 2 * Math.PI;  // rotate each ring
                for (int k = 0; k < CELLS_PER_RING; k++) {
                    double angle = phase + 2 * Math.PI * k / CELLS_PER_RING;
                    cells.add(cell(cx + RING_RADIUS * Math.cos(angle),
                            cy + RING_RADIUS * Math.sin(angle)));
                }
            }
        }
        return cells;
    }

    /**
     * Scatters cells uniformly over the right half of the annotation.
     *
     * @param nCells (int) Number of cells to place.
     * @return (List of PathObject) Random cells.
     */
    private static List<PathObject> randomCells(int nCells) {
        List<PathObject> cells = new ArrayList<>();
        Random random = new Random(SEED);
        double halfWidth = ANNOTATION_WIDTH / 2;
        for (int i = 0; i < nCells; i++) {
            cells.add(cell(ANNOTATION_X + halfWidth + random.nextDouble() * halfWidth,
                    ANNOTATION_Y + random.nextDouble() * ANNOTATION_HEIGHT));
        }
        return cells;
    }

    /**
     * Creates one round cell with a concentric nucleus, like Analyze > Cell detection does.
     *
     * @param x (double) Centre x, in slide pixels.
     * @param y (double) Centre y, in slide pixels.
     * @return (PathObject) Cell object.
     */
    private static PathObject cell(double x, double y) {
        ROI boundary = ROIs.createEllipseROI(x - CELL_RADIUS, y - CELL_RADIUS, 2 * CELL_RADIUS,
                2 * CELL_RADIUS, null);
        ROI nucleus = ROIs.createEllipseROI(x - NUCLEUS_RADIUS, y - NUCLEUS_RADIUS,
                2 * NUCLEUS_RADIUS, 2 * NUCLEUS_RADIUS, null);
        PathObject cellObject = PathObjects.createCellObject(boundary, nucleus);
        return cellObject;
    }
}
