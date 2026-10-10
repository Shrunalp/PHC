/*
 * Rotatable 3D scatter plot of the 3D MDS embedding of a PHC run's cells.
 *
 * Contents
 * --------
 * MDSScatter3D : class
 *     3D scatter of shaded spheres with labelled axes; drag rotates, scroll zooms, click
 *     selects.
 */

package qupath.ext.phc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntConsumer;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.scene.AmbientLight;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.PerspectiveCamera;
import javafx.scene.PointLight;
import javafx.scene.SceneAntialiasing;
import javafx.scene.SubScene;
import javafx.scene.control.Label;
import javafx.scene.input.PickResult;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.Sphere;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.transform.Rotate;

/**
 * Rotatable 3D scatter of the 3D embedding: one shaded sphere per cell, three labelled axes,
 * perspective camera; drag rotates, scroll zooms, click selects. Holds the rotation, zoom and
 * selection as state.
 */
final class MDSScatter3D extends Pane {

    /** Data fills a cube of side 2 * CUBE_HALF, viewed from CAMERA_DISTANCE. */
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

    private final SubScene subScene;
    private final Label tooltip = MDSScatter.createTooltip();
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
     *
     * @param points (List of MDSScatter.Point) Embedded cells to plot.
     * @param onClick (IntConsumer) Called with the index of a clicked point in points.
     */
    MDSScatter3D(List<MDSScatter.Point> points, IntConsumer onClick) {
        Group world = new Group();
        Rotate rotateX = new Rotate(0, Rotate.X_AXIS);
        Rotate rotateY = new Rotate(0, Rotate.Y_AXIS);
        rotateX.angleProperty().bind(angleX);
        rotateY.angleProperty().bind(angleY);
        world.getTransforms().addAll(rotateX, rotateY);

        // --- Spheres, isotropically scaled into the cube ---
        List<double[]> coordinates = points.stream().map(MDSScatter.Point::mds3).toList();
        double[][] axes = {MDSScatter.centreAndHalfRange(coordinates, 0),
                MDSScatter.centreAndHalfRange(coordinates, 1),
                MDSScatter.centreAndHalfRange(coordinates, 2)};
        double half = Math.max(axes[0][1], Math.max(axes[1][1], axes[2][1]));
        double scale = CUBE_HALF / half;
        double radius = points.size() > MDSScatter.DENSE_POINTS ? DENSE_SPHERE_RADIUS
                : SPHERE_RADIUS;
        Map<Color, PhongMaterial> materials = new HashMap<>();
        for (int i = 0; i < points.size(); i++) {
            MDSScatter.Point point = points.get(i);
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
        subScene.setFill(MDSScatter.SURFACE);
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
                MDSScatter.showTooltip(tooltip, points.get(hit), event.getX(), event.getY(),
                        this);
            } else {
                tooltip.setVisible(false);
            }
        });
        subScene.setOnMouseExited(event -> tooltip.setVisible(false));
        subScene.setOnMouseClicked(event -> {
            int hit = pickedPoint(event.getPickResult());
            if (event.isStillSincePress() && hit >= 0) {
                onClick.accept(hit);
            }
        });
    }

    /**
     * Enlarges the selected cell's sphere so it stands out, restoring the previous one.
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
     * along it, with "MDS 1/2/3" at their positive ends. Labels counter-rotate so they always
     * face the camera.
     *
     * @param world (Group) Rotating group the axes join.
     * @param axisHalf (double[] - size (3)) Half-length of the MDS 1, 2 and 3 axes, in scene
     *        units.
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
        label.setFill(MDSScatter.INK_PRIMARY);
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
