/*
 * QuPath entry point for the Persistent Homology Convolutions (PHC) extension.
 *
 * Contents
 * --------
 * PHCExtension : class
 *     Adds the Extensions > PHC menu and the PHC preferences to QuPath.
 */

package qupath.ext.phc;

import javafx.beans.property.StringProperty;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;

import org.controlsfx.control.PropertySheet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Registers PHC with QuPath: a menu to run PHC on the selected annotation (tiled or per-cell
 * windows), show its MDS plot and clear its results, plus preferences for the Python
 * environment that holds the PHC library.
 */
public class PHCExtension implements QuPathExtension {

    private static final Logger logger = LoggerFactory.getLogger(PHCExtension.class);

    private static final String PREFERENCE_CATEGORY = "PHC";
    private static final String DEFAULT_PYTHON = "python3";

    /** Python executable with gudhi, scikit-learn, opencv and joblib installed. */
    private static final StringProperty PYTHON_PATH =
            PathPrefs.createPersistentPreference("phc.pythonPath", DEFAULT_PYTHON);

    /** Folder that contains the PHC package; blank if PHC is installed in that Python. */
    private static final StringProperty PHC_LIBRARY_DIR =
            PathPrefs.createPersistentPreference("phc.libraryDir", "");

    private boolean isInstalled = false;

    /**
     * Adds the PHC menu items and preferences; QuPath calls this once at start-up.
     *
     * @param qupath (QuPathGUI) Running QuPath instance.
     * @return (void)
     */
    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled) {
            return;
        }
        isInstalled = true;

        PropertySheet.Item pythonItem = new PropertyItemBuilder<>(PYTHON_PATH, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.FILE)
                .name("Python executable")
                .category(PREFERENCE_CATEGORY)
                .description("Full path to a Python with gudhi, scikit-learn, opencv and "
                        + "joblib, e.g. ~/miniconda3/envs/phc/bin/python")
                .build();
        PropertySheet.Item libraryItem = new PropertyItemBuilder<>(PHC_LIBRARY_DIR, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.DIRECTORY)
                .name("PHC library folder")
                .category(PREFERENCE_CATEGORY)
                .description("Folder that contains the PHC package (leave blank if it is "
                        + "pip-installed)")
                .build();
        qupath.getPreferencePane().getPropertySheet().getItems().addAll(pythonItem, libraryItem);

        PHCCommand command = new PHCCommand(qupath, PYTHON_PATH, PHC_LIBRARY_DIR);
        MenuItem runItem = new MenuItem("Run PHC on selected annotation...");
        runItem.setOnAction(event -> command.run());
        MenuItem clearItem = new MenuItem("Clear PHC heatmap from selected annotation");
        clearItem.setOnAction(event -> command.clear());
        MenuItem mdsItem = new MenuItem("Show MDS plot");
        mdsItem.setOnAction(event -> command.showMds());
        Menu menu = qupath.getMenu("Extensions>PHC", true);
        menu.getItems().addAll(runItem, mdsItem, clearItem);
        logger.info("PHC extension installed (Extensions > PHC)");
    }

    /**
     * Gives the name shown in QuPath's extension manager.
     *
     * @return (String) Extension name.
     */
    @Override
    public String getName() {
        return "PHC extension";
    }

    /**
     * Gives the description shown in QuPath's extension manager.
     *
     * @return (String) One-sentence description.
     */
    @Override
    public String getDescription() {
        return "Alpha Persistent Homology Convolutions on the centroids of detected cells in "
                + "an annotation, in one window centred on each cell, shown as an L2 / "
                + "agglomerative clustering of the cells and an MDS plot.";
    }

    /**
     * Declares the QuPath version this extension was built against.
     *
     * @return (Version) QuPath 0.7.0.
     */
    @Override
    public Version getQuPathVersion() {
        return Version.parse("0.7.0");
    }
}
