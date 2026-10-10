#!/bin/bash
# Builds the PHC QuPath extension jar by compiling against the jars of an installed QuPath.
#
# Usage
# -----
#   ./build.sh          build the extension jar
#   ./build.sh test     also compile and run the headless pipeline check (synthetic cells,
#                       per-cell windows, plain and Delaunay-constrained clustering)
#   ./build.sh guitest  also run the on-screen progress dialog and MDS viewer check (opens windows)
#
# Arguments (environment variables)
# ---------------------------------
#   JAVA_HOME    JDK 25 or newer, default ~/.local/jdk-25.0.4.1+1/Contents/Home
#   QUPATH_APP   QuPath app bundle, default /Applications/QuPath-0.7.0-arm64.app
#   PHC_PYTHON   Python used by the test, default the python3 on PATH
#   PHC_LIBRARY  Folder holding the PHC package, used by the test, default the folder above
#                this one (the root of the PHC repository)
#
# Outputs
# -------
#   build/libs/qupath-extension-phc-<VERSION>.jar   drop this onto QuPath to install it
#   build/test/blank_slide.png                      blank image the synthetic cells sit on
#   build/test/PHC plots/synthetic_ring_random_*    matplotlib MDS plots and CSV from Python
#                                                   (*_cells_delaunay.png when spatial)
#   build/test/progress_dialog.png                  mid-run screenshot from the dialog check
#   build/test/mds_cells_2d_view.png, mds_cells_3d_view.png   MDS viewer snapshots from the
#                                                   dialog check
#   exit status 0 on success, non-zero if compilation or the test fails

set -euo pipefail

VERSION="0.6.2"
JAVA_RELEASE=25   # QuPath 0.7 runs on Java 25
JAVA_HOME="${JAVA_HOME:-$HOME/.local/jdk-25.0.4.1+1/Contents/Home}"
QUPATH_APP="${QUPATH_APP:-/Applications/QuPath-0.7.0-arm64.app}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PHC_PYTHON="${PHC_PYTHON:-$(command -v python3)}"
PHC_LIBRARY="${PHC_LIBRARY:-$(dirname "$ROOT")}"
BUILD="$ROOT/build"
JAR="$BUILD/libs/qupath-extension-phc-$VERSION.jar"
QUPATH_CP="$(ls "$QUPATH_APP"/Contents/app/*.jar | tr '\n' ':')"

rm -rf "$BUILD"
mkdir -p "$BUILD/classes" "$BUILD/libs"

"$JAVA_HOME/bin/javac" --release "$JAVA_RELEASE" -Xlint:all,-processing,-path -cp "$QUPATH_CP" \
    -d "$BUILD/classes" $(find "$ROOT/src/main/java" -name '*.java')
cp -R "$ROOT/src/main/resources/." "$BUILD/classes/"
# QuPath reads the extension version from Implementation-Version
printf 'Implementation-Title: qupath-extension-phc\nImplementation-Version: %s\n' "$VERSION" \
    > "$BUILD/MANIFEST.MF"
"$JAVA_HOME/bin/jar" --create --file "$JAR" --manifest "$BUILD/MANIFEST.MF" -C "$BUILD/classes" .
echo "Built $JAR"

if [[ "${1:-}" == "test" || "${1:-}" == "guitest" ]]; then
    mkdir -p "$BUILD/test-classes" "$BUILD/test"
    # the checks write their own blank slide and synthetic cell detections into build/test
    "$JAVA_HOME/bin/javac" --release "$JAVA_RELEASE" -cp "$QUPATH_CP:$JAR" \
        -d "$BUILD/test-classes" $(find "$ROOT/src/test/java" -name '*.java')
    "$JAVA_HOME/bin/java" -Djava.awt.headless=true \
        -cp "$QUPATH_CP:$JAR:$BUILD/test-classes" qupath.ext.phc.PHCPipelineCheck \
        "$PHC_PYTHON" "$PHC_LIBRARY" "$BUILD/test"
    if [[ "${1:-}" == "guitest" ]]; then
        "$JAVA_HOME/bin/java" --enable-native-access=ALL-UNNAMED \
            -cp "$QUPATH_CP:$JAR:$BUILD/test-classes" qupath.ext.phc.PHCTaskDialogCheck \
            "$PHC_PYTHON" "$PHC_LIBRARY" "$BUILD/test"
    fi
fi
