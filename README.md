# Classification of Histopathology Slides with Persistent Homology Convolutions
GitHub repository for Persistent Homology Convolutions (PHC). This method computes localized
persistent homology on greyscale image data, and on point clouds of detected cells.
Check out the corresponding paper at: https://arxiv.org/abs/2507.14378

![alt text](https://github.com/Shrunalp/PHC/blob/main/PHC_visual.png?raw=true#center)

## Abstract
Convolutional neural networks (CNNs) are a standard tool for computer vision tasks such as image classification. However, typical model architectures may result in the loss of topological information. In specific domains such as histopathology, topology is an important descriptor that can be used to distinguish between disease-indicating tissue by analyzing the shape characteristics of cells. Current literature suggests that reintroducing topological information using persistent homology can improve medical diagnostics; however, previous methods utilize global topological summaries which do not contain information about the locality of topological features. To address this gap, we present a novel method that generates local persistent homology-based data using a modified version of the convolution operator called Persistent Homology Convolutions. This method captures information about the locality and translation invariance of topological features. We perform a comparative study using various representations of histopathology slides and find that models trained with persistent homology convolutions outperform conventionally trained models and are less sensitive to hyperparameters. These results indicate that persistent homology convolutions extract meaningful geometric information from the histopathology slides.

## Repository layout

```
PHC/                    the Python library (import PHC)
├── local_ph.py         PHC class: sliding-window and per-cell persistence
├── filtrations.py      alpha, lower star, adjacency and cubical filtrations
├── image_conditioning.py  preprocess class: threshold and dilate images
├── cells.py            cell centroids from QuPath GeoJSON exports
├── clustering.py       L2 measures, agglomerative clustering, MDS
├── spatial.py          clustering restricted to Delaunay neighbours
├── constrained_linkage.py  optional numba backend for spatial clustering
├── plotting.py         matplotlib figures
└── utils.py            shared helpers
qupath-extension-phc/   QuPath extension that runs PHC on detected cells
└── dist/             ready-built extension jar (drag onto QuPath)
experiments/            scripts used for the paper (data generation, training)
PHC_Tutorial.ipynb      tutorial notebook
requirements.txt        pinned dependencies of the library, the extension and the tutorial
```

## Installation

PHC requires **Python 3.10 or greater**.

1. Clone the GitHub repository:
   ```
   git clone https://github.com/Shrunalp/PHC.git
   cd PHC
   ```
   or using the GitHub CLI: `gh repo clone Shrunalp/PHC`.

2. Create a Python environment, either with conda
   ```
   conda create -n phc python=3.11
   conda activate phc
   ```
   or with venv
   ```
   python3 -m venv phc-venv
   source phc-venv/bin/activate        # Windows: phc-venv\Scripts\activate
   ```

3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
   `numba` (fast spatially constrained clustering) and `matplotlib` (plots, tutorial) are
   optional; the library works without them. To run the tutorial, also install Jupyter
   (`pip install notebook`).

4. Check the installation from the repository folder:
   ```
   python -c "from PHC import PHC, preprocess; print('PHC is ready')"
   ```

`PHC` is imported from the repository folder. To use it from another folder, add the
repository to your Python path, e.g. `export PYTHONPATH=/path/to/PHC:$PYTHONPATH`.

The scripts in `experiments/` need extra packages (TensorFlow, Weights & Biases, ...):
`pip install -r experiments/requirements.txt`. Note that tensorflow-metal is only required for
M1 (or greater) MacBook users.

## Quick start

```python
import numpy as np
from PHC import PHC, preprocess

img = np.load("my_greyscale_images.npy")[0]            # (height, width) greyscale image

conditioning = preprocess(thresh=230, iterate=2)        # remove background, thicken cells
prepped_image = conditioning.dilate(conditioning.threshold(img))

localhom = PHC(persistence_type="cubical_complex", window_size=128, stride=128)
windows = localhom.convolve(prepped_image)              # one persistence image per window
```

`persistence_type` is one of `"lower_star"`, `"ext_lower_star"`, `"alpha"`, `"ext_alpha"`,
`"adj_complex"`, `"ext_adj_complex"` or `"cubical_complex"`; `vectorization` is `"PI"`
(persistence image) or `"PL"` (persistence silhouette). For cell centroids, use
`PHC(persistence_type="alpha", window_size=...)` with `convolve_points` (tiled windows) or
`convolve_cells` (one window per cell).

## Tutorial

Check out our notebook `PHC_Tutorial.ipynb` to get started! It covers PHC on greyscale
images and PHC on cell centroids, including spatially constrained clustering.

## QuPath extension

`qupath-extension-phc/` adds a **PHC** menu to [QuPath](https://qupath.github.io) 0.7. It runs
PHC on the cells detected inside an annotation (alpha complex persistence of the cell
centroids), clusters the windows (or the cells), and shows the clusters as a heatmap on the
slide together with an interactive 2D / 3D MDS plot. The computation runs in the PHC Python
library installed above.

### 1. Install the extension in QuPath

1. Download the ready-built extension
   [`qupath-extension-phc-0.6.0.jar`](https://github.com/Shrunalp/PHC/releases/latest/download/qupath-extension-phc-0.6.0.jar)
   from the [latest release](https://github.com/Shrunalp/PHC/releases/latest). The same file is
   also in this repository at
   [`qupath-extension-phc/dist/`](qupath-extension-phc/dist/qupath-extension-phc-0.6.0.jar).
   It needs **QuPath 0.7** and works on macOS, Windows and Linux.
2. Drag the `.jar` file onto the QuPath window and restart QuPath. A **PHC** entry appears
   under **Extensions**.

### 2. Point QuPath to the PHC Python library

The extension runs the computation in the PHC library, so complete the
[Installation](#installation) steps above first. Then open **Edit > Preferences > PHC** and set:

- **Python executable**: the full path to the Python of the environment you installed the
  requirements into, e.g. `~/miniconda3/envs/phc/bin/python` (run `which python` with the
  environment active to find it; on Windows, `where python`). Use the full path: QuPath
  started from Finder does not see your shell's `PATH`.
- **PHC library folder**: the root of the cloned repository (the folder that contains the
  `PHC` package), e.g. `~/PHC`.

### 3. Run PHC on a slide

1. Open an image and draw or select an area annotation.
2. Detect the cells in it: **Analyze > Cell detection > Cell detection** (or any detection
   command, e.g. StarDist).
3. Choose **Extensions > PHC > Run PHC on selected annotation...**, pick the settings (window
   size and stride in µm, tiled or per-cell windows, number of clusters, ...) and press **Run**.
4. The annotation is covered with tiles coloured by cluster (or, with per-cell windows, the
   cells get PHC measurements), and the MDS viewer opens. **Extensions > PHC > Clear PHC
   heatmap** removes the results.

See [`qupath-extension-phc/README.md`](qupath-extension-phc/README.md) for every setting,
the measurements written to QuPath, the plot files and the known limits.

### Building the extension yourself (optional)

Only needed if you change the Java code. Building needs QuPath 0.7 and a **JDK 25** or newer
(for example [Eclipse Temurin](https://adoptium.net)). From the repository folder:

```
cd qupath-extension-phc
JAVA_HOME=/path/to/jdk-25 QUPATH_APP=/Applications/QuPath-0.7.0-arm64.app ./build.sh
```

This writes `qupath-extension-phc/build/libs/qupath-extension-phc-0.6.0.jar`. `QUPATH_APP` is
the QuPath installation whose jars are compiled against (on macOS, the `.app` bundle). On macOS,
`JAVA_HOME=$(/usr/libexec/java_home -v 25)` finds an installed JDK 25. Optionally,
`PHC_PYTHON=/path/to/python ./build.sh test` also runs the end-to-end check on synthetic cells.

## Authors

Shrunal Pothagoni - spothago@gmu.edu

Benjamin Schweinhart - bschwei@gmu.edu
