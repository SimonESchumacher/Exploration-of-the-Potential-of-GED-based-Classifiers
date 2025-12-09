# Exploration of Graph Edit Distance based Classifiers
this is the Repository for the Bachorthesis of Simon Schumacher.
## Installation & Setup

### 1. System Requirements

#### 1.1 Operating System
This project has been developed and tested on **Ubuntu 20.04.6 LTS (Focal Fossa)** 
The following operating systems are supported:

- **Linux**: Ubuntu 20.04 LTS or later (recommended)
- **Windows**: Windows 10/11 with Windows Subsystem for Linux 2 (WSL2) running Ubuntu 20.04 or later
- **macOS**: macOS 11 (Big Sur) or later (Intel and Apple Silicon)

The precompiled binaries included in this repository are specifically compiled for **Ubuntu 20.04 LTS** with **glibc 2.31** or compatible systems.

#### 1.2 Programming Languages & Compilers
- **Python**: Version 3.8.10 or later (tested with Python 3.8.10)
- **C++**: C++17 compatible compiler (GCC 9.4.0 or later recommended)
- **Build Tools**: CMake 3.16 or later, Make 4.2.1 or later

### 2. Python Environment Setup

#### 2.1 Install Python Dependencies
First, ensure you have Python 3.8 or later installed:

The required Python packages are listed in the `requirements.txt` file. You can install them using the following command:
```bash
pip install -r requirements.txt
```
it as aditionally recomended to use a virtual envirnment



### 3. GEDLIBPY Setup

This project uses **GEDLIBPY** (Python bindings for the GEDLIB library) for approximate Graph Edit Distance computations. GEDLIBPY is based on the original repository: [https://github.com/Ryurin/gedlibpy](https://github.com/Ryurin/gedlibpy).

#### 3.1 Using the Included Version
The repository includes a version of GEDLIBPY. To ensure compatibility with your system, you may need to recompile it:

```bash
# Navigate to the GEDLIBPY directory
cd gedlibpy/

# Recompile the Python bindings
python setup.py build_ext --inplace

# Test the compilation
python test.py
```

### 4. Exact GED Computation Setup
For exact Graph Edit Distance computations, this project uses a modified version of the Graph Edit Distance tool.

#### 4.1 Original and Modified Repositories
Original repository: https://github.com/LijunChang/Graph_Edit_Distance

Modified fork (used in this project): https://github.com/simon-forb/Graph_Edit_Distance

The modified version includes additional functionality for extended GED computations.

#### 4.2 Compiling from Source
If the precompiled binary is incompatible with your system, compile from source:

```bash
# Navigate to the Graph Edit Distance directory
cd Graph_Edit_Distance

# Clean any previous builds and compile
make clean
make
```

## Using the Repository

This repository provides a framework for benchmarking Graph Edit Distance (GED) based classifiers against baseline methods. The system computes pairwise GED values and uses them as distance metrics in various classifiers.

### Quick Start Example

To run an experiment with the MUTAG dataset:

```bash
# 1. Ensure the MUTAG dataset is in Datasets/TUD/MUTAG/
# 2. Run the main experiment script
python main_experiment_run.py
```

### Detailed Workflow
#### Step 1: Dataset Preparation

1. Place your dataset in the `Datasets/TUD/` directory in TUDataset format:
```
Datasets/TUD/DATASET_NAME/
├── DATASET_NAME_A.txt          # Graph adjacency lists
├── DATASET_NAME_graph_indicator.txt  # Node-to-graph mapping
├── DATASET_NAME_graph_labels.txt     # Graph class labels
├── DATASET_NAME_node_labels.txt      # Node labels (if available)
└── DATASET_NAME_edge_labels.txt      # Edge labels (if available)
```
#### 2. GED Precomputation
Before running experiments with exact GED calculations, you must precompute the pairwise distance matrix:

Edit exact_GED_builder.py:

Set the `datasets_to_compute` list to include your dataset name

Example: `datasets_to_compute = ['MUTAG', 'BZR_MD']`

Run the precomputation:

```bash
python Calculators/exact_GED_builder.py
```
The results are saved as `.joblib` files in `presaved_data/` (e.g., `Exact_GED_MUTAG_0_0.joblib`)

#### 3.Configuration
All experiment parameters are configured in `confgis/Config.ini`

#### 4. Running Experimetns
- Single dataset experiment:
    - Set `Datasets_to_run = 'MUTAG'` in `main_experiment_run.py`
    - Run: `python main_experiment_run.py`

- Multiple dataset experiment:
    - Run: `python main_experiment_run.py`
    - Specify the Datasets, either in the Arguments, or provide them as a system input.



#### Step 5: Understanding Results
The system generates two types of output files:

1. Final Results (`results/` directory):
    - Excel files with with auto gernated name or spcifically specified
    - Contains evaluation metrics (accuracy, precision, recall, F1-score)
    - Includes computation times and model parameters

2. Intermediate Results (`results/intermediate/` directory):
    - Automatically saved during long-running experiments
    - Prevents data loss if the experiment is interrupted
    - same contetent as final results
3. Hyperparameter Tuning Results (`results/Hyperparameter_tuning_results/`):
    - Detailed logs of hyperparameter search
    - Performance for each parameter combination

#### Sept 6 Methodlogy

- Find that in the Attached Bacelor Thesis `Exploration of GED-based Classifers Simon Schumacher.pdf`
- Chapter Experimental Design.




## **📂 Project Structure (current)**
```
The-GED-classifier-bakeoff/
├── README.md
├── requirements.txt
├── config_loader.py
├── Experiment.py
├── Dataset_loader.py
├── Graph_Tools.py
├── main_experiment_run.py
├── Run_helpers.py
├── Exploration of GED-based Classifiers Simon Schumacher.pdf
├── The-GED-classifier-bakeoff.code-workspace
├── Calculators/
│   ├── __init__.py
│   ├── Base_Calculator.py
│   ├── GED_Calculator.py
│   ├── exact_GED_builder.py
│   ├── GEDLIB_Caclulator.py
│   ├── Product_GRaphs.py
│   ├── Prototype_Selction.py
│   ├── Dummy_Calculator.py
│   └── Random_walk_edit _Calculator.py
├── Models/
│   ├── __init__.py
│   ├── graph_classifier.py
│   ├── SVC.py
│   ├── blind_classifier.py
│   ├── random_classifer.py
│   ├── graph_Classifier.py
│   ├── KNN.py
│   ├── support_vector_models/
│   │   ├── GED_SVC.py
│   │   ├── baseline_SVC.py
│   │   ├── WL_ST_SVC.py
│   │   ├── rw_SVC.py
│   │   └── GED/
│   │       ├── Triv_GED_SVC.py
│   │       ├── Zero_GED_SVC.py
│   │       ├── prototype_GED_SVC.py
│   │       ├── Diff_GED_SVC.py
│   │       ├── hybrid_prototype_selector.py
│   |       └── rwe_SVC.py
│   └── k_nearest_neigbour/
│       ├── GED_KNN.py
│       └── feature_KNN.py
├── Datasets/
│   ├── ged/
│   ├── TUD/
│   └── Test_graphs/
├── gedlibpy/           # bundled GEDLIB Python bindings (sub-repo)
├── Graph_Edit_Distance/ # C++ exact GED tool and sources
├── graph_mixup/         # experimental module (not referenced by core code)
├── configs/
│   ├── Config.ini
│   └── example_config.json
├── presaved_data/
├── results/
├── Graphics_builders/
├── tests/
└── bin/
```

### Key directories and notes

1. **`Calculators/`** — Graph Edit Distance calculators and helpers. Contains both exact and approximate GED implementations and the prototype selection utilities. Precomputed calculators/matrices are stored under `presaved_data/`.

2. **`Models/`** — ML classifiers and SVC/KNN implementations. The package exposes base classes in `graph_classifier.py` and various concrete models under `Models/SVC/` and `Models/KNN/`.

3. **`Datasets/`** — TUDataset-format datasets and preprocessed GED-ready data (`ged/` holds preprocessed matrices used by GED calculators).

4. **`gedlibpy/`** — Bundled copy of the GEDLIB Python bindings (kept as a nested repository/submodule). It's registered as a submodule and used by some calculators; the working tree is included for convenience.

5. **`Graph_Edit_Distance/`** — External C++ project (fork) providing exact GED computation. Contains sources and precompiled binaries.

6. **`Custom_Kernels/`** — Optional custom kernel implementations (e.g., `GEDLIB_kernel.py`) used by some models and notebooks.

7. **`graph_mixup/`** — Experimental code and notebooks for graph mixup; currently self-contained and not referenced by the main run scripts.

8. **`configs/`** — Experiment and hyperparameter tuning configuration files. `Config.ini` is the central configuration.

9. **`presaved_data/` & `results/`** — Precomputed artifacts and output directories used by experiments. Keep backups of large `.joblib` artifacts when cleaning the repo.

10. **`tests/`** & **`Graphics_builders/`** — Notebooks and lightweight tests used for development and visualization (some rely on the experimental modules).

---

If you want, I can also:
- generate a compact tree (`tree -L 2`) and insert it verbatim, or
- add a one-line map of which modules import which major subpackages (helpful for cleanup).
Which of those would you prefer?
