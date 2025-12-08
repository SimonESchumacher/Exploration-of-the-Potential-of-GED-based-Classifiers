# Exploration of Graph Edit Distance based Classifiers
this is the Repository for the Bachorthesis of Simon Schumacher.

link to file. Exploration

## How to run the code
The Main files to use are the Run_Experiment_main.py
Here a couple of parameters can be defined at the TOP, and than the file can be run, and will conduct the experiment with the selected selections.

This requires already conducted GED precumputation (build of GED clalculators) for the Datasets specified.
To precompute GEDs the specific Dataset must be in the Datassets/TUD/ folder in the TUD file format.
In the script Calculators/exact_GED_Calculator.py 
Here the Dataset Name must be Specified, in the List of Datasets, and the file can be run, and will precompute the GED-values.


There is not much to say yet, other than my code is a mess

libaires used:
link to requiremnts. txt

Furthermore 

gedlibpy the python binding for GEDLIB was used 
And The Graph Edit Distance Calculation from the following Repository.
This was however modified. 


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
cd path/to/gedlibpy/directory

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

## **📂 Project Structure**
```
├── README.md # This documentation file
├── requirements.txt # Python dependencies
├── config_loader.py # Configuration file parser
├── Experiment.py # Main experiment class
├── Dataset.py # Dataset loading and preprocessing
├── Graph_Tools.py # Graph utilities and helper functions
├── io_Manager.py # Input/output management
├── Run_Experiment_main.py # Main experiment runner
├── Run_helpers.py # Helper functions for experiments
├── Timeout_handler.py # Timeout handling utilities
├── init.py # Package initialization
├── Exploration of GED-based Classifiers Simon Schumacher.pdf # Thesis document
├── The-GED-classifier-bakeoff.code-workspace # IDE workspace file
│
├── Calculators/ # Graph Edit Distance calculators
│   ├── Base_Calculator.py # Abstract base calculator class
│   ├── GED_Calculator.py # Main GED calculator interface
│   ├── exact_GED_Calculator.py # Exact GED computation using external tool
│   ├── GEDLIB_Calculator.py # GEDLIB-based approximate GED
│   ├── Random_walk_edit_Calculator.py # Random walk based GED approximation
│   ├── Dummy_Calculator.py # Placeholder/dummy calculator for testing
│   ├── Product_Graphs.py # Product graph representations
│   ├── Prototype_Selection.py # Prototype selection algorithms
│   ├── exact_GED_results_summary.txt # Summary of exact GED results
│   └── init.py
│
├── Models/ # Machine learning classifiers
│   ├── Graph_Classifier.py # Base graph classifier class
│   ├── KNN_Classifier.py # k-Nearest Neighbors classifier
│   ├── SupportVectorMachine_Classifier.py # SVM classifier base
│   ├── Blind_Classifier.py # Baseline random classifier
│   ├── Random_Classifier.py # Random prediction classifier
│   ├── init.py
│   │
│   ├── KNN/ # KNN implementations
│   │   ├── feature_KNN.py # Feature-based KNN
│   │   └── GEDLIB_KNN.py # GEDLIB-based KNN
│   │
│   └── SVC/ # Support Vector Classifier implementations
│       ├── Base_GED_SVC.py # Base GED-based SVC
│       ├── Baseline_SVC.py # Baseline SVC implementation
│       ├── WeisfeilerLehman_SVC.py # Weisfeiler-Lehman kernel SVC
│       ├── random_walk.py # Random walk kernel SVC
│       └── init.py
│
├── Custom_Kernels/ # Custom kernel implementations
│   ├── GEDLIB_kernel.py # GEDLIB-based kernel functions
│   └── init.py
│
├── Datasets/ # Graph datasets
│   ├── TUD/ # TUDataset format collections
│   │   ├── MUTAG/ # Mutagenicity dataset
│   │   ├── BZR/ # Benzodiazepine receptor dataset
│   │   ├── BZR_MD/ # BZR with additional metadata
│   │   ├── COX2_MD/ # Cyclooxygenase-2 dataset
│   │   ├── ENZYMES/ # Enzyme protein structures
│   │   ├── IMDB-BINARY/ # IMDB movie collaboration network
│   │   ├── IMDB-MULTI/ # Multi-class IMDB dataset
│   │   ├── KKI/ # KKI medical imaging dataset
│   │   ├── Letter-high/ # Letter recognition dataset
│   │   ├── MSRC_9/ # Microsoft Research Cambridge dataset
│   │   ├── PTC_FR/ # Predictive toxicology challenge
│   │   └── ... (other TUDatasets)
│   │
│   ├── ged/ # Preprocessed datasets for GED computation
│   │   ├── MUTAG_0_0/ # MUTAG with label normalization scheme 0_0
│   │   ├── MUTAG_1_1/ # MUTAG with label normalization scheme 1_1
│   │   ├── BZR_0_0/ # BZR with label normalization scheme 0_0
│   │   ├── BZR_1_1/ # BZR with label normalization scheme 1_1
│   │   └── ... (other preprocessed datasets)
│   │
│   └── Test_graphs/ # Test graph files for debugging
│       ├── G.txt
│       ├── G2.txt
│       ├── Ge1.txt
│       └── Ge2.txt
├── gedlipy Repo foked for approximate GEDs
├── Graph_Edit_Distance/ # Exact GED computation tool (C++)
│   ├── ged # Precompiled binary (Ubuntu 20.04 compatible)
│   ├── Application.cpp/.h # Main application logic
│   ├── Graph.h # Graph data structure
│   ├── Timer.h # Timing utilities
│   ├── Utility.h # Utility functions
│   ├── main.cpp # Entry point
│   ├── makefile # Build configuration
│   ├── popl.hpp # Command-line argument parser
│   ├── LICENSE.md # License information
│   ├── README.md # Original documentation
│   ├── config.yml # Configuration for documentation
│   │
│   └── datasets/ # Example datasets for testing
│         ├── AIDS.txt
│         ├── AIDS_query100.txt
│         ├── graph_g.txt
│         └── graph_q.txt
│
├── configs/ # Experiment configurations
|    └──── Config.ini # Main configuration file (INI format)
│
├── presaved_data/ # Precomputed GED matrices and calculators
│    ├── Exact_GED_.joblib # Precomputed exact GED matrices
│    ├── GED_Calculator_.joblib # Saved calculator states
│    ├── Heuristic_Calculator_.joblib # Heuristic calculator states
│    └── Randomwalk_GED_Calculator_*.joblib # Random walk calculator states
│
├── results/ # Current experiment results
│    ├── *Some_Result.xlsx # Result files in Excel format
|    ├── Hyperparameter_tuning_results/ # Hyperparameter tuning outputs
|    |     └── HP_Some_hypertuning_data.xlsx
│    └── intermediate/ # Intermediate computation files
│          └── Some_Result_inter.xlsx
├── Graphics_builders/ # Visualization tools and figures
│    ├── SVM_visualizations.ipynb # Jupyter notebook for SVM visualizations
│    ├── Kernel_Matrix.ipynb # Kernel matrix visualization
│    ├── visulaize_graphs.ipynb # Graph visualization tools
│    ├── *.pdf # Generated figures and diagrams
│    └── *.ipynb # Jupyter notebooks for analysis
│
├── tests/ # Unit and integration tests
│    ├── test_clone.py # Cloning functionality tests
│    ├── test_exact_GED.ipynb # Exact GED computation tests
│    └── Calculator_path_test.ipynb # Calculator path testing
│
└── bin/ # Binary directory (empty/utility)
```

### Key Directories Explained:

1. **`Calculators/`** - Implements different Graph Edit Distance computation methods
Here the GEDs Are precomputed.
2. **`Models/`** - Contains machine learning classifiers using GED-based kernels
3. **`Datasets/`** - Stores graph datasets in TUDataset format and preprocessed versions
4. **`Graph_Edit_Distance/`** - C++ tool for exact GED computation with precompiled binary
5. **`configs/`** - Configuration files for experiments
    - the main file of intrest here is configs.ini
6. **`presaved_data/`** - Cache of precomputed GED matrices to speed up experiments
    - Saved GED Distance matrices, for the models to use
7. **`Graphics_builders/`** - Tools for visualizing results and algorithms
8. **`tests/`** - Test suite for validation
