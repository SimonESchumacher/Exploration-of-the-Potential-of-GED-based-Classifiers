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
[GEDLIBPY_PLACEHOLDER]

### 4. Exact GED Computation Setup
[EXACT_GED_PLACEHOLDER]