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