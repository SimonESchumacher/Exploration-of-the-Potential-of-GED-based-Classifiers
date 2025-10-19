import sys
import os
# add the root directory to the sys.path
sys.path.append(os.getcwd())
from Dataset import Dataset
from Calculators.GED_Calculator import build_exact_ged_calculator

DATASET = Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels="label", use_edge_labels="label",load_now=False)
DATASET.load()
build_exact_ged_calculator(DATASET.get_graphs(),DATASET.name, n_jobs=8)






