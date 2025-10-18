import sys
import os 
# append the dircetotry this gets caled from to the sys.path
sys.path.append(os.getcwd())
import subprocess
from asyncio.log import logger
import re
from tempfile import NamedTemporaryFile
from Calculators.GED_Calculator import GED_Calculator
from torch_geometric.utils import from_networkx
import os
from Dataset import Dataset
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed







DATASET = Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels="label", use_edge_labels="label",load_now=False)
DATASET.load()
calculate_exact_ged_distance_matrix(DATASET, n_jobs=8)


