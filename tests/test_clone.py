from sklearn.base import clone
# add the root directory to the sys.path
import sys
import os
sys.path.append(os.getcwd())
from Models.support_vector_models.rw_SVC import rw_SVC
from Dataset_loader import Dataset_loader
ged_calculator = "Exact_GED"
DATASET= Dataset_loader(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="weight",load_now=False)
DATASET.load()
ged_calculator = DATASET.get_calculator()
estimator = rw_SVC(normalize_kernel=True, rw_kernel_type="geometric", p_steps=3,C=1.0, kernel_type="precomputed")
try:
    cloned = clone(estimator)
    print("Cloning works!")
    print("Original params:", estimator.get_params())
    print("Cloned params:", cloned.get_params())
except Exception as e:
    print("Cloning failed:", e)
    print("You need to fix get_params()")