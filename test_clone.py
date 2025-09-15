from sklearn.base import clone

from Models_single.SVC.Base_GED_SVC import Base_GED_SVC
from Models_single.SVC.GED.Trivial_GED_SVC import Trivial_GED_SVC
from Models_single.SVC.GED.RandomWalk_edit import Random_walk_edit_SVC
from Models_single.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC
from Dataset import Dataset
ged_calculator = "GEDLIB_Calculator"
DATASET= Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="weight",load_now=False)
DATASET.load()
ged_calculator = DATASET.get_calculator()
estimator = DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound="Mean-Distance", diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5)
try:
    cloned = clone(estimator)
    print("Cloning works!")
    print("Original params:", estimator.get_params())
    print("Cloned params:", cloned.get_params())
except Exception as e:
    print("Cloning failed:", e)
    print("You need to fix get_params()")