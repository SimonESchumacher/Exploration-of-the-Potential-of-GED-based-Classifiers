from Dataset import Dataset
from Experiment import experiment
import sys
import os
# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.Graph_Classifier import GraphClassifier
from Models.Baseline_SVC import VertrexHistogram_SVC,EdgeHistogram_SVC, CombinedHistogram_SVC
# from Models.GED_SVC import GED_SVC as GED_SVC
# from Models.GED_gkl_SVC import GED_SVC as GED_gkl_SVC
from Models.Blind_Classifier import Blind_Classifier 
# from Custom_Kernels.GED_kernel import GEDKernel
from Models.KNN_Classifer import KNN
from Models.GEDLIB_SVC import GED_SVC
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Custom_Kernels.Trivial_GED_Kernel import Trivial_GED_Kernel
from Dummy_Calculator import Dummy_Calculator
from Base_Calculator import Base_Calculator
# from GEDLIB_Caclulator import GEDLIB_Calculator
# from Models.GED_KNN import GED_KNN
import pandas as pd

# classifier = GED_SVC(gamma=1.0, method='BIPARTITE', normalize_ged=True, similarity=True, C=1.0)
if __name__ == "__main__":
    ged_calculator = Dummy_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    # ged_calculator = None
    DATASET= Dataset(name="AIDS", source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="label",load_now=False)
    DATASET.load()
    Kernel = Trivial_GED_Kernel(ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1")
    # Kernel = GEDKernel(ged_calculator=ged_calculator, comparison_method="Mean-Similarity")
    classifier = GED_SVC(kernel=Kernel, kernel_name="GEDLIB", C=1.0)

    # Kernel = GEDKernel(method="BRANCH", edit_cost="CONSTANT",comparison_method="Mean-Similarity")
    # classifier = GED_SVC(kernel=Kernel, kernel_name="GEDLIB", C=1.0)
    # ged_calculator = classifier.get_calculator()
    # Define the Dataset

    # dataset = Dataset(name='ENZYMES', source='TUD', domain='Bioinformatics',)

    # define the model

    # classifier = WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True) # You can tune n_iter and normalize
    # classifier = CombinedHistogram_SVC(kernel_type="rbf",C=1.0)
    # classifier = GED_SVC(gamma=1.0, node_del_cost=1.0, node_ins_cost=1.0, edge_del_cost=1.0, edge_ins_cost=1.0, approximation=None, kernel_type="rbf", C=0.1)
    # classifier = Blind_Classifier()
    # classifier = Random_Classifier(random_state=42, strategy='uniform', constant=None)
    # classifier = GED_KNN(approximation=None, node_del_cost=1.0, node_ins_cost=1.0, edge_del_cost=1.0, edge_ins_cost=1.0, n_neighbors=5, weights='uniform', algorithm='auto')



    # define experiment 
    exp_instance = experiment(f"Fucntionality_test_{DATASET.name}_with_{classifier.get_name}",
                            dataset=DATASET,
                                datset_name=DATASET.name,
                                model=classifier,
                                model_name=classifier.get_name)
    accuracy, report = exp_instance.run_simple()
# accuracy, report = exp_instance.run_kfold()
# print(report)

# param_grid = classifier.get_param_grid()
# param_grid


# results ,best_model, best_params = exp_instance.run_hyperparameter_tuning(tuning_method='grid', scoring='f1', cv=5, verbose=1, n_jobs=1)