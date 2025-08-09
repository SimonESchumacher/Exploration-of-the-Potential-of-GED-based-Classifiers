# Array Classifier Test
from Dataset import Dataset
from Experiment import experiment
import sys
import os
# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.SVC.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.Graph_Classifier import GraphClassifier
from Models.SVC.Baseline_SVC import VertexHistogram_SVC,EdgeHistogram_SVC, CombinedHistogram_SVC, NX_Histogram_SVC
from Models.Blind_Classifier import Blind_Classifier
from Models.Random_Classifer import Random_Classifier
from Models.KNN_Classifer import KNN
from Models.SVC.GED.Trivial_GED_SVC import Trivial_GED_SVC
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Calculators.Dummy_Calculator import Dummy_Calculator
from Calculators.Base_Calculator import Base_Calculator
from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
from Models.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
import pandas as pd

if __name__ == "__main__":
    # ged_calculator = GEDLIB_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    ged_calculator = GEDLIB_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    # ged_calculator = Base_Calculator()
    DATASET= Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="label",load_now=False)
    DATASET.load()
    classifiers: list[GraphClassifier] = [
        WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True), 
        VertexHistogram_SVC(),
        EdgeHistogram_SVC(kernel_type='precomputed'),
        CombinedHistogram_SVC(kernel_type='precomputed'),
        NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=DATASET.get_edge_labels, get_node_labels=DATASET.get_node_labels,Histogram_Type="combined"),
        Blind_Classifier(),
        Random_Classifier(),
        GED_KNN(ged_calculator=ged_calculator, comparison_method="Mean-Distance", n_neighbors=1, weights='uniform', algorithm='auto'),
        Base_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0, kernel_type="precomputed", class_weight='balanced'),
        DIFFUSION_GED_SVC(kernel_type='precomputed',C=1.0, KERNEL_llambda=1.0, ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_diffusion_function="exp_diff_kernel", class_weight='balanced'),
        Trivial_GED_SVC(kernel_type='precomputed',ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1")
        ]
    for classifier in classifiers:
        print(f"Running experiment for {classifier.__class__.__name__}")
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,model=classifier,model_name=classifier.get_name)
        # accuracy, report = expi.run_simple()
        accuracy, report = expi.run_kfold(k=5)
        print(classifier.get_name)
        # results ,best_model, best_params = expi.run_hyperparameter_tuning(tuning_method='grid', scoring='f1', cv=5, verbose=1, n_jobs=1)
