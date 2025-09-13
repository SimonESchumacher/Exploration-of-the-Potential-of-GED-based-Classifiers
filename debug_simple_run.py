from Dataset import Dataset
from Experiment import experiment
import sys
import os
# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.SVC.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.Graph_Classifier import GraphClassifier
from Models.SVC.Baseline_SVC import VertexHistogram_SVC,EdgeHistogram_SVC, CombinedHistogram_SVC,NX_Histogram_SVC
# from Models.GED_SVC import GED_SVC as GED_SVC
# from Models.GED_gkl_SVC import GED_SVC as GED_gkl_SVC
from Models.Blind_Classifier import Blind_Classifier 
# from Custom_Kernels.GED_kernel import GEDKernelq
from Models.KNN_Classifer import KNN
from Models.SVC.GED.Trivial_GED_SVC import Trivial_GED_SVC
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Calculators.Dummy_Calculator import Dummy_Calculator,Dummy_Calculator2D
from Calculators.NetworkX_GED_Calculator import NetworkXGEDCalculator
from Calculators.Base_Calculator import Base_Calculator
from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
from Models.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.SVC.GED.Zero_GED_SVC import ZERO_GED_SVC
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
# from Models.SVC.GED.RandomWalk_edit import Random_walk_edit_SVC
# import os
print("Current Working Directory:", os.getcwd())
# from Models.GED_KNN import GED_KNN
import pandas as pd

# classifier = GED_SVC(gamma=1.0, method='BIPARTITE', normalize_ged=True, similarity=True, C=1.0)
if __name__ == "__main__":
    ged_calculator = GEDLIB_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    # ged_calculator = None
    # ged_calculator = "GEDLIB_Calculator"
    DATASET= Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="weight",load_now=False)
    DATASET.load()
    ged_calculator = DATASET.get_calculator()
    # classifier = ZERO_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="precomputed", KERNEL_classwise=False,KERNEL_I_size=8, KERNEL_aggregation_method="sum")
    # classifier = Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',I_size=5, selection_method="random")
    # classifier = Random_walk_edit_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_decay_lambda=0.1, KERNEL_max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
    # classifier = Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',KERNEL_prototype_size=5, KERNEL_selection_method="RPS", KERNEL_classwise=False, KERNEL_single_class=False)

    # classifier = Base_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced')
    # classifier = DIFFUSION_GED_SVC(C=1.0, KERNEL_llambda=1.0, ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_diffusion_function="exp_diff_kernel", class_weight='balanced')
    classifier = GED_KNN(ged_calculator=ged_calculator, comparison_method="Mean-Distance", n_neighbors=5, weights='uniform', algorithm='auto')    
    # Kernel = Trivial_GED_Kernel(ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1")
    # Kernel = GEDKernel(ged_calculator=ged_calculator, comparison_method="Mean-Similarity")
    # classifier = GED_SVC(kernel=Kernel, kernel_name="GEDLIB", class_weight='balanced', C=1.0)
    # classifier = NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=DATASET.get_edge_labels, get_node_labels=DATASET.get_node_labels,Histogram_Type="combined")
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
                                dataset_name=DATASET.name,
                                model=classifier,
                                model_name=classifier.get_name
                                ,ged_calculator=ged_calculator)
    start_time = pd.Timestamp.now()
    accuracy, report = exp_instance.run_simple()
    runtime = pd.Timestamp.now() - start_time
    print(f"Experiment completed in {runtime}")
# accuracy, report = exp_instance.run_kfold()
# print(report)

    param_grid = classifier.get_param_grid()
    param_grid
    print()
    print(param_grid)
    print()
    print("Starting hyperparameter tuning...")

    results ,best_model, best_params = exp_instance.run_hyperparameter_tuning(tuning_method='grid', scoring='accuracy', cv=5, verbose=1, n_jobs=1)