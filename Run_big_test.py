# Array Classifier Test
from Dataset import Dataset
from Experiment import experiment
import sys
import os
# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.SVC.GED.RandomWalk_edit import Random_walk_edit_SVC
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
from Models.SVC.GED.Zero_GED_SVC import ZERO_GED_SVC
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
import pandas as pd

N_JOBS =1

def nonGEd_classifiers():
    return [
        # WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True), 
        # VertexHistogram_SVC(),
        # EdgeHistogram_SVC(kernel_type='precomputed'),
        CombinedHistogram_SVC(kernel_type='precomputed'),
        # NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=DATASET.get_edge_labels, get_node_labels=DATASET.get_node_labels,Histogram_Type="combined"),
        Blind_Classifier(),
        Random_Classifier(),
        ]


def ged_classifiers(ged_calculator: Base_Calculator):
    return [
        # GED_KNN(ged_calculator=ged_calculator, comparison_method="Mean-Distance", n_neighbors=7, weights='uniform', algorithm='auto'),
        # Base_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0, kernel_type="precomputed", class_weight='balanced'),
        # DIFFUSION_GED_SVC(kernel_type='precomputed',C=1.0, KERNEL_llambda=1.0, ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_diffusion_function="exp_diff_kernel", class_weight='balanced'),
        # Trivial_GED_SVC(kernel_type='precomputed',ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1"),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',KERNEL_prototype_size=8, KERNEL_selection_method="k-CPS", KERNEL_classwise=False, KERNEL_single_class=False,dataset_name=DATASET.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="precomputed", KERNEL_classwise=False,KERNEL_I_size=8, KERNEL_aggregation_method="sum",dataset_name=DATASET.name),
        Random_walk_edit_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_decay_lambda=0.1, KERNEL_max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
        ]
def reference_classifiers(ged_calculator: Base_Calculator):
    return [
        # GED_KNN(ged_calculator=ged_calculator, comparison_method="Mean-Distance", n_neighbors=7, weights='uniform', algorithm='auto'),
        Base_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0, kernel_type="precomputed", class_weight='balanced'),
        # DIFFUSION_GED_SVC(kernel_type='precomputed',C=1.0, KERNEL_llambda=1.0, ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", KERNEL_diffusion_function="exp_diff_kernel", class_weight='balanced'),
        Trivial_GED_SVC(kernel_type='precomputed',ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1"),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',KERNEL_prototype_size=8, KERNEL_selection_method="SPS", KERNEL_classwise=False, KERNEL_single_class=False,dataset_name=DATASET.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, KERNEL_comparison_method="Mean-Distance", C=1.0,kernel_type="precomputed", KERNEL_classwise=False,KERNEL_I_size=8, KERNEL_aggregation_method="sum",dataset_name=DATASET.name),
        ]


def get_Dataset(dataset_name: str, ged_calculator):
    DATASET= Dataset(name=dataset_name, source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="label",load_now=False)
    DATASET.load()
    return DATASET, DATASET.get_calculator()
# run a list of classifiers on a dataset and return the results in a dataframe
def run_classifiers(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator, testDF: pd.DataFrame):
    for classifier in classifier_list:
        
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)

        instance_dict =expi.run_extensive_test(should_print=True, cv=5,test_DF= dict(), n_jobs=N_JOBS)
        # add the values form instance_dict as the last row of testDF
        instance_df = pd.DataFrame([instance_dict])
        testDF = pd.concat([testDF, instance_df], ignore_index=True)
        del classifier
        del expi
    return testDF
def estimate_experiment_duration(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator):
    total_duration = pd.Timedelta(0)
    for classifier in classifier_list:
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=None)
        estimated_time = expi.get_estimated_tuning_time(cv=5)
        print(f"Estimated time for {classifier.get_name}: {estimated_time}")
        total_duration += estimated_time
        del classifier
        del expi
    return total_duration



if __name__ == "__main__":
    testDF = pd.DataFrame()
    experiment_name = "initial_test_run"
    dataset_name = "MUTAG"
    only_estimate_duration = False
    preloaded=True
    start_time = pd.Timestamp.now()
    print(f"Experiment started at {start_time}")
    # ged_calculator = GEDLIB_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    # ged_calculator = Base_Calculator()
    # ged_calculator = "GEDLIB_Calculator"

    # test the Non GED classifiers first
    if preloaded:
        ged_calculator = "GEDLIB_Calculator"
    else:
        ged_calculator = GEDLIB_Calculator(GED_calc_method="BRANCH", GED_edit_cost="CONSTANT",need_node_map=True)

    DATASET, ged_calculator = get_Dataset(dataset_name, ged_calculator)
    classifiers_without_calculator: list[GraphClassifier] = nonGEd_classifiers()

    if only_estimate_duration:
        est_total_duration = pd.Timedelta(0)
        total_duration_nonGED = estimate_experiment_duration(classifiers_without_calculator, DATASET, ged_calculator)
        print(f"Estimated total duration for non-GED classifiers: {total_duration_nonGED}")
    else:
        print(f"Running non-GED classifiers on {DATASET.name} dataset.")
        testDF = run_classifiers(classifiers_without_calculator, DATASET, ged_calculator, testDF)

    # first round with the real GED calculator
    classifiers_with_calculator: list[GraphClassifier] = ged_classifiers(ged_calculator)
    if only_estimate_duration:
        total_duration_GED = estimate_experiment_duration(classifiers_with_calculator, DATASET, ged_calculator)
        est_total_duration += total_duration_GED + total_duration_nonGED
        print(f"Estimated total duration for GED classifiers: {total_duration_GED}")
    else:
        print(f"Running GED-based classifiers on {DATASET.name} dataset.")
        testDF = run_classifiers(classifiers_with_calculator, DATASET, ged_calculator, testDF)
    # reference calculator for sanity check
    if preloaded:
        reference_calculator = "Dummy_Calculator"
    else:
        reference_calculator = Dummy_Calculator(GED_calc_method="BRANCH", GED_edit_cost="CONSTANT",need_node_map=False)
    DATASET, reference_calculator = get_Dataset(dataset_name, reference_calculator)
    classifiers_with_reference_calculator: list[GraphClassifier] = reference_classifiers(reference_calculator)


    if only_estimate_duration:
        total_duration_reference = estimate_experiment_duration(classifiers_with_reference_calculator, DATASET, reference_calculator)
        est_total_duration += total_duration_reference
        print(f"Estimated total duration for reference GED classifiers: {total_duration_reference}")
        print(f"Estimated total duration for all classifiers: {est_total_duration}")
    else:
        print(f"Running GED-based classifiers with reference calculator on {DATASET.name} dataset.")
        testDF = run_classifiers(classifiers_with_reference_calculator, DATASET, reference_calculator, testDF)
        # save the results to a csv file
        results_dir = os.path.join("configs", "results")
        results_path = os.path.join(results_dir, f"{experiment_name}_{DATASET.name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        end_time = pd.Timestamp.now()
        print(f"Experiment ended at {end_time}")
        total_duration = end_time - start_time
        print(f"Total experiment duration: {total_duration}")

    




    
    
    

    

