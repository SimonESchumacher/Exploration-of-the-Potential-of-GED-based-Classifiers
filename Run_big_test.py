# Array Classifier Test
from Dataset import Dataset
from Experiment import experiment
import sys
import os
import traceback
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
from io_Manager import IO_Manager
N_JOBS =8

def nonGEd_classifiers():
    return [
        Random_Classifier(),
        Blind_Classifier(),
        WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True), 
        VertexHistogram_SVC(),
        EdgeHistogram_SVC(kernel_type='precomputed'),
        CombinedHistogram_SVC(kernel_type='precomputed'),
        NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=DATASET.get_edge_labels, get_node_labels=DATASET.get_node_labels,Histogram_Type="combined")
        ]


def ged_classifiers(ged_calculator: Base_Calculator):
    return [
        GED_KNN(ged_calculator=ged_calculator, ged_bound="Mean-Distance", n_neighbors=7, weights='uniform', algorithm='auto'),
        Base_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced'),
        Trivial_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1'),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound="Mean-Distance", diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=8, selection_method="k-CPS", selection_split="all",dataset_name=DATASET.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=7, aggregation_method="sum",dataset_name=DATASET.name,selection_method="k-CPS"),
        Random_walk_edit_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
        ]
def reference_classifiers(ged_calculator: Base_Calculator):
    return [
        GED_KNN(ged_calculator=ged_calculator, ged_bound="Mean-Distance", n_neighbors=7, weights='uniform', algorithm='auto'),
        Base_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced'),
        Trivial_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1'),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound="Mean-Distance", diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=4, selection_method="k-CPS", selection_split="all",dataset_name=DATASET.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=7, aggregation_method="sum",dataset_name=DATASET.name,selection_method="k-CPS"),
        ]


def get_Dataset(dataset_name: str, ged_calculator):
    DATASET= Dataset(name=dataset_name, source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels="label",load_now=False)
    DATASET.load()
    return DATASET, DATASET.get_calculator()
# run a list of classifiers on a dataset and return the results in a dataframe
last_save_time = pd.Timestamp.now()
def save_progress(testDF: pd.DataFrame, experiment_name: str, dataset_name: str):
    global last_save_time
    current_time = pd.Timestamp.now()
    if (current_time - last_save_time).seconds >= 600:  # Save every 10 minutes
        results_dir = os.path.join("configs", "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{experiment_name}_{dataset_name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        IO_Manager.save_prototype_selector()
        print(f"Progress saved at {current_time}")
        last_save_time = current_time
def run_classifiers(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator, testDF: pd.DataFrame):
    for classifier in classifier_list:
        
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)
        try:
            instance_dict =expi.run_extensive_test(should_print=True, cv=5,test_DF= dict(), n_jobs=N_JOBS)
        except Exception as e:
            #  print the full traceback
            traceback.print_exc()
            print(f"Error running {classifier.get_name()} on {DATASET.name}: {e}")
            instance_dict = {
                "Model": classifier.get_name(),
                "Dataset": DATASET.name,
                "Error": str(e)
            }
        # add the values form instance_dict as the last row of testDF
        instance_df = pd.DataFrame([instance_dict])
        testDF = pd.concat([testDF, instance_df], ignore_index=True)
        save_progress(testDF, "Run_big_test", DATASET.name)
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
    dataset_name = "AIDS"
    experiment_name = f"{dataset_name}_run"
    only_estimate_duration = True
    preloaded=False
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
    print(f"Finished loading at {pd.Timestamp.now()}")
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
        IO_Manager.save_prototype_selector()
        end_time = pd.Timestamp.now()
        print(f"Experiment ended at {end_time}")
        total_duration = end_time - start_time
        print(f"Total experiment duration: {total_duration}")

    




    
    
    

    

