# Array Classifier Test
from Calculators.GED_Calculator import build_GED_calculator, build_Heuristic_calculator, build_Randomwalk_GED_calculator
from Calculators.Product_GRaphs import RandomWalkCalculator
from Dataset import Dataset
from Experiment import experiment
import sys
import os
import traceback

# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.KNN.feature_KNN import Feature_KNN
from Models.SVC.GED.RandomWalk_edit import Random_walk_edit_SVC, Random_Walk_edit_accelerated
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
from Models.SVC.GED.hybrid_prototype_selector import HybridPrototype_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
import pandas as pd
import networkx as nx
from io_Manager import IO_Manager
N_JOBS =1
SPLIT=0.2
NUM_TRIALS=3
SEARCH_METHOD="random"  # "grid" or "random"
 # 10% test size alternatively 0.2 for 20%
EXPERIMENT_NAME="BIG_TEST"
GED_BOUND="IPFP"  # "UpperBound-Distance", "Mean-Distance", "LowerBound-Distance"
HEURISTIC_BOUND="Combined"  # "Vertex", "Edge", "Combined"
GED_CALC_METHOD="IPFP"  # "BIPARTITE", "ANCHOR_AWARE_GED", "IPFP"
ONLY_ESTIMATE=False
PRELOAD_CALCULATORS=True
ONLY_LOAD_CALCULATORS=False
GED_EDIT_COST="CONSTANT"  # "CONSTANT"
TEST_TRAIL=False
MULTI=False
DATASET_STR="PTC_FR"
DATASET_EDGE_LABELS="label"
DATASETS = ["MUTAG","MSRC_9","PTC_FR"]
DATASET_Labels =["label",None,"label"]
# DATASETS = ["MSRC_9"]
DATASET= None
def nonGEd_classifiers(ged_calculator: Base_Calculator, dataset: Dataset):
    return [
        Random_Classifier(),
        Blind_Classifier(),
        WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True), 
        VertexHistogram_SVC(),
        EdgeHistogram_SVC(),
        CombinedHistogram_SVC(kernel_type='precomputed'),
        # NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=dataset.get_edge_labels, get_node_labels=dataset.get_node_labels,Histogram_Type="combined")
        ]


def ged_classifiers(ged_calculator: Base_Calculator, dataset: Dataset):
    random_walk_calculator = build_Randomwalk_GED_calculator(ged_calculator=ged_calculator)
    return [
        GED_KNN(ged_calculator=ged_calculator, ged_bound=GED_BOUND, n_neighbors=10, weights='distance', algorithm='auto'),
        Feature_KNN(vector_feature_list=["VertexHistogram","density","Prototype-Distance"], dataset_name=dataset.name, prototype_size=5, selection_split="all", selection_method="TPS", metric="minkowski", ged_calculator=ged_calculator, ged_bound=GED_BOUND, n_neighbors=5, weights='uniform', algorithm='auto', node_label_tag=dataset.Node_label_name, edge_label_tag=dataset.Edge_label_name),
        Base_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced'),
        Trivial_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1'),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound=GED_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=8, selection_method="k-CPS", selection_split="all",dataset_name=dataset.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=1, aggregation_method="sum",dataset_name=dataset.name,selection_method="k-CPS"),
        HybridPrototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=5, selection_method="TPS", selection_split="all",dataset_name=dataset.name, vector_feature_list=["VertexHistogram","density"],node_label_tag=dataset.Node_label_name, edge_label_tag=dataset.Edge_label_name),
        Random_Walk_edit_accelerated(ged_calculator=ged_calculator, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1,random_walk_calculator=random_walk_calculator, C=1.0,kernel_type="precomputed", class_weight='balanced')
        ]
# def get_Random_walk_edit_SVC(ged_calculator: Base_Calculator, dataset: Dataset):
#     random_walk_calculator = RandomWalkCalculator(ged_calculator=ged_calculator, llambda_samples=[0.005,0.01,0.03,0.05,0.1,0.2,0.45], dataset=dataset)
#     return [Random_Walk_edit_accelerated(ged_calculator=ged_calculator, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1,random_walk_calculator=random_walk_calculator, C=1.0,kernel_type="precomputed", class_weight='balanced')]
def reference_classifiers(ged_calculator: Base_Calculator, dataset: Dataset):
    return [
        GED_KNN(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, n_neighbors=7, weights='uniform', algorithm='auto'),
        # Feature_KNN(vector_feature_list=["VertexHistogram","density","Prototype-Distance"], dataset_name=dataset.name, prototype_size=5, selection_split="all", selection_method="TPS", metric="minkowski", ged_calculator=ged_calculator, ged_bound=GED_BOUND, n_neighbors=5, weights='uniform', algorithm='auto', node_label_tag=dataset.Node_label_name, edge_label_tag=dataset.Edge_label_name),
        Base_GED_SVC(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced'),
        Trivial_GED_SVC(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1'),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=4, selection_method="k-CPS", selection_split="all",dataset_name=dataset.name),
        ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=1, aggregation_method="sum",dataset_name=dataset.name,selection_method="k-CPS"),
        HybridPrototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=5, selection_method="TPS", selection_split="all",dataset_name=dataset.name, vector_feature_list=["VertexHistogram","density"],node_label_tag=dataset.Node_label_name, edge_label_tag=dataset.Edge_label_name)
        ]


def get_Dataset(dataset_name: str, ged_calculator, edge_labels=None):
    DATASET= Dataset(name=dataset_name, source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels="label", use_edge_labels=edge_labels,load_now=False,use_node_attributes=None, use_edge_attributes=None)
    DATASET.load()
    # DATASET.load_with_attributes(new_attributes=["x","y"], encoding_dimension=2, remove_old=True)
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
def run_classifiers(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator, testDF: pd.DataFrame,experiment_name: str="unknown"):
    for classifier in classifier_list:
        
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)
        try:
            instance_dict =expi.run_extensive_test(should_print=True, cv=int(1/SPLIT),test_DF= dict(), n_jobs=N_JOBS,get_all_tuning_results=False,search_method="random",scoring="f1_macro")
        except Exception as e:
            #  print the full traceback
            traceback.print_exc()
            print(f"Error running {classifier.get_name} on {DATASET.name}: {e}")
            instance_dict = {
                "model_name": classifier.get_name,
                "Calculator_name": ged_calculator.get_Name(),
                "Dataset": DATASET.name,
                "Error": str(e)
            }
        # add the values form instance_dict as the last row of testDF
        instance_df = pd.DataFrame([instance_dict])
        testDF = pd.concat([testDF, instance_df], ignore_index=True)
        save_progress(testDF, experiment_name+"_inter", DATASET.name)
        del classifier
        del expi
    return testDF
def run_classifiers_new(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator, testDF: pd.DataFrame,experiment_name: str="unknown", search_method: str="grid"):
    for classifier in classifier_list:
        
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)
        cv= int(1/SPLIT)
        instance_dict = dict()
        try:
            instance_dict =expi.run_joblib_parallel_nested_cv(outer_cv=cv,inner_cv=cv,num_trials=NUM_TRIALS,scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'], verbose=0, n_jobs=N_JOBS, search_method=search_method,should_print=True,test_trail=TEST_TRAIL,get_all_results=True)
        except Exception as e:
            #  print the full traceback
            traceback.print_exc()
            print(f"Error running {classifier.get_name} on {DATASET.name}: {e}")
            instance_dict = {
                "model_name": classifier.get_name,
                "Calculator_name": ged_calculator.get_Name(),
                "Dataset": DATASET.name,
                "Error": str(e)
            }
        # add the values form instance_dict as the last row of testDF
        instance_df = pd.DataFrame([instance_dict])
        testDF = pd.concat([testDF, instance_df], ignore_index=True)
        save_progress(testDF, experiment_name+"_inter", DATASET.name)
        del classifier
        del expi
    return testDF
    
def estimate_experiment_duration(classifier_list: list[GraphClassifier], DATASET: Dataset, ged_calculator: Base_Calculator, search_method: str="grid"):
    total_duration = pd.Timedelta(0)
    for classifier in classifier_list:
        expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                        model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)
        estimated_time = expi.estimate_nested_cv_time(cv=int(1/SPLIT),num_trials=NUM_TRIALS,search_method=search_method)
        total_duration += estimated_time
        del classifier
        del expi
    return total_duration


def run_classifier_group(get_classifiers_funct: callable, dataset_name: str, calculator_type:str= "GEDLIB_Calculator",
                          experiment_name: str=EXPERIMENT_NAME, testDF: pd.DataFrame=pd.DataFrame(),edge_labels=None):
    # if PRELOAD_CALCULATORS:
    #     ged_calculator = "GED_Calculator"
    # else:
    #     ged_calculator = lambda dataset, labels: build_GED_calculator(GED_edit_cost="CONSTANT", GED_calc_methods=[("IPFP","upper")], dataset=dataset, labels=labels, need_node_map=True)
    # DATASET, ged_calculator = get_Dataset(dataset_name, ged_calculator)
    
    
    if PRELOAD_CALCULATORS:
        ged_calculator = calculator_type
    else:
        if calculator_type == "GEDLIB_Calculator":
            ged_calculator = GEDLIB_Calculator(GED_calc_method=GED_CALC_METHOD, GED_edit_cost=GED_EDIT_COST, need_node_map=True)
        elif calculator_type == "GED_Calculator":
            ged_calculator = lambda dataset, labels: build_GED_calculator(GED_edit_cost=GED_EDIT_COST, GED_calc_methods=[(GED_CALC_METHOD,"upper")], dataset=dataset, labels=labels, need_node_map=True)
        elif calculator_type == "Base_Calculator":
            ged_calculator = Base_Calculator(GED_calc_method=GED_CALC_METHOD, GED_edit_cost=GED_EDIT_COST, need_node_map=False)
        elif calculator_type == "Dummy_Calculator":
            ged_calculator = Dummy_Calculator(GED_calc_method=GED_CALC_METHOD, GED_edit_cost=GED_EDIT_COST, need_node_map=False)
        elif calculator_type == "Heuristic_Calculator":
            ged_calculator = lambda dataset, labels: build_Heuristic_calculator(GED_edit_cost=GED_EDIT_COST, GED_calc_methods=["Vertex","Edge","Combined"], dataset=dataset, labels=labels, need_node_map=True)
        elif calculator_type == None:
            ged_calculator = None
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
    DATASET, ged_calculator = get_Dataset(dataset_name, ged_calculator, edge_labels=edge_labels)
     
    print(f"Running {calculator_type} on {DATASET.name} dataset.")
    classifier_list: list[GraphClassifier] = get_classifiers_funct(ged_calculator, DATASET)

    if ONLY_ESTIMATE:
        total_duration = estimate_experiment_duration(classifier_list, DATASET, ged_calculator, search_method=SEARCH_METHOD)
        print(f"Estimated total duration for group {get_classifiers_funct.__name__} classifiers: {total_duration}")
        return None,total_duration
    elif not ONLY_LOAD_CALCULATORS:
        print(f"Running non-GED classifiers on {DATASET.name} dataset.")
        
        testDF = run_classifiers_new(classifier_list, DATASET, ged_calculator, testDF, experiment_name, search_method=SEARCH_METHOD)

    
    return testDF,None


def run_big_test(dataset_name: str="MUTAG", preloaded: bool=True, only_estimate_duration: bool=False, only_load_calculators: bool=False, edge_labels=None):
    testDF = pd.DataFrame()
    start_time = pd.Timestamp.now()

    experiment_name = f"{dataset_name}_{1/SPLIT}_{start_time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Experiment started at {start_time}")
    # ged_calculator = GEDLIB_Calculator(GED_calc_method="BIPARTITE", GED_edit_cost="CONSTANT")
    # ged_calculator = Base_Calculator()
    # ged_calculator = "GEDLIB_Calculator"

    # test the Non GED classifiers first
    
    testDF, total_duration_nonGED = run_classifier_group(nonGEd_classifiers, dataset_name=dataset_name, calculator_type=None, experiment_name=experiment_name, testDF=testDF, edge_labels=edge_labels)
    
    
    # if preloaded:
    #     ged_calculator = "GEDLIB_Calculator"
    # else:
    #     ged_calculator = GEDLIB_Calculator(GED_calc_method=GED_CALC_METHOD, GED_edit_cost="CONSTANT",need_node_map=True)
    # DATASET, ged_calculator = get_Dataset(dataset_name, ged_calculator, edge_labels=edge_labels)
    # classifiers_without_calculator: list[GraphClassifier] = nonGEd_classifiers(DATASET)
    # print(f"Finished loading at {pd.Timestamp.now()}")


    # if only_estimate_duration:
    #     est_total_duration = pd.Timedelta(0)
    #     total_duration_nonGED = estimate_experiment_duration(classifiers_without_calculator, DATASET, ged_calculator, search_method=SEARCH_METHOD)
    #     print(f"Estimated total duration for non-GED classifiers: {total_duration_nonGED}")
    # elif not only_load_calculators:
    #     print(f"Running non-GED classifiers on {DATASET.name} dataset.")
        
    #     testDF = run_classifiers_new(classifiers_without_calculator, DATASET, ged_calculator, testDF, experiment_name, search_method=SEARCH_METHOD)
    
    
    testDF, total_duration_GED = run_classifier_group(ged_classifiers, dataset_name=dataset_name, calculator_type="GED_Calculator", experiment_name=experiment_name, testDF=testDF, edge_labels=edge_labels)
    
    
    # first round with the real GED calculator
    # classifiers_with_calculator: list[GraphClassifier] = ged_classifiers(ged_calculator, DATASET)
    # if only_estimate_duration:
    #     total_duration_GED = estimate_experiment_duration(classifiers_with_calculator, DATASET, ged_calculator, search_method=SEARCH_METHOD)
    #     est_total_duration += total_duration_GED + total_duration_nonGED
    #     print(f"Estimated total duration for GED classifiers: {total_duration_GED}")
    # elif not only_load_calculators:
    #     print(f"Running GED-based classifiers on {DATASET.name} dataset.")
    #     testDF = run_classifiers_new(classifiers_with_calculator, DATASET, ged_calculator, testDF, experiment_name, search_method=SEARCH_METHOD)
    # reference calculator for sanity check
    # testDF, total_duration_rw = run_classifier_group(get_Random_walk_edit_SVC, dataset_name=dataset_name, calculator_type="GEDLIB_Calculator", experiment_name=experiment_name, testDF=testDF, edge_labels=edge_labels)

    testDF, total_duration_reference = run_classifier_group(reference_classifiers, dataset_name=dataset_name, calculator_type="Heuristic_Calculator", experiment_name=experiment_name, testDF=testDF, edge_labels=edge_labels)
    # if preloaded:
    #     reference_calculator = "Dummy_Calculator"
    # else:
    #     reference_calculator = Dummy_Calculator(GED_calc_method=GED_CALC_METHOD, GED_edit_cost="CONSTANT",need_node_map=False)
    # DATASET, reference_calculator = get_Dataset(dataset_name, reference_calculator, edge_labels=edge_labels)
    # classifiers_with_reference_calculator: list[GraphClassifier] = reference_classifiers(reference_calculator, DATASET)

    if only_estimate_duration:
        est_total_duration = total_duration_nonGED + total_duration_GED + total_duration_reference
        print(f"Estimated total duration for all classifiers: {est_total_duration}")
        # total_duration_reference = estimate_experiment_duration(classifiers_with_reference_calculator, DATASET, reference_calculator, search_method=SEARCH_METHOD)
        # est_total_duration += total_duration_reference
        # print(f"Estimated total duration for reference GED classifiers: {total_duration_reference}")
        # print(f"Estimated total duration for all classifiers: {est_total_duration}")
        return est_total_duration
    elif not only_load_calculators:
        # print(f"Running GED-based classifiers with reference calculator on {DATASET.name} dataset.")
        # testDF = run_classifiers_new(classifiers_with_reference_calculator, DATASET, reference_calculator, testDF, experiment_name, search_method=SEARCH_METHOD)
        # # save the results to a csv file
        results_dir = os.path.join("configs", "results")
        results_path = os.path.join(results_dir, f"{experiment_name}_{dataset_name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        IO_Manager.save_prototype_selector()
        end_time = pd.Timestamp.now()
        print(f"Experiment ended at {end_time}")
        total_duration = end_time - start_time
        print(f"Total experiment duration: {total_duration}")
if __name__ == "__main__":
    if MULTI:
        total_toal_time = pd.Timedelta(0)
        for i,dataset in enumerate(DATASETS):

            total_time =run_big_test(dataset_name=dataset, preloaded=PRELOAD_CALCULATORS, only_estimate_duration=ONLY_ESTIMATE, only_load_calculators=ONLY_LOAD_CALCULATORS,edge_labels=DATASET_Labels[i])
            if total_time is not None:
                total_toal_time += total_time
        if total_time is not None:
            print(f"Estimated total duration for all datasets: {total_toal_time}")
    else:
        run_big_test(dataset_name=DATASET_STR, preloaded=PRELOAD_CALCULATORS, only_estimate_duration=ONLY_ESTIMATE, only_load_calculators=ONLY_LOAD_CALCULATORS,edge_labels=DATASET_EDGE_LABELS)





    
    
    

    

