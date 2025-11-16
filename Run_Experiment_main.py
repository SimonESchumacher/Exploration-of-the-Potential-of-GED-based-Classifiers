# Run Experiment 
# imports 
from Calculators.GED_Calculator import build_GED_calculator, build_Heuristic_calculator, build_Randomwalk_GED_calculator, load_exact_GED_calculator, try_load_else_build_rw_calculator
from Calculators.Product_GRaphs import RandomWalkCalculator
from Dataset import Dataset
from Experiment import experiment
import sys
import os
import traceback

from Run_helpers import end_run, save_progress, set_Mode, set_global_ged_calculator_All

# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.KNN.feature_KNN import Feature_KNN
from Models.SVC.GED.RandomWalk_edit import Random_walk_edit_SVC, Random_Walk_edit_accelerated
from Models.SVC.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.SVC.random_walk import RandomWalk_SVC
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
from Models.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC,Diffusion_GED_new
from Models.SVC.GED.Zero_GED_SVC import ZERO_GED_SVC
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Models.SVC.GED.hybrid_prototype_selector import HybridPrototype_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
import pandas as pd



# to set the mode:
TESTING_MODE= "ALL" # "SINGLE" or "ALL", "MULTI"
CALCULATOR_NAME= "Exact_GED"
HEURISTIC_CALCULATOR_NAME="Heuristic_Calculator"
N_JOBS=16

# Testing Level, indecating stages from only testing for fucntionality, to wanting a full result
# 1 only Speed Test
# 2 a test trail, with only 1 trail
# 3 test trail, but with full trials
# 4 Full Run with all tuning results saved
testing_level= 4 # Number from 1 to 4
NUM_TRIALS, TEST_TRIAL, ONLY_ESTIMATE, GET_ALL_TUNING_RESULTS = set_Mode(testing_level)

# DATASET
DATASET_NAME="MUTAG" if TESTING_MODE != "MULTI" else "MULTI"  # e.g. "MUTAG", "PTC_MR", "IMDB-MULTI", "PROTEINS", "NCI1", "NCI109", "DD", "COLLAB", "REDDIT-BINARY"
DATASET_ARRAY=["MUTAG", "PTC_FR", "KKI","BZR_MD","MSRC_9","IMDB-MULTI"]
TUNING_METRIC="f1_macro"  # e.g. "accuracy", "f1_macro", "roc_auc"
DATASET_EDGE_LABELS="label"
DATASET_NODE_LABELS="label"
DATASET_NODE_ATTRIBUTES=None  # e.g. ["x","y"]
DATASET_EDGE_ATTRIBUTES=None  # e.g. ["weight"]bb


# Not Meant to be changed
SPLIT= 0.2
SEARCH_METHOD="random"  # "grid" or "random"
GED_BOUND="Exact"   # outdated
HEURISTIC_BOUND="Vertex" 
now = pd.Timestamp.now().strftime("%d_%m_%Y_%H_%M")
EXPERIMENT_NAME=f"{now}_{DATASET_NAME}_{TUNING_METRIC}_{int(DATASET_NODE_LABELS!=None)}_{int(DATASET_EDGE_LABELS!=None)}_{TESTING_MODE}" # Day_Month_Year_Hour_Minute_TESTING_MODE


def get_Dataset(ged_calculator):
    DATASET= Dataset(name=DATASET_NAME,ged_calculator=ged_calculator, use_node_labels=DATASET_NODE_LABELS, use_edge_labels=DATASET_EDGE_LABELS,load_now=False,use_node_attributes=DATASET_NODE_ATTRIBUTES, use_edge_attributes=DATASET_EDGE_ATTRIBUTES)
    DATASET.load()
    # DATASET.load_with_attributes(new_attributes=["x","y"], encoding_dimension=2, remove_old=True)
    return DATASET, DATASET.get_calculator()

def get_single_classifier(ged_calculator):
    calculator_id = set_global_ged_calculator_All(ged_calculator)

    # return [ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=7, aggregation_method="sum",dataset_name=DATASET.name,selection_method="k-CPS")
    # return [Random_walk_edit_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
    # random_walk_calculator = RandomWalkCalculator(ged_calculator=ged_calculator, llambda_samples=[0.005,0.01,0.03,0.05,0.1,0.2,0.45,0.89], dataset=DATASET,ged_method=GED_BOUND)
    # random_walk_calculator = build_Randomwalk_GED_calculator(ged_calculator=ged_calculator)
    # return [Random_Walk_edit_accelerated(calculator_id=calculator_id, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced', random_walk_calculator_id=random_walk_calculator.get_identifier_name())]
    return[ Trivial_GED_SVC(calculator_id=calculator_id,ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1',llambda=100)]
    # return [ Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=1, selection_method="TPS", selection_split="all",dataset_name=DATASET.name)]
    # return [RandomWalk_SVC(normalize_kernel=True, rw_kernel_type="exponential", p_steps=1,C=1.0, kernel_type="precomputed")]
    # return Feature_KNN(vector_feature_list=["VertexHistogram","density","Prototype-Distance"], dataset_name=DATASET.name, prototype_size=5, selection_split="all", selection_method="TPS", metric="minkowski", calculator_id=calculator_id, ged_bound=GED_BOUND, n_neighbors=5, weights='uniform', algorithm='auto')
    # return [Diffusion_GED_new(C=0.1, llambda=0.5, calculator_id=calculator_id, ged_bound=GED_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5)]
    # return HybridPrototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=5, selection_method="TPS", selection_split="all",dataset_name=DATASET.name, vector_feature_list=["density"], node_label_tag="label", edge_label_tag="label")
    # return GED_KNN(ged_calculator=ged_calculator, ged_bound=GED_BOUND, n_neighbors=7, weights='uniform', algorithm='auto')
    # return [CombinedHistogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced')]
    # classifier = Random_walk_edit_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
    # return Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=8, selection_method="k-CPS", selection_split="all",dataset_name=DATASET_NAME)


def nonGEd_classifiers(ged_calculator: Base_Calculator):
    return [
        Random_Classifier(),
        Blind_Classifier(),
        WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True),
        VertexHistogram_SVC(kernel_type='precomputed'),
        EdgeHistogram_SVC(kernel_type='precomputed'), 
        CombinedHistogram_SVC(kernel_type='precomputed'),
        # RandomWalk_SVC(normalize_kernel=True, rw_kernel_type="geometric", p_steps=3,C=1.0, kernel_type="precomputed"),
        ]
def ged_classifiers(ged_calculator: Base_Calculator):
    random_walk_calculator = try_load_else_build_rw_calculator(ged_calculator=ged_calculator)
    random_walk_calculator_id = random_walk_calculator.get_identifier_name()
    calculator_id = set_global_ged_calculator_All(ged_calculator)
    return [
        GED_KNN(calculator_id=calculator_id, ged_bound=GED_BOUND, n_neighbors=10, weights='distance', algorithm='auto'),
        Trivial_GED_SVC(calculator_id=calculator_id, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1',llambda=1.0),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, calculator_id=calculator_id, ged_bound=GED_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Random_Walk_edit_accelerated(calculator_id=calculator_id, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1,random_walk_calculator_id=random_walk_calculator_id, C=1.0,kernel_type="precomputed", class_weight='balanced')
        ]
def reference_classifiers(ged_calculator: Base_Calculator):
    calculator_id = set_global_ged_calculator_All(ged_calculator)
    return [
        # GED_KNN(calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, n_neighbors=7, weights='uniform', algorithm='auto'),
        # Trivial_GED_SVC(calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1',llambda=1.0),
        # Diffusion_GED_new(C=1.0, llambda=1.0, calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        ]

def run_classifier(classifier: GraphClassifier,expi: experiment,cv:int,testDF: pd.DataFrame):        
    try:
        instance_dict =expi.run_joblib_parallel_nested_cv(outer_cv=cv,inner_cv=cv,num_trials=NUM_TRIALS,scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'],tuning_metric=TUNING_METRIC, verbose=0, n_jobs=N_JOBS, search_method=SEARCH_METHOD,should_print=True,test_trail=TEST_TRIAL, get_all_results=GET_ALL_TUNING_RESULTS)
    except Exception as e:
        #  print the full traceback
        traceback.print_exc()
        print(f"Error running {classifier.get_name} on {DATASET_NAME}: {e}")
        instance_dict = {
            "Model": classifier.get_name,
            "Dataset": DATASET_NAME,
            "Error": str(e)
        }
    # add the values form instance_dict as the last row of testDF
    instance_df = pd.DataFrame([instance_dict])
    testDF = pd.concat([testDF, instance_df], ignore_index=True)
    save_progress(testDF, EXPERIMENT_NAME+"_inter")
    del classifier
    del expi
    return testDF

def run_classifier_group(get_classifiers_funct: callable, calculator_type:str,
                        testDF: pd.DataFrame):
 
    DATASET, ged_calculator = get_Dataset(calculator_type)

    print(f"Running {calculator_type} on {DATASET_NAME} dataset.")
    classifier_list: list[GraphClassifier] = get_classifiers_funct(ged_calculator)
    get_expi = lambda classifier: experiment(f"{EXPERIMENT_NAME}_{classifier.get_name}",DATASET,dataset_name=DATASET_NAME,
                    model=classifier,model_name=classifier.get_name,ged_calculator=None)
    cv = int(1/SPLIT)

    if ONLY_ESTIMATE:
        total_duration = pd.Timedelta(0)
        for classifier in classifier_list:
            expi = get_expi(classifier)
            estimated_time = expi.estimate_nested_cv_time(cv=cv,num_trials=NUM_TRIALS,search_method=SEARCH_METHOD)
            total_duration += estimated_time
        print(f"Estimated total duration for group {get_classifiers_funct.__name__} classifiers: {total_duration}")
        return None,total_duration
    else:
        for classifier in classifier_list:
            expi = get_expi(classifier)
            testDF = run_classifier(classifier, expi, cv, testDF)
    return testDF,0

if __name__ == "__main__":
    start_time = pd.Timestamp.now()
    Test_df = pd.DataFrame()
    print(f"Starting experiment {EXPERIMENT_NAME} on dataset {DATASET_NAME}) at {start_time}")
    if TESTING_MODE == "SINGLE":
        Test_df, total_duration = run_classifier_group(get_single_classifier,calculator_type=CALCULATOR_NAME,testDF=Test_df)
    
    elif TESTING_MODE == "ALL":
        Test_df, total_duration_nonGED = run_classifier_group(nonGEd_classifiers,  calculator_type=None, testDF=Test_df)

        Test_df, total_duration_GED = run_classifier_group(ged_classifiers,  calculator_type=CALCULATOR_NAME,testDF=Test_df)

        Test_df, total_duration_reference = run_classifier_group(reference_classifiers,  calculator_type=HEURISTIC_CALCULATOR_NAME, testDF=Test_df)
        total_duration = total_duration_nonGED + total_duration_GED + total_duration_reference
    elif TESTING_MODE == "MULTI":
        total_duration = 0
        for ds in DATASET_ARRAY:
            DATASET_NAME=ds
            Test_df["Dataset"] = ds
            Test_df, total_duration_nonGED = run_classifier_group(nonGEd_classifiers,  calculator_type=None, testDF=Test_df)

            Test_df, total_duration_GED = run_classifier_group(ged_classifiers,  calculator_type=CALCULATOR_NAME,testDF=Test_df)

            # Test_df, total_duration_reference = run_classifier_group(reference_classifiers,  calculator_type=HEURISTIC_CALCULATOR_NAME, testDF=Test_df)
            total_duration += total_duration_nonGED + total_duration_GED
    else:
        raise ValueError(f"Invalid TESTING_MODE: {TESTING_MODE}. Use 'SINGLE' or 'ALL'.")
    
    if ONLY_ESTIMATE:
        print(f"Estimated total duration for all classifiers: {total_duration}")
    else:
        end_run(Test_df, start_time, EXPERIMENT_NAME)


    

