# Run Experiment 
# imports
# standart libraries
import sys
import os
import traceback
import pandas as pd
# Main support modules
from Run_helpers import configure_run, define_dataset_processing, define_experiment_name, end_run, save_progress,set_Mode, set_global_ged_calculator_All 
from Calculators.GED_Calculator import reset_calculators_cache, try_load_else_build_rw_calculator
from Dataset import Dataset
from Experiment import experiment
sys.path.append(os.getcwd())


# Model imports
from Models.SVC.GED.RandomWalk_edit import Random_Walk_edit_accelerated, set_global_random_walk_calculator
from Models.SVC.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.SVC.random_walk import RandomWalk_SVC
from Models.Graph_Classifier import GraphClassifier
from Models.SVC.Baseline_SVC import VertexHistogram_SVC,EdgeHistogram_SVC, CombinedHistogram_SVC
from Models.Blind_Classifier import Blind_Classifier
from Models.Random_Classifer import Random_Classifier
from Models.SVC.GED.Trivial_GED_SVC import Trivial_GED_SVC
from Calculators.Base_Calculator import Base_Calculator
from Models.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC
from Models.SVC.GED.Zero_GED_SVC import ZERO_GED_SVC
from Models.KNN.GEDLIB_KNN import GED_KNN
from config_loader import get_conifg_param
module="Run_Experiment_main"

# User Configurable Parameters

# to set the mode:
# Testing Level, indecating stages from only testing for fucntionality, to wanting a full result
# 0 Speed Test Mode (only for for testing runtime and support vectors only)
# 1 only Speed Test
# 2 a test trail, with only 1 trail
# 3 test trail, but with full trials
# 4 Full Run with all tuning results saved
Datasets_to_run = ["MUTAG","PTC_FR"]  # singel string or list of dataset names e.g. "MUTAG", "PTC_MR", "IMDB-MULTI", "PROTEINS", "NCI1", "NCI109", "DD", "COLLAB", "REDDIT-BINARY"
testing_level= get_conifg_param(module, 'testing_level', type='int') # 0-4
MODELS_TO_RUN= get_conifg_param(module, 'models_to_run', type='str') # ALL for all models, or "String of a specific model or list of models e.g. ["Random_Walk_edit_accelerated", "VertexHistogram_SVC"]
Nodes_and_edges = get_conifg_param(module, 'nodes_and_edges') # eg "labels", "attributes" or None for both
TUNING_METRIC= get_conifg_param(module, 'tuning_metric', type='str') # e.g. "accuracy", "f1_macro", "roc_auc"
N_JOBS= get_conifg_param(module, 'n_jobs') # Auto or specified number of jobs for parallel processing
SPLIT= get_conifg_param(module, 'split', type='float') # e.g. 0.1 for 10 fold CV
EXPERIMENT_NAME=get_conifg_param(module, 'experiment_name',type=str) # if AUTO it will be set automatically, with a desicrptive name for result files. aditionnally the name can be set manually here.




# Not meant to be changed by user
CALCULATOR_NAME= get_conifg_param(module, 'CALCULATOR_NAME', type='str') # Exact_GED
HEURISTIC_CALCULATOR_NAME= get_conifg_param(module, 'HEURISTIC_CALCULATOR_NAME', type='str') # Heuristic_Calculator
SEARCH_METHOD= get_conifg_param(module, 'SEARCH_METHOD', type='str') # random
GED_BOUND= get_conifg_param(module, 'GED_BOUND', type='str') # Exact
HEURISTIC_BOUND= get_conifg_param(module, 'HEURISTIC_BOUND', type='str') # Vertex


# Configure Run
DATASET_NAME, DATASET_ARRAY,TESTING_MODE,MODELS_TO_RUN = configure_run(Datasets_to_run,MODELS_TO_RUN,testing_level)
NUM_TRIALS, TEST_TRIAL, ONLY_ESTIMATE, GET_ALL_TUNING_RESULTS,N_JOBS= set_Mode(testing_level,N_JOBS,SPLIT)
DATASET_EDGE_LABELS, DATASET_NODE_LABELS, DATASET_NODE_ATTRIBUTES, DATASET_EDGE_ATTRIBUTES = define_dataset_processing(Nodes_and_edges)
EXPERIMENT_NAME=define_experiment_name(EXPERIMENT_NAME, TUNING_METRIC, DATASET_NAME,Nodes_and_edges, TESTING_MODE,MODELS_TO_RUN)

# Parameters configured, start experiment




# isntanciate ged calculator and dataset
def get_Dataset(ged_calculator,name=DATASET_NAME):
    DATASET= Dataset(name=name,ged_calculator=ged_calculator, use_node_labels=DATASET_NODE_LABELS, use_edge_labels=DATASET_EDGE_LABELS,load_now=False,use_node_attributes=DATASET_NODE_ATTRIBUTES, use_edge_attributes=DATASET_EDGE_ATTRIBUTES)
    DATASET.load()
    # DATASET.load_with_attributes(new_attributes=["x","y"], encoding_dimension=2, remove_old=True)
    return DATASET, DATASET.get_calculator()
# Select single classifier to run
def get_single_classifier(ged_calculator):
    set_global_random_walk_calculator(None)
    set_global_ged_calculator_All(ged_calculator)
    random_walk_calculator = try_load_else_build_rw_calculator(ged_calculator=ged_calculator)
    try:
        random_walk_calculator_id = random_walk_calculator.get_identifier_name()
        calculator_id = set_global_ged_calculator_All(ged_calculator)
    except Exception as e:
        print(f"Error occurred while getting identifiers: {e}")
    if MODELS_TO_RUN == "VH":
        return [VertexHistogram_SVC()]
    elif MODELS_TO_RUN == "EH":
        return [EdgeHistogram_SVC()]
    elif MODELS_TO_RUN == "CH":
        return [CombinedHistogram_SVC(C=1.0, class_weight='balanced')]
    elif MODELS_TO_RUN == "RW":
        return [RandomWalk_SVC(normalize_kernel=True, rw_kernel_type="exponential", p_steps=1,C=1.0, kernel_type="precomputed")]
    elif MODELS_TO_RUN == "WL-ST":
        return [WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True)]
    elif MODELS_TO_RUN == "GED-KNN":
        return [GED_KNN(calculator_id=calculator_id, ged_bound=GED_BOUND, n_neighbors=5, weights='distance', algorithm='auto')]
    elif MODELS_TO_RUN == "Diff-GED":
        return [DIFFUSION_GED_SVC(C=1.0, llambda=0.1, calculator_id=calculator_id, ged_bound=GED_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5)]
    elif MODELS_TO_RUN == "Triv-GED":
        return [Trivial_GED_SVC(calculator_id=calculator_id, ged_bound=GED_BOUND, C=0.5,kernel_type="precomputed", class_weight='balanced',similarity_function='k4',llambda
            =0.1)]
    elif MODELS_TO_RUN == "RWE":
        return [Random_Walk_edit_accelerated(calculator_id=calculator_id, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1,random_walk_calculator_id=random_walk_calculator_id, C=1.0,kernel_type="precomputed", class_weight='balanced')]
    elif MODELS_TO_RUN == "Zero-GED":
        # Probably Broken
        return [ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=7, aggregation_method="sum",dataset_name=DATASET.name,selection_method="k-CPS")]
    else:
        return []

# Get all non-GED classifiers
def nonGEd_classifiers(ged_calculator: Base_Calculator):
    return [
        Random_Classifier(),
        Blind_Classifier(),
        WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True),
        VertexHistogram_SVC(),
        EdgeHistogram_SVC(),
        CombinedHistogram_SVC(C=1.0, class_weight='balanced'),
        RandomWalk_SVC(normalize_kernel=True, rw_kernel_type="geometric", p_steps=3,C=1.0, kernel_type="precomputed"),
        ]
# Get all GED classifiers
def ged_classifiers(ged_calculator: Base_Calculator):
    set_global_random_walk_calculator(None)
    random_walk_calculator = try_load_else_build_rw_calculator(ged_calculator=ged_calculator)
    random_walk_calculator_id = random_walk_calculator.get_identifier_name()
    calculator_id = set_global_ged_calculator_All(ged_calculator)
    return [
        GED_KNN(calculator_id=calculator_id, ged_bound=GED_BOUND, n_neighbors=5, weights='distance', algorithm='auto'),
        Trivial_GED_SVC(calculator_id=calculator_id, ged_bound=GED_BOUND, C=0.5,kernel_type="precomputed", class_weight='balanced',similarity_function='k4',llambda=0.1),
        DIFFUSION_GED_SVC(C=1.0, llambda=0.1, calculator_id=calculator_id, ged_bound=GED_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
        Random_Walk_edit_accelerated(calculator_id=calculator_id, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1,random_walk_calculator_id=random_walk_calculator_id, C=1.0,kernel_type="precomputed", class_weight='balanced')
        ]

# GED calculators with for basic GED node count heuristics calculators
# Unsused currently
def reference_classifiers(ged_calculator: Base_Calculator):
    calculator_id = set_global_ged_calculator_All(ged_calculator)
    return [
        GED_KNN(calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, n_neighbors=7, weights='uniform', algorithm='auto'),
        Trivial_GED_SVC(calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1',llambda=1.0),
        DIFFUSION_GED_SVC(C=1.0, llambda=1.0, calculator_id=calculator_id, ged_bound=HEURISTIC_BOUND, diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=5),
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
            "Error": str(e),
            "traceback": traceback.format_exc()
        }
    # add the values form instance_dict as the last row of testDF
    instance_df = pd.DataFrame([instance_dict])
    testDF = pd.concat([testDF, instance_df], ignore_index=True)
    save_progress(testDF, EXPERIMENT_NAME+"_inter")
    del classifier
    del expi
    return testDF
def run_speed_test(get_classifiers_funct: callable, calculator_type:str, iterations:int,testDF: pd.DataFrame,dataset_name=DATASET_NAME):
    DATASET, ged_calculator = get_Dataset(calculator_type,dataset_name)

    print(f"Running speed test for {calculator_type} on {DATASET_NAME} dataset.")
    classifier_list: list[GraphClassifier] = get_classifiers_funct(ged_calculator)
    get_expi = lambda classifier: experiment(f"{EXPERIMENT_NAME}_{classifier.get_name}",DATASET,dataset_name=DATASET_NAME,
                    model=classifier,model_name=classifier.get_name,ged_calculator=None)

    for classifier in classifier_list:
        expi = get_expi(classifier)
        new_row = expi.run_large_speed_test(iterations=iterations)
        new_row["Model"] = classifier.get_name
        new_row["Dataset"] = DATASET_NAME
        new_row["tuning_metric"] = TUNING_METRIC
        testDF = pd.concat([testDF, pd.DataFrame([new_row])], ignore_index=True)
    return testDF


def run_classifier_group(get_classifiers_funct: callable, calculator_type:str,
                        testDF: pd.DataFrame,dataset_name:str=DATASET_NAME):
 
    DATASET, ged_calculator = get_Dataset(calculator_type,dataset_name)
    classifier_list: list[GraphClassifier] = get_classifiers_funct(ged_calculator)
    get_expi = lambda classifier: experiment(f"{EXPERIMENT_NAME}_{classifier.get_name}",DATASET,dataset_name=DATASET_NAME,
                    model=classifier,model_name=classifier.get_name,ged_calculator=None)
    cv = int(1/SPLIT)

    if ONLY_ESTIMATE:
        total_duration = pd.Timedelta(0)
        for classifier in classifier_list:
            expi = get_expi(classifier)
            estimated_time, _ = expi.estimate_nested_cv_time(cv=cv,num_trials=NUM_TRIALS,search_method=SEARCH_METHOD)
            total_duration += estimated_time
        print(f"Estimated total duration for group {get_classifiers_funct.__name__} classifiers: {total_duration}")
        return testDF,total_duration
    else:
        for classifier in classifier_list:
            expi = get_expi(classifier)
            testDF = run_classifier(classifier, expi, cv, testDF)
    return testDF, pd.Timedelta(0)

if __name__ == "__main__":
    start_time = pd.Timestamp.now()
    Test_df = pd.DataFrame()
    print(f"Starting experiment {EXPERIMENT_NAME} on dataset {DATASET_NAME}) at {start_time}")
    if TESTING_MODE == "SINGLE":
        Test_df, total_duration = run_classifier_group(get_single_classifier,calculator_type=CALCULATOR_NAME,testDF=Test_df)
    elif TESTING_MODE == "SINGLEMULTI":
        for ds in DATASET_ARRAY:
            DATASET_NAME=ds
            print(DATASET_NAME)
            set_global_random_walk_calculator(None)
            Test_df["Dataset"] = ds
            Test_df, total_duration = run_classifier_group(get_single_classifier,calculator_type=CALCULATOR_NAME,testDF=Test_df,dataset_name=ds)
            reset_calculators_cache()
    elif TESTING_MODE == "ALL":
        Test_df, total_duration_nonGED = run_classifier_group(nonGEd_classifiers,  calculator_type=None, testDF=Test_df)
        Test_df, total_duration_GED = run_classifier_group(ged_classifiers,  calculator_type=CALCULATOR_NAME,testDF=Test_df)

        # Test_df, total_duration_reference = run_classifier_group(reference_classifiers,  calculator_type=HEURISTIC_CALCULATOR_NAME, testDF=Test_df)
        total_duration = total_duration_nonGED + total_duration_GED
    elif TESTING_MODE == "SPEEDTEST":
        Test_df = run_speed_test(nonGEd_classifiers, calculator_type=None, iterations=10,testDF=Test_df)
        Test_df = run_speed_test(ged_classifiers, calculator_type=CALCULATOR_NAME, iterations=10,testDF=Test_df)
        
        total_duration = 0
    elif TESTING_MODE == "MULTI":
        total_duration = pd.Timedelta(0)
        for ds in DATASET_ARRAY:
            DATASET_NAME=ds
            Test_df["Dataset"] = ds
            Test_df, total_duration_nonGED = run_classifier_group(nonGEd_classifiers,  calculator_type=None, testDF=Test_df,dataset_name=ds)

            Test_df, total_duration_GED = run_classifier_group(ged_classifiers,  calculator_type=CALCULATOR_NAME,testDF=Test_df,dataset_name=ds)
            reset_calculators_cache()
            # Test_df, total_duration_reference = run_classifier_group(reference_classifiers,  calculator_type=HEURISTIC_CALCULATOR_NAME, testDF=Test_df)
            total_duration += total_duration_nonGED + total_duration_GED
    elif TESTING_MODE == "SPEEDMULTI":
        total_duration = 0
        for ds in DATASET_ARRAY:
            DATASET_NAME=ds
            Test_df["Dataset"] = ds
            Test_df = run_speed_test(nonGEd_classifiers, calculator_type=None, iterations=10,testDF=Test_df,dataset_name=ds)
            Test_df = run_speed_test(ged_classifiers, calculator_type=CALCULATOR_NAME, iterations=10,testDF=Test_df,dataset_name=ds)
            reset_calculators_cache()

    else:
        raise ValueError(f"Invalid TESTING_MODE: {TESTING_MODE}. Use 'SINGLE' or 'ALL'.")
    
    if ONLY_ESTIMATE:
        print(f"Estimated total duration for all classifiers: {total_duration}")
    else:
        end_run(Test_df, start_time, EXPERIMENT_NAME)


    

