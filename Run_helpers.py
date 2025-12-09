# file for main Run helper Functions
# Imports
import os
import pandas as pd
from Calculators.Base_Calculator import Base_Calculator
import multiprocessing
from Models.KNN.GEDLIB_KNN import set_global_ged_calculator_KNN
from Models.SVC.Base_GED_SVC import set_global_ged_calculator
last_save_time = pd.Timestamp.now()
from config_loader import get_conifg_param
INTERMEDIATE_SAVE_INTERVAL = get_conifg_param("Run_Experiment_main", "INTERMEDIATE_SAVE_INTERVAL")  # seconds, how often to save the intermediate results

def save_progress(testDF: pd.DataFrame, experiment_name: str):
    global last_save_time
    current_time = pd.Timestamp.now()
    if (current_time - last_save_time).seconds >= INTERMEDIATE_SAVE_INTERVAL:  # Save every 5 seconds
        results_dir = os.path.join("results","intermediate")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{experiment_name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        print(f"Progress saved at {current_time}")



def set_global_ged_calculator_All(calculator: Base_Calculator):
    set_global_ged_calculator(calculator)
    set_global_ged_calculator_KNN(calculator)
    if calculator is None:
        return "None"
    return calculator.get_identifier_name()


def end_run(testDF: pd.DataFrame, start_time: pd.Timestamp, EXPERIMENT_NAME: str):
    results_dir = os.path.join("results")
    results_path = os.path.join(results_dir, f"{EXPERIMENT_NAME}.xlsx")
    testDF.to_excel(results_path, index=False)
    end_time = pd.Timestamp.now()
    print(f"Experiment ended at {end_time}")
    total_duration = end_time - start_time
    print(f"Total experiment duration: {total_duration}")


def set_Mode(mode_level: int,N_JOBS:int,SPLIT:float):
    num_trials =None
    Test_TRAIL=False
    Only_estimate=False
    Get_All_tuning_results=False
    if mode_level == 0:
        num_trials = 0
        Test_TRAIL=False
        Only_estimate=False
        Get_All_tuning_results=False
    elif mode_level == 1:
        num_trials = 1
        Test_TRAIL=True
        Only_estimate=True
        Get_All_tuning_results=False
    elif mode_level == 2:
        num_trials = 1
        Test_TRAIL = True
        Only_estimate = False
        Get_All_tuning_results = False
    elif mode_level == 3:
        num_trials = 3
        Test_TRAIL = True
        Only_estimate = False
        Get_All_tuning_results = False
    elif mode_level == 4:
        num_trials = 3
        Test_TRAIL = False
        Only_estimate = False
        Get_All_tuning_results = True
    elif mode_level == 5:
        num_trials = 5
        Test_TRAIL = False
        Only_estimate = False
        Get_All_tuning_results = True
    else:
        raise ValueError(f"Unknown mode level: {mode_level}")
    num_runs = num_trials * int(1/SPLIT)
    try:
        cpu_count = multiprocessing.cpu_count() - 1
    except:
        try:
            cpu_count = os.cpu_count() - 1
        except:
            cpu_count = 1
    if N_JOBS == "AUTO":
        N_JOBS = min(cpu_count,num_runs)
    elif isinstance(N_JOBS, int):
        if N_JOBS < 1:
            print("warning: N_JOBS must be at least 1, setting to AUTO")
            N_JOBS = min(cpu_count,num_runs)
        else:
            N_JOBS = min(N_JOBS,num_runs,cpu_count)
    else:
        raise ValueError(f"Unknown N_JOBS type: {type(N_JOBS)}")
    print(f"Number of parallel jobs set to: {N_JOBS}")
    return num_trials, Test_TRAIL, Only_estimate, Get_All_tuning_results,N_JOBS

def define_dataset_processing(Nodes_and_edges):
    """
    Nodes_and_edges = None # eg "labels", "attributes" or None for both
    """
    if Nodes_and_edges == "labels":
        return "label","label",None, None
    elif Nodes_and_edges == "attributes":
        return None, None, "attributes", "attributes"
    elif Nodes_and_edges == None:
        return None, None, None, None
    else:
        raise ValueError(f"Unknown node and edge handling: {Nodes_and_edges}")

def configure_run(Datasets_to_run, MODELS_TO_RUN,testing_level):
    """
    Datasets_to_run = "PTC_FR"  # singel string or list of dataset names e.g. "MUTAG", "PTC_MR", "IMDB-MULTI", "PROTEINS", "NCI1", "NCI109", "DD", "COLLAB", "REDDIT-BINARY"
    MODELS_TO_RUN= "ALL" # ALL for all models, or "String of a specific model or list of models e.g. ["Random_Walk_edit_accelerated", "VertexHistogram_SVC"]
    """
    multi_ds_mode=False
    if isinstance(Datasets_to_run, str):
        # only a single dataset
        DATASET_NAME = Datasets_to_run
        DATASET_ARRAY = [Datasets_to_run]
    elif isinstance(Datasets_to_run, list):
        # multiple datasets
        multi_ds_mode=True
        DATASET_NAME = Datasets_to_run[0]
        DATASET_ARRAY = Datasets_to_run
    else:
        raise ValueError(f"Unknown Datasets_to_run type: {type(Datasets_to_run)}")
    if testing_level == 0:
        print("Running in SPEEDTEST mode.")
        print("automatically setting MODELS_TO_RUN to ALL")
        MODELS_TO_RUN = "ALL"
        if multi_ds_mode:
            TESTING_MODE = "SPEEDMULTI"
        else:
            TESTING_MODE = "SPEEDTEST"
    else:
        if MODELS_TO_RUN == "ALL":
            if  multi_ds_mode:
                TESTING_MODE = "MULTI"
            else:
                TESTING_MODE = "ALL"
        elif isinstance(MODELS_TO_RUN, str) and MODELS_TO_RUN in ["VH","EH","CH","RW","WL-ST","GED-KNN","Diff-GED","Triv-GED","RWE","Zero-GED"]:
            if  multi_ds_mode:
                TESTING_MODE = "SINGLEMULTI"
            else:
                TESTING_MODE = "SINGLE"
        else:
            raise ValueError(f"Unknown MODELS_TO_RUN type: {type(MODELS_TO_RUN)}")
   
   
    return DATASET_NAME, DATASET_ARRAY,TESTING_MODE,MODELS_TO_RUN

# f"{TUNING_METRIC}_{now}_{DATASET_NAME}_{int(DATASET_NODE_LABELS!=None)}_{int(DATASET_EDGE_LABELS!=None)}_{TESTING_MODE}"

def define_experiment_name(EXPERIMENT_NAME, TUNING_METRIC, DATASET_NAME, Nodes_and_edges, TESTING_MODE, MODELS_TO_RUN):
    now = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    if EXPERIMENT_NAME is None or EXPERIMENT_NAME == "" or EXPERIMENT_NAME =="AUTO":
        if TESTING_MODE == "SINGLE":
            experiment_name = f"{TUNING_METRIC}_{MODELS_TO_RUN}_{DATASET_NAME}_{0 if Nodes_and_edges is None else Nodes_and_edges}_{now}"
        elif TESTING_MODE == "SINGLEMULTI":
            experiment_name = f"{TUNING_METRIC}_{MODELS_TO_RUN}_{0 if Nodes_and_edges is None else Nodes_and_edges}_MULTI_{now}"
        elif TESTING_MODE == "ALL":
            experiment_name = f"{TUNING_METRIC}_{DATASET_NAME}_{0 if Nodes_and_edges is None else Nodes_and_edges}_{now}"
        elif TESTING_MODE == "MULTI":
            experiment_name = f"{TUNING_METRIC}_MULTI_{0 if Nodes_and_edges is None else Nodes_and_edges}_{now}"
        elif TESTING_MODE == "SPEEDTEST":
            experiment_name = f"SPEEDTEST_{DATASET_NAME}_{0 if Nodes_and_edges is None else Nodes_and_edges}_{now}"
        elif TESTING_MODE == "SPEEDMULTI":
            experiment_name = f"SPEED_MULTI_{0 if Nodes_and_edges is None else Nodes_and_edges}_{now}"
    else:
        experiment_name = EXPERIMENT_NAME.strip()
    print(f"Experiment {experiment_name} started at {now}")
    print(f"Testing Mode: {TESTING_MODE}")
    print(f"Dataset Name: {DATASET_NAME}")
    print(f"Node and Edge Handling: {Nodes_and_edges if Nodes_and_edges is not None else 'unlabeld'}")
    print(f"Tuning Metric: {TUNING_METRIC}")
    print(f"Models to Run: {MODELS_TO_RUN}")
    print("-----------------------------------------------------")
    return experiment_name
