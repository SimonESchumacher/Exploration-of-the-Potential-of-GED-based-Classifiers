# file for main Run helper Functions

# Imports
import os
import pandas as pd
import traceback
from Calculators.Base_Calculator import Base_Calculator
from Dataset import Dataset
from Models.KNN.GEDLIB_KNN import set_global_ged_calculator_KNN
from Models.SVC.Base_GED_SVC import set_global_ged_calculator
last_save_time = pd.Timestamp.now()


def save_progress(testDF: pd.DataFrame, experiment_name: str):
    global last_save_time
    current_time = pd.Timestamp.now()
    if (current_time - last_save_time).seconds >= 5:  # Save every 5 seconds
        results_dir = os.path.join("configs", "results","intermediate")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{experiment_name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        print(f"Progress saved at {current_time}")



def set_global_ged_calculator_All(calculator: Base_Calculator):
    set_global_ged_calculator(calculator)
    set_global_ged_calculator_KNN(calculator)
    return calculator.get_identifier_name()


def end_run(testDF: pd.DataFrame, start_time: pd.Timestamp, EXPERIMENT_NAME: str):
    results_dir = os.path.join("configs", "results")
    results_path = os.path.join(results_dir, f"{EXPERIMENT_NAME}_results.xlsx")
    testDF.to_excel(results_path, index=False)
    end_time = pd.Timestamp.now()
    print(f"Experiment ended at {end_time}")
    total_duration = end_time - start_time
    print(f"Total experiment duration: {total_duration}")


def set_Mode(mode_level: int):
    num_trials =None
    Test_TRAIL=False
    Only_estimate=False
    Get_All_tuning_results=False
    if mode_level == 1:
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
    return num_trials, Test_TRAIL, Only_estimate, Get_All_tuning_results
