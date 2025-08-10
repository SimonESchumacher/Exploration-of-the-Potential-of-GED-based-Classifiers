# Experiment
# Imports
import random
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  # For progress bar
import traceback
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from scipy import stats
from Models.Graph_Classifier import GraphClassifier
# import classweights
from sklearn.utils.class_weight import compute_class_weight
from config_loader import get_conifg_param

RANDOM_STATE = get_conifg_param('Experiment', 'random_state', type='int') # default 42
RANDOM_STATE = random.randint(0, 1000)
TEST_SIZE = get_conifg_param('Experiment', 'test_size', type='float') # default 0.2
RESULTS_FILE = get_conifg_param('Experiment', 'results_filepath', type='str') # default "experiment_log.xlsx"
DIAGNOSTIC_FILE =get_conifg_param('Experiment', 'disgnostics_filepath', type='str') 
# DATASET_NAMES = ["MUTAG", "PROTEINS_full", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]
DEBUG = get_conifg_param(module="Experiment",parametername="DEBUG",type="bool")# Set to False to disable debug prints
# load config file with the models, with teir specifcations
# load the config file
REPORT_SETTING=get_conifg_param(module="Experiment",parametername="report_setting",type='str') # for f1_score, precision, recall

ERRORINTERVAL_SETTING = get_conifg_param(module="Experiment",parametername="errorinterval_setting",type='str') # "std" or "confidence interval"
SAVE_MODELS = get_conifg_param(module="Experiment",parametername="save_models",type='bool') # default True
SAVE_LOGS = get_conifg_param(module="Experiment",parametername="save_logs",type='bool') # default True

DATASET_HYPERPARAM = get_conifg_param(module="Experiment",parametername="dataset_hyperparams",type='bool') # default False
CALCULATOR_HYPERPARAM = get_conifg_param(module="Experiment",parametername="calculator_hyperparams",type='bool') # default False
class experiment:


    # for now asssue, initialized Dataset and Model in the constructor
    def __init__(self, name, dataset, dataset_name, model, model_name=None, ged_calculator=None):
        self.experiment_name = name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model = model
        self.model_name = model_name if model_name else model.__class__.__name__
        self.ged_calculator = ged_calculator
        # Results file Collums (to be added):
        # experiment_name, dataset_name, model_name,filename,saving timestamp,timstamp_experiment_run,testsize,has_edge_labels,has_node_labels,dataset_mean_Nodes,dataset_mean_edges,dataset_Num_clases,datset_domain, accuracy, f1_score, precision, recall, 
        self.diagnostic_log = {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "filename": None,  # To be set after saving the model
            "dataset_load_duration": None,  # To be set after loading the dataset
            "saving_timestamp": None,  # To be set after saving the model
            "timestamp_experiment_run": pd.Timestamp.now(),
            "Error": None,  # To be set if an error occurs

        }
        self.results_log = {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "testsize": TEST_SIZE,
            "training_duration": None,  # To be set after training
            "testing_duration": None,  # To be set after testing
            "accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "roc_auc": None,
            "saving_timestamp": None,  # To be set after saving the model
            "timestamp_experiment_run": pd.Timestamp.now(),
        }
    def save_log_to_excel(self):
        """
        Writes the content of self.log_dict to an Excel file.
        If the file exists, it appends the new entry as a new row.
        """
        if not SAVE_LOGS:
            return
        # Create a DataFrame from the current log_dict.
        # We wrap it in a list to ensure it's treated as a single row.
        new_entry__diagnostics_df = pd.DataFrame([self.diagnostic_log])
        new_entry_results_df = pd.DataFrame([self.results_log])
        # Ensure the directory exists

        # Check if the file already exists
       
        try:
            # Read the existing Excel file
            results_old = pd.read_excel(RESULTS_FILE)
            diagnostics_old = pd.read_excel(DIAGNOSTIC_FILE)
            # Check if headers match, or if the new entry has extra columns
            # This is a robust check to prevent misaligned appends

            # results_Save
            if not results_old.empty and not new_entry_results_df.columns.isin(results_old.columns).all():
                print(f"Warning: New entry columns do not exactly match existing Excel columns in {RESULTS_FILE}.")                         
            # Concatenate the existing data with the new entry
            combined_df = pd.concat([results_old, new_entry_results_df], ignore_index=True)
            # Write the combined DataFrame back to the Excel file
            combined_df.to_excel(RESULTS_FILE, index=False)
            if DEBUG:
                print(f"Appended experiment log to existing file: {RESULTS_FILE}")
            
            # diagnostics_Save
            if not diagnostics_old.empty and not new_entry__diagnostics_df.columns.isin(diagnostics_old.columns).all():
                print(f"Warning: New entry columns do not exactly match existing Excel columns in {DIAGNOSTIC_FILE}.")                
            # Concatenate the existing data with the new entry
            combined_df = pd.concat([diagnostics_old, new_entry__diagnostics_df], ignore_index=True)            
            # Write the combined DataFrame back to the Excel file
            combined_df.to_excel(DIAGNOSTIC_FILE, index=False)
            if DEBUG:
                print(f"Appended experiment log to existing file: {DIAGNOSTIC_FILE}")

        except Exception as e:
            print(f"Error while saving to Excel: {e}")

  
            

        # Update filename in the original log_dict for internal reference
    
    # not done
    def __str__(self):
        return f"Experiment(name={self.experiment_name}, dataset={self.dataset_name}, model={self.model_name})"
    def split_data(self):
        
        X_train, X_test, y_train, y_test = self.dataset.train_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        # if DEBUG:
        #     print(f"Successfully split Data:{self.dataset_name} into training (len:{len(X_train)}) and testing (len:{len(X_test)})sets.")
        #     print("Dataset Attributes:")
        #     print(self.dataset.attributes())
        return X_train, X_test, y_train, y_test
    
    def inner_model_fit(self, G_train, y_train):
        """
        Fits the model to the training data.
        This method is called by the fit_model method.
        """
        if hasattr(self.model, 'fit'):
            
            self.model.fit(G_train, y_train)
            
            return self.model
        else:
            raise ValueError("Model does not have a fit method.")
    def inner_model_predict(self, G_test):
        """
        Predicts the labels for the test data.
        This method is called by the test_model method.
        """
        if hasattr(self.model, 'predict'):
            
            y_pred = self.model.predict(G_test)
           
            return y_pred
        else:
            raise ValueError("Model does not have a predict method.")

    def fit_model(self, G_train, y_train):
        fit_start_time = datetime.now()
        self.inner_model_fit(G_train, y_train)
        print(f"Model {self.model_name} trained successfully.")
        duration = datetime.now() - fit_start_time
        self.results_log["training_duration"] = duration

        filename = f"{self.model_name}_trained_on_{self.dataset_name}.joblib"
        filepath = os.path.join("Models_Storage", filename)
        self.diagnostic_log["filename"] = filename
        self.diagnostic_log["saving_timestamp"] = datetime.now()
        self.results_log["saving_timestamp"] = self.diagnostic_log["saving_timestamp"]
        if SAVE_MODELS:
            self.model.save(filepath)
        return self.model

    def test_model(self, G_test, y_test):
        test_start_time = datetime.now()
        y_pred=self.inner_model_predict(G_test)          
        self.results_log["testing_duration"] = datetime.now() - test_start_time
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=REPORT_SETTING)
        roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        precision = precision_score(y_test, y_pred, average=REPORT_SETTING)
        recall = recall_score(y_test, y_pred, average=REPORT_SETTING)
        # maybe i shoudl split this to a defirent file and make a diffrence between k_fold and simple
        self.results_log["accuracy"] = accuracy
        self.results_log["f1_score"] = f1
        self.results_log["roc_auc"] = roc_auc
        self.results_log["precision"] = precision
        self.results_log["recall"] = recall

        classification_report_str = classification_report(y_test, y_pred)
        print(f"Model {self.model_name} tested on {len(G_test)} graphs with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report_str)
        return accuracy, y_pred,classification_report_str
      
    # train test in K-Fold (perfoming cross validation)
    def train_test_kfold(self, k=5):
       
        # save values for metrics
        is_first_fold = True
        durations_fit = []
        durattions_test = []
        accuracies = []
        f1_scores = []
        roc_aucs = []
        precisions = []
        recalls = []
        for X_train, X_test, y_train_fold, y_test_fold in self.dataset.split_k_fold(k=k, random_state=RANDOM_STATE):
            fit_start_time = datetime.now()
            self.inner_model_fit(X_train, y_train_fold)
            durations_fit.append((datetime.now() - fit_start_time).total_seconds())
            test_start_time = datetime.now()
            y_pred = self.inner_model_predict(X_test)
            durattions_test.append((datetime.now() - test_start_time).total_seconds())

            accuracies.append(accuracy_score(y_test_fold, y_pred))
            f1_scores.append(f1_score(y_test_fold, y_pred, average=REPORT_SETTING))
            roc_aucs.append(roc_auc_score(y_test_fold, y_pred, multi_class='ovr'))
            precisions.append(precision_score(y_test_fold, y_pred, average=REPORT_SETTING))
            recalls.append(recall_score(y_test_fold, y_pred, average=REPORT_SETTING))
            if DEBUG:
                print(f"Accuracy for fold: {accuracies[-1]:.4f}, F1 Score: {f1_scores[-1]:.4f}, ")
            if is_first_fold:
                self.diagnostic_log["filename"] = f"{self.model_name}_trained_on_{self.dataset_name}_kfold.joblib"
                self.diagnostic_log["saving_timestamp"] = datetime.now()
                self.results_log["saving_timestamp"] = self.diagnostic_log["saving_timestamp"]
                if SAVE_MODELS:
                    self.model.save(os.path.join("Models_Storage", self.diagnostic_log["filename"]))
                classification_report_str = classification_report(y_test_fold, y_pred)
                is_first_fold = False
        # Calculate mean values for metrics
        # save the mean values for the metrics, along with the standard deviation 
        erroraccnolagement = ERRORINTERVAL_SETTING
        if erroraccnolagement == "std":
            self.results_log["training_duration"] = np.mean(durations_fit) +" " + np.std(durations_fit)
            self.results_log["testing_duration"] = np.mean(durattions_test) + " " + np.std(durattions_test)
            self.results_log["accuracy"] = np.mean(accuracies) + " " + np.std(accuracies)
            self.results_log["f1_score"] = np.mean(f1_scores) + " " + np.std(f1_scores)
            self.results_log["roc_auc"] = np.mean(roc_aucs) + " " + np.std(roc_aucs)
            self.results_log["precision"] = np.mean(precisions) + " " + np.std(precisions)
            self.results_log["recall"] = np.mean(recalls) + " " + np.std(recalls)
        elif erroraccnolagement == "confidence interval":
            # Calculate confidence intervals for each metric
            n = len(accuracies)
            confidence = 0.95
            z_score = stats.norm.ppf((1 + confidence) / 2)
            self.results_log["training_duration"] = f"{np.mean(durations_fit)} ∓ { z_score * np.std(durations_fit) / np.sqrt(n)}"
            self.results_log["testing_duration"] = f"{np.mean(durattions_test)} ∓ { z_score * np.std(durattions_test) / np.sqrt(n)}"
            self.results_log["accuracy"] = f"{np.mean(accuracies)} ∓ { z_score * np.std(accuracies) / np.sqrt(n)}"
            self.results_log["f1_score"] = f"{np.mean(f1_scores)} ∓ { z_score * np.std(f1_scores) / np.sqrt(n)}"
            self.results_log["roc_auc"] = f"{np.mean(roc_aucs)} ∓ { z_score * np.std(roc_aucs) / np.sqrt(n)}"
            self.results_log["precision"] = f"{np.mean(precisions)} ∓ { z_score * np.std(precisions) / np.sqrt(n)}"
            self.results_log["recall"] = f"{np.mean(recalls)} ∓ { z_score * np.std(recalls) / np.sqrt(n)}"
        elif erroraccnolagement == "none":
            self.results_log["training_duration"] = np.mean(durations_fit)
            self.results_log["testing_duration"] = np.mean(durattions_test)
            self.results_log["accuracy"] = np.mean(accuracies)
            self.results_log["f1_score"] = np.mean(f1_scores)
            self.results_log["roc_auc"] = np.mean(roc_aucs)
            self.results_log["precision"] = np.mean(precisions)
            self.results_log["recall"] = np.mean(recalls)
        else:
            raise ValueError(f"Unknown error acknowledgment method: {erroraccnolagement}. Use 'std', 'confidence interval', or 'none'.")
        print(f"Model {self.model_name} trained and tested with K-Fold cross-validation (k={k}).")
        print(f"Mean Accuracy: {self.results_log['accuracy']}")
        return np.mean(accuracies) ,classification_report_str
    
    def run_hyperparameter_tuning(self, tuning_method='grid',scoring='accuracy',cv=5, verbose=2, n_jobs=1):



        X_train, X_test, y_train, y_test = self.split_data()
        param_grid = self.model.get_param_grid()
        # currently not in use, porbaly not needed and not a smart idea to use it
        if DATASET_HYPERPARAM:
            param_grid.update(self.dataset.get_param_grid())
        if CALCULATOR_HYPERPARAM and self.ged_calculator is not None:
            param_grid.update(self.ged_calculator.get_param_grid())


        if DEBUG:
            print(f"Starting hyperparameter tuning for {self.model_name} on dataset {self.dataset_name} with parameters: {param_grid}")
        hyperparameter_tuning_start_time = datetime.now()
        if tuning_method == 'grid':
            hyperparameter_tuner = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs, error_score='raise')
        elif tuning_method == 'random':
            hyperparameter_tuner = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs, n_iter=10)
        if DEBUG:
            print("stating hyperparameter tuning...")  
        hyperparameter_tuner.fit(X_train, y_train)
        if DEBUG:
            print("Hyperparameter tuning completed.")
        best_model = hyperparameter_tuner.best_estimator_
        best_params = hyperparameter_tuner.best_params_
        best_score = hyperparameter_tuner.best_score_
        tuning_duration = datetime.now() - hyperparameter_tuning_start_time
        if DEBUG:
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"Tuning duration: {tuning_duration}")
        # Save the best model
        # TODO
        if SAVE_MODELS:
            best_model.save(os.path.join("Models_Storage", f"{self.model_name}_best_model.joblib"))
            self.diagnostic_log["filename"] = f"{self.model_name}_best_model.joblib"
            self.diagnostic_log["saving_timestamp"] = datetime.now()
            self.results_log["saving_timestamp"] = self.diagnostic_log["saving_timestamp"]
        # try the model on the test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=REPORT_SETTING)
        precision = precision_score(y_test, y_pred, average=REPORT_SETTING)
        recall = recall_score(y_test, y_pred, average=REPORT_SETTING)
        roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        self.results_log["accuracy"] = accuracy
        self.results_log["f1_score"] = f1
        self.results_log["precision"] = precision
        self.results_log["recall"] = recall
        self.results_log["roc_auc"] = roc_auc
        if DEBUG:
            classification_report_str = classification_report(y_test, y_pred)
            print(f"Best Model tested on the test set with accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report_str)
        # Save results to the log
        self.save_log_to_excel()
        if DEBUG and False:
            all_results =hyperparameter_tuner.cv_results_
            print("\nAll hyperparameter tuning results:")
            for key in all_results.keys():
                print(f"{key}: {all_results[key]}")
            
        return hyperparameter_tuner.cv_results_,best_model, best_params
    
    def run_simple(self,use_class_weights=False):
        
        print(f"Running experiment: {self.experiment_name}")
        G_train, G_test, y_train, y_test = self.split_data()
        # not really used
        # could be added to hyperparameter tuning
        if use_class_weights:
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = dict(zip(np.unique(y_train), class_weights))
            if DEBUG:
                print(f"Computed class weights: {class_weights_dict}")
            self.model.set_class_weights(class_weights_dict)
        else:
            if DEBUG:
                print("Not using class weights.")
        
        trained_model = self.fit_model(G_train, y_train)
        
        if DEBUG:
            print(f"Trained model: {trained_model}")
        if trained_model is None:
            print("Model training failed. Exiting experiment.")
            return None
        accuracy, y_pred, report = self.test_model(G_test, y_test)
        print(f"Model {self.model_name} achieved accuracy: {accuracy:.4f}")
        # Save results
        self.save_log_to_excel()
        return accuracy, report
    
    def run_kfold(self, k=5):
        print(f"Running K-Fold experiment: {self.experiment_name} with k={k}")
        G_train , y_train = self.dataset.data
        accuracy, report = self.train_test_kfold(k=k)
        print(f"K-Fold experiment completed with mean accuracy: {accuracy:.4f}")
        # Save results
        self.save_log_to_excel()
        return accuracy, report
    