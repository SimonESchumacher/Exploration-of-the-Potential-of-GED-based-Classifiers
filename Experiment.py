# Experiment
# Imports
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from time import time
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  # For progress bar
import traceback
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from scipy import stats
import multiprocessing
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

        # Update filename in the original log_dict for internal reference
    
    # not done
    def __str__(self):
        return f"Experiment(name={self.experiment_name}, dataset={self.dataset_name}, model={self.model_name})"
 
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
            y_pred,y_score = self.model.predict_both(G_test)
            return y_pred, y_score
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
        y_pred,y_score = self.inner_model_predict(G_test)
        self.results_log["testing_duration"] = datetime.now() - test_start_time
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        try:
            roc_auc = roc_auc_score(y_test, y_score=y_score, labels=self.model.classes_, multi_class='ovr')
        except np.AxisError:
            roc_auc = 0.0
        precision = precision_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        recall = recall_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        # maybe i shoudl split this to a defirent file and make a diffrence between k_fold and simple
        self.results_log["accuracy"] = accuracy
        self.results_log["f1_score"] = f1
        self.results_log["roc_auc"] = roc_auc
        self.results_log["precision"] = precision
        self.results_log["recall"] = recall
        self.results_log["Action"] = "Simple Train-Test Split"
        self.results_log["Calculator_name"] = self.ged_calculator.get_Name() if self.ged_calculator else "None"
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
            y_pred,y_score = self.inner_model_predict(X_test)
            durattions_test.append((datetime.now() - test_start_time).total_seconds())

            accuracies.append(accuracy_score(y_test_fold, y_pred))
            f1_scores.append(f1_score(y_test_fold, y_pred, average=REPORT_SETTING, zero_division=0.0))
            roc_aucs.append(roc_auc_score(y_test_fold, y_score, labels=self.model.classes_, multi_class='ovr'))
            precisions.append(precision_score(y_test_fold, y_pred, average=REPORT_SETTING, zero_division=0.0))
            recalls.append(recall_score(y_test_fold, y_pred, average=REPORT_SETTING, zero_division=0.0))
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
        self.results_log["Action"] = "K-Fold Cross-Validation"
        self.results_log["Calculator_name"] = self.ged_calculator.get_Name() if self.ged_calculator else "NO Calculator"
        print(f"Model {self.model_name} trained and tested with K-Fold cross-validation (k={k}).")
        print(f"Mean Accuracy: {self.results_log['accuracy']}")
        return np.mean(accuracies) ,classification_report_str
    
    def run_hyperparameter_tuning(self, tuning_method='grid',scoring='accuracy',cv=5, verbose=2, n_jobs=5):



        X_train, X_test, y_train, y_test = self.dataset.train_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        if tuning_method == 'grid':
            param_grid = self.model.get_param_grid()
        elif tuning_method == 'random':
            param_grid = self.model.get_random_param_space()
        else:
            param_grid = {}
        # currently not in use, porbaly not needed and not a smart idea to use it
        if DATASET_HYPERPARAM:
            param_grid.update(self.dataset.get_param_grid())
        if CALCULATOR_HYPERPARAM and self.ged_calculator is not None:
            param_grid.update(self.ged_calculator.get_param_grid())


        if DEBUG:
            print(f"Starting hyperparameter tuning for {self.model_name} \n on dataset {self.dataset_name} with parameters: \n {param_grid}")
        hyperparameter_tuning_start_time = datetime.now()
        if tuning_method == 'grid':
            hyperparameter_tuner = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs, error_score='raise', refit=scoring)
        elif tuning_method == 'random':
            hyperparameter_tuner = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=self.model.random_search_iterations(), scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs, refit=scoring)
        # if DEBUG:
            # print("stating hyperparameter tuning...")  
        hyperparameter_tuner.fit(X_train, y_train)
        # if DEBUG:
            # print("Hyperparameter tuning completed.")
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
        # y_score =
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        precision = precision_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        recall = recall_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        try:
            roc_auc = roc_auc_score(y_test, y_pred, labels=self.model.classes_, multi_class='ovr')
        except np.AxisError:
            roc_auc = 0.0
        self.results_log["accuracy"] = accuracy
        self.results_log["f1_score"] = f1
        self.results_log["precision"] = precision
        self.results_log["recall"] = recall
        self.results_log["roc_auc"] = roc_auc
        self.results_log["Action"] = f"Hyperparameter Tuning ({tuning_method})"
        self.results_log["Calculator_name"] = self.ged_calculator.get_Name() if self.ged_calculator else "no Calculator"

        if DEBUG:
            classification_report_str = classification_report(y_test, y_pred)
            print(f"Best Model tested on the test set with accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report_str)
        # Save results to the log
        self.save_log_to_excel()
        if DEBUG:
            all_results = hyperparameter_tuner.cv_results_
            print("\nAll hyperparameter tuning results:")
            for key in all_results.keys():
                print(f"{key}: {all_results[key]}")

        return hyperparameter_tuner.cv_results_, best_model, best_params
    
    def run_simple(self,use_class_weights=False):
        
        print(f"Running experiment: {self.experiment_name}")
        G_train, G_test, y_train, y_test = self.dataset.train_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
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
    
    def run_speed_test(self):
        start_time = pd.Timestamp.now()
        G_train, G_test, y_train, y_test = self.dataset.train_test_split(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        self.inner_model_fit(G_train, y_train)
        trained_model = self.model
        if trained_model is None:
            print("Model training failed. Exiting experiment.")
            return None
        self.inner_model_predict(G_test)
        # Save results
        training_duration = pd.Timestamp.now() - start_time
        return training_duration


    def overfitting_simple_run(self,X_train, X_test, y_train, y_test,test_DF):
        fit_start_time = datetime.now()
        self.inner_model_fit(X_train, y_train)
        y_test_pred, y_test_score = self.inner_model_predict(X_test)
        duration = datetime.now() - fit_start_time
        test_DF["Best_model_duration"] = str(duration)
        
        accuracy_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred, average=REPORT_SETTING, zero_division=0.0)
        try:
            roc_auc_test = roc_auc_score(y_test, y_test_score, labels=self.model.classes_, multi_class='ovr')
        except np.AxisError:
            roc_auc_test = 0.0
        classification_report_str = classification_report(y_test, y_test_pred)


        # now test on train set
        y_train_pred, y_train_score = self.inner_model_predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred, average=REPORT_SETTING, zero_division=0.0)
        try:
            roc_auc_train = roc_auc_score(y_train, y_train_score, labels=self.model.classes_, multi_class='ovr')
        except np.AxisError:
            roc_auc_train = 0.0

        # caclulate the diffrence of f1 , roc and accuracy between train and test
        diff_accuracy = accuracy_train - accuracy_test
        diff_f1 = f1_train - f1_test
        diff_roc_auc = roc_auc_train - roc_auc_test
        test_DF["OF_diff_accuracy"] = diff_accuracy
        test_DF["OF_diff_f1"] = diff_f1
        test_DF["OF_diff_roc_auc"] = diff_roc_auc

        test_DF["OF_test_accuracy"] = accuracy_test
        test_DF["OF_test_f1"] = f1_test
        test_DF["OF_test_roc_auc"] = roc_auc_test

        test_DF["OF_train_accuracy"] = accuracy_train
        test_DF["OF_train_f1"] = f1_train
        test_DF["OF_train_roc_auc"] = roc_auc_train

        test_DF["Classifcation_report"] = classification_report_str

        # Save results
        return classification_report_str
    
    def run_nested_cross_validation(self, outer_cv=5, inner_cv=5, num_trials=3, scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'], verbose=0, n_jobs=-1, random_seed=RANDOM_STATE, search_method="random",test_DF=None,stratify=False):
        random_gen = random.Random(random_seed)
        outer_accuracies = []
        outer_f1_scores = []
        outer_roc_aucs = []
        outer_precisions = []
        outer_recalls = []
        outer_durations = []
        for trial in range(num_trials):
            if DEBUG:
                print(f"Starting trial {trial + 1}/{num_trials} of nested cross-validation.")
            inner_cv_object = None
            outer_cv_object = None
            if stratify:
                inner_cv_object = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_gen.randint(0, 1000))
                outer_cv_object = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_gen.randint(0, 1000))
            else:
                inner_cv_object = KFold(n_splits=inner_cv, shuffle=True, random_state=random_gen.randint(0, 1000))
                outer_cv_object = KFold(n_splits=outer_cv, shuffle=True, random_state=random_gen.randint(0, 1000))
            tuner = None
            if search_method == "grid":
                param_grid = self.model.get_param_grid()
                tuner = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=inner_cv_object, verbose=verbose, n_jobs=n_jobs,) 
            elif search_method == "random":
                param_grid = self.model.get_random_param_space()
                tuner = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=self.model.random_search_iterations(), scoring=scoring, cv=inner_cv_object, verbose=verbose, n_jobs=n_jobs,refit=scoring[0])
            else:
                raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'.")
            nested_score = cross_val_score(tuner, X=self.dataset.data[0], y=self.dataset.data[1], cv=outer_cv_object, scoring=scoring[0], n_jobs=n_jobs, )
            outer_accuracies.append(np.mean(nested_score['test_accuracy']))
            outer_f1_scores.append(np.mean(nested_score['test_f1_macro']))
            outer_roc_aucs.append(np.mean(nested_score['test_roc_auc']))
            outer_precisions.append(np.mean(nested_score['test_precision']))    
            outer_recalls.append(np.mean(nested_score['test_recall']))
            outer_durations.append(np.mean(nested_score['fit_time']))
    
    def score_best_model(self,X_test,y_test):
        y_pred, y_score = self.inner_model_predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        precision = precision_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        recall = recall_score(y_test, y_pred, average=REPORT_SETTING, zero_division=0.0)
        try:
            roc_auc = roc_auc_score(y_test, y_score, labels=self.model.classes_, multi_class='ovr') if y_score is not None else 0.0
        except np.AxisError:
            roc_auc = 0.0
        classification_report_str = classification_report(y_test, y_pred, zero_division=0.0)
        # get confusion matrix as a string
        return accuracy,f1,precision,recall,roc_auc,classification_report_str
    
    def extensive_model_score(self,best_params,X_train,X_test,y_train,y_test,classes):
        # create a new model instance with the best params
        results_dict = dict()
        self.model = self.model.__class__(**best_params)
        # get the current time in seconds and milliseconds
        start_time = time.time()
        self.inner_model_fit(X_train, y_train)
        done_training_time = time.time()
        
        training_duration = done_training_time - start_time
        results_dict["training_duration"] = str(training_duration)
        y_pred_test, y_score_test = self.inner_model_predict(X_test)
        testing_duration = time.time() - done_training_time
        results_dict["testing_duration"] = str(testing_duration)

        # measure test scores
        results_dict["accuracy_test"] = accuracy_score(y_test, y_pred_test)
        results_dict["f1_macro_test"] = f1_score(y_test, y_pred_test, average='macro', zero_division=0.0)
        results_dict["f1_micro_test"] = f1_score(y_test, y_pred_test, average='micro', zero_division=0.0)
        results_dict["f1_weighted_test"] = f1_score(y_test, y_pred_test, average='weighted', zero_division=0.0)
        results_dict["precision_test"] = precision_score(y_test, y_pred_test, average=REPORT_SETTING, zero_division=0.0)
        results_dict["recall_test"] = recall_score(y_test, y_pred_test, average=REPORT_SETTING, zero_division=0.0)
        # figure out if this is a binary or multi class classification
        if len(classes) == 2:
            try:
                results_dict["roc_auc_test"] = roc_auc_score(y_test, y_score_test[:, 1], pos_label=classes[1]) if y_score_test is not None else 0.0
            except np.AxisError:
                results_dict["roc_auc_test"] = 0.0
            # get the confusion matrix entries for binary classification
            results_dict["tn"], results_dict["fp"], results_dict["fn"], results_dict["tp"] = confusion_matrix(y_test, y_pred_test, labels=classes).ravel()
        else:
            try:
                results_dict["roc_auc_test"] = roc_auc_score(y_test, y_score_test, labels=classes, multi_class='ovr') if y_score_test is not None else 0.0
            except np.AxisError:
                results_dict["roc_auc_test"] = 0.0
            # get the f1 score for each class
            f1_per_class = f1_score(y_test, y_pred_test, labels=classes, average=None, zero_division=0.0)
            for i, cls in enumerate(classes):
                results_dict[f"f1_class_{cls}_test"] = f1_per_class[i]
        # get the classification report as a string
        results_dict["classification_report_test"] = classification_report(y_test, y_pred_test, zero_division=0.0)

        # test for overfitting
        y_pred_train, y_score_train = self.inner_model_predict(X_train)
        results_dict["accuracy_train"] = accuracy_score(y_train, y_pred_train)
        results_dict["f1_macro_train"] = f1_score(y_train, y_pred_train, average='macro', zero_division=0.0)
        results_dict["f1_micro_train"] = f1_score(y_train, y_pred_train, average='micro', zero_division=0.0)
        results_dict["f1_weighted_train"] = f1_score(y_train, y_pred_train, average='weighted', zero_division=0.0)
        if len(classes) == 2:
            try:
                results_dict["roc_auc_train"] = roc_auc_score(y_train, y_score_train[:, 1], pos_label=classes[1]) if y_score_train is not None else 0.0
            except np.AxisError:
                results_dict["roc_auc_train"] = 0.0
        else:
            try:
                results_dict["roc_auc_train"] = roc_auc_score(y_train, y_score_train, labels=classes, multi_class='ovr') if y_score_train is not None else 0.0
            except np.AxisError:
                results_dict["roc_auc_train"] = 0.0
        return results_dict





    
    def run_inner_hyperparameter_tuning(self,X_train,y_train,Y_test,y_test,inner_cv=5,scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'], verbose=0, n_jobs=-1, random_seed=RANDOM_STATE, search_method="random",test_trail=False,fold_index=None):
        random_gen = random.Random(random_seed)
        if search_method == "grid":
            param_grid = self.model.get_param_grid()
            tuner = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring=scoring, cv=inner_cv, verbose=verbose, n_jobs=n_jobs,) 
        elif search_method == "random":
            param_grid = self.model.get_random_param_space()
            iterations = self.model.random_search_iterations() if not test_trail else 1
            tuner = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=iterations, scoring=scoring[0], cv=inner_cv, verbose=verbose, n_jobs=n_jobs,error_score="raise",refit=scoring[0])
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'grid' or 'random'.")
        print(f"starting inner tuning fold {fold_index}")
        tuner.fit(X_train,y_train)
        print(f"completed inner tuning fold {fold_index}")
        best_model = tuner.best_estimator_
        best_params = tuner.best_params_
        best_score = tuner.best_score_
        self.model = best_model
        results_dict = tuner.cv_results_

        accuracy,f1,precision,recall,roc_auc,classification_report_str = self.score_best_model(Y_test,y_test)
        # scores = self.extensive_model_score(best_params,X_train,Y_test,y_train,y_test,classes=self.model.classes_)

        # return best_model,best_params,scores, results_dict
        return best_model, best_params, best_score, accuracy,f1,precision,recall,roc_auc,classification_report_str, results_dict

    

    def run_joblib_parallel_nested_cv(self,outer_cv=5,inner_cv=5,num_trials=3,scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'],tuning_metric="f1_macro", verbose=0, n_jobs=-1, random_seed=RANDOM_STATE, search_method="random",should_print=DEBUG,get_all_results=True,test_trail=False):
        # insert the main scoring metric at the first position
        if tuning_metric != scoring[0]:
            scoring.remove(tuning_metric)
            scoring.insert(0, tuning_metric)
        X, y = self.dataset.data
        random_gen = random.Random(random_seed)
        test_Dict = dict()
        if should_print:
           print("\n--------------------------------------------------------------------")
           print(f"Running extensive test for model:\n {self.model_name}")
        estimated_test_duration = None
        test_Dict["model_name"] = self.model_name
        test_Dict["Calculator_name"] = self.model.get_calculator().get_Name() if self.model.get_calculator() else "None"
        try:
            estimated_test_duration,duration1train =self.estimate_nested_cv_time(cv=outer_cv,num_trials=num_trials,search_method=search_method)
        except Exception as e:
            print(f"Error occurred while running speed test: {e}")
            traceback.print_exc()
            test_Dict["Error source"] = "Speed Test"
            test_Dict["Error"] = str(e)
            return test_Dict
        test_Dict["train_test_duration"] = str(duration1train)
        if should_print:
            print(f"Estimated test duration: {estimated_test_duration}")
       



        accuracy_scores = []
        f1_scores = []
        roc_auc_scores = []
        precision_scores = []
        recall_scores = []
        time_Start = pd.Timestamp.now()
        results_df = pd.DataFrame()
        # build a progress bar which tracks the progress of both loops.

        if should_print:
            print(f"{self.model_name} start nested CV at {pd.Timestamp.now()}")
            # get the Parameter Grid
            if search_method == "grid":
                param_grid = self.model.get_param_grid()
                print(f"Parameter grid: {param_grid}")
            elif search_method == "random":
                param_grid = self.model.get_random_param_space()
                print(f"Search Space: {param_grid}")
        maximum_jobs = min(n_jobs, outer_cv*num_trials) if n_jobs != -1 else outer_cv*num_trials
        inner_jobs = n_jobs // maximum_jobs if n_jobs != -1 else 1
        delayed_calls =[
            joblib.delayed(self.run_inner_hyperparameter_tuning)(X_train, y_train_fold, X_test, y_test_fold,
                                                                inner_cv=inner_cv, scoring=scoring, verbose=verbose, n_jobs=inner_jobs,
                                                                random_seed=random_gen.randint(0, 1000),
                                                                search_method=search_method,test_trail=test_trail,
                                                                fold_index=i
                                                                )
            for i, (X_train, X_test, y_train_fold, y_test_fold) in enumerate(self.dataset.split_k_fold(k=inner_cv, random_state=random_gen.randint(0, 1000),repeat=num_trials,stratify=True))
        ]
        all_folds_results = joblib.Parallel(n_jobs=maximum_jobs,verbose=1)(delayed_calls)
        all_scores_list = pd.DataFrame()
        # for fold_index, (best_model,best_params, scores, results_dict) in enumerate(all_folds_results):
        #     if fold_index ==0:
        #         test_Dict["best_params"] = str(best_params)
        #         test_Dict["best_score"] = scores["accuracy_test"]
        #         test_Dict["classification_report_train"] = results_dict["classification_report_test"]


        #     # Append the results_dict to the results_df
        #     results_dict['fold_index'] = fold_index
        #     results_df = pd.concat([results_df, pd.DataFrame([results_dict])], ignore_index=True)
        #     all_scores_list = pd.concat([all_scores_list, pd.DataFrame([scores])], ignore_index=True)
        
        for fold_index, (best_model, best_params, best_score, accuracy_train,f1_train,precision_train,recall_train,roc_auc_train,classification_report_train, results_dict) in enumerate(all_folds_results):
            if fold_index ==0:
                test_Dict["best_params"] = str(best_params)
                test_Dict["best_score"] = best_score
                test_Dict["classification_report_train"] = classification_report_train
            accuracy_scores.append(accuracy_train)
            f1_scores.append(f1_train)
            roc_auc_scores.append(roc_auc_train)
            precision_scores.append(precision_train)
            recall_scores.append(recall_train)

            # Append the results_dict to the results_df
            results_dict['fold_index'] = fold_index
            results_df = pd.concat([results_df, pd.DataFrame(results_dict)], ignore_index=True)



        time_End = pd.Timestamp.now()
        total_duration = time_End - time_Start
        if should_print:
            print(f"Nested CV completed in {total_duration}.\n")
            print(f"Time {pd.Timestamp.now()}")
            print("--------------------------------------------------------------------\n")
        if test_Dict is not None:
            erroracknowledgment = ERRORINTERVAL_SETTING
        if erroracknowledgment == "std":
            test_Dict["k_fold_accuracy"] = np.mean(accuracy_scores)
            test_Dict["k_fold_acc_std"] = np.std(accuracy_scores)
            test_Dict["k_fold_f1_score"] = np.mean(f1_scores)
            test_Dict["k_fold_f1_std"] = np.std(f1_scores)
            test_Dict["k_fold_roc_auc"] = np.mean(roc_auc_scores)
            test_Dict["k_fold_roc_auc_std"] = np.std(roc_auc_scores)
            test_Dict["k_fold_precision"] = np.mean(precision_scores)
            test_Dict["k_fold_recall"] = np.mean(recall_scores)
        elif erroracknowledgment == "confidence interval":
            # Calculate confidence intervals for each metric
            n = len(accuracy_scores)
            confidence = 0.95
            z_score = stats.norm.ppf((1 + confidence) / 2)
            test_Dict["k_fold_accuracy"] = np.mean(accuracy_scores)
            test_Dict["K_fold_acc_CI"] = z_score * np.std(accuracy_scores) / np.sqrt(n)
            test_Dict["k_fold_f1_score"] = np.mean(f1_scores)
            test_Dict["K_fold_f1_CI"] = z_score * np.std(f1_scores) / np.sqrt(n)
            test_Dict["k_fold_roc_auc"] = np.mean(roc_auc_scores)
            test_Dict["K_fold_roc_auc_CI"] = z_score * np.std(roc_auc_scores) / np.sqrt(n)
            test_Dict["k_fold_precision"] = np.mean(precision_scores)
            test_Dict["k_fold_recall"] = np.mean(recall_scores)
        else:
            raise ValueError(f"Unknown error acknowledgment method: {erroracknowledgment}. Use 'std', 'confidence interval', or 'none'.")
        test_Dict["nested_total_duration"] = str(total_duration)
        if get_all_results:
            results_dir = os.path.join("configs", "results", "Hyperparameter_tuning_results")
            results_path = os.path.join(results_dir, f"HP_{pd.Timestamp.now().strftime('%Y%m%d')}_{self.model_name}_{self.dataset_name}.xlsx")
            os.makedirs(results_dir, exist_ok=True)
            results_df.to_excel(results_path, index=False)
        return test_Dict
        


    def estimate_nested_cv_time(self,cv=5,num_trials=3,search_method="grid"):
        duration1train =self.run_speed_test()
        tested_configs_num= 0
        if search_method == "grid":
            param_grid = self.model.get_param_grid()
            tested_configs_num = 1
            for key in param_grid:
                tested_configs_num *= len(param_grid[key])
        elif search_method == "random":
            tested_configs_num = self.model.random_search_iterations()
        total_combinations = cv * cv * tested_configs_num * num_trials
        estimated_test_duration = duration1train * total_combinations
        print(f" {self.model_name}: {duration1train} x {total_combinations}  = Estimated nested CV duration: {estimated_test_duration}")
        return estimated_test_duration, duration1train
      



    def get_estimated_tuning_time(self,cv=5,search_method="grid",max_combinations=-1):
        duration1train =self.run_speed_test()
        if search_method == "grid":
            param_grid = self.model.get_param_grid()
            total_combinations = cv
            for key in param_grid:
                total_combinations *= len(param_grid[key])
            total_combinations += cv +2
            # multipy all posibilities for the param grid times 5       
        elif search_method == "random":
            total_combinations = self.model.random_search_iterations()*cv+cv+2

        estimated_test_duration = duration1train * total_combinations
        print(f" {self.model_name}: {duration1train} x {total_combinations} combinations = Estimated tuning duration: {estimated_test_duration}")
        return estimated_test_duration

    def run_extensive_cross_validation(self,test_DF,should_print=False,cv=5, verbose=0, n_jobs=-1,random_seed=RANDOM_STATE,get_all_tuning_results=False,search_method="grid"):
        # get a random number generator with the seed
        random_gen = random.Random(random_seed)


        if should_print:
           print("\n--------------------------------------------------------------------")
           print(f"Running extensive cross-validation for model:\n {self.model_name}")
        duration1train = None
        test_DF["model_name"] = self.model_name
        test_DF["Calculator_name"] = self.model.get_calculator().get_Name() if self.model.get_calculator() else "None"
        try:
            duration1train =self.run_speed_test()
        except Exception as e:
            print(f"Error occurred while running speed test: {e}")
            test_DF["Error source"] = "Speed Test"
            test_DF["Error"] = str(e)
            return test_DF
        test_DF["train_test_duration"] = str(duration1train)
        if search_method == "grid":
            param_grid = self.model.get_param_grid()
            total_combinations = cv
            for key in param_grid:
                total_combinations *= len(param_grid[key])
            total_combinations += cv +1           
        elif search_method == "random":
            param_grid = self.model.get_random_param_space()
            total_combinations = self.model.random_search_iterations()*cv+cv+2
        estimated_test_duration = duration1train * total_combinations
        if should_print:
            print(f"Estimated test duration: {estimated_test_duration}")
        if should_print:
            print(f"Parameter grid: {param_grid}")
        # run the nested cross validation
        try:
            self.run_nested_cross_validation(outer_cv=cv, inner_cv=cv, num_trials=3, scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'], verbose=verbose, n_jobs=n_jobs, random_seed=random_gen.randint(0, 1000), search_method=search_method,test_DF=test_DF)
        except Exception as e:
            print(f"Error occurred while running nested cross-validation: {e}")
            test_DF["Error source"] = "Nested Cross-Validation"
            test_DF["Error"] = str(e)
            return test_DF
        if should_print:
            print("Extensive cross-validation completed.")
            print(f"Time {pd.Timestamp.now()}")
            print("--------------------------------------------------------------------\n")
        return test_DF




