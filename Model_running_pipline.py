# Imports
from grakel.datasets import fetch_dataset
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import json
import pandas as pd
from Dataset import Dataset
CONFIG_FILE = "config.json"



RANDOM_STATE = 42
TEST_SIZE = 0.2
RESULTS_FILE = "results.xlsx"
DIAGNOSTIC_FILE = "diagnostic.xlsx"
DATASET_NAMES = ["MUTAG", "PROTEINS_full", "COLLAB", "IMDB-BINARY", "IMDB-MULTI"]
# load config file with the models, with teir specifcations
# load the config file
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        MODELS = config.get("models", [])
        DATASETS = config.get("datasets", DATASET_NAMES)
        TEST_SIZE = config.get("test_size", TEST_SIZE)
        DIAGNOSTIC_FILE = config.get("diagnostic_file", DIAGNOSTIC_FILE)
        RESULTS_FILE = config.get("results_file", RESULTS_FILE)
except FileNotFoundError:
    print(f"Config file {CONFIG_FILE} not found. Please create it with the required model configurations.")
    sys.exit(1)


class experiment:
    def __init__(self, name, dataset, datset_name, model, model_name=None):
        self.name = name
        self.dataset = dataset
        self.dataset_name = datset_name
        self.model = model
        self.model_name = model_name if model_name else model.__class__.__name__
    

    # not done
    def __str__(self):
        return f"Experiment(name={self.name}, dataset={self.dataset_name}, model={self.model_name})"
    
    def split_data(self):
        G_train, G_test, y_train, y_test = train_test_split(self.dataset.data, self.dataset.target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return G_train, G_test, y_train, y_test
    
    def fit_model(self, G_train, y_train):
        if hasattr(self.model, 'fit'):
            self.model.fit(G_train, y_train)
        else:
            raise ValueError("Model does not have a fit method.")
        print(f"Model {self.model_name} training on {len(G_train)} graphs.")
        self.model.fit(G_train, y_train)
        print(f"Model {self.model_name} trained successfully.")
        filename = f"{self.model_name}_trained_on_{self.dataset_name}.joblib"
        filepath = f"Models/{filename}"
        self.model.save(filepath)
        return self.model
    
    def test_model(self, G_test, y_test):
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(G_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model {self.model_name} tested on {len(G_test)} graphs with accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            return accuracy, y_pred,classification_report(y_test, y_pred)
        else:
            raise ValueError("Model does not have a predict method.")
    
    def run(self):
        print(f"Running experiment: {self.name}")
        G_train, G_test, y_train, y_test = self.split_data()
        trained_model = self.fit_model(G_train, y_train)
        accuracy, y_pred = self.test_model(G_test, y_test)
        
        # Save results
        results = {
            "experiment_name": self.name,
            "dataset": self.dataset_name,
            "model": self.model_name,
            "accuracy": accuracy
        }
        
        # Save results to Excel file
        if os.path.exists(RESULTS_FILE):
            df = pd.read_excel(RESULTS_FILE)
            df = df.append(results, ignore_index=True)
        else:
            df = pd.DataFrame([results])
        
        df.to_excel(RESULTS_FILE, index=False)
        
        # Save diagnostic information
        diagnostic_info = {
            "experiment_name": self.name,
            "dataset": self.dataset_name,
            "model": self.model_name,
            "accuracy": accuracy,
            "y_pred": y_pred.tolist(),
            "y_test": y_test.tolist()
        }
        
        with open(DIAGNOSTIC_FILE, 'a') as f:
            json.dump(diagnostic_info, f)
            f.write('\n')
        
        print(f"Experiment {self.name} completed successfully.")
        
    
def run_experiments():
    # TODO define the model_storage path
    # load results and dignostic files

    # Diagnostic file Collums (to be added):
    # model_name, dataset_name, training_time, testing_time, errors,saved_filename ,saving timestamp

    for dataset_name in DATASETS:
        Dataset_instance = Dataset(name=dataset_name)
        print(f"Dataset {dataset_name} loaded with {Dataset_instance.num_graphs} graphs and {Dataset_instance.num_classes} classes.")
        
        for model_config in MODELS:
            model_class = model_config.get("class")
            model_params = model_config.get("params", {})
            model_name = model_config.get("name", model_class.__name__)
            
            # Create the model instance
            model = model_class(**model_params)
            
            # Create an experiment instance
            experiment_instance = experiment(name=f"{model_name}_{dataset_name}", dataset=Dataset_instance, datset_name=dataset_name, model=model, model_name=model_name)
            
            # Run the experiment
            experiment_instance.run()