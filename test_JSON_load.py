import sys
import os
import json

from Dataset import Dataset
from Experiment import experiment
# import descion tree classifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.join(os.getcwd(), 'Models'))
from WL_SVC_Kernel import SVC_WeisfeilerLehman
from Graph_Classifier import GraphClassifier
from random_model import RandomGuesser

MODEL_REGISTRY = {
    "SVC_WeisfeilerLehman": SVC_WeisfeilerLehman,
    "RandomGuesser": RandomGuesser,  # Assuming RandomGuesser is in random_model.py
    # Add other scikit-learn models or your custom models here
    # "RandomForestClassifier": RandomForestClassifier,
}
def load_model_from_JSON(model_config) -> GraphClassifier:
    model_id = model_config.get("model_id", "UnnamedModel")
    model_type_str = model_config.get("model_type")
    model_params = model_config.get("parameters", {})

    print(f"\n--- Running Model: {model_id} ({model_type_str}) ---")
    print(f"Parameters: {model_params}")

    model_class = MODEL_REGISTRY.get(model_type_str)
    if model_class is None:
        print(f"Error: Unknown model type '{model_type_str}'. Skipping this model.")
        return None

    model_instance = None
    try:
        # Dynamically instantiate the model
        # Pass random_state if the model accepts it
        if 'random_state' in model_params:
            model_instance = model_class(**model_params)
        else:
            # If random_state is a global param, try to pass it if model accepts
            # This requires checking the model's __init__ signature,
            # which is more complex. For simplicity, we'll only pass if in model_params.
            model_instance = model_class(**model_params)
        print(f"Model {model_id} instantiated successfully.")
        # print(f"Model Name: {model_instance.model_name}")
        print(f"Model: {model_instance.to_string()}")
        print(f"Model attributes: {model_instance.model_attributes}")


    except Exception as e:
        print(f"Error instantiating model '{model_id}': {e}")
        return None
    return model_instance

def parse_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Please create it with the required model configurations.")
        sys.exit(1)

def load_config(config_file):
    config = parse_config(config_file)
    models = config.get("models", [])
    datasets = config.get("datasets", [])
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    return models, datasets, test_size, random_state

# model={      
#     "model_id": "WL_SVM_Default",
#       "model_type": "SVC_WeisfeilerLehman",
#       "parameters": {
#         "n_iter": 5,
#         "C": 1.0,
#         "normalize_kernel": True
#       }
#         }
# model :GraphClassifier = load_model_from_JSON(model)
# MUTAG = Dataset(name="MUTAG", source="TUD", domain="molecular")
# experiment_instance = experiment(name="MUTAG_Experiment", dataset=MUTAG, datset_name="MUTAG", model=model, model_name=model.get_name)
# experiment_instance.run_simple()

models, datasets, test_size, random_state = load_config("configs/example_config.json")
results_summary = []
for dataset_config in datasets:
    dataset_name = dataset_config.get("name", "MUTAG")
    dataset_source = dataset_config.get("source", "TUD")
    dataset_domain = dataset_config.get("domain", "molecular")
    dataset = Dataset(name=dataset_name, source=dataset_source, domain="molecular")
    for model_config in models:
        model: GraphClassifier = load_model_from_JSON(model_config)
        if model is None:
            continue
        experiment_instance = experiment(name=f"{dataset_name}_{model.model_name}_Experiment",
                                         dataset=dataset,
                                         datset_name=dataset_name,
                                         model=model,
                                         model_name=model.get_name())
        accuracy,y_pred,report=experiment_instance.run_simple()
        results_summary.append(f"{model.model_name} scored accuracy {accuracy} on dataset: {dataset_name}")

for result in results_summary:
    print(result)
    print("\n")
