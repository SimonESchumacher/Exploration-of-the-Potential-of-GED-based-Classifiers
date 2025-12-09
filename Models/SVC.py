import pandas as pd
from sklearn.svm import SVC
import numpy as np
import sys
import os
import traceback
sys.path.append(os.getcwd())
from Grakel_Converter import  get_grakel_graphs_from_nx
from Models.graph_classifier import graph_classifier
from scipy.stats import loguniform
from config_loader import get_conifg_param
module="SVC"
DEBUG = get_conifg_param(module, 'debuging_prints')  # Set to False to disable debug prints
PROBABILITY_ESTIMATES = get_conifg_param(module, 'probability_estimates')  # Enable probability estimates for SVC
FITTING_TOLERANCE = get_conifg_param(module, 'fitting_tolerance', type='float')  # Tolerance for SVC fitting
CHACHE_SIZE = get_conifg_param(module, 'cache_size', type='int')  # Cache size for SVC
DEFAULT_SVC_RANDOM_STATE = get_conifg_param(module, 'default_random_state', type='int')  # Default random state for SVC
class support_vector_classifier(graph_classifier):
    model_specific_iterations = get_conifg_param("Hyperparameter_fields", "tuning_iterations", type="int")
    # Support Vector Machine Classifier for Graphs
    # with different Kernels
    def __init__(self, kernel_type="precomputed", C=1.0, random_state=None,kernelfunction=None,kernel_name="unspecified",class_weight=None,classes=[0,1],attributes=None, **kwargs):
        
        self.kernel_type = kernel_type
        #     self.kernel = "None"
        self.kernel_fuct = kernelfunction
        self.kernel_name = kernel_name
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.probability = PROBABILITY_ESTIMATES  # Enable probability estimates
        max_iter = kwargs.get('max_iter', get_conifg_param('SVC', 'max_iter', type='int'))
        classifier = SVC(kernel=self.kernel_type, C=self.C, random_state=self.random_state, class_weight=class_weight, probability=self.probability, tol=FITTING_TOLERANCE, cache_size=CHACHE_SIZE, max_iter=max_iter)
        default_attributes = {
            "Kernel_type": self.kernel_type,
            "Kernel": self.kernel_name,
            "Classifier_C": self.C,
            "model_random_state": self.random_state,
            "class_weight": self.class_weight
        }
        if attributes is None:
            attributes = default_attributes
        else:
            attributes.update(default_attributes)
        if kernel_name == "unspecified":
            kernel_name = "SVC"
        super().__init__(
            classifier=classifier,
            model_name=self.kernel_name,
            modelattributes=attributes,
            **kwargs
        )
        self.is_fitted_ = False
        if DEBUG:
            print(f"Initialized SupportVectorMachine with kernel={self.kernel_type}, C={self.C}, in child class")
            print(f"Model Name: {self.get_name}")
    # TODO: this does not feel like a good move
    def fit_transform(self, X, y=None):
        # X2 = [convert_nx_to_grakel_graph(x) for x in X]
        X = get_grakel_graphs_from_nx([x for x in X], node_label_tag="label", edge_label_tag="label")
        # convert generator to list
        return self.kernel_fuct.fit_transform(X,y)
    # TODO: this does not feel like a good move
    def transform(self, X):
        # X2 = [convert_nx_to_grakel_graph(x) for x in X]
        X = get_grakel_graphs_from_nx([x for x in X], node_label_tag="label", edge_label_tag="label")
        return self.kernel_fuct.transform(X)
    def get_params(self, deep=True):
        # params = super().get_params(deep=deep)
        params = {}
        params.update({
            "C": self.C,
            "kernel_type": self.kernel_type,
            "class_weight": self.class_weight,
        })
        return params
    def fit(self, X, y=None):
        """
        Fits the SVC model to the graph data.
        """

        if DEBUG:
            print("Fitting SVC model...")
        self.prepare_fit(X, y)
        X = self.fit_transform(X, y)
        start_time = pd.Timestamp.now()
        self.classifier.fit(X, y)
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        if duration > get_conifg_param('SVC', 'fit_warning_threshold', type='int'):
            print(f"Warning: SVC fitting took {duration} seconds, which is longer than expected.")
            print(self.get_params())
            # print(f"Fitting details: X shape: {X.shape}, y length: {len(y)}")
        # print(f"Completed fit {random_number}")
        self.post_fit(X, y)
        if DEBUG:
            
            print("SVC model fitted successfully.")
        # print(self.get_params())
        return self
    def predict(self, X):
        """
        Predicts the labels for the input graph data.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"SVC is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            if self.probability:
                y_pred = self.classifier.predict_proba(X)
                y_pred = self.classes_[np.argmax(y_pred, axis=1)]
            else:
                y_pred = self.classifier.predict(X)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            raise e
        return y_pred
    def predict_proba(self, X):
        """
        Predicts the class probabilities for the input graph data.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            y_conf = self.classifier.predict_proba(X)
        except Exception as e:
            print(f"Error during probability prediction: {e}")
            traceback.print_exc()
            raise e
        return y_conf
    def predict_both(self, X):
        if self.probability:
            probabilities = self.predict_proba(X)
            if self.classes_.shape[0] == 2:
                return self.classes_[np.argmax(probabilities, axis=1)], probabilities[:,0]
            else:
                return self.classes_[np.argmax(probabilities, axis=1)], probabilities
        else:
            # simply return 1 for the class we prectict and else 0
            predictions = self.predict(X)
            probabilities = np.zeros((len(predictions), len(self.classes_)))
            for i, pred in enumerate(predictions):
                class_index = np.where(self.classes_ == pred)[0][0]
                probabilities[i, class_index] = 1.0
            if self.classes_.shape[0] == 2:
                return predictions, probabilities[:,0]
            else:
                return predictions, probabilities
    def __str__(self):
        return (f"SVC_{self.kernel_name}(kernel_type={self.kernel_type}, C={self.C}, "
                f"random_state={self.random_state})")
    def to_string(self):
            return (f"SVC_{self.kernel_name}(kernel_type={self.kernel_type}, C={self.C}, "
                f"random_state={self.random_state})")
    @classmethod
    def get_param_grid(cls):
        param_grid = graph_classifier.get_param_grid()
        param_grid.update({
            'C': [0.1, 0.5,0.25],
            # 'kernel_type': ['poly', 'linear'],
            "kernel_type": ['precomputed'],
            # 'kernel_type': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            # 'class_weight': [None, 'balanced'],
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = graph_classifier.get_random_param_space()
        param_space.update({
            'C': loguniform(a=get_conifg_param('Hyperparameter_fields', 'lower_C'), b=get_conifg_param('Hyperparameter_fields', 'upper_C')),
            # 'kernel_type': ['poly', 'linear', 'rbf', 'sigmoid'],
            'kernel_type': ['precomputed'],
            'class_weight': [None, 'balanced'],
        })
        return param_space
        

     
