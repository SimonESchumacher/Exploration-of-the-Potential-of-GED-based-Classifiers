# K-NN Classifer 
# imports 
import traceback
import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.stats import randint
from typing import Dict, Any
from config_loader import get_conifg_param

# imports from custom modules
import sys
import os
sys.path.append(os.getcwd())
from Models.Graph_Classifier import GraphClassifier
DEBUG = get_conifg_param('KNN', 'debuging_prints')  # Set to False to disable debug prints

class KNN(GraphClassifier):
    def __init__(self, n_neighbors=1, weights='uniform', leaf_size=30,
                  metric=None,metric_name="unspecified",random_state=None,
                  attributes:dict=dict(), **kwargs): 
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.metric = metric
        if metric == "precomputed":
            self.algorithm = 'brute'  # 'brute' is required for precomputed metrics
        else:
            self.algorithm = 'auto'  # Let sklearn choose the best algorithm
        self.metric_name = metric_name
        self.random_state = random_state
        classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm,
                                          leaf_size=self.leaf_size, metric=self.metric)
        if attributes is None:
            attributes = {
                "n_neighbors": self.n_neighbors,
                "weights": self.weights,
                "metric": self.metric_name,
            }
        else:
            attributes["n_neighbors"] = self.n_neighbors
            attributes["weights"] = self.weights
            attributes["metric"] = self.metric_name
        model_name = kwargs.pop("name", "KNN")
        super().__init__(
            classifier=classifier,
            model_name=model_name,
            modelattributes=attributes,
            **kwargs
        )
        
        if DEBUG:
            print(f"Initialized KNNClassifier with n_neighbors={self.n_neighbors}, weights={self.weights}")
            print(f"Model Name: {self.get_name}")
    
    def fit_transform(self, X, y=None):
        return self.metric.fit_transform(X, y)
    def transform(self, X):
        return self.metric.transform(X)
    def get_params(self, deep=True):
        return super().get_params(deep)
    def set_params(self, **params):
        """
        Set the parameters of this estimator and its underlying classifier.
        Uses the underlying classifier's set_params for all matching parameters.
        """
        for parameter, value in params.items():
            if DEBUG:
                print(f"KNN: set_params: Setting {parameter} to {value}")
            if parameter == 'metric':
                self.metric = value
                # If metric changes to 'precomputed', update the algorithm accordingly
                if self.metric == 'precomputed':
                    self.algorithm = 'brute'
                else:
                    self.algorithm = 'auto'
                self.classifier.set_params(metric=self.metric, algorithm=self.algorithm)
            # Directly set attribute if it exists
            if hasattr(self, parameter):
                setattr(self, parameter, value)
                # If the parameter also exists in the classifier, update it there too
                if hasattr(self.classifier, parameter):
                    self.classifier.set_params(**{parameter: value})
        

        return self
    def fit(self, X, y=None):
        """
        Fits the KNN classifier model to the provided graph data.
        """
        if DEBUG:
            print("Fitting KNNClassifier...")
        self.prepare_fit(X, y)
        X =self.fit_transform(X, y)
        self.classifier.fit(X, y)
        self.post_fit(X, y)
        if DEBUG:
            print("KNNClassifier fitted successfully.")
        return self
    def predict(self, X):
        """
        Predicts class labels for graphs in X using the fitted KNN classifier.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            
            y_pred = self.classifier.predict(X)
            
        except Exception as e:
            print(f"Error during KNN prediction: {e}")
            traceback.print_exc()
            raise e
        return y_pred
    def predict_proba(self, X):
        """
        Predicts class probabilities for graphs in X using the fitted KNN classifier.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            y_proba = self.classifier.predict_proba(X)
        except Exception as e:
            print(f"Error during probability prediction: {e}")
            print(self.attributes)
            traceback.print_exc()
            raise e
        return y_proba
    def predict_both(self, X):
        probabilities = self.predict_proba(X)
        if self.classes_.shape[0] == 2:
            return self.classes_[np.argmax(probabilities, axis=1)], probabilities[:,0]
        else:
            return self.classes_[np.argmax(probabilities, axis=1)], probabilities
    def save(self, filename):
        """
        Saves the fitted KNN classifier model to a file.
        """
        if DEBUG:
            print(f"Saving KNNClassifier model to {filename}")
        joblib.dump(self, filename=filename)
    @classmethod
    def load(cls, filename):
        """
        Loads a KNN classifier model from a file.
        """
        if DEBUG:
            print(f"Loading KNNClassifier model from {filename}")
        return joblib.load(filename)
    def __str__(self):
        return f"{self.metric_name} - ({self.n_neighbors})-NN"
        
    def to_string(self):
        return f"{self.metric_name} - ({self.n_neighbors})-NN"
        

    @classmethod
    def get_param_grid(cls):
        param_grid = GraphClassifier.get_param_grid()
        param_grid.update({
            'n_neighbors': [1, 3, 5],
            'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            # 'leaf_size': [10, 20, 30, 40, 50],
            'metric': ['euclidean', 'precomputed']
            # 'metric': ['euclidean','precomputed']  # KNN with precomputed metric
        })

        return param_grid
    @classmethod
    def get_random_param_space(cls) -> Dict[str, Any]:
        param_space = super().get_random_param_space()
        param_space.update({
            'n_neighbors': randint(get_conifg_param('Hyperparameter_fields', 'min_neighbors', type='int'),
                                    get_conifg_param('Hyperparameter_fields', 'max_neighbors', type='int')),
            'weights': ['uniform', 'distance'],
            'leaf_size': randint(10, 50)
            # 'metric': ['precomputed', 'euclidean']
            # 'metric': ['euclidean','precomputed']  # KNN with precomputed metric
        })
        return param_space
