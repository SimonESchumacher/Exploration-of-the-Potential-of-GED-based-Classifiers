# this model, is supposed, to just give a random prediction
from sklearn.dummy import DummyClassifier
import numpy as np
import sys
import os
import traceback
sys.path.append(os.path.join(os.getcwd(), 'Models'))
import random as rnd
from graph_classifier import graph_classifier
DEBUG = False  # Set to True for debug prints
class random_classifier(graph_classifier):
    model_specific_iterations = 10
    def __init__(self, random_state=None,strategy='uniform',constant=None,attributes=None):
        self.random_state = random_state
        self.strategy = strategy
        if strategy == 'constant' and constant is None:
            raise ValueError("If strategy is 'constant', you must provide a constant value.")
        self.constant = constant
        if self.random_state is None:
            rnd.seed(None)  # Use system time or os.urandom for randomness
            self.random_state = rnd.randint(0, 10000)  # Random state for reproducibility
            if DEBUG:
                print(f"Random state not provided, using random value: {self.random_state}")
        classifier :DummyClassifier= DummyClassifier(strategy=strategy, random_state=self.random_state,constant=constant)
        if attributes is None:
            attributes = {
                "model_random_state": self.random_state,
                "model_strategy": self.strategy,
                "model_constant": self.constant if self.strategy == 'constant' else None
            }
        else:
            attributes["model_random_state"] = self.random_state
            attributes["model_strategy"] = self.strategy
            attributes["model_constant"] = self.constant if self.strategy == 'constant' else None
        super().__init__(
            classifier=classifier,
            model_name="RandomGuesser_" + str(self.strategy),
            modelattributes=attributes
        )
        if DEBUG:
            print(f"Initialized RandomGuesser with strategy={self.strategy}, random_state={self.random_state}")
            print(f"Model Name: {self.get_name}")
    def set_params(self, **params):
        # Iterate over provided parameters
        for parameter, value in params.items():
            if DEBUG:
                print(f"RandomGuesser: set_params: Setting {parameter} to {value}")

            # Handle parameters that belong to the RandomGuesser itself
            if parameter == 'strategy':
                self.strategy = value
                self.classifier.set_params(strategy=value)
            elif parameter == 'constant':
                self.constant = value
                self.classifier.set_params(constant=value)
            elif parameter == 'random_state':
                self.random_state = value
                rnd.seed(value)
            else:
                super().set_params(**{parameter: value})
        if DEBUG:
            print(f"RandomGuesser: set_params: Updated parameters: {self.get_params()}")
        return self
    def fit(self, X, y=None):
        # call the parent class's prepare_fit method
        self.prepare_fit(X, y)       
        self.classifier.fit(X, y=y)    
        # post-fit operations
        self.post_fit(X, y)
        return self
    
    def predict(self, X):
        # Check if the model is fitted
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted: {e}")
            traceback.print_exc()
            raise e
        # Ensure X is a valid array

        
        # Use the classifier to predict
        try:
            y_pred = self.classifier.predict(X)
        except Exception as e:
            # print traceback
            print("Error during prediction:")  
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {e}")
        
        return y_pred
    
    def predict_proba(self, X):
        # Check if the model is fitted
        
        # Ensure X is a valid array
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted: {e}")
            traceback.print_exc()
            raise e

        # Use the classifier to predict
        return self.classifier.predict_proba(X)
    def predict_both(self, X):
        probabilities = self.predict_proba(X)
        if self.classes_.shape[0] == 2:
            return self.classes_[np.argmax(probabilities, axis=1)], probabilities[:,0]
        else:
            return self.classes_[np.argmax(probabilities, axis=1)], probabilities 
         
    def __str__(self):  
        return f"RandomGuesser(strategy={self.strategy}, random_state={self.random_state}, constant={self.constant})"
    
    def to_string(self):
        return f"RandomGuesser(strategy={self.strategy}, random_state={self.random_state}, constant={self.constant})"
    @classmethod
    def get_param_grid(cls):
        """
        Returns a dictionary of parameters for grid search.
        """
        param_grid = graph_classifier.get_param_grid()
        param_grid.update({
            'strategy': ['most_frequent', 'stratified', 'uniform']
            # ,'constant': [0, 1]  # Only relevant if strategy is 'constant'
            })
        return param_grid
    @classmethod
    def get_random_param_space(cls) -> dict:
        """
        Returns a dictionary of parameters for random search.
        """
        param_space = graph_classifier.get_random_param_space()
        param_space.update({
            'strategy': ['most_frequent', 'stratified', 'uniform']
            # ,'constant': [0, 1]  # Only relevant if strategy is 'constant'
            })
        return param_space