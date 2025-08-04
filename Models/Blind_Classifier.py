# desperate Fitter 
# model, that gets random info and tries to fit it
# import dummy classifier
# import perceptron classifier
from sklearn.linear_model import Perceptron
# liabry to save Model:
import sys
import os
import traceback
sys.path.append(os.getcwd())
import random as rnd
DEBUG = False  # Set to True for debug prints
from Models.Graph_Classifier import GraphClassifier

class Blind_Classifier(GraphClassifier):
    def __init__(self,random_state=42,attributes=None):
        classifier = Perceptron(random_state=random_state)
        if attributes is None:
            attributes = {
                "random_state": random_state
            }
        else:
            attributes["random_state"] = random_state
        
        super().__init__(classifier=classifier, model_name="DesperateFitter",modelattributes=attributes)
        if DEBUG:
            print(f"Initialized DesperateFitter")
    def get_parmas(self, deep=True):
        return super().get_parmas(deep)
    def set_params(self, **params):
        # Iterate over provided parameters
        for parameter, value in params.items():
            if DEBUG:
                print(f"DesperateFitter: set_params: Setting {parameter} to {value}")
            if parameter == "random_state":
                self.classifier.random_state = value
            else:
                super().set_params(**{parameter: value})
        if DEBUG:
            print(f"DesperateFitter: set_params: Finished setting parameters")
        return self
    def fit(self, X, y=None):
        self.prepare_fit(X, y)
        # fill every entry in the shape of X with random values
        X = [[rnd.random(),rnd.random() ]for _ in X]

        self.classifier.fit(X, y)
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
        print("successfully checked if model is fitted")
        # Ensure X is a valid array
        X = [[rnd.random(),rnd.random() ]for _ in X]
        
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
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted: {e}")
            traceback.print_exc()
            raise e
        print("successfully checked if model is fitted")
        # Ensure X is a valid array
        X = [[rnd.random(),rnd.random() ]for _ in X]

        
        # Use the classifier to predict
        try:
            y_pred = self.classifier.predict_proba(X)
        except Exception as e:
            # print traceback
            print("Error during prediction:")  
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {e}")
        
        return y_pred
    def __str__(self):
        return f"DesperateFitter with random_state={self.classifier.random_state}"
    def to_string(self):
        return self.__str__()
    @classmethod
    def get_param_grid(cls):
        return super().get_param_grid()
