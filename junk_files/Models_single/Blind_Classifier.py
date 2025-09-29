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
    def __init__(self,random_state=42,attributes=dict(),**kwargs):
        self.random_state = random_state
        classifier = Perceptron(random_state=random_state)
        attributes.update({
            "random_state": self.random_state,
        })
        super().__init__(classifier=classifier, model_name="Blind_Classifier",modelattributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized Blind_Classifier")
    def get_params(self, deep=True):
        return super().get_params(deep)
    def set_params(self, **params):
        # Iterate over provided parameters
        for parameter, value in params.items():
            if DEBUG:
                print(f"Blind_Classifier: set_params: Setting {parameter} to {value}")
            if parameter == "random_state":
                self.classifier.random_state = value
            else:
                super().set_params(**{parameter: value})
        if DEBUG:
            print(f"Blind_Classifier: set_params: Finished setting parameters")
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
        return f"Blind_Classifier with random_state={self.classifier.random_state}"
    def to_string(self):
        return self.__str__()
    @classmethod
    def get_param_grid(cls):
        return super().get_param_grid()
