# GED K-NN Classifier
# imports
import traceback
from sklearn.neighbors import KNeighborsClassifier


# imports from custom modules
import sys
import os
sys.path.append(os.getcwd())
from Models.Graph_Classifier import GraphClassifier
from Models.KNN_Classifer import KNN
from Graph_Tools import convert_nx_to_grakel_graph
from Calculators import Base_Calculator, Dummy_Calculator
# from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
# from Calculators.Dummy_Calculator import Dummy_Calculator
DEBUG = False  # Set to False to disable debug prints

class GED_KNN(KNN):

    def __init__(self,
                 ged_calculator:Base_Calculator=None, ged_bound="Mean-Similarity",
                 attributes : dict=dict(),similarity=False ,**kwargs):
        """
        Initialize the GED K-NN Classifier with the given parameters.
        """

        self.ged_calculator = ged_calculator
        self.ged_bound = ged_bound
        self.node_del_cost = 1.0
        self.similarity = similarity
        attributes.update({
            "ged_calculator": ged_calculator.get_name() if ged_calculator else "None",
            "comparison_method": ged_bound
        })
        super().__init__(
            metric="precomputed",
            metric_name="GED",
            attributes=attributes,
            **kwargs
        )
        if DEBUG:
            print(f"Initialized GED_KNNClassifier")
    def get_calculator(self):
        return self.ged_calculator
    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        """
        save the traiing Graphs and transform Data into matrix.
        """
        self.X_fit =X
        distance_matrix=self.ged_calculator.get_complete_matrix(method=self.ged_bound,x_graphindexes=self.X_fit)
        self.max_distance = distance_matrix.max() 
        similarity_matrix = self.max_distance - distance_matrix
        if DEBUG:
            print(f"Fitting {len(X)} graphs into distance matrix.")
            print("Fitted Graphs:")
            print(self.X_fit)
            print("Distance Matrix:")
            print(distance_matrix)
            print("Initial Data")
            print(X)
        if self.similarity:
            return similarity_matrix
        return distance_matrix
    def transform(self, X):
        """
        Transform the input graphs into a distance matrix using the GED calculator.
        """
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming {len(X)} graphs into distance matrix.")
            print(self.X_fit)
            print(X)
        distance_matrix = self.ged_calculator.get_complete_matrix(method=self.ged_bound, x_graphindexes=X, y_graphindexes=self.X_fit)
        similarity_matrix = self.max_distance - distance_matrix 
        if DEBUG:
            print("Transformed Data:")
        if self.similarity:
            return similarity_matrix
        return distance_matrix
    def get_params(self,deep=True):
        """
        Get the parameters of the GED K-NN Classifier.
        """
        params = super().get_params(deep=deep)
        params.update({
            "ged_calculator": self.ged_calculator,
            "comparison_method": self.ged_bound
        })
        return params
    def set_params(self, **params):
        need_new_GED =False
        calculator_params = {}
        if DEBUG:
            print(f"Setting parameters for GED_SVC")
        for parameter, value in params.items():
            if parameter.startswith("GED_"):
                if DEBUG:
                    print(f"Setting GED parameter {parameter} to {value}")
                # pass to the calculator
                calculator_params[parameter] = value
                need_new_GED = True
            else:
                if DEBUG:
                    print(f"Setting parameter {parameter} to {value}")
                if hasattr(self, parameter):
                    setattr(self, parameter, value)
                else:
                    print(f"Warning: Parameter {parameter} not found in GED_SVC. Skipping.")
        if need_new_GED:
            if DEBUG:
                print(f"Reinitializing GED calculator with parameters: {calculator_params}")
            self.ged_calculator.set_params(**calculator_params)
        return self
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        param_grid = super().get_param_grid()
        param_grid.update({
        })
        return param_grid

        

        
        
