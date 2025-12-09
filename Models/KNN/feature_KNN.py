# GED K-NN Classifier
# imports
import traceback
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint, uniform
from typing import Dict, Any, List

# imports from custom modules
import sys
import os

from Models.KNN.GEDLIB_KNN import abstract_GED_KNN
sys.path.append(os.getcwd())
from Models.Graph_Classifier import GraphClassifier
from Models.KNN_Classifer import KNN
from Calculators import Base_Calculator, Dummy_Calculator
from config_loader import get_conifg_param
# from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
# from Calculators.Dummy_Calculator import Dummy_Calculator
DEBUG = get_conifg_param('GED_models', 'debuging_prints', type='bool')

class Feature_KNN(abstract_GED_KNN):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', '', type='int')

    def __init__(self,
                vector_feature_list:list,
                dataset_name:str,
                prototype_size:int,
                selection_split:str,
                selection_method:str,
                ged_bound: str,
                calculator_id:str,
                node_label_tag:str="label",
                edge_label_tag:str="label",
                attributes: dict=dict(),
                **kwargs):
        """
        Initialize the GED K-NN Classifier with the given parameters.
        """
        self.vector_feature_list = vector_feature_list
        self.dataset_name = dataset_name
        self.prototype_size = prototype_size
        self.selection_split = selection_split
        self.selection_method = selection_method
        self.node_label_tag = node_label_tag
        self.edge_label_tag = edge_label_tag
        if self.selection_method not in ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        if kwargs.get("name") is None:
            self.name="feature-KNN"
        else:
            self.name=kwargs.pop("name")
        attributes.update({
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "dataset_name": self.dataset_name,
            "vector_feature_list": self.vector_feature_list
        })
        
        super().__init__(
            attributes=attributes,
            name=self.name,
            ged_bound=ged_bound,
            calculator_id=calculator_id,
            **kwargs
        )
        self.vector_creator = None # Removed
        self.add_vector_extractors()
        if DEBUG:
            print(f"Initialized GED_KNNClassifier")
    def add_vector_extractors(self):
        for feature in self.vector_feature_list:
            if feature == "EdgeHistogram":
                self.vector_creator.add_edge_histogram_extractor()
            elif feature == "VertexHistogram":
                self.vector_creator.add_vertex_histogram_extractor()
            elif feature == "density":
                self.vector_creator.add_density_extractor()
            elif feature == "Prototype-Distance":
                self.vector_creator.add_prototype_dis_vector_extractor(selection_split=self.selection_split, selection_method=self.selection_method, size=self.prototype_size, ged_bound=self.ged_bound,dataset_name=self.dataset_name)
            else:
                raise ValueError(f"Unknown feature: {feature}")
            

    def get_calculator(self):
        return self.ged_calculator
    def fit_transform(self, X, y=None):

        """
        save the traiing Graphs and transform Data into matrix.
        """
        self.X_fit =X
        #measure the time the create vetor takes
        self.features= self.vector_creator.create_vector(X, is_fitted=False,node_label_tag=self.node_label_tag, edge_label_tag=self.edge_label_tag)
        self.max_distance = self.features.max()
        return self.features
    def transform(self, X):
        """
        Transform the input graphs into a distance matrix using the GED calculator.
        """

        features = self.vector_creator.create_vector(X, is_fitted=True,node_label_tag=self.node_label_tag, edge_label_tag=self.edge_label_tag)
        return features
    def get_params(self,deep=True):
        """
        Get the parameters of the GED K-NN Classifier.
        """
        params = super().get_params(deep=deep)
        params.update({
            "ged_bound": self.ged_bound,
            "calculator_id": self.calculator_id,
            "vector_feature_list": self.vector_feature_list,
            "dataset_name": self.dataset_name,
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "metric": self.metric
        })
        return params
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        param_grid = super().get_param_grid()
        param_grid.update({
            'vector_feature_list': [
                ["VertexHistogram", "density","Prototype-Distance"],
                # ["WeisfeilerLehman", "density"],
                ["EdgeHistogram", "density","Prototype-Distance"],
                # ["VertexHistogram", "density"],
                ["VertexHistogram","EdgeHistogram","Prototype-Distance"],
                # ["WeisfeilerLehman"],
                ["Prototype-Distance"],
                # Different combinations of features
            ],
            "prototype_size": [1, 2, 4, 6],
            "selection_split": ["all", "classwise"],
            "selection_method": ["CPS", "TPS", "k-CPS"],
            "metric": ["minkowski", "manhattan","euclidean"],
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls) -> Dict[str, Any]:
        """
        Get the random parameter space for hyperparameter tuning.
        """
        param_space = super().get_random_param_space()
        param_space.update({
            'vector_feature_list': [
                ["VertexHistogram", "density","Prototype-Distance"],
                ["EdgeHistogram", "density","Prototype-Distance"],
                ["VertexHistogram","EdgeHistogram","Prototype-Distance"],
                ["VertexHistogram", "density"],
                ["Prototype-Distance"],
                ["Prototype-Distance","density"],
                ["VertexHistogram","Prototype-Distance"],
                ["EdgeHistogram","Prototype-Distance"],
                # Different combinations of features
            ],
            "prototype_size": randint(1, 11),  # Random integer between 1 and 10
            "selection_split": ["all", "classwise"],
            "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"],
            "metric": ["minkowski", "manhattan","euclidean"],
        })
        return param_space
   
        
    

        

        
        
