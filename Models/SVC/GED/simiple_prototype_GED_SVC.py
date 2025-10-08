# Class for Graph Edit Distance Kernel
# imports
import sys
import os
import numpy as np
import tqdm
from scipy.stats import randint, uniform
from typing import Dict, Any, List

sys.path.append(os.getcwd())
from grakel.kernels import Kernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Calculators.Prototype_Selction import select_Prototype, Prototype_Selector,Select_Prototypes, buffered_prototype_selection
DEBUG = False  # Set to True for debug prints

class Simple_Prototype_GED_SVC(Base_GED_SVC):
    model_specific_iterations = 100  # Base number of iterations for this model
    def __init__(self,
                prototype_size,
                selection_split,
                selection_method,
                dataset_name:str,
                attributes: dict = dict(),
                 **kwargs):
        self.prototype_size = prototype_size
        self.selection_split = selection_split
        self.selection_method = selection_method
        self.dataset_name = dataset_name
        if self.selection_method not in ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        if kwargs.get("name") is None:
            self.name="Simple-Prototype-GED"
        else:
            self.name=kwargs.pop("name")
        attributes.update({
            "pototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "dataset_name": self.dataset_name
        })
        # print("initialitized")
        super().__init__(attributes=attributes, name=self.name, **kwargs)
    def build_feature_vector(self, g):
        feature_vector = np.empty((self.prototype_size,), dtype=float)
        for i, g0 in enumerate(self.prototypes):
            feature_vector[i] = self.ged_calculator.compare(g, g0, method=self.ged_bound)
        return feature_vector
    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        self.X_fit_graphs_ = X # Store the training graphs for transform method
        self.prototypes = Select_Prototypes(X,  ged_calculator=self.ged_calculator,y=y, size=self.prototype_size, selection_split=self.selection_split, selection_method=self.selection_method, comparison_method=self.ged_bound)
        feature_matrix = self.build_feature_matrix(X)
        return feature_matrix


    def transform(self, X):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        feature_matrix = self.build_feature_matrix(X)
        return feature_matrix
    def build_feature_matrix(self, X):
        feature_vectors = np.zeros((len(X), self.prototype_size))
        for i, g in enumerate(X):
            feature_vectors[i, :] = self.build_feature_vector(g)
        return feature_vectors
    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "ged_bound": self.ged_bound
        })
        return params
    
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({            
            'kernel_type': ['poly', 'rbf', 'linear'],
            "prototype_size": [1, 5, 8],
            "selection_split": ["all", "classwise", "single_class"],
            "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"],
            # "selection_method": ["k-CPS","RPS"]
            # "C": [0.1]
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = Base_GED_SVC.get_random_param_space()
        param_space.update({
            'kernel_type': ['poly', 'rbf', 'linear'],
            "prototype_size": randint(1, 8),
            "selection_split": ["all", "classwise", "single_class"],
            "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"],
            # "selection_method": ["k-CPS","RPS"]
            # "C": [0.1, 1.0, 10.0, 100.0]
        })
        return param_space

