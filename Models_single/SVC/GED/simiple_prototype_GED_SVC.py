# Class for Graph Edit Distance Kernel
# imports
import sys
import os
import numpy as np
import tqdm
sys.path.append(os.getcwd())
from grakel.kernels import Kernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Models_single.SVC.Base_GED_SVC import Base_GED_SVC
from Calculators.Prototype_Selction import select_Prototype, Prototype_Selector,Select_Prototypes, buffered_prototype_selection
DEBUG = False  # Set to True for debug prints

class Simple_Prototype_GED_SVC(Base_GED_SVC):
    def __init__(self,
                prototypes_size,
                classwise,
                single_class,
                selection_method,
                dataset_name:str,

                attributes: dict = dict(),
                 **kwargs):
        self.prototypes_size = prototypes_size
        self.classwise = classwise
        self.single_class = single_class
        self.selection_method = selection_method
        self.dataset_name = dataset_name
        if self.selection_method not in ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        self.name="Simple-Prototype-GED"
        attributes.update({
            "pototypes_size": self.prototypes_size,
            "classwise": self.classwise,
            "single_class": self.single_class,
            "selection_method": self.selection_method
            "dataset_name": self.dataset_name
        })
        super().__init__(attributes=attributes, name=self.name, **kwargs)
    def build_feature_vector(self, g):
        feature_vector = np.zeros(self.prototypes_size)
        for i, g0 in enumerate(self.prototypes):
            feature_vector[i] = self.ged_calculator.compare(g, g0, method=self.ged_bound)
        return feature_vector
    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        self.X_fit_graphs_ = X # Store the training graphs for transform method
       
        self.prototypes = buffered_prototype_selection(X, y=y, ged_calculator=self.ged_calculator, size=self.prototypes_size, classwise=self.classwise, single_class=self.single_class, selection_method=self.selection_method, comparison_method=self.comparison_method, dataset_name=self.dataset_name)
        return self.transform(X)
    
    def transform(self, X):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        feature_vectors = np.zeros((len(X), self.prototypes_size))
        for i, g in enumerate(X):
            feature_vectors[i, :] = self.build_feature_vector(g)
        return feature_vectors
    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            "prototypes_size": self.prototypes_size,
            "classwise": self.classwise,
            "single_class": self.single_class,
            "selection_method": self.selection_method,
        })
        return params
    
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({            
            'kernel_type': ['poly', 'rbf', 'linear'],
            "prototypes_size": [1, 3, 5, 8, 10],
            "classwise": [False, True],
            "single_class": [False, True],
            "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]
        })
        return param_grid

