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
from Models.SVC.Base_GED_SVC import Base_GED_SVC, Base_Kernel
DEBUG = False  # Set to True for debug prints

class Simple_Prototype_GED_SVC(Base_GED_SVC):
    def __init__(self,
                 I_size=1,
                 attributes: dict = dict(),
                 ged_calculator: Base_Calculator = None,
                 **kwargs):
        self.I_size = I_size
        self.selection_method = "random"
        attributes.update({"I_size": I_size})
        self.kernel_name = "Simple_Prototype_GED"
        super().__init__(attributes=attributes, ged_calculator=ged_calculator, **kwargs)
    def initKernel(self, ged_calculator: Base_Calculator = None, KERNEL_comparison_method="Mean-Distance", **kernel_kwargs):
        self.kernel = None
        self.comparison_method = KERNEL_comparison_method

    def fit_transform(self, X, y=None):
        """ Fit the kernel to the data and return the kernel matrix.
        """
        X=[int(X[i].name) for i in range(len(X))]
        self.I=np.zeros(self.I_size)
        # pick random indices of the graphs of X
        if self.selection_method == "random":
            self.I = np.random.choice(X, size=self.I_size, replace=False)
        elif self.selection_method == "stratified_random":
            # get a random subset of the graphs in X of the cardinality of I_size
            unique_labels, counts = np.unique([g.get_label() for g in X], return_counts=True)
            if len(unique_labels) > self.I_size:
                raise ValueError(f"To many labels in the dataset to select {self.I_size} graphs.")
            self.I = np.random.choice(X, size=self.I_size, replace=False)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        return self.real_transform(X)
    def transform(self, X):
       X=[int(i.name) for i in X]
       return self.real_transform(X)
    
    def real_transform(self, X):
        feature_vectors = np.zeros((len(X), self.I_size))
        for i, g in enumerate(X):
            for j, g0 in enumerate(self.I):
                feature_vectors[i, j] = self.ged_calculator.compare(g, g0, method=self.comparison_method)
        return feature_vectors
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({            
            'kernel_type': ['poly', 'rbf', 'linear'],
            'I_size': [ 3, 5, 10],
            # 'selection_method': ['random', 'stratified_random']
        })
        return param_grid

