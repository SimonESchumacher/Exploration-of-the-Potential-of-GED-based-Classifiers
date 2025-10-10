# Base GED SVC Parent Class
# SVC that uses a Kernel based on Graph Edit Distance (GED)
# Has a Caclulator and a Kernel
# In this file the Base Parent Kernel and Base Parent SVC are defined

import sys
import os

import numpy as np
import tqdm
sys.path.append(os.getcwd())
from grakel.kernels import Kernel
from Calculators.Base_Calculator import Base_Calculator

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
DEBUG = False  # Set to True for debug prints

class Base_GED_SVC(SupportVectorMachine):
    model_specific_iterations = 50
    def __init__(self,
            ged_calculator,
            ged_bound,
            attributes:dict=dict(),
            name="Base_GED_SVC",
                **kwargs):
        # chekc if kwargs has the key "KERNEL_comparison_method", if not, set it to "Mean-Distance"
        if ged_calculator is None:
            raise ValueError("ged_calculator must be provided.")
        # get all the kwargs for the Kernel that start with "KERNEL_"
        
        self.ged_calculator = ged_calculator
        self.ged_bound = ged_bound
        self.name = name
        # if needed.



        attributes.update({
            "ged_calculator_name": ged_calculator.get_name() if ged_calculator else None,
            "ged_bound": ged_bound
        })
        # Initialize the Support Vector Machine with the GED kernel
        super().__init__(kernelfunction=None,
                        kernel_name=self.name,
                        attributes=attributes,
                        **kwargs)
        if DEBUG:
            print(f"Initialized {self.__class__.__name__}")
    
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        return self.ged_calculator
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "ged_calculator": self.ged_calculator,
            "ged_bound": self.ged_bound
        })
       
        return params
    def compare(self, g1, g2):
        return 1/(1+ self.ged_calculator.compare(g1, g2, method=self.ged_bound))
    
    def _calculate_kernel_matrix(self,X_graphs ,Y_graphs=None):
        """Compute the kernel matrix for the input graphs."""
        if self.ged_calculator is None or not self.ged_calculator.isactive or not self.ged_calculator.isclalculated:

            raise RuntimeError("GED calculator is not initialized or not active. Call init() first.")

        if Y_graphs is None:
            Y_graphs = X_graphs       
            K = np.zeros((len(X_graphs), len(Y_graphs)), dtype=np.float64)
            if DEBUG:
                iter = tqdm.tqdm(enumerate(X_graphs), desc="Calculating GED Kernel Matrix", total=len(X_graphs))
            else:
                iter = enumerate(X_graphs)
            for i, g1 in iter:
                for j in range(i, len(Y_graphs)):
                    K[i, j] = self.compare(g1, Y_graphs[j])
                    K[j, i] = K[i, j]  # Exploit symmetry
            return K
        else:
            K = np.zeros((len(X_graphs), len(Y_graphs)), dtype=np.float64) 
            if DEBUG:
                iter = tqdm.tqdm(enumerate(X_graphs), desc="Calculating GED Kernel Matrix", total=len(X_graphs))
            else:
                iter = enumerate(X_graphs)
            for i, g1 in iter:
                for j, g2 in enumerate(Y_graphs):
                    K[i, j] = self.compare(g1, g2)
            return K


    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        self.X_fit_graphs_ = X # Store the training graphs for transform method

        # Calculate the kernel matrix for the training data
        K_train = self._calculate_kernel_matrix(X_graphs=X)
        
        # Check if the generated matrix is approximately positive semi-definite (optional but good practice)
        # E.g., check eigenvalues, but SVC is usually robust enough for RBF on metric spaces.
        
        return K_train
    def transform(self, X):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        if not hasattr(self, 'X_fit_graphs_'):
            raise RuntimeError("The model must be fitted before calling transform.")
        
        # Calculate the cross-kernel matrix between X (test) and X_fit_graphs_ (train)
        K_test = self._calculate_kernel_matrix(X_graphs=X, Y_graphs=self.X_fit_graphs_)
        
        return K_test

    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({
            # "ged_bound": ['UpperBound-Distance', 'Mean-Distance', 'LowerBound-Distance']
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = SupportVectorMachine.get_random_param_space()
        param_space.update({
            # "ged_bound": ['UpperBound-Distance', 'Mean-Distance', 'LowerBound-Distance']
        })
        return param_space

    