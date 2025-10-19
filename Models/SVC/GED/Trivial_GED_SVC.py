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
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from scipy.stats import randint, uniform, loguniform

DEBUG = False  # Set to True for debug prints

class Trivial_GED_SVC(Base_GED_SVC):
    model_specific_iterations = 100  # Base number of iterations for this model
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """
    def __init__(self,
                similarity_function,
                llambda,
                attributes:dict=dict(),
                **kwargs):
        self.similarity_function = similarity_function
        self.llambda = llambda
        if self.similarity_function not in ["k1", "k2", "k3", "k4", "frac"]:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
        self.name="Trivial-GED"
        attributes.update({
            "similarity_function": self.similarity_function
        })
        super().__init__(attributes=attributes, name=self.name, **kwargs)
    def compare(self, g1, g2):
        ged = self.ged_calculator.compare(g1, g2, method=self.ged_bound)
        HIGH = 1e100  # high but non-infinite cap

        if self.similarity_function == 'k1':
            val = -((self.llambda * ged) ** 2)
        elif self.similarity_function == 'k2':
            val = -(self.llambda * ged)
        elif self.similarity_function == 'k3':
            val = np.tanh(-(self.llambda * ged))
        elif self.similarity_function == 'k4':
            val = np.exp(-(self.llambda * ged))
        elif self.similarity_function == 'frac':
            # avoid division by zero if ged is -1/(llambda) (very unlikely), handled by isfinite below
            val = 1.0 / (1.0 + (self.llambda * ged))
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")

        # handle non-finite results: replace infinities with a large finite number, NaNs with 0.0
        if not np.isfinite(val):
            if np.isnan(val):
                val = 0.0
            else:  # +/- inf
                sign = np.sign(val)
                # if sign is 0 for some reason, use positive HIGH
                val = sign * HIGH if sign != 0 else HIGH

        return val
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "similarity_function": self.similarity_function,
            "llambda": self.llambda
        })
       
        return params
    def set_params(self, **params):
        return super().set_params(**params)
        
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({
            "similarity_function": ['k1', 'k2', 'k3', 'k4'],
            "llambda": [0.01, 0.1, 1, 10, 100]
        })

        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_grid = Base_GED_SVC.get_random_param_space()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({
            "similarity_function": ['k1', 'k2', 'k3', 'k4'],
            "llambda": loguniform(0.01, 100)
        })
        return param_grid
        # param_grid.update({
    
# Kernels that this is designed for:
# GEDKernel, Trivial_GED_Kernel
