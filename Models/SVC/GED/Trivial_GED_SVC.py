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

class Trivial_GED_SVC(Base_GED_SVC):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """   
    def initKernel(self,ged_calculator, **kernel_kwargs):
        self.kernel = Trivial_GED_Kernel(ged_calculator=ged_calculator, **kernel_kwargs)
   
        
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(Trivial_GED_Kernel.get_param_grid())

        return param_grid
    
# Kernels that this is designed for:
# GEDKernel, Trivial_GED_Kernel


class Trivial_GED_Kernel(Base_Kernel):

    def __init__(self,ged_calculator,KERNEL_comparison_method,KERNEL_similarity_function,attributes:dict=dict(),**kwargs):

        self.similarity_function = KERNEL_similarity_function
        if self.similarity_function == "k1":
            self.similarity_function_eq = "-d(g1, g2)^2"
        elif self.similarity_function == "k2":
            self.similarity_function_eq = "-d(g1, g2)"
        elif self.similarity_function == "k3":
            self.similarity_function_eq = "tanh(-d(g1, g2))"
        elif self.similarity_function == "frac":
            self.similarity_function_eq = "1 / (1 + d(g1, g2))"
        elif self.similarity_function == "k4":
            self.similarity_function_eq = "exp(-d(g1, g2))"
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")

        attributes.update({"KERNEL_similarity_function": self.similarity_function})
        super().__init__(ged_calculator=ged_calculator,
                         KERNEL_comparison_method=KERNEL_comparison_method,
                         KERNEL_name="Trivial-GED",
                        attributes=attributes,
                         **kwargs)
        if DEBUG:
            print(f"Initialized GEDKernel with comparison_method={self.comparison_method}")
    
    def compare(self, g1, g2):
        if self.similarity_function == 'k1':
            return -(self.ged_calculator.compare(g1, g2, method=self.comparison_method) ** 2)
        elif self.similarity_function == 'k2':
            return -self.ged_calculator.compare(g1, g2, method=self.comparison_method)
        elif self.similarity_function == 'k3':
            return np.tanh(-self.ged_calculator.compare(g1, g2, method=self.comparison_method))
        elif self.similarity_function == 'k4':
            return np.exp(-self.ged_calculator.compare(g1, g2, method=self.comparison_method))
        elif self.similarity_function == 'frac':
            return 1 / (1 + self.ged_calculator.compare(g1, g2, method=self.comparison_method))
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
    
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        param_grid = Base_Kernel.get_param_grid()
        param_grid.update({
            "KERNEL_similarity_function": ['k1', 'k2', 'k3', 'k4', 'frac']
        })
        return param_grid
       