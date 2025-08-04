# Class for Graph Edit Distance Kernel
# imports
import sys
import os
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
DEBUG = False  # Set to True for debug prints

class GED_SVC(SupportVectorMachine):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """
    def __init__(self, C=1.0,
            kernel_type="precomputed",
            kernel=None,
            kernel_name=None,
                attributes:dict=None):
        
        if kernel is None:
            raise ValueError("Kernel must be provided.")
        self.kernel = kernel
        self.kernel_name = kernel_name
        # Initialize the Support Vector Machine with the GED kernel
        super().__init__(kernel_type=kernel_type, 
                         C=C, 
                         kernelfunction=self.kernel,
                         kernel_name=kernel_name,
                         attributes=attributes)	
        if DEBUG:
            print(f"Initialized GED_SVC")
    # Override fit_transform and transform methods
    # NO grakel conversion
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        if self.kernel is None:
            raise RuntimeError("GED calculator is not initialized.")
        return self.kernel.get_calculator()
    
    def fit_transform(self, X, y=None):
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        k_train=self.kernel.fit_transform(X, y)
        return  k_train
    def transform(self, X):
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        k_test = self.kernel.transform(X)
        return k_test
    
    def set_params(self, **params):
        need_new_GED =False
        calculator_params = {}
        kernel_params = {}
        if DEBUG:
            print(f"Setting parameters for GED_SVC")
        for parameter, value in params.items():
            if parameter.startswith("GED_"):
                if DEBUG:
                    print(f"Setting GED parameter {parameter} to {value}")
                # pass to the calculator
                calculator_params[parameter] = value
                need_new_GED = True
            elif parameter.startswith("KERNEL_"):
                if DEBUG:
                    print(f"Setting KERNEL parameter {parameter} to {value}")
                # pass to the kernel
                kernel_params[parameter] = value
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
            self.kernel.get_calculator().set_params(**calculator_params)
        self.kernel.set_params(**kernel_params)
        return self    
        
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(GEDKernel.get_param_grid())

        return param_grid