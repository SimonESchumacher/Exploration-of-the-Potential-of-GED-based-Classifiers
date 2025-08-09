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
    
    def __init__(self,
            ged_calculator=None,
            attributes:dict=dict(),
                **kwargs):
        
        if ged_calculator is None:
            raise ValueError("ged_calculator must be provided.")
        # get all the kwargs for the Kernel that start with "KERNEL_"
        kernel_kwargs =dict()
        svc_kwargs = dict()
        for key, value in kwargs.items():
            if key.startswith("KERNEL_"):
                kernel_kwargs[key[len("KERNEL_"):]] = value
            else:
                svc_kwargs[key] = value
        self.ged_calculator = ged_calculator
        self.initKernel(ged_calculator=ged_calculator, **kernel_kwargs)
        self.kernel_name = self.kernel.kernel_name
        attributes.update(self.kernel.attributes)
        attributes.update({
            "ged_calculator_name": ged_calculator.get_name() if ged_calculator else None
        })
        # Initialize the Support Vector Machine with the GED kernel
        super().__init__(kernelfunction=self.kernel,
                        kernel_name=self.kernel_name,
                        attributes=attributes,
                        **kwargs)
        if DEBUG:
            print(f"Initialized {self.__class__.__name__}")
    
    def initKernel(self,ged_calculator=None, **kernel_kwargs):
        self.kernel = Base_Kernel(ged_calculator=ged_calculator, **kernel_kwargs)
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        if self.kernel is None:
            raise RuntimeError("GED calculator is not initialized.")
        return self.kernel.get_calculator()
    def set_params(self, **params):
        # all parameters that start with "GED_" are passed to the GED calculator
        # the subset of all parameters that start with GED are to be set to calculator_params
        svc_kwargs = dict()
        calculator_params = dict()
        for key, value in params.items():
            if key.startswith("GED_"):
                calculator_params[key] = value
            else:
                svc_kwargs[key] = value
        if len(calculator_params)>0:
            self.ged_calculator.set_params(**calculator_params)
        super().set_params(**svc_kwargs)
        return self
    
    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        k_train=self.kernel.fit_transform(X, y)
        return  k_train
    def transform(self, X):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        k_test = self.kernel.transform(X)
        return k_test

    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(Base_Kernel.get_param_grid())

        return param_grid

class Base_Kernel(Kernel):
    def __init__(self,ged_calculator=None,KERNEL_comparison_method="Mean-Distance",KERNEL_name="Base-GED",attributes:dict=dict(),**kwargs):
        if ged_calculator is None:
            if Base_Calculator.backup is not None:
                ged_calculator = Base_Calculator.backup
            else:    
                raise ValueError("ged_calculator must be provided")
        self.ged_calculator = ged_calculator
        self.comparison_method = KERNEL_comparison_method
        self.kernel_name = KERNEL_name
        attributes.update({
            "kernel_name": self.kernel_name,
            "comparison_method": self.comparison_method
        })
        self.attributes = attributes
        super().__init__()
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        if self.ged_calculator is None:
            raise RuntimeError("GED calculator is not initialized.")
        return self.ged_calculator
    def add_graphs(self, graphs, labels=None):
        """
        Add graphs to the GED calculator.
        """
        if DEBUG:
            print(f"Adding {len(graphs)} graphs to GED calculator")
        self.ged_calculator.add_graphs(graphs, labels)
        return self.ged_calculator.get_indexes()
    def init(self):
        # should not be called currently
        # activations should always happen from outside of the kernel directly on the calculator
        print("Warning: init() is called, but it should not be used directly. Use activate() on the calculator instead.")
        if DEBUG:
            print("activating Kernel")
        self.ged_calculator.activate()
        self.ged_calculator.calculate()
        self.runtime = self.ged_calculator.get_runtime()
    def compare(self, g1, g2):
        return 1/(1+ self.ged_calculator.compare(g1, g2, method=self.comparison_method))
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
            for j, g2 in enumerate(Y_graphs):
                K[i, j] = self.compare(g1, g2)
        return K
    def fit_transform(self, X, y=None):
        """
        Fits the kernel (computes and stores the training kernel matrix).
        """
        if DEBUG:
            print("fit_transform called: Calculating training kernel matrix...")
        # X should be a list of NetworkX graphs
        self.X_fit_graphs_ = X # Store the training graphs for transform method

        # Calculate the kernel matrix for the training data
        K_train = self._calculate_kernel_matrix(X_graphs=X)
        
        # Check if the generated matrix is approximately positive semi-definite (optional but good practice)
        # E.g., check eigenvalues, but SVC is usually robust enough for RBF on metric spaces.
        
        return K_train
    def transform(self, X):
        """
        Transforms new graphs into the kernel space relative to the fitted training data.
        """
        if DEBUG:
            print("transform called: Calculating test kernel matrix...")
        # X should be a list of NetworkX graphs for the test set
        if not hasattr(self, 'X_fit_graphs_'):
            raise RuntimeError("The model must be fitted before calling transform.")
        
        # Calculate the cross-kernel matrix between X (test) and X_fit_graphs_ (train)
        K_test = self._calculate_kernel_matrix(X_graphs=X, Y_graphs=self.X_fit_graphs_)
        
        return K_test
    def get_params(self, deep=True):
        """
        Returns the parameters of the GEDLIB_Calculator.
        """
        return self.attributes
    def set_params(self, **params):
        """
        Set parameters for the GED kernel.
        will probably be called by the SVC using this kernel.

        """
        for key, value in params.items():
            if key.startswith("KERNEL_"):
                key = key[len("KERNEL_"):]  # Remove the prefix
                # set the parameter in the kernel
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Parameter {key} not found in GEDKernel. Skipping.")
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        return {
            # "KERNEL_comparison_method": ['Mean-Distance', 'UpperBound-Distance', 'LowerBound-Distance']
        }