# GED Kernel
from grakel.kernels import Kernel
import networkx as nx
import numpy as np
import os
import sys
import tqdm
sys.path.append(os.getcwd())
from Calculators.Base_Calculator import Base_Calculator
from Custom_Kernels.GEDLIB_kernel import GEDKernel

DEBUG = False  # Set to True for debug prints
class Trivial_GED_Kernel(Kernel):

    def __init__(self,ged_calculator=None,comparison_method="Mean-Distance",similarity_function="k1"):
        if ged_calculator is None:
            if Base_Calculator.backup is not None:
                ged_calculator = Base_Calculator.backup
            else:    
                raise ValueError("ged_calculator must be provided")
        
        self.ged_calculator = ged_calculator
        self.comparison_method = comparison_method
        self.similarity_function = similarity_function
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

        self.kernel_name = f"{self.comparison_method}-GEDLIB"
        self.attributes ={
            "kernel_name": self.kernel_name,
            "comparison_method": self.comparison_method,
            "similarity_function": self.similarity_function_eq
        }
        if DEBUG:
            print(f"Initialized GEDKernel with comparison_method={self.comparison_method}")
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
        self.runntime = self.ged_calculator.get_runtime()

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
        return {
            "comparison_method": self.comparison_method,
            "similarity_function": self.similarity_function,
            # "ged_calculator": self.ged_calculator
        }
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
            #"method": ['BRANCH_FAST', 'BRANCH_SLOW', 'BRANCH_OPTIMAL'],
            "KERNEL_similarity_function": ['k1','k2','k3','k4', 'frac'],
            "KERNEL_comparison_method": ['Mean-Distance', 'UpperBound-Distance', 'LowerBound-Distance']
        }