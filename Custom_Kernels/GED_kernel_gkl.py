# GED Kernel
from grakel.kernels import Kernel
from grakel.graph import Graph
import networkx as nx
import numpy as np
import gmatch4py as gm
import os
import sys
import tqdm
sys.path.append(os.getcwd())
from GED_gkl import gkl_calculator
from GED import GraphEditDistanceCalculator
DEBUG = True  # Set to True for debug prints
class GEDKernel(Kernel):

    def __init__(self, gamma=0.1, method=None,normalize_ged=True,similarity=False):
        self.gamma = gamma
        self.normalize_ged = normalize_ged
        self.similarity = similarity
        # GED costs (specific to gmatch4py or your GED implementation)
        self.method= method if method else 'BRANCH_FAST'
        # Initialize the Graph Edit Distance calculator
        self.ged_calculator = gkl_calculator(method=self.method,gamma=self.gamma)
        super().__init__()
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        if self.ged_calculator is None:
            raise RuntimeError("GED calculator is not initialized.")
        return self.ged_calculator
    
    def _calculate_kernel_matrix(self,X_graphs ,Y_graphs=None):
        """Compute the kernel matrix for the input graphs."""
        if self.ged_calculator is None or self.ged_calculator.isclalculated is False:
            raise RuntimeError("GED calculator is not initialized or GED matrix is not computed.")
        
        if Y_graphs is None:
            Y_graphs = X_graphs

        # TODO idk if this makes sense
        if self.similarity:
            K = self.ged_calculator.partial_similarity_matrix(X_graphs, Y_graphs)
            if self.normalize_ged:
                print("not Normalizing ")
        else:
            K = self.ged_calculator.partial_matrix(X_graphs, Y_graphs)
            if self.normalize_ged:
                K = np.exp(-self.gamma * K)
        
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