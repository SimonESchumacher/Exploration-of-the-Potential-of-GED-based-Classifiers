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

from GED import GraphEditDistanceCalculator
DEBUG = True  # Set to True for debug prints
class GEDKernel(Kernel):

    def __init__(self, gamma=0.1, random_state=None,
                 node_del_cost=1.0, node_ins_cost=1.0, node_sub_cost=1.0,
                 edge_del_cost=1.0, edge_ins_cost=1.0, edge_sub_cost=1.0,
                 normalize_ged=True,approximation=None):
        self.gamma = gamma
        self.random_state = random_state
        
        # GED costs (specific to gmatch4py or your GED implementation)
        self.node_del_cost = node_del_cost
        self.node_ins_cost = node_ins_cost
        self.edge_del_cost = edge_del_cost
        self.edge_ins_cost = edge_ins_cost
        self.normalize_ged = normalize_ged
        self.approximation = approximation
        # Initialize the Graph Edit Distance calculator
        self.ged_calculator = GraphEditDistanceCalculator(
            node_deletion_cost=self.node_del_cost,
            node_insertion_cost=self.node_ins_cost,
            edge_deletion_cost=self.edge_del_cost,
            edge_insertion_cost=self.edge_ins_cost,
            approximation=self.approximation
        )
        super().__init__()
    def _calculate_kernel_matrix(self,X_graphs ,Y_graphs=None):
        """Compute the kernel matrix for the input graphs."""
        if Y_graphs is None:
            symetric_kernel = True
            Y_graphs = X_graphs
        else:
            symetric_kernel = False

        n = len(X_graphs)
        m = len(Y_graphs)
        
        K = np.zeros((n, m))
        if DEBUG:
            iters = tqdm.tqdm(range(n), desc="Calculating Kernel Matrix")
        else:
            iters = range(n)

        for i in iters:
            for j in range(m):
                ged_distance = self.ged_calculator.distance(X_graphs[i], Y_graphs[j])
                if self.normalize_ged:
                    K[i, j] = np.exp(-self.gamma * ged_distance)
                else:
                    K[i, j] = ged_distance
        
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