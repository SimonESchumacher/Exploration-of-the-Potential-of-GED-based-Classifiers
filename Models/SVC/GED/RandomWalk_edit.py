import numpy as np
import pandas as pd
import time
from Calculators.Product_GRaphs import build_restricted_product_graph, limited_length_approx_random_walk_similarity, infinte_length_random_walk_similarity
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from io_Manager import IO_Manager
DEBUG = False  # Set to True for debug prints


class Random_walk_edit_SVC(Base_GED_SVC):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """
    def __init__(self,
                decay_lambda,
                max_walk_length,
                attributes:dict=dict(),
                **kwargs):
        self.name="Random-Walk-Edit"
        self.decay_lambda = decay_lambda
        self.max_walk_length = max_walk_length
        if self.max_walk_length == -1:
            self.random_walk_function = lambda pg: infinte_length_random_walk_similarity(pg, llamda=decay_lambda)
        else:
            self.random_walk_function = lambda pg: limited_length_approx_random_walk_similarity(pg, llamda=decay_lambda, max_length=max_walk_length)
        # inner metrics 
        self.sum_bulid_product_graph_time = 0
        self.sum_random_walk_time = 0
        attributes.update({
            "decay_lambda": decay_lambda,
            "max_walk_length": max_walk_length
        })
        super().__init__(attributes=attributes, name=self.name, **kwargs)
    def _calculate_kernel_matrix(self,X_graphs ,Y_graphs=None):
        # buffered, to see if calculation maybe has already been done
        if Y_graphs is None:
            rw_kernel_matrix_key = f"{self.decay_lambda}_{self.max_walk_length}_train"
        else:
            rw_kernel_matrix_key = f"{self.decay_lambda}_{self.max_walk_length}_test"
        kernel_matrix =IO_Manager.get_rw_kernel_matrix(rw_kernel_matrix_key)
        if kernel_matrix is None:
            kernel_matrix = super()._calculate_kernel_matrix(X_graphs, Y_graphs)
            IO_Manager.save_rw_kernel_matrix(rw_kernel_matrix_key, kernel_matrix)
        return kernel_matrix
    def compare(self, g1, g2):
        node_map = self.ged_calculator.get_node_map(g1, g2)
        if DEBUG:
            print(f"Node map between graphs: {node_map}")
        graph1 = self.ged_calculator.get_dataset()[g1]
        graph2 = self.ged_calculator.get_dataset()[g2]
        start_time = time.time()
        product_graph = build_restricted_product_graph(graph1, graph2, node_map)
        end_time = time.time()
        self.sum_bulid_product_graph_time += end_time - start_time
        if DEBUG:
            print(f"Product graph has {product_graph.number_of_nodes()} nodes and {product_graph.number_of_edges()} edges.")
        start_time = time.time()
        similarity = self.random_walk_function(product_graph)
        end_time = time.time()
        self.sum_random_walk_time += end_time - start_time
        if DEBUG:
            print(f"Random walk similarity: {similarity}")
        return similarity
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "decay_lambda": self.decay_lambda,
            "max_walk_length": self.max_walk_length
        })
        return params
    
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({
            "decay_lambda": [0.01, 0.1],
            "max_walk_length": [5, -1]  # -1 indicates infinite length
        })
        return param_grid
    

        
