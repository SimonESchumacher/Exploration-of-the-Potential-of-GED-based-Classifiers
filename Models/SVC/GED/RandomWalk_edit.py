import numpy as np
import pandas as pd
import sys
import time
import os
from Calculators.Product_GRaphs import build_restricted_product_graph, limited_length_approx_random_walk_similarity, infinte_length_random_walk_similarity
from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
from Calculators.Base_Calculator import Base_Calculator
from Models.SVC.Base_GED_SVC import Base_GED_SVC, Base_Kernel
DEBUG = False  # Set to True for debug prints


class Random_walk_edit_SVC(Base_GED_SVC):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """

    def initKernel(self, ged_calculator, **kernel_kwargs):
        self.kernel = random_walk_edit_Kernel(ged_calculator, **kernel_kwargs)

    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(random_walk_edit_Kernel.get_param_grid())

        return param_grid
    
class random_walk_edit_Kernel(Base_Kernel):
    """
    Random Walk Edit Kernel
    """

    def __init__(self, ged_calculator, KERNEL_decay_lambda,KERNEL_max_walk_length, attributes: dict = dict(), **kwargs):
        super().__init__(ged_calculator,KERNEL_name="Random-Walk-Edit",attributes=attributes,**kwargs)
        self.ged_calculator = ged_calculator
        self.decay_lambda = KERNEL_decay_lambda
        self.max_walk_length = KERNEL_max_walk_length
        self.sum_bulid_product_graph_time = 0
        self.sum_random_walk_time = 0
        if KERNEL_max_walk_length == -1:
            self.random_walk_function = lambda pg: infinte_length_random_walk_similarity(pg, llamda=KERNEL_decay_lambda)
        else:
            self.random_walk_function = lambda pg: limited_length_approx_random_walk_similarity(pg, llamda=KERNEL_decay_lambda, max_length=KERNEL_max_walk_length)
        attributes.update({"KERNEL_decay_lambda": KERNEL_decay_lambda, "KERNEL_max_walk_length": KERNEL_max_walk_length})
        super().__init__(ged_calculator=ged_calculator,
                         KERNEL_name="Random-Walk-Edit",
                         attributes=attributes,
                         **kwargs)
        if DEBUG:
            print(f"Initialized random_walk_edit_Kernel with comparison_method={self.comparison_method}, decay_lambda={KERNEL_decay_lambda}, max_walk_length={KERNEL_max_walk_length}")
        
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
    
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_Kernel.get_param_grid()
        param_grid.update({
            "KERNEL_decay_lambda": [0.01, 0.1],
            "KERNEL_max_walk_length": [5, -1]  # -1 indicates infinite length
        })
        return param_grid
        
