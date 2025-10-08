# GED computation with Graphkit Learn
# impoerts

from time import time
import numpy as np
import networkx as nx
import tqdm
import sys
import os
# add the current directory to the system path
sys.path.append(os.getcwd())
from gedlibpy import librariesImport
from gedlibpy import gedlibpy
from Calculators.Base_Calculator import Base_Calculator
DEBUG = False

class GEDLIB_Calculator(Base_Calculator):

    def activate(self):
       
        gedlibpy.restart_env()
        if DEBUG:
            iters = tqdm.tqdm(self.dataset, desc='Adding graphs to GEDLIB', total=len(self.dataset))
        else:
            iters = self.dataset
        for graph in iters:
            if not isinstance(graph, nx.Graph):
                raise TypeError("All graphs must be of type networkx.Graph")
            gedlibpy.add_nx_graph(graph, "")
        gedlibpy.set_edit_cost(self.GED_edit_cost)
        gedlibpy.init()
        gedlibpy.set_method(self.GED_calc_method, "")
        gedlibpy.init_method()
        return super().activate()

    def run_method(self, graph1_index, graph2_index):
        """        Runs the GED method for the specified graph indexes.
        """
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        gedlibpy.run_method(graph1_index, graph2_index)
        self.upperbound_matrix[graph1_index][graph2_index] = gedlibpy.get_upper_bound(graph1_index, graph2_index)
        self.lowerbound_matrix[graph1_index][graph2_index] = gedlibpy.get_lower_bound(graph1_index, graph2_index)

        self.upperbound_matrix[graph2_index][graph1_index] = self.upperbound_matrix[graph1_index][graph2_index]
        self.lowerbound_matrix[graph2_index][graph1_index] = self.lowerbound_matrix[graph1_index][graph2_index]
        if self.need_node_map:
            node_map = gedlibpy.get_node_map(graph1_index, graph2_index)
            self.node_map_matrix[graph1_index][graph2_index] = node_map
            self.node_map_matrix[graph2_index][graph1_index] = [(b,a) for (a,b) in node_map]
    # Things that the Real GEDLIB Calculator can output
    def get_all_map(self, graph1_index, graph2_index):
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        return gedlibpy.get_all_map(graph1_index, graph2_index)
    def get_forward_map(self, graph1_index, graph2_index):
        return gedlibpy.get_forward_map(graph1_index, graph2_index)
    def get_backward_map(self, graph1_index, graph2_index):
        return gedlibpy.get_backward_map(graph1_index, graph2_index)
    def get_assignment_matrix(self, graph1_index, graph2_index):
        return gedlibpy.get_assignment_matrix(graph1_index, graph2_index)
    def get_node_image(self, graph1_index, graph2_index,node_index):
        return gedlibpy.get_node_image(graph1_index, graph2_index,node_index)
    # special funtions handmade
    def get_Name(self):
        return "GEDLIB_Calculator"
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        return {
            "GED_calc_method": ['BRANCH', 'BIPARTITE'],
            "GED_edit_cost": ['CONSTANT']
            #, "gamma": [0.1, 0.5, 1.0]
        }
    
    


