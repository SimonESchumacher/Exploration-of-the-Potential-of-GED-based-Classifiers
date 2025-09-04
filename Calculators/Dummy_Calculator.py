# GED computation with Graphkit Learn
# impoerts

import math
from time import time
import numpy as np
import networkx as nx
import tqdm
from Calculators.Base_Calculator import Base_Calculator
DEBUG = True

class Dummy_Calculator(Base_Calculator):

    def activate(self):
        self.dataset_edge_count = np.zeros(len(self.dataset))
        self.dataset_node_count = np.zeros(len(self.dataset))
        if DEBUG:
            iters = tqdm.tqdm(enumerate(self.dataset), desc='Adding graphs to Dummy_Calculator', total=len(self.dataset))
        else:
            iters = enumerate(self.dataset)
        for idx, graph in iters:
            if not isinstance(graph, nx.Graph):
                raise TypeError("All graphs must be of type networkx.Graph")
            rnd=2
            self.dataset_edge_count[idx] = graph.number_of_edges() #+ np.random.randint(-rnd, rnd)  # add some random noise to edges
            self.dataset_node_count[idx] = graph.number_of_nodes() #+ np.random.randint(-rnd, rnd)  # add some random noise to nodes
            # self.dataset_edge_count[idx] = (idx)/20
            # self.dataset_node_count[idx] = (idx)/20
            # self.dataset_edge_count[idx]= idx +1
            # self.dataset_node_count[idx]= idx
        return super().activate()

    def count_nodes(self, graph_index):
        """Counts the number of nodes in the specified graph."""
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        return self.dataset_node_count[graph_index]
    def count_edges(self, graph_index):
        """Counts the number of edges in the specified graph."""
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        return self.dataset_edge_count[graph_index]
    def run_method(self, graph1_index, graph2_index):
        """        Runs the GED method for the specified graph indexes.
        """
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        self.upperbound_matrix[graph1_index][graph2_index] = math.fabs(self.count_edges(graph1_index) - self.count_edges(graph2_index))
        self.lowerbound_matrix[graph1_index][graph2_index] = math.fabs(self.count_nodes(graph1_index) - self.count_nodes(graph2_index))

        self.upperbound_matrix[graph2_index][graph1_index] = self.upperbound_matrix[graph1_index][graph2_index]
        self.lowerbound_matrix[graph2_index][graph1_index] = self.lowerbound_matrix[graph1_index][graph2_index]
        if self.need_node_map:
            node_map = []
            self.node_map_matrix[graph1_index][graph2_index] = node_map
            self.node_map_matrix[graph2_index][graph1_index] = node_map
            raise NotImplementedError("Node map functionality is not implemented in Dummy_Calculator.")
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        return {
            # "GED_calc_method": ['BRANCH', 'BIPARTITE'],
            "GED_edit_cost": ['CONSTANT']
            #, "gamma": [0.1, 0.5, 1.0]
        }
    
class Dummy_Calculator2D(Dummy_Calculator):
    def run_method(self, graph1_index, graph2_index):
        
        cords_graph1 =np.array(math.fabs(self.count_edges(graph1_index)), math.fabs(self.count_nodes(graph1_index)))
        cords_graph2 =np.array(math.fabs(self.count_edges(graph2_index)), math.fabs(self.count_nodes(graph2_index)))
        # Euclidean distance between the two points
        distance = ((cords_graph1[0] - cords_graph2[0]) ** 2 + (cords_graph1[1] - cords_graph2[1]) ** 2)
        self.upperbound_matrix[graph1_index][graph2_index] = distance
        self.lowerbound_matrix[graph1_index][graph2_index] = distance

        self.upperbound_matrix[graph2_index][graph1_index] = self.upperbound_matrix[graph1_index][graph2_index]
        self.lowerbound_matrix[graph2_index][graph1_index] = self.lowerbound_matrix[graph1_index][graph2_index]
        
    
    
    


