# GED computation with Graphkit Learn
# impoerts

import gc
import math
import os
import sys
from time import time
import numpy as np
import networkx as nx
import tqdm
sys.path.append(os.getcwd())
from Calculators.Base_Calculator import Base_Calculator
DEBUG = True

class NetworkXGEDCalculator(Base_Calculator):

    def activate(self):
        self.node_match :callable = lambda x, y: x.label == y.label
        self.edge_match :callable = lambda x, y: x.label == y.label
        self.Map_Cost_Functions(self.GED_edit_cost)
        self.timeout = 60
        return super().activate()

    def Map_Cost_Functions(self, GED_edit_cost):
        """
        Maps the GED_edit_cost to a function that can be used to calculate the cost of the edit operation.
        """
        if GED_edit_cost == "CONSTANT":
            self.node_subst_cost = lambda x, y: 1.0
            self.node_ins_cost = lambda x: 1.0
            self.node_del_cost = lambda x: 1.0
            self.edge_subst_cost = lambda x, y: 1.0
            self.edge_ins_cost = lambda x: 1.0
            self.edge_del_cost = lambda x: 1.0
        else:
            raise ValueError(f"Unknown GED_edit_cost: {GED_edit_cost}")

    def run_method(self, graph1_index, graph2_index):
        self.upperbound_matrix[graph1_index][graph2_index] = 0
        self.lowerbound_matrix[graph1_index][graph2_index] = np.inf
        count = 0
        sum_cost = 0

        gernator = nx.optimize_graph_edit_distance(
            G1=self.dataset[graph1_index],
            G2=self.dataset[graph2_index],
            node_match=self.node_match,
            edge_match=self.edge_match,
            node_subst_cost=self.node_subst_cost,
            node_ins_cost=self.node_ins_cost,
            node_del_cost=self.node_del_cost,
            edge_subst_cost=self.edge_subst_cost,
            edge_ins_cost=self.edge_ins_cost,
            edge_del_cost=self.edge_del_cost,
            upper_bound=100    
            # timeout=self.timeout
        )
        try:
            dist = min(gernator)
        except sys.exception as e:
            print("Waring, no results")
        self.lowerbound_matrix[graph1_index][graph2_index] = dist
        self.upperbound_matrix[graph1_index][graph2_index] = dist
        gc.collect()

        # print(v)
        # mean = sum_cost / count if count > 0 else 0
        # print(f"Count of edit operations: {count}")
        # print(f"Distance: {self.upperbound_matrix[graph1_index][graph2_index]}")
        
         
            
