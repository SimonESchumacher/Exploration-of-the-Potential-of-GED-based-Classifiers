# GED computation with Graphkit Learn
# impoerts

import math
from time import time
import numpy as np
import networkx as nx
import tqdm
from Base_Calculator import Base_Calculator
DEBUG = True

class Dummy_Calculator(Base_Calculator):
    # add class variable, as copy of itself for backup

    def __init__(self, GED_edit_cost="CONSTANT", GED_calc_method="BIPARTITE", dataset=None, labels=None, activate: bool = True):
        """
        Initialize the Dummy_Calculator with the specified edit cost and method.

        """
        # check if there is backup, which has the same parameters as the requested one
        if ((hasattr(Base_Calculator, 'backup') and Base_Calculator.backup is not None)
            and (Base_Calculator.backup.GED_edit_cost == GED_edit_cost and Base_Calculator.backup.GED_calc_method == GED_calc_method)):
            self = Base_Calculator.backup
            if DEBUG:
                print("Dummy_Calculator initialized from backup.")
        else:
            self.GED_edit_cost = GED_edit_cost
            self.GED_calc_method = GED_calc_method
            self.isclalculated = False
            self.dataset_edge_count = None
            self.dataset_node_count = None
            self.lowerbound_matrix = None # is in reality the diffrence of number of nodes
            self.upperbound_matrix = None # is in reality the diffrence of number of edges
            if dataset is not None:
                self.dataset : list[nx.Graph] = dataset
                if labels is not None:
                    if len(labels) != len(dataset):
                        raise ValueError("Labels length must match the number of graphs.")
                    self.labels = labels
                else:
                    self.labels = []
                if activate:
                    self.activate()
            else:
                self.dataset = []
                self.graphindexes = []
                self.labels = []
                self.isactive = False     
            self.runtime = None
            if DEBUG:
                print(f"Initialized Dummy_Calculator with GED_edit_cost={self.GED_edit_cost} and GED_calc_method={self.GED_calc_method}")
            self.make_backup()  # backup itself for later use

    def add_graphs(self, graphs,labels=None):
        self.isclalculated = False
        self.isactive = False
        if not hasattr(self, 'dataset'):
            self.dataset = []
        if not hasattr(self, 'labels'):
            self.labels = []
        for graph in graphs:
            if not isinstance(graph, nx.Graph):
                raise TypeError("All graphs must be of type networkx.Graph")
            self.dataset.append(graph)
        if labels is not None:
            if len(labels) != len(graphs):
                raise ValueError("Labels length must match the number of graphs.")
            # self.labels.extend(labels)  
             

    def set_method(self,GED_calc_method):
        self.GED_calc_method = GED_calc_method
        self.isclalculated = False
        self.isactive = False

    def set_edit_cost(self,GED_edit_cost):     
        self.GED_edit_cost = GED_edit_cost
        self.isactive = False
        self.isclalculated = False

    def activate(self):
        self.isclalculated = False
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
            # self.dataset_edge_count[idx] = 0
            # self.dataset_node_count[idx] = 0
        self.lowerbound_matrix = np.zeros((len(self.dataset), len(self.dataset)))
        self.upperbound_matrix = np.zeros((len(self.dataset), len(self.dataset)))
        self.graphindexes = range(len(self.dataset))
        self.isactive = True
        return self.graphindexes

    def get_dataset(self):
        """        Returns the dataset of graphs.
        """
        return self.dataset
    def get_indexes(self):
        """        Returns the indexes of the graphs in the GEDLIB environment.
        """
        return self.graphindexes
    def get_labels(self):
        """        Returns the labels of the graphs in the GEDLIB environment.
        """
        if hasattr(self, 'labels'):
            return self.labels
        else:
            return None

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
    def calculate(self):
        """
        Computes the GED matrix for the dataset.
        """
        
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        if self.isclalculated:
            print("GED matrix already calculated.")
        else:
            # Start the timer
            self.maxUpperBound = 0
            self.maxLowerBound = 0
            self.max_MeanDistance = 0
            if DEBUG:
                iters = tqdm.tqdm(self.graphindexes, desc='Computing GED Matrix', total=len(self.graphindexes))
            else:
                iters = self.graphindexes
            for i in iters:
                for j in self.graphindexes:
                    self.run_method(i, j)
                    upper_bound = self.upperbound_matrix[i][j]
                    lower_bound = self.lowerbound_matrix[i][j]
                    mean_distance = (upper_bound + lower_bound) / 2
                    if upper_bound > self.maxUpperBound:
                        self.maxUpperBound = upper_bound
                    if lower_bound > self.maxLowerBound:
                        self.maxLowerBound = lower_bound
                    if mean_distance > self.max_MeanDistance:
                        self.max_MeanDistance = mean_distance
            self.isclalculated = True
            print("GED matrix computed.")
     
    def get_runtime(self):
        return self.runtime if hasattr(self, 'runtime') else None
    
    def get_lower_bound(self, graph1_index, graph2_index):
        return self.lowerbound_matrix[graph1_index][graph2_index]
    def get_upper_bound(self, graph1_index, graph2_index):
        return self.upperbound_matrix[graph1_index][graph2_index]
    def get_node_map(self, graph1_index, graph2_index):
        return None  # Dummy implementation, as gedlibpy is not available in this context
    def get_all_map(self, graph1_index, graph2_index):
        return None  # Dummy implementation, as gedlibpy is not available in this context
    def get_forward_map(self, graph1_index, graph2_index):
        return None  # Dummy implementation, as gedlibpy is not available in this context
    def get_backward_map(self, graph1_index, graph2_index):
        return None  # Dummy implementation, as gedlibpy is not available in this context
    def get_assignment_matrix(self, graph1_index, graph2_index):
        return None  # Dummy implementation, as gedlibpy is not available in this context
    # special funtions handmade
    def get_mean_distance(self, graph1_index, graph2_index):
        return (self.get_lower_bound(graph1_index, graph2_index) + self.get_upper_bound(graph1_index, graph2_index)) / 2
    def get_distance(self, graph1_index, graph2_index, method="Mean"):
        if method == "Mean":
            return self.get_mean_distance(graph1_index, graph2_index)
        elif method == "LowerBound":
            return self.get_lower_bound(graph1_index, graph2_index)
        elif method == "UpperBound":
            return self.get_upper_bound(graph1_index, graph2_index)
        else:
            raise ValueError("Invalid method. Choose from 'Mean', 'LowerBound', or 'UpperBound'.")
    
    def get_similarity(self, graph1_index, graph2_index,method="LowerBound"):    
        distance = self.get_distance(graph1_index, graph2_index, method=method)
        if method == "LowerBound":
            return 1 - (distance / self.maxLowerBound) if self.maxLowerBound > 0 else 0
        elif method == "UpperBound":
            return 1 - (distance / self.maxUpperBound) if self.maxUpperBound > 0 else 0
        elif method == "Mean":
            return 1 - (distance / self.max_MeanDistance) if self.max_MeanDistance > 0 else 0
        else:
            raise ValueError("Invalid method. Choose from 'LowerBound', 'UpperBound', or 'Mean'.")
        

    def compare(self, graph1_index, graph2_index, method):
        bound, distance = method.split("-")
        if distance == "Distance":
            return self.get_distance(graph1_index, graph2_index, method=bound)
        elif distance == "Similarity":
            return self.get_similarity(graph1_index, graph2_index, method=bound)
        else:
            raise ValueError("Invalid method. Choose from 'LowerBound-Distance', 'UpperBound-Distance', 'Mean-Distance', 'LowerBound-Similarity', 'UpperBound-Similarity', or 'Mean-Similarity'.")
    def deactivate(self):
        self.isclalculated = False
        self.isactive = False
    def delete_calculation(self):
        self.isclalculated = False
    def delete_dataset(self):
        self.dataset = []
        self.graphindexes = []
        self.labels = []
        self.isactive = False
        self.isclalculated = False
    def get_params(self, deep=True):
        """
        Returns the parameters of the GEDLIB_Calculator.
        """
        return {
            "GED_edit_cost": self.GED_edit_cost,
            "GED_calc_method": self.GED_calc_method,
            # "isactive": self.isactive,
            # "isclalculated": self.isclalculated,
            # "dataset_length": len(self.dataset),
            # "graphindexes_length": len(self.graphindexes),
            # "labels_length": len(self.labels) if hasattr(self, 'labels') else 0
        }
    def set_params(self, **params):
        """
        Sets the attributes of the GEDLIB_Calculator.
        """
        was_active = self.isactive
        was_calculated = self.isclalculated
        for key, value in params.items():
            if key == "GED_edit_cost":
                self.set_edit_cost(value)
            elif key == "GED_calc_method":
                self.set_method(value)
            else:
                raise ValueError(f"Unknown attribute: {key}")
        if was_active:
            self.activate()
            if was_calculated:
                self.calculate()
        return self             
    
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
    
    
    
    


