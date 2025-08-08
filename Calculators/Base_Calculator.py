# GED computation with Graphkit Learn
# impoerts

import math
from time import time
import numpy as np
import networkx as nx
import tqdm
DEBUG = True

class Base_Calculator():
    # add class variable, as copy of itself for backup
    backup = None
    
    def __init__(self, GED_edit_cost="CONSTANT", GED_calc_method="BIPARTITE", dataset=None, labels=None, activate: bool = True):
        """
        Initialize the Dummy_Calculator with the specified edit cost and method.

        """
        # check if there is backup, which has the same parameters as the requested one
        if ((hasattr(Base_Calculator, 'backup') and Base_Calculator.backup is not None)
            and (Base_Calculator.backup.GED_edit_cost == GED_edit_cost and Base_Calculator.backup.GED_calc_method == GED_calc_method)):
            backup = Base_Calculator.backup
            self.GED_edit_cost = backup.GED_edit_cost
            self.GED_calc_method = backup.GED_calc_method
            self.isclalculated = backup.isclalculated
            self.dataset_edge_count = backup.dataset_edge_count
            self.dataset_node_count = backup.dataset_node_count
            self.lowerbound_matrix = backup.lowerbound_matrix
            self.upperbound_matrix = backup.upperbound_matrix
            self.dataset = backup.dataset
            self.graphindexes = backup.graphindexes
            self.labels = backup.labels
            self.isactive = backup.isactive
            self.runtime = backup.runtime

            # if DEBUG:
            #     print("Base_Calculator initialized from backup.")
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
        for graph in graphs:
            if not isinstance(graph, nx.Graph):
                raise TypeError("All graphs must be of type networkx.Graph")
            self.dataset.append(graph)
        if labels is not None:
            if len(labels) != len(graphs):
                raise ValueError("Labels length must match the number of graphs.")
            self.labels.extend(labels)  
        self.graphindexes = range(len(self.dataset))
      

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
            print(f"This thing is just spitting random numbers, so activating is pretty fast")

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

    def run_method(self, graph1_index, graph2_index):
        """        Runs the GED method for the specified graph indexes.
        """
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        # generate 2 random integers, the higer one is upper bound the lower one is lower bound
        # n1 = np.random.randint(0, 100)
        # n2 = np.random.randint(0, 100)
        n1= 0
        n2= 0
        if n1 > n2:
            self.upperbound_matrix[graph1_index][graph2_index] = n1
            self.lowerbound_matrix[graph1_index][graph2_index] = n2
        else:
            self.upperbound_matrix[graph1_index][graph2_index] = n2
            self.lowerbound_matrix[graph1_index][graph2_index] = n1
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
    def get_node_image(self, graph1_index, graph2_index,node_index):
        return None
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
    def get_complete_matrix(self, method="Mean-Distance",x_graphindexes=None, y_graphindexes=None):
        if x_graphindexes is None:
            x_graphindexes = self.graphindexes
        if y_graphindexes is None:
            y_graphindexes =  x_graphindexes
        matrix = np.zeros((len(x_graphindexes), len(y_graphindexes)))
        if not self.isclalculated:
            raise ValueError("GED matrix has not been computed yet. Call compute_complete_matrix() first.")
        else:
            for i in range(len(x_graphindexes)):
                for j in range(len(y_graphindexes)):
                    matrix[i, j] = self.compare(x_graphindexes[i], y_graphindexes[j], method)
        return matrix
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
    def make_backup(self):
        # set itself to the backup class variable
        Base_Calculator.backup = self
    def get_name(self):
        """
        Returns the name of the calculator.
        """
        return f"{self.__class__.__name__}"
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
    
    
    
    


