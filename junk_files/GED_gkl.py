# GED computation with Graphkit Learn
# impoerts
import gklearn as gkl
from gklearn.ged.model import distances, ged_com, ged_model, optim_costs
from gklearn.ged.edit_costs import edit_cost
from gklearn.ged import env
from gklearn.ged.env import ged_data
import numpy as np
import networkx as nx
import tqdm
from gklearn.ged.env import Options
from gklearn.ged.methods.ged_method import GEDMethod
DEBUG = True

class gkl_calculator:

    def __init__(self,method=Options.GEDMethod.BIPARTITE,gamma=0.1):
        self.env = env.GEDEnv()
        self.set_method(Options.GEDMethod.BRANCH, recalculate=False)
        self.graphindexes = []
        
        self.isclalculated = False
        self.ged_matrix = None
        
    
    def add_graphs(self, graphs,labels=None):
        if labels is not None:
            if len(labels) != len(graphs):
                raise ValueError("Labels length must match the number of graphs.")
            else:
                self.labels = labels
        for i in range(len(graphs)):
            graph = graphs[i]
            if isinstance(graph, (nx.Graph, nx.DiGraph)):
                gi=self.env.add_nx_graph(graph,labels[i],ignore_duplicates=True)
                self.graphindexes.append(gi)
            else:
                raise TypeError("Unsupported graph type. Use gklearn Graph or NetworkX Graph.")
        return self.graphindexes   
        

    def set_method(self,method,recalculate=True):
        """
        Set the method for GED computation.
        """
        self.env.set_method(Options.GEDMethod.BIPARTITE,dict())
        self.isclalculated = False
        if DEBUG:
            print(f"GED method set to {method}")
        if recalculate:
            self.compute_complete_matrix()
    
    
    def compute_complete_matrix(self):
        self.env.set_edit_cost(Options.EditCosts.CONSTANT, [1,1,1,1,1,1])
        self.env.init()
        self.env.init_method()
        self.ged_matrix = np.zeros((len(self.graphindexes), len(self.graphindexes)))
        if DEBUG:
            iters = tqdm.tqdm(range(len(self.graphindexes)), desc='Computing GED Matrix', total=len(self.graphindexes))
        else:
            iters = range(len(self.graphindexes))
        for i in iters:
            for j in range(len(self.graphindexes)):
                distance=self.env.run_method(self.graphindexes[i], self.graphindexes[j])
                self.ged_matrix[j, i] = distance
        self.isclalculated = True
        self.max_distance = np.max(self.ged_matrix)
        return self.ged_matrix
    def compute_similarity(self,graph1_index, graph2_index):
        if not self.isclalculated:
            raise ValueError("GED matrix has not been computed yet. Call compute_complete_matrix() first.")
        else:
            distance = self.compute_distance(graph1_index, graph2_index)
            return 1 - (distance / self.max_distance) if self.max_distance > 0 else 0

    def compute_distance(self, graph1_index, graph2_index):
        if not self.isclalculated:
            return self-env.compute_ged(graph1_index, graph2_index)
        else:
            return self.ged_matrix[graph1_index, graph2_index]
        
    def partial_matrix(self, graph_indices1, graph_indices2=None):
        if not self.isclalculated:
            raise ValueError("GED matrix has not been computed yet. Call compute_complete_matrix() first.")
        if graph_indices2 is None:
            graph_indices2 = graph_indices1
        partial_matrix = np.zeros((len(graph_indices1), len(graph_indices2)))
        for i, g1 in enumerate(graph_indices1):
            for j, g2 in enumerate(graph_indices2):
                partial_matrix[i, j] = self.compute_distance(g1, g2)
        return partial_matrix
    def partial_similarity_matrix(self, graph_indices1, graph_indices2=None):
        if not self.isclalculated:
            raise ValueError("GED matrix has not been computed yet. Call compute_complete_matrix() first.")
        if graph_indices2 is None:
            graph_indices2 = graph_indices1
        partial_similarity_matrix = np.zeros((len(graph_indices1), len(graph_indices2)))
        for i, g1 in enumerate(graph_indices1):
            for j, g2 in enumerate(graph_indices2):
                partial_similarity_matrix[i, j] = self.compute_similarity(g1, g2)
        return partial_similarity_matrix
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        return {
            "ged_method": ['BRANCH_FAST', 'BRANCH_SLOW', 'BRANCH_OPTIMAL']
            #, "gamma": [0.1, 0.5, 1.0]
        }
    
    


