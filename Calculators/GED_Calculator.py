import joblib
from gedlibpy import gelibpy
import numpy as np
import networkx as nx



class GED_Calculator:
    def __init__(self, distance_matrix_dict: dict[str, np.ndarray], node_map: dict[str, np.ndarray], dataset: list, labels: list, **kwargs):
        self.distance_matrix_dict = distance_matrix_dict
        self.node_map = node_map
        self.dataset = dataset
        self.labels = labels
        self.params = kwargs
        self.name = "GED_Calculator"
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in GED_Calculator.")
        return self
    def get_params(self, deep=True):
        return self.params
    def get_Name(self):
        return self.name
    
    def get_node_map(self, graph1_index, graph2_index, method):
        if method not in self.node_map:
            raise ValueError(f"Node map for method {method} not available.")
        return self.node_map[method][graph1_index][graph2_index]
    def get_distance(self, graph1_index, graph2_index, method):
        if method not in self.distance_matrix_dict:
            raise ValueError(f"Distance matrix for method {method} not available.")
        return self.distance_matrix_dict[method][graph1_index][graph2_index]
    def get_complete_matrix(self, method):
        if method not in self.distance_matrix_dict:
            raise ValueError(f"Distance matrix for method {method} not available.")
        return self.distance_matrix_dict[method]
    def save_calculator(self, dataset_name):
        filename = self.get_Name() + f"_{dataset_name}.joblib"
        filepath = "preserved_data/" + filename
        joblib.dump(self, filepath)

class Heuristic_Calculator:
    def __init__(self,distance_matrix_dict: dict[str, np.ndarray],node_map: dict[str, np.ndarray], dataset: list, labels: list, **kwargs):
        self.distance_matrix_dict = distance_matrix_dict
        self.node_map = node_map
        self.dataset = dataset
        self.labels = labels
        self.params = kwargs
        self.name = "Heuristic_Calculator"
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in Heuristic_Calculator.")
        return self
    def get_params(self, deep=True):
        return self.params
    def get_Name(self):
        return self.name
    def save_calculator(self, dataset_name):
        filename = self.get_Name() + f"_{dataset_name}.joblib"
        filepath = "preserved_data/" + filename
        joblib.dump(self, filepath)   


def build_GED_calculator(GED_edit_cost="CONSTANT", GED_calc_methods=[("BIPARTITE","upper")], dataset=None, labels=None, **kwargs) -> GED_Calculator:
    if dataset is None or labels is None:
        raise ValueError("Dataset and labels must be provided to build GED_Calculator.")
    distance_matrix_dict = {}
    node_map_dict = {}
    graph_indices = list(range(len(dataset)))

    for method, bound in GED_calc_methods:
        gelibpy.restart_env()
        for graph in dataset:
            if not isinstance(graph, nx.Graph):
                raise TypeError("All graphs must be of type networkx.Graph")
            gelibpy.add_nx_graph(graph, "")
        gelibpy.set_edit_cost(GED_edit_cost)
        gelibpy.init()
        gelibpy.set_method(method, "")
        gelibpy.init_method()
        n = len(dataset)
        distance_matrix = np.zeros((n, n))
        node_map_matrix = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(i, n):
                gelibpy.run_method(i, j)
                if bound == "upper":
                    distance = gelibpy.get_upper_bound(i, j)
                else:
                    distance = gelibpy.get_lower_bound(i, j)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance  # Symmetric
                node_map = gelibpy.get_node_map(i, j)
                node_map_matrix[i][j] = node_map
                node_map_matrix[j][i] = [(b, a) for (a, b) in node_map]  # Reverse mapping
        distance_matrix_dict[method] = distance_matrix
        node_map_dict[method] = node_map_matrix
    return GED_Calculator(distance_matrix_dict=distance_matrix_dict, node_map=node_map_dict, dataset=dataset, labels=labels, GED_edit_cost=GED_edit_cost, GED_calc_methods=GED_calc_methods, **kwargs), graph_indices

def load_GED_calculator(dataset_name: str) -> GED_Calculator:
    filename = "GED_Calculator_" + dataset_name + ".joblib"
    filepath = "preserved_data/" + filename
    return joblib.load(filepath)

        

