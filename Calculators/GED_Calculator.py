import math
import re
import traceback
import joblib
import tqdm
# from Calculators.Product_GRaphs import build_restricted_product_graph
from gedlibpy import librariesImport
from gedlibpy import gedlibpy
import numpy as np
from scipy.linalg import inv
import networkx as nx
import subprocess
from joblib import Parallel, delayed
from config_loader import get_conifg_param
ENABLE_NODE_MAPPING = get_conifg_param('GED_Calculator', 'enable_node_mapping', type='bool')
PRINT_GED_DEBUG_INFO = get_conifg_param('GED_Calculator', 'debuging_prints', type='bool')
MENTIONED_MAPPING_FAIL = False
APROXIMATION_METHOD = get_conifg_param('GED_Calculator', 'approximation_method', type='str')
APROXIMATION_BOUND = get_conifg_param('GED_Calculator', 'approximation_bound', type='str')
GEDLIB_EDIT_COST = get_conifg_param('GED_Calculator', 'gedlib_edit_cost', type='str')
GED_distance_matrix_dict_cache = {}
GED_node_map_dict_cache = {}
_dataset_cache = None
Heuristic_distance_matrix_dict_cache = {}
Heuristic_node_map_dict_cache = {}
class abstract_Calculator:
    def get_Name(self):
        raise NotImplementedError
    def get_name(self):
        raise NotImplementedError
    def set_params(self, **params):
        raise NotImplementedError
    def get_params(self, deep=True):
        raise NotImplementedError
    def save_calculator(self, dataset_name):
        raise NotImplementedError
    def get_param_grid(self):
        raise NotImplementedError
    def get_dataset(self):
        raise NotImplementedError
    def compare(self, graph1_index, graph2_index, method):
        raise NotImplementedError
    def get_complete_matrix(self, method, x_graphindexes=None, y_graphindexes=None):
        raise NotImplementedError
    

class GED_Calculator(abstract_Calculator):
    def __init__(self, dataset_name=None, **kwargs):
        global GED_distance_matrix_dict_cache
        global GED_node_map_dict_cache
        global _dataset_cache
        self.distance_matrix_dict = GED_distance_matrix_dict_cache
        self.dataset_name = dataset_name
        self.identifier_name = f"GED_Calculator_{dataset_name}"
        self.node_map = GED_node_map_dict_cache
        self.dataset = _dataset_cache
        self.params = kwargs
        self.name = "GED_Calculator"
        self.isactive = True
        self.isclalculated = True
        if self.distance_matrix_dict is None or self.distance_matrix_dict == {}:
            print("Warning: Distance matrix dictionary is not set or empty.")
            backup = load_GED_calculator(dataset_name)  # default to MUTAG
            self.distance_matrix_dict = backup.distance_matrix_dict
            self.node_map = backup.node_map
            self.dataset = backup.dataset
            self.params = backup.params
            self.isactive = backup.isactive
            self.isclalculated = backup.isclalculated
            self.name = backup.name

            GED_distance_matrix_dict_cache = self.distance_matrix_dict
            GED_node_map_dict_cache = self.node_map
            _dataset_cache = self.dataset

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in GED_Calculator.")
        return self
    def get_params(self, deep=True):
        return self.params
    def get_name(self):
        return self.name
    def get_Name(self):
        return self.name
    def get_node_map(self, graph1_index, graph2_index, method):
        if method is None:
            method = self.distance_matrix_dict.keys()[0]
        elif method not in self.node_map:
            if self.node_map is None:
                raise ValueError("Node map dictionary is not set.")
            raise ValueError(f"Node map for method {method} not available.")
        return self.node_map[method][graph1_index][graph2_index]
    def compare(self, graph1_index, graph2_index, method):
        if self.distance_matrix_dict is None or self.distance_matrix_dict == {}:
            raise ValueError("Distance matrix dictionary is not set.")
        if method is None:
           method = self.distance_matrix_dict.keys()[0]
        elif method not in self.distance_matrix_dict:
            raise ValueError(f"Distance matrix for method {method} not available.")
        return self.distance_matrix_dict[method][graph1_index][graph2_index]
    
    def get_complete_matrix(self, method, x_graphindexes=None, y_graphindexes=None):
        if method is None:
            method = self.distance_matrix_dict.keys()[0]
        elif method not in self.distance_matrix_dict:
            if self.distance_matrix_dict is None or self.distance_matrix_dict == {}:
                raise ValueError("Distance matrix dictionary is not set.")
            raise ValueError(f"Distance matrix for method {method} not available.")
        if x_graphindexes is None and y_graphindexes is None:
            return self.distance_matrix_dict[method]
        elif y_graphindexes is None:
            return self.distance_matrix_dict[method][np.ix_(x_graphindexes, x_graphindexes)]
        else:
            return self.distance_matrix_dict[method][np.ix_(x_graphindexes, y_graphindexes)]
    def get_dataset(self):
        return self.dataset
        # If specific graph indexes are provided, return the submatrix
    def save_calculator(self, dataset_name):
        self.identifier_name = self.get_Name() + f"_{dataset_name}"
        filename = self.get_Name() + f"_{dataset_name}.joblib"
        filepath = "presaved_data/" + filename
        joblib.dump(self, filepath)
    def get_identifier_name(self):
        return self.identifier_name
    def get_param_grid(self):
        return {"method": list(self.distance_matrix_dict.keys())}
        

class Heuristic_Calculator(abstract_Calculator):
    def __init__(self, **kwargs):
        self.distance_matrix_dict = Heuristic_distance_matrix_dict_cache
        self.node_map = Heuristic_node_map_dict_cache
        self.dataset = _dataset_cache
        self.params = kwargs
        self.name = "Heuristic_Calculator"
        self.isactive = True
        self.isclalculated = True
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
    def get_name(self):
        return self.name
    def get_identifier_name(self):
        return self.identifier_name
    def save_calculator(self, dataset_name):
        self.identifier_name = self.get_Name() + f"_{dataset_name}"
        filename = self.get_Name() + f"_{dataset_name}.joblib"
        filepath = "presaved_data/" + filename
        joblib.dump(self, filepath)
    def get_param_grid(self):
        return {"method": list(self.distance_matrix_dict.keys())}
    def compare(self, graph1_index, graph2_index, method):
        if method is None:
            print("Warning, no method provided, using the first available method.")
            method = list(self.distance_matrix_dict.keys())[0]
        elif method not in self.distance_matrix_dict:
            if self.distance_matrix_dict is None or self.distance_matrix_dict == {}:
                raise ValueError("Distance matrix dictionary is not set.")
            raise ValueError(f"Distance matrix for method {method} not available.")
        return self.distance_matrix_dict[method][graph1_index][graph2_index]
    def get_complete_matrix(self, method, x_graphindexes=None, y_graphindexes=None):
        if method is None:
            print("Warning, no method provided, using the first available method.")
            method = list(self.distance_matrix_dict.keys())[0]
        elif method not in self.distance_matrix_dict:
            if self.distance_matrix_dict is None or self.distance_matrix_dict == {}:
                raise ValueError("Distance matrix dictionary is not set.")
            raise ValueError(f"Distance matrix for method {method} not available.")
        if x_graphindexes is None and y_graphindexes is None:
            return self.distance_matrix_dict[method]
        elif y_graphindexes is None:
            return self.distance_matrix_dict[method][np.ix_(x_graphindexes, x_graphindexes)]
        else:
            return self.distance_matrix_dict[method][np.ix_(x_graphindexes, y_graphindexes)]
    def get_node_map(self, method, x_graphindexes=None, y_graphindexes=None):
        raise NotImplementedError("Node map retrieval not implemented in Heuristic_Calculator.")
    


def build_GED_calculator(GED_edit_cost=GEDLIB_EDIT_COST, GED_calc_methods=[(APROXIMATION_METHOD,APROXIMATION_BOUND)], dataset=None, labels=None,dataset_name=None, **kwargs) -> GED_Calculator:
    if dataset is None:
        raise ValueError("Dataset and labels must be provided to build GED_Calculator.")
    with tqdm.tqdm(total=((len(dataset)*(len(dataset)+1)/2)+2)*len(GED_calc_methods)) as pbar:
        for method, bound in GED_calc_methods:
            if method == "Dummy":
                pass

            else:
                gedlibpy.restart_env()
                for graph in dataset:
                    if not isinstance(graph, nx.Graph):
                        raise TypeError("All graphs must be of type networkx.Graph")
                    gedlibpy.add_nx_graph(graph, "")
                gedlibpy.set_edit_cost(GED_edit_cost)
                gedlibpy.init()
                gedlibpy.set_method(method, "")
                gedlibpy.init_method()
                pbar.update(1)
                n = len(dataset)
                distance_matrix = np.zeros((n, n))
                node_map_matrix = np.empty((n, n), dtype=object)
                for i in range(n):
                    for j in range(i, n):
                        gedlibpy.run_method(i, j)
                        if bound == APROXIMATION_BOUND:
                            distance = gedlibpy.get_upper_bound(i, j)
                        else:
                            distance = gedlibpy.get_lower_bound(i, j)
                        distance_matrix[i][j] = distance
                        distance_matrix[j][i] = distance  # Symmetric
                        node_map = gedlibpy.get_node_map(i, j)
                        node_map_matrix[i][j] = node_map
                        node_map_matrix[j][i] = [(b, a) for (a, b) in node_map]  # Reverse mapping
                        pbar.update(1)
                GED_distance_matrix_dict_cache[method] = distance_matrix
                GED_node_map_dict_cache[method] = node_map_matrix
                pbar.update(1)
    global _dataset_cache
    _dataset_cache = dataset
    return GED_Calculator(dataset_name=dataset_name)

def build_Heuristic_calculator(GED_edit_cost=GEDLIB_EDIT_COST, GED_calc_methods=["Vertex","Edge"], dataset=None, labels=None, **kwargs) -> Heuristic_Calculator:
    if dataset is None or labels is None:
        raise ValueError("Dataset and labels must be provided to build Heuristic_Calculator.")
    def run_method(graph1, graph2, method):
        if method == "Vertex":
            return math.fabs(graph1.number_of_nodes() - graph2.number_of_nodes())
        elif method == "Edge":
            return math.fabs(graph1.number_of_edges() - graph2.number_of_edges())
        elif method == "Combined":
            return (math.fabs(graph1.number_of_nodes() - graph2.number_of_nodes()) +
                    math.fabs(graph1.number_of_edges() - graph2.number_of_edges()))
        else:
            raise ValueError(f"Unknown heuristic method: {method}")
    with tqdm.tqdm(total=((len(dataset)*(len(dataset))/2)+2)*len(GED_calc_methods)) as pbar:
        for method in GED_calc_methods:
            if method == "Dummy":
                pass

            else:
                pbar.update(1)
                n = len(dataset)
                distance_matrix = np.zeros((n, n))
                node_map_matrix = np.empty((n, n), dtype=object)
                for i, g1 in enumerate(dataset):
                    for j, g2 in enumerate(dataset):
                        if i < j:  # Ensure unique pairs
                            distance = run_method(g1, g2, method)
                            distance_matrix[i][j] = distance
                            distance_matrix[j][i] = distance  # Symmetric
                            pbar.update(1)
                Heuristic_distance_matrix_dict_cache[method] = distance_matrix
                pbar.update(1)
    global _dataset_cache
    _dataset_cache = dataset
    return Heuristic_Calculator()
    
def load_GED_calculator(dataset_name: str) -> GED_Calculator:
    filename = "GED_Calculator_" + dataset_name + ".joblib"
    filepath = "presaved_data/" + filename
    ged_calculator: GED_Calculator = joblib.load(filepath)
    global GED_distance_matrix_dict_cache
    global GED_node_map_dict_cache
    global _dataset_cache
    ged_calculator.identifier_name = ged_calculator.get_Name() + f"_{dataset_name}"
    GED_distance_matrix_dict_cache = ged_calculator.distance_matrix_dict
    GED_node_map_dict_cache = ged_calculator.node_map
    _dataset_cache = ged_calculator.dataset

    return ged_calculator
def load_Heuristic_calculator(dataset_name: str) -> Heuristic_Calculator:
    filename = "Heuristic_Calculator_" + dataset_name + ".joblib"
    filepath = "presaved_data/" + filename
    heuristic_calculator: Heuristic_Calculator = joblib.load(filepath)
    heuristic_calculator.identifier_name = heuristic_calculator.get_Name() + f"_{dataset_name}"
    global Heuristic_distance_matrix_dict_cache
    global Heuristic_node_map_dict_cache
    global _dataset_cache
    Heuristic_distance_matrix_dict_cache = heuristic_calculator.distance_matrix_dict
    Heuristic_node_map_dict_cache = heuristic_calculator.node_map
    _dataset_cache = heuristic_calculator.dataset

    return heuristic_calculator

def load_calculator_from_id(identifier_name: str):
    if identifier_name.startswith("GED_Calculator_"):
        return  load_GED_calculator(identifier_name[len("GED_Calculator_"):])
    elif identifier_name.startswith("Heuristic_Calculator_"):
        return load_Heuristic_calculator(identifier_name[len("Heuristic_Calculator_"):])
    elif identifier_name.startswith("Randomwalk_GED_Calculator_"):
        return load_Randomwalk_GED_calculator(identifier_name[len("Randomwalk_GED_Calculator_"):])
    elif identifier_name.startswith("Exact_GED_"):
        return load_exact_GED_calculator(identifier_name[len("Exact_GED_"):])
    else:
        raise ValueError(f"Unknown calculator identifier: {identifier_name}")
    

_adj_matrices_dict_cache = {}
_precomputed_walk_traces = {}  
class Randomwalk_GED_Calculator:
    def __init__(self, **kwargs):
        self.dataset = _dataset_cache
        self.adj_matrices_dict = _adj_matrices_dict_cache
        self.precomputed_walk_traces = _precomputed_walk_traces
        self.params = kwargs
        self.isactive = True
        self.is_calculated = True
        self.name = "Randomwalk_GED_Calculator"
    def get_Name(self):
        return self.name
    def get_name(self):
        return self.name
    def get_identifier_name(self):
        return self.identifier_name
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in Randomwalk_GED_Calculator.")
        return self
    def get_params(self, deep=True):
        return self.params
    def save_calculator(self, dataset_name):
        self.identifier_name = self.get_Name() + f"_{dataset_name}"
        filename = self.get_Name() + f"_{dataset_name}.joblib"
        filepath = "presaved_data/" + filename
        joblib.dump(self, filepath)
    def get_adj_matrix(self, g1_index, g2_index,method):
        if method is None:
            method = list(self.adj_matrices_dict.keys())[0]
        elif method not in self.adj_matrices_dict:
            if self.adj_matrices_dict is None:
                raise ValueError("Adjacency matrix dictionary is not set.")
            raise ValueError(f"Adjacency matrices for method {method} not available.")
        try:
            return self.adj_matrices_dict[method][g1_index][g2_index]
        except IndexError:
            raise IndexError(f"Graph indexes {g1_index}, {g2_index} out of bounds for adjacency matrices.")
    def get_limited_length_walk(self, g1_index, g2_index, llambda, max_length,method):
        if g1_index > g2_index:
            g1_index, g2_index = g2_index, g1_index
        if g1_index == g2_index:
            return 1.0
        total_similarity = 0.0
        for step in range(max_length):
            trace = self.precomputed_walk_traces[method][g1_index][g2_index][step]
            if trace is None:
                raise ValueError("Traces not precomputed for this pair/method or max_length too large.")
            total_similarity += (llambda ** step) * trace
        return total_similarity
    
    def get_exact_inflength_walk(self, g1_index, g2_index, llambda,method):
        if g1_index > g2_index:
            g1_index, g2_index = g2_index, g1_index
        # check if indexes are equal
        if g1_index == g2_index:
            return 1.0
        adj_matrix = self.get_adj_matrix(g1_index, g2_index,method)
        identity_matrix = np.eye(adj_matrix.shape[0])
        try:
            kernel_matrix = inv(identity_matrix - llambda * adj_matrix)
            kernel_value = np.sum(kernel_matrix)
             
        except np.linalg.LinAlgError as e:
            print(f"Warning: Matrix is not invertible. {e}")
            kernel_value = 18446744073709551614
        if np.isnan(kernel_value):
            raise ValueError(f"Exact similarity for graphs {g1_index} and {g2_index} with lambda={llambda} is negative or NaN ({kernel_value})")
        float_result = float(kernel_value)
        if np.isnan(float_result):
            print(f"Warning: Similarity for graphs {g1_index} and {g2_index} with lambda={llambda} is negative or NaN ({float_result}). Setting to 0.")
            return 0.0
        return float_result
    
def build_restricted_product_graph(g1: nx.Graph, g2: nx.Graph, node_matches : list[(int,int)]):
    restricted_graph = nx.Graph()

    lowest_node_id_g1 = min(g1.nodes)
    lowest_node_id_g2 = min(g2.nodes)
    higest_node_id_g1 = max(g1.nodes)
    max_id = max(max(g1.nodes), max(g2.nodes))
    # sorted_node_matches = sorted(node_matches, key=lambda x: (x[0], x[1]))
    for (node1, node2) in node_matches.items():
        if node1 == 18446744073709551614 or node2 == 18446744073709551614:
            continue
        else:
            node1_id = node1 + lowest_node_id_g1
            node2_id = node2 + lowest_node_id_g2
            # label = (g1.nodes[node1_id].get('label', None), g2.nodes[node2_id].get('label', None))
            restricted_graph.add_node((node1_id, node2_id,))
    # rather inefficient O(n^2) approach,
    # possible better to iterate over the edges of g1, and check if the corresponding edge exists in g2
    # map the node id of g1 to the node id of g2
    g1_node_map = {node1 + lowest_node_id_g1: node2 + lowest_node_id_g2 for (node1, node2) in node_matches.items()}
    # for (u2,v2) in g2.edges:
    for (u1,v1) in g1.edges:
        u2 = g1_node_map.get(u1, None)
        v2 = g1_node_map.get(v1, None)
        if u2 is not None and v2 is not None and g2.has_edge(u2, v2):
            label = g1.edges[(u1, v1)].get('label', None)
            restricted_graph.add_edge((u1, u2), (v1, v2))
            # print(f"Added edge between ({u1}, {u2}) and ({v1}, {v2}) with label {label}")
        
        
    

    # for (u1, u2) in sorted_node_matches:
    #     u1_id = u1 + lowest_node_id_g1
    #     u2_id = u2 + lowest_node_id_g2
    #     for (v1, v2) in sorted_node_matches:
    #         v1_id = v1 + lowest_node_id_g1
    #         v2_id = v2 + lowest_node_id_g2
    #         if g1.has_edge(u1_id, v1_id)  and g2.has_edge(u2_id, v2_id):
    #             label =g1.edges[(u1_id, v1_id)].get('label', None)
    #             restricted_graph.add_edge((u1_id, u2_id), (v1_id, v2_id),label=label)
    return restricted_graph
def precompute_walk_traces(adj_matrix, max_length=7):
    traces = np.zeros(max_length)
    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = adj_matrix / row_sums[:, np.newaxis]
    
    walk_distribution = np.identity(adj_matrix.shape[0])
    traces[0] = np.trace(walk_distribution)
    for step in range(1, max_length):
        walk_distribution = np.dot(walk_distribution, transition_matrix)
        trace = np.trace(walk_distribution)
        traces[step] = trace
    return traces



def build_Randomwalk_GED_calculator(ged_calculator,max_walk_length=7, **kwargs) -> Randomwalk_GED_Calculator:
    if ged_calculator is None or not ged_calculator.isactive or not ged_calculator.isclalculated:
        raise RuntimeError("GED calculator is not initialized or not active. Call init() first.")
    dataset = ged_calculator.get_dataset()
    def calculate_prod_graph(i,j,g1,g2,method):
        if i == j:
            return nx.to_numpy_array(g1)
        else:
            node_map = ged_calculator.get_node_map(i, j, method=method)
            product_graph = build_restricted_product_graph(g1, g2, node_map)
            nodelist = list(product_graph.nodes())
            return nx.to_numpy_array(product_graph, nodelist=nodelist)
    if isinstance(ged_calculator, exact_GED_Calculator):
        GED_calc_methods = ["Exact"]
    elif isinstance(ged_calculator, GED_Calculator):
        GED_calc_methods = list(ged_calculator.distance_matrix_dict.keys())
    elif( isinstance(ged_calculator, Heuristic_Calculator)):
        GED_calc_methods = list(ged_calculator.distance_matrix_dict.keys())
    else:
        raise ValueError("Unknown type of ged_calculator provided.")
    with tqdm.tqdm(total=((len(dataset)*(len(dataset)+1)/2)+2)*len(GED_calc_methods)) as pbar:
        n = len(dataset)
        adj_matrices_dict = {}
        precomputed_walk_traces = {}
        for method in GED_calc_methods:
            adj_matrices = [[None for _ in range(n)] for _ in range(n)]
            walk_traces = [[[None for _ in range(max_walk_length)] for _ in range(n)] for _ in range(n)]
            if method == "Dummy":
                pass
            
            else:
                for i, g1 in enumerate(dataset):
                    for j, g2 in enumerate(dataset):
                        if i <= j:  # Ensure unique pairs and include self-pairs
                            adj_matrices[i][j] = calculate_prod_graph(i,j,g1,g2,method)
                            walk_traces[i][j] = precompute_walk_traces(adj_matrices[i][j], max_length=max_walk_length)
                            pbar.update(1)
            adj_matrices_dict[method] = adj_matrices
            precomputed_walk_traces[method] = walk_traces
            pbar.update(1)
    global _dataset_cache
    global _adj_matrices_dict_cache
    global _precomputed_walk_traces
    _dataset_cache = dataset
    _adj_matrices_dict_cache = adj_matrices_dict
    _precomputed_walk_traces = precomputed_walk_traces
    calculator = Randomwalk_GED_Calculator()
    dataset_name = ged_calculator.get_identifier_name()[len("Exact_GED_"):]
    calculator.save_calculator(dataset_name)
    return calculator

def load_Randomwalk_GED_calculator(dataset_name: str) -> Randomwalk_GED_Calculator:
    filename = "Randomwalk_GED_Calculator_" + dataset_name + ".joblib"
    filepath = "presaved_data/" + filename
    randomwalk_ged_calculator: Randomwalk_GED_Calculator = joblib.load(filepath)
    global _adj_matrices_dict_cache
    global _precomputed_walk_traces
    global _dataset_cache
    randomwalk_ged_calculator.identifier_name = randomwalk_ged_calculator.get_Name() + f"_{dataset_name}"
    _adj_matrices_dict_cache = randomwalk_ged_calculator.adj_matrices_dict
    _precomputed_walk_traces = randomwalk_ged_calculator.precomputed_walk_traces
    _dataset_cache = randomwalk_ged_calculator.dataset

    return randomwalk_ged_calculator
def load_Randomwalk_calculator_from_id(identifier_name: str):
    if identifier_name.startswith("Randomwalk_GED_Calculator_"):
        return  load_Randomwalk_GED_calculator(identifier_name[len("Randomwalk_GED_Calculator_"):])
    else:
        raise ValueError(f"Unknown calculator identifier: {identifier_name}")

def try_load_else_build_rw_calculator(ged_calculator, max_walk_length=7):
    dataset_name = ged_calculator.get_identifier_name()[len("Exact_GED_"):]
    try:
        rw_calculator = load_Randomwalk_GED_calculator(dataset_name)
        if rw_calculator is None:
            raise ValueError("Loaded Randomwalk_GED_Calculator is None.")
        print(f"Loaded precomputed Randomwalk_GED_Calculator for dataset {dataset_name}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Precomputed Randomwalk_GED_Calculator for dataset {dataset_name} not found or failed to load ({e}). Building new one...")
        rw_calculator = build_Randomwalk_GED_calculator(ged_calculator, max_walk_length=max_walk_length)
        print(f"Built and saved new Randomwalk_GED_Calculator for dataset {dataset_name}.")
    return rw_calculator

_ged_matrix: np.ndarray = None
class exact_GED_Calculator(GED_Calculator):
    def __init__(self, dataset_name, **kwargs):
        global _ged_matrix
        global GED_node_map_dict
        global _dataset_cache
        self.distance_matrix = _ged_matrix
        self.node_map_dict = GED_node_map_dict
        self.dataset = _dataset_cache
        self.name = "Exact_GED"
        self.dataset_name = dataset_name
        self.identifier_name = f"Exact_GED_{dataset_name}"
        self.params = kwargs
        self.isactive = True
        self.isclalculated = True
        self.save_calculator(dataset_name)
        # a dict, that counts for every model, the numer of times a ged value was accsessed.
        accesses_counts_dict = {}
        # Any specific initialization for exact GED can be added here
    def get_node_map(self, graph1_index, graph2_index, method):
        return self.node_map_dict[graph1_index][graph2_index]
    def compare(self, g1, g2,method=None,**kwargs):
        return self.distance_matrix[g1, g2]
    def get_dataset(self):
        return self.dataset
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            # no specific parameters for exact GED
        })
        return params
    def get_complete_matrix(self, method, x_graphindexes=None, y_graphindexes=None):
        if x_graphindexes is None and y_graphindexes is None:
            return self.distance_matrix
        elif y_graphindexes is None:
            return self.distance_matrix[np.ix_(x_graphindexes, x_graphindexes)]
        else:
            return self.distance_matrix[np.ix_(x_graphindexes, y_graphindexes)]
   



def calculate_ged_between_two_graphs(dataset_name,g_id1, g_id2,node_size_i,node_size_j,nx_1,nx_2,timeout=5, lb=0,gedlibpy_edit_cost="CONSTANT",gedlibpy_method="IPFP",use_node_mapping=ENABLE_NODE_MAPPING):
    # load the graphs from files
    filepath1 = None
    filepath2 = None
    
    if node_size_i < node_size_j:
        filepath1 = f"Datasets/ged/{dataset_name}/g_{g_id1}.txt"
        filepath2 = f"Datasets/ged/{dataset_name}/g_{g_id2}.txt"
    else:
        filepath1 = f"Datasets/ged/{dataset_name}/g_{g_id2}.txt"
        filepath2 = f"Datasets/ged/{dataset_name}/g_{g_id1}.txt"
   
    try:
        command = ["Graph_Edit_Distance/ged", "-q", filepath1, "-d", filepath2, "-g"]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        output: str = process.stdout.decode()
        err_output: str = process.stderr.decode()

        # Extract GED
        ged_match = re.search(r"GED: (\d+)", output)
        ged = None
        if ged_match:
            ged = int(ged_match.group(1))
            
        else:
            raise Exception(
                "GED value not found in output:"
                + "\nSTDOUT:\n "
                + output
                + "\nSTDERR:\n "
                + err_output
            )

        # Extract mapping
        mapping_match = re.search(r"Mapping: (.+)", output)
        mapping = None
        if mapping_match:
            mapping: dict[int, int] = {}
            pairs = mapping_match.group(1).split(", ")
            for pair in pairs:
                if "->" in pair:
                    q, g = map(int, pair.split(" -> "))
                    mapping[q] = g
            # print(mapping)
        else:
            if use_node_mapping:
                if not MENTIONED_MAPPING_FAIL:
                    print("WARNING:")
                    print("There was no mapping found in the output of the ged binary.")
                    print("likely, the wrong version is being used.")
                    print("The mapping is required for enabling the Randomwalk GED calculation.")
                    print(
                        f"Mapping not found in output for graphs {g_id1} and {g_id2}."
                        + "\nSTDOUT:\n "
                        + output
                        + "\nSTDERR:\n "
                        + err_output
                    )
                    print("Continuing without mapping... might produce errors for later")
                    print("Not Showing this warning again...")
                    MENTIONED_MAPPING_FAIL = True

        # ===
        # Extract total time. For some unknown reason, time is not always
        # present in the binary's output, hence None is also accepted here.
        # ===
        total_time_match = re.search(
            r"Total time: ([\d,]+) \(microseconds\)", output
        )
        time = (
            int(total_time_match.group(1).replace(",", ""))
            if total_time_match
            else None
        )
        if PRINT_GED_DEBUG_INFO:
            print(f"Computed exact GED {ged} for graphs {g_id1} and {g_id2} in {time} microseconds.")
        if use_node_mapping:
            return ged, mapping, time, 0
        else:
            return ged, None, time, 0
        



    except subprocess.TimeoutExpired:
        if PRINT_GED_DEBUG_INFO:
            print(f"Timeout expired for graphs {g_id1} and {g_id2} in {timeout} seconds.")
        return None, None, None, None

        # global _ged_matrix
def run_ged_calculation(args):
    i, j, dataset_name, node_size_i, node_size_j, nx_1, nx_2, timeout = args
    # Unpack all your arguments    
    try:
        # Call your original function
        # NOTE: Your original function must return (i, j) so we know which task is finished
        (ged, mapping_dict, time, approx_ged) = calculate_ged_between_two_graphs(
            dataset_name, i, j, 
            node_size_i=node_size_i, 
            node_size_j=node_size_j, 
            nx_1=nx_1, nx_2=nx_2, 
            timeout=timeout
        )
        return (i, j, ged, mapping_dict, time, approx_ged, None) # Add 'None' for no error
    except Exception as e:
        # Return the error to be handled in the main process
        return (i, j, None, None, None, None, e)
    
def build_exact_ged_calculator(dataset=None, dataset_name=None, n_jobs=1, timeout=10, **kwargs) -> exact_GED_Calculator:
    # we assume the Dataset is already loadded, but also we have the files of the Graphs in the directories.
    n = len(dataset)
    # we distribute the Jobs, sot that every Jobs needs to caclulate the same amount of GEDs
    global _ged_matrix
    global GED_node_map_dict
    global _dataset_cache
    _ged_matrix = np.zeros((n,n), dtype=np.int32)
    GED_node_map_dict = np.empty((n,n), dtype=object)
    node_sizes = [len(g.nodes()) for g in dataset]
    # first we compute the diagonal
    _dataset_cache = [None for _ in range(n)]
    for i in range(n):
        _dataset_cache[i] = dataset[i]
        _ged_matrix[i,i] = 0
        if ENABLE_NODE_MAPPING:
            GED_node_map_dict[i,i] = {k:k for k in range(len(dataset[i].nodes()))}
        else:
            GED_node_map_dict[i,i] = {}
        node_sizes[i] = len(dataset[i].nodes())
    # then we compute the upper triangle
    # we distribute the Jobs so that every Jobs needs to caclulate the same amount of GEDs
    tasks = [] # will be n/2 tasks
    
    for i in range(n):
        for j in range(i+1, n):
            task_args = (
                i, j, dataset_name, 
                node_sizes[i], node_sizes[j], 
                dataset[i], dataset[j], 
                timeout
                )
            tasks.append(task_args)
    # start parallel processing
    # for every entry in task the GED needs to be calculated
    # build a Normal GedLIbpy Calculator as backup Approx values
    gedlipy_calculator: GED_Calculator = build_GED_calculator(dataset=dataset, dataset_name=dataset_name, GED_calc_methods=[(APROXIMATION_METHOD, APROXIMATION_BOUND)])
    approx_ged_matrix = gedlipy_calculator.get_complete_matrix(method=APROXIMATION_METHOD)
    approx_mappings = gedlipy_calculator.node_map[APROXIMATION_METHOD]
    # save the calculator
    gedlipy_calculator.save_calculator(dataset_name)
    print(f"Starting calculation of exact GED distance matrix with {n_jobs} parallel jobs...")



    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_ged_calculation)(task_args)
            for task_args in tasks
        )
    except Exception as e:
        print(f"Error during parallel GED calculation: {e}")
        traceback.print_exc()
    # write results into the distance matrix (symmetric)
    approximation_counter = 0
    deviation_sum = 0.0
    times = 0
    num_times =0
    for (i, j, ged, mapping_dict, time, approx_ged, error) in tqdm.tqdm(results, desc="Processing GED results", total=len(results)):
        if error is not None:
            print(f"Error calculating GED between graphs {i} and {j}: {error}")
            raise error
        else:
            if ged is None:
                ged = approx_ged_matrix[i, j]
                approximation_counter += 1
            else:
                # calculate the devation between approx and exact
                deviation_sum += abs(ged - approx_ged_matrix[i, j])
        times += time if time is not None else 0
        num_times += 1 if time is not None else 0

        _ged_matrix[i, j] = ged
        _ged_matrix[j, i] = ged
        if mapping_dict is None:
            mapping = approx_mappings[i][j]
            approx_mapping_dict = {}
            for (a, b) in mapping:
                if a == 18446744073709551614 or b == 18446744073709551614:
                    continue
                approx_mapping_dict[a] = b           
            mapping_dict = approx_mapping_dict
        GED_node_map_dict[i, j] = mapping_dict
        # reverse mapping
        reverse_mapping = {v: k for k, v in mapping_dict.items()}
        GED_node_map_dict[j, i] = reverse_mapping
    # process results
    print(f"Number of approximations used due to timeouts: {approximation_counter} out of {len(tasks)}")
    rel_deviation = deviation_sum / (len(tasks) - approximation_counter) if len(tasks) > 0 else 0.0
    print(f"Average deviation between approximate and exact GED: {rel_deviation:.4f}")
    # create GED_Calculator_object
    ged_calculator = exact_GED_Calculator(dataset_name=dataset_name)
    average_time = times / num_times if num_times > 0 else 0
    print(f"Average time per GED computation: {average_time} microseconds ({average_time/1e6} seconds)")
    print(f"Total number of GED computations: {len(tasks)}")

    print(f"total time for GED computations: {times/1e6} seconds")
    return ged_calculator, approximation_counter, rel_deviation, average_time

def load_exact_GED_calculator(dataset_name: str) -> exact_GED_Calculator:
    filename = f"Exact_GED_{dataset_name}.joblib"
    if PRINT_GED_DEBUG_INFO:
        print(f"Loading Exact_GED_Calculator for {dataset_name}...")
    filepath = f"presaved_data/{filename}"
    exact_ged_calculator: exact_GED_Calculator = joblib.load(filepath)
    global _ged_matrix
    global GED_node_map_dict
    global _dataset_cache
    exact_ged_calculator.identifier_name = exact_ged_calculator.get_Name() + f"_{dataset_name}"
    _ged_matrix = exact_ged_calculator.distance_matrix
    GED_node_map_dict = exact_ged_calculator.node_map_dict
    _dataset_cache = exact_ged_calculator.dataset
    return exact_ged_calculator
    
def reset_calculators_cache():
    global _dataset_cache
    global _adj_matrices_dict_cache
    global _precomputed_walk_traces
    global _ged_matrix
    global GED_node_map_dict
    _dataset_cache = None
    _adj_matrices_dict_cache = None
    _precomputed_walk_traces = None
    _ged_matrix = None
    GED_node_map_dict = None
    
