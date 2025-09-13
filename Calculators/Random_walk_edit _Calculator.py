# GED computation with Graphkit Learn
# impoerts

from time import time
import numpy as np
import networkx as nx
import tqdm
from gedlibpy import gedlibpy
from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator 
from Calculators.Base_Calculator import Base_Calculator
DEBUG = False

class Random_walk_edit_Calculator(GEDLIB_Calculator):
    def __init__(self, decay_lambda=0.1,max_walk_path_length=-1, **kwargs):
        super().__init__(**kwargs)
        self.decay_lambda = decay_lambda
        self.max_walk_path_length = max_walk_path_length
        if max_walk_path_length == -1:
            self.random_walk_function = lambda pg: infinte_length_random_walk_similarity(pg, llamda=decay_lambda)
        else:
            self.random_walk_function = lambda pg: limited_length_approx_random_walk_similarity(pg, llamda=decay_lambda, max_length=max_walk_path_length)

        self.sum_bulid_product_graph_time = 0
        self.sum_random_walk_time = 0
    def get_Name(self):
        return "Random_walk_edit_Calculator"
    def run_method(self, graph1_index, graph2_index):
        if not self.isactive:
            raise ValueError("Calculator is not active. Call activate() first.")
        gedlibpy.run_method(graph1_index, graph2_index)
        node_map = gedlibpy.get_node_map(graph1_index, graph2_index)
        if DEBUG:
            print(f"Node map between graphs: {node_map}")
        graph1 = self.dataset[graph1_index]
        graph2 = self.dataset[graph2_index]
        start_time = time()
        product_graph = build_restricted_product_graph(graph1, graph2, node_map)
        build_product_graph_time = time() - start_time
        self.sum_bulid_product_graph_time += build_product_graph_time
        if DEBUG:
            print(f"Built product graph with {product_graph.number_of_nodes()} nodes and {product_graph.number_of_edges()} edges in {build_product_graph_time:.4f} seconds")
        start_time = time()
        similarity = self.random_walk_function(product_graph)
        random_walk_time = time() - start_time
        self.sum_random_walk_time += random_walk_time
        if DEBUG:
            print(f"Computed random walk similarity: {similarity} in {random_walk_time:.4f} seconds")

        self.upperbound_matrix[graph1_index][graph2_index] = similarity
        self.lowerbound_matrix[graph1_index][graph2_index] = similarity
        self.upperbound_matrix[graph2_index][graph1_index] = similarity
        self.lowerbound_matrix[graph2_index][graph1_index] = similarity
        







def build_restricted_product_graph(g1: nx.Graph, g2: nx.Graph, node_matches : list[(int,int)]):
    restricted_graph = nx.Graph()

    lowest_node_id_g1 = min(g1.nodes)
    lowest_node_id_g2 = min(g2.nodes)
    higest_node_id_g1 = max(g1.nodes)
    max_id = max(max(g1.nodes), max(g2.nodes))
    # sorted_node_matches = sorted(node_matches, key=lambda x: (x[0], x[1]))
    sorted_node_matches = node_matches
    for (node1, node2) in sorted_node_matches:
        if node1 == 18446744073709551614 or node2 == 18446744073709551614:
            continue
        else:
            node1_id = node1 + lowest_node_id_g1
            node2_id = node2 + lowest_node_id_g2
            # label = (g1.nodes[node1_id].get('label', None), g2.nodes[node2_id].get('label', None))
            restricted_graph.add_node((node1_id, node2_id))
    # rather inefficient O(n^2) approach,
    # possible better to iterate over the edges of g1, and check if the corresponding edge exists in g2
    # map the node id of g1 to the node id of g2
    g1_node_map = {node1 + lowest_node_id_g1: node2 + lowest_node_id_g2 for (node1, node2) in sorted_node_matches}
    # for (u2,v2) in g2.edges:
    for (u1,v1) in g1.edges:
        u2 = g1_node_map.get(u1, None)
        v2 = g1_node_map.get(v1, None)
        if u2 is not None and v2 is not None and g2.has_edge(u2, v2):
            # label = g1.edges[(u1, v1)].get('label', None)
            restricted_graph.add_edge((u1, u2), (v1, v2))
            # print(f"Added edge between ({u1}, {u2}) and ({v1}, {v2}) with label {label}")
def get_Name(self):
        return "Random_walk_Edit_Calculator"



def limited_length_approx_random_walk_similarity(product_graph: nx.Graph,llamda =0.1,max_length=10):
    nodelist = list(product_graph.nodes())
    adj_matrix =nx.to_numpy_array(product_graph)

    row_sums = adj_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    transition_matrix = adj_matrix / row_sums[:, np.newaxis]
    total_similarity = 0.0

    walk_distribution = np.identity(len(nodelist))

    for step in range(max_length):
        
        trace = np.trace(walk_distribution)

        total_similarity += (llamda ** step) * trace
        walk_distribution = np.dot(walk_distribution, transition_matrix)
    return total_similarity

def infinte_length_random_walk_similarity(product_graph: nx.Graph, llamda=0.1):
    nodelist = product_graph.nodes()
    adj_matrix = nx.to_numpy_array(product_graph, nodelist=nodelist)
    # eigenvalues = np.linalg.eigvals(adj_matrix)
    # max_eigenvalue = np.max(np.abs(eigenvalues))
    identity_matrix = np.eye(adj_matrix.shape[0])
    try:
        # The core of the kernel computation is in this single matrix inversion.
        # This operation efficiently sums all possible walks of all lengths.
        kernel_matrix = np.linalg.inv(identity_matrix - llamda * adj_matrix)

        # D. Sum all elements of the resulting matrix to get the final kernel value.
        kernel_value = np.sum(kernel_matrix)
        return kernel_value
        
    except np.linalg.LinAlgError as e:
        print(f"Warning: Matrix is not invertible. {e}")
        return 18446744073709551614