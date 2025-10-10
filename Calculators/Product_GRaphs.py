# bult the restricted Product Graph
import numpy as np
import networkx as nx
from scipy.linalg import inv
from scipy.interpolate import interp1d

from Calculators import Base_Calculator
import Dataset
DEBUG = True
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
            restricted_graph.add_node((node1_id, node2_id,))
    # rather inefficient O(n^2) approach,
    # possible better to iterate over the edges of g1, and check if the corresponding edge exists in g2
    # map the node id of g1 to the node id of g2
    g1_node_map = {node1 + lowest_node_id_g1: node2 + lowest_node_id_g2 for (node1, node2) in sorted_node_matches}
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

def build_product_graph_adj_matrix(g1: nx.Graph, g2: nx.Graph,node_matches : list[(int,int)]):
    clean_node_matches = [(u,v) for (u,v) in node_matches if u != 18446744073709551614 and v != 18446744073709551614]
    # get all edges in g_x
    min_g1_id = min(g1.nodes)
    min_g2_id = min(g2.nodes)
    g_x=nx.Graph()
    for (u1,v1) in clean_node_matches:
        u1_id = u1 + min_g1_id
        u2_id = v1 + min_g2_id
        label = (g1.nodes[u1_id].get('label', None), g2.nodes[u2_id].get('label', None))
        g_x.add_node((u1_id,u2_id), label=label)
    g1_node_map = {node1 + min_g1_id: node2 + min_g2_id for (node1, node2) in clean_node_matches}
    for (u1,v1) in g1.edges:
        u2 = g1_node_map.get(u1, None)
        v2 = g1_node_map.get(v1, None)
        if u2 is not None and v2 is not None and g2.has_edge(u2, v2):
            label = g1.edges[(u1, v1)].get('label', None)
            g_x.add_edge((u1,u2), (v1,v2), label=label)
    A_x =np.zeros((len(clean_node_matches),len(clean_node_matches)), dtype=int)




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
    nodelist = list(product_graph.nodes())
    adj_matrix = nx.to_numpy_array(product_graph, nodelist=nodelist)
    identity_matrix = np.eye(adj_matrix.shape[0])

    # Use scipy for faster matrix inversion and sum
    try:
        kernel_matrix = inv(identity_matrix - llamda * adj_matrix)
        kernel_value = np.sum(kernel_matrix)
        return kernel_value
    except np.linalg.LinAlgError as e:
        print(f"Warning: Matrix is not invertible. {e}")
        return 18446744073709551614
def infinte_length_random_walk_similarity2(adj_matrix, llamda=0.1):
    identity_matrix = np.eye(adj_matrix.shape[0])
    # 
    # Use scipy for faster matrix inversion and sum
    try:
        kernel_matrix = inv(identity_matrix - llamda * adj_matrix)
        kernel_value = np.sum(kernel_matrix)
        return kernel_value
    except np.linalg.LinAlgError as e:
        print(f"Warning: Matrix is not invertible. {e}")
        return 18446744073709551614
    
class RandomWalkCalculator():
    backup = None
    def __init__(self, ged_calculator: Base_Calculator, llambda_samples: list[float], dataset: Dataset):
        if (hasattr(RandomWalkCalculator, 'backup') and RandomWalkCalculator.backup is not None and dataset.__str__() == RandomWalkCalculator.backup.dataset_str):
            backup = RandomWalkCalculator.backup
            self.ged_calculator = backup.ged_calculator
            self.llambda_samples = backup.llambda_samples
            self.dataset_str = backup.dataset_str
            self.adj_matrices = backup.adj_matrices
            self.interpolators = backup.interpolators
            self.is_calculated = backup.is_calculated
        else:
            self.ged_calculator = ged_calculator
            self.llambda_samples = llambda_samples
            self.is_calculated = False
            self.dataset_str = dataset.__str__()
            self.calculate_prod_graphs()
            # self.calculate_sample_walks()
            self.is_calculated = True
            if DEBUG:
                print(f"Initialized RandomWalkCalculator with {len(self.ged_calculator.get_dataset())} graphs and {len(self.llambda_samples)} lambda samples.")
        self.make_backup()
    def calculate_prod_graphs(self):
        n = len(self.ged_calculator.get_dataset())
        # initialize a 2 D of length n x n array to store the nodelists and adjacency matrices of the product graphs
        self.adj_matrices = [[None for _ in range(n)] for _ in range(n)]
        for i, g1 in enumerate(self.ged_calculator.get_dataset()):
            for j, g2 in enumerate(self.ged_calculator.get_dataset()):
                if i > j:
                    continue
                elif i == j:
                    # for the same graph, the product graph is just the graph itself
                    self.adj_matrices[i][j] = nx.to_numpy_array(g1)
                    continue
                node_map = self.ged_calculator.get_node_map(i, j)
                product_graph = build_restricted_product_graph(g1, g2, node_map)
                nodelist = list(product_graph.nodes())
                self.adj_matrices[i][j] = nx.to_numpy_array(product_graph, nodelist=nodelist)
    def get_adj_matrix(self, i, j):
        return self.adj_matrices[i][j]
    def calculate_sample_walks(self):
        if DEBUG:
            print("Calculating sample walks for all graph pairs...")
        n = len(self.ged_calculator.get_dataset())
        self.interpolators = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):                
                adj_matrix = self.get_adj_matrix(i, j)
                sample_sums = np.zeros((len(self.llambda_samples)), dtype=float)
                for k, llambda in enumerate(self.llambda_samples):
                    similarity = infinte_length_random_walk_similarity2(adj_matrix, llamda=llambda)
                    sample_sums[k] = similarity
                self.interpolators[i][j] = interp1d(self.llambda_samples, sample_sums, kind='cubic', fill_value="extrapolate")

    def get_approx_inflength_walk(self, i, j, llambda):
        if i > j:
            i, j = j, i
        if self.interpolators[i][j] is None:
            raise ValueError(f"Interpolator for graphs {i} and {j} has not been calculated.")
        result = self.interpolators[i][j](llambda)
        if np.isnan(result):
            raise ValueError(f"Interpolator for graphs {i} and {j} returned NaN for lambda={llambda}")
        float_result = float(result)
        if float_result < 0 or np.isnan(float_result):
            print(f"Warning: Similarity for graphs {i} and {j} with lambda={llambda} is negative or NaN ({float_result}). Setting to 0.")   
            return 0.0
        return float(result)
    
    def get_exact_inflength_walk(self, i, j, llambda):
        if i > j:
            i, j = j, i
        adj_matrix = self.get_adj_matrix(i, j)
        result = infinte_length_random_walk_similarity2(adj_matrix, llamda=llambda)
        if result < 0 or np.isnan(result):
            raise ValueError(f"Exact similarity for graphs {i} and {j} with lambda={llambda} is negative or NaN ({result})")
        float_result = float(result)
        if float_result < 0 or np.isnan(float_result):
            print(f"Warning: Similarity for graphs {i} and {j} with lambda={llambda} is negative or NaN ({float_result}). Setting to 0.")   
            return 0.0
        return infinte_length_random_walk_similarity2(adj_matrix, llamda=llambda)
    def get_limited_length_walk(self, i, j, llambda, max_length):
        if i > j:
            i, j = j, i
        adj_matrix = self.get_adj_matrix(i, j)

        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = adj_matrix / row_sums[:, np.newaxis]
        total_similarity = 0.0

        walk_distribution = np.identity(adj_matrix.shape[0])

        for step in range(max_length):
            
            
            trace = np.trace(walk_distribution)

            total_similarity += (llambda ** step) * trace
            walk_distribution = np.dot(walk_distribution, transition_matrix)
            if np.isnan(total_similarity):
                raise ValueError(f"Limited length similarity for graphs {i} and {j} with lambda={llambda} and step={step} is NaN")
            # check if there are nan values in the walk_distribution
            if np.isnan(walk_distribution).any():
                raise ValueError(f"Walk distribution for graphs {i} and {j} with lambda={llambda} and step={step} contains NaN values")
        return total_similarity
    def make_backup(self):
        RandomWalkCalculator.backup = self
    
