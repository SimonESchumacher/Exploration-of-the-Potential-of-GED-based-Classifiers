# bult the restricted Product Graph
import numpy as np
import networkx as nx

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