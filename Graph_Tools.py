# Tools
from grakel.graph import Graph # To convert NetworkX to GraKeL Graph objects
import networkx as nx

def convert_nx_to_grakel_graph(nx_graph, use_edge_labels=True, use_node_labels=True, use_node_attributes=False, use_edge_attributes=False):
    """
    Converts a NetworkX Graph object to a grakel.Graph object
    using the constructor signature:
    Graph(initialization_object, node_labels, edge_labels, ...)
    """

    # Extract edges as a list of tuples (required for initialization_object)
    edges_list = list(nx_graph.edges())
    
    # Extract node labels as a dictionary {node_id: label}
    if use_node_labels:
        node_labels_dict = {node: nx_graph.nodes[node].get('label', None) for node in nx_graph.nodes()}
    elif use_node_attributes:
        node_labels_dict = {node: nx_graph.nodes[node].get('attributes', None) for node in nx_graph.nodes()}
    else:
        node_labels_dict = None
    # Extract edge labels if they exist, otherwise set to None
    if use_edge_labels:
        edge_labels_dict = {edge: nx_graph.edges[edge].get('label', None) for edge in nx_graph.edges()}
    elif use_edge_attributes:
        edge_labels_dict = {edge: nx_graph.edges[edge].get('attributes', None) for edge in nx_graph.edges()}
    else:
        edge_labels_dict = None
    return Graph(initialization_object=edges_list, node_labels=node_labels_dict,edge_labels=edge_labels_dict,graph_format="dictionary")

