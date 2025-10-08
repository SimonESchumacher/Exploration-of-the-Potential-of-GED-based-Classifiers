# Tools
import grakel
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

def graph_from_networkx(nx_graph, node_label_tag="label", edge_label_tag="label"):
    """
    Converts a NetworkX graph to a GraKeL graph.
    Assumes node labels are stored under the 'label' attribute.
    """
    return grakel.graph_from_networkx(nx_graph, node_labels_tag=node_label_tag, edge_labels_tag=edge_label_tag)

def get_grakel_graphs_from_nx(graph_list, node_label_tag="label", edge_label_tag="label"):
    """
    Converts a list of NetworkX graphs to a list of GraKeL graphs.
    Assumes node labels are stored under the 'label' attribute.
    """
    # detect if the graphs have edge labels
    test_graph = graph_list[0]
    test_graph2 = graph_list[1]
    test_graph3 = graph_list[2]
    edge_labels_set = {test_graph.edges[edge].get(edge_label_tag, None) for edge in test_graph.edges()}
    if any(label is not None for label in edge_labels_set):
        edge_label_tag = edge_label_tag
    else:
        edge_labels_set2 = {test_graph2.edges[edge].get(edge_label_tag, None) for edge in test_graph2.edges()}
        edge_labels_set3 = {test_graph3.edges[edge].get(edge_label_tag, None) for edge in test_graph3.edges()}
        if any(label is not None for label in edge_labels_set2) or any(label is not None for label in edge_labels_set3):
            edge_label_tag = edge_label_tag
        else:
            edge_label_tag = None
    node_labels_set = {test_graph.nodes[node].get(node_label_tag, None) for node in test_graph.nodes()}
    if any(label is not None for label in node_labels_set) or any(label is not None for label in {test_graph2.nodes[node].get(node_label_tag, None) for node in test_graph2.nodes()}) or any(label is not None for label in {test_graph3.nodes[node].get(node_label_tag, None) for node in test_graph3.nodes()}):
        node_label_tag = node_label_tag
    else:
        node_label_tag = None
    # detect if the graphs have edge labels
 
    #     edge_label_tag = None
    # # detect if the graphs have node labels
    # if any('label' in g.nodes[g.nodes()[0]] for g in graph_list if g.number_of_nodes() > 0):
    #     node_label_tag = node_label_tag
    # else:
    #     node_label_tag = None

    return grakel.graph_from_networkx(graph_list, node_labels_tag=node_label_tag, edge_labels_tag=edge_label_tag,as_Graph=True, val_node_labels=0, val_edge_labels=0)
    