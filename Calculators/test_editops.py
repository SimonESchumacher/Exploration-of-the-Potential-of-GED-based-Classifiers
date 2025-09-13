# add calculators to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Calculators')))
from Edit_Set import EditOperation, NodeExchangeOperation, NodeDeleteOperation, EdgeDeletionOperation, EdgeInsertionOperation,EdgeExchangeOperation,NodeInsertOperation,pathGenerator
import networkx as nx
from GEDLIB_Caclulator import GEDLIB_Calculator
from Dataset import Dataset
from Calculators.Product_GRaphs import build_restricted_product_graph, limited_length_approx_random_walk_similarity

def createNxGraph() :
    G = nx.Graph()
    G.add_node("C1", label = "C1", attr= "0.5")
    G.add_node("02", label = "02", attr= "0.5")
    # G.add_edge("C1", "02", label = "1", attr= "0.5")
    G.add_node("N3", label = "N3", attr= "0.5")
    G.add_node("C4", label = "C4", attr= "0.5")
    G.add_node("H5", label = "H5", attr= "0.5")
    G.add_node("H6", label = "H6", attr= "0.5")
    # G.add_edge("N3", "C4", label = "1", attr= "0.5")
    G.add_edge("C1", "C4", label = "a", attr= "0.5")
    G.add_edge("C4", "02", label = "a", attr= "0.5")
    G.add_edge("C4", "N3", label = "a", attr= "0.5")
    G.add_edge("C1", "H5", label = "a", attr= "2")
    return G
def createNxGraph2() :
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_node(3, label = "N3", attr= "0.5")   
    G.add_edge(3, 4, label = "b", attr= "0.5")
    G.add_edge(3, 2, label = "b", attr= "0.5")
    return G
def createNxGraph3():
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_node(3, label = "G3", attr= "0.5")  
    G.add_node(5, label = "H5", attr= "0.5")
    G.add_node(6, label = "H6", attr= "0.5") 
    G.add_edge(3, 4, label = "b", attr= "0.5")
    G.add_edge(3, 2, label = "b", attr= "0.5")

    G.add_edge(4, 5, label = "b", attr= "0.5")
    G.add_edge(5, 6, label = "b", attr= "0.5")
    return G
def createNxGraph4A():
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_node(3, label = "A3", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_edge(2, 3, label = "b", attr= "0.5")
    G.add_edge(3, 1, label = "b", attr= "0.5")
    G.add_edge(2, 4, label = "b", attr= "0.5")
    G.add_edge(4, 3, label = "b", attr= "0.5")
    return G
def createNxGraph4B():
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_node(3, label = "A3", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_node(5, label = "C5", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_edge(2, 3, label = "b", attr= "0.5")
    G.add_edge(3, 1, label = "b", attr= "0.5")
    G.add_edge(3, 4, label = "b", attr= "0.5")
    G.add_edge(4, 5, label = "b", attr= "0.5")
    G.add_edge(5, 3, label = "b", attr= "0.5")
    G.add_edge(2, 4, label = "b", attr= "0.5")
    return G
def createNxGraph4A1():
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_node(3, label = "A3", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_edge(2, 3, label = "b", attr= "0.5")
    G.add_edge(3, 1, label = "b", attr= "0.5")
    G.add_edge(1, 4, label = "b", attr= "0.5")
    G.add_edge(4, 3, label = "b", attr= "0.5")
    return G
def createNxGraph4A2():
    G = nx.Graph()
    G.add_node(1, label = "C1", attr= "0.5")
    G.add_node(2, label = "O2", attr= "0.5")
    G.add_node(3, label = "A3", attr= "0.5")
    G.add_node(4, label = "C4", attr= "0.5")
    G.add_edge(1, 2, label = "b", attr= "0.5")
    G.add_edge(2, 3, label = "b", attr= "0.5")
    G.add_edge(3, 1, label = "b", attr= "0.5")
    G.add_edge(1, 4, label = "b", attr= "0.5")
    G.add_edge(4, 3, label = "b", attr= "0.5")
    return G
calculator = GEDLIB_Calculator(GED_calc_method="BRANCH", GED_edit_cost="CONSTANT", need_node_map=True)
DATASET= Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels="label", use_edge_labels="weight",load_now=False)
DATASET.load()
g1= createNxGraph4A1()
g2 = createNxGraph4A2()
g1 = DATASET.get_graphs()[0]
g2 = DATASET.get_graphs()[1]
dataset= [g1, g2]
calculator.add_graphs(dataset)
calculator.activate()
calculator.run_method(0, 1)
node_map = calculator.get_node_map(0, 1)
print(calculator.get_mean_distance(0,1))
print(calculator.get_upper_bound(0,1))
print(calculator.get_lower_bound(0,1))
# operations = pathGenerator.get_edit_ops(g1, g2, node_map)
# print(len(operations))
# for op in operations:
#     print(op)
prod_graph = build_restricted_product_graph(g1, g2, node_map)