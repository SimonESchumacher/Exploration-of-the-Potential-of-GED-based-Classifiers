# Methods for GED comupation
import gmatch4py as gm
import networkx as nx
from Graph_Tools import convert_nx_to_grakel_graph
import numpy as np
DEBUG = False  # Set to False to disable debug prints
class GraphEditDistanceCalculator:
    def __init__(self, node_deletion_cost=1.0, node_insertion_cost=1.0, node_substitution_cost=1.0, edge_deletion_cost=1.0, edge_insertion_cost=1.0, edge_substitution_cost=1.0,approximation=None):
        
        self.node_deletion_cost = node_deletion_cost
        self.node_insertion_cost = node_insertion_cost
        self.node_substitution_cost = node_substitution_cost
        self.edge_deletion_cost = edge_deletion_cost
        self.edge_insertion_cost = edge_insertion_cost
        self.edge_substitution_cost = edge_substitution_cost
        if approximation is None:
            self.ged= gm.GraphEditDistance(node_del=self.node_deletion_cost,
                                     node_ins=self.node_insertion_cost,
                                     edge_del=self.edge_deletion_cost,
                                     edge_ins=self.edge_insertion_cost)
        elif approximation == "greedy":
            self.ged = gm.GreedyEditDistance(
                                     node_del=self.node_deletion_cost,
                                     node_ins=self.node_insertion_cost,
                                     edge_del=self.edge_deletion_cost,
                                     edge_ins=self.edge_insertion_cost)
        elif approximation == "approximate":
           raise NotImplementedError("Approximate GED is not implemented in gmatch4py")
        elif approximation == "Hausdorff":
            self.ged = gm.HED(
                                     node_del=self.node_deletion_cost,
                                     node_ins=self.node_insertion_cost,
                                     edge_del=self.edge_deletion_cost,
                                     edge_ins=self.edge_insertion_cost)
        elif approximation == "Bipartite":
            self.ged = gm.BP_2(
                                     node_del=self.node_deletion_cost,
                                     node_ins=self.node_insertion_cost,
                                     edge_del=self.edge_deletion_cost,
                                     edge_ins=self.edge_insertion_cost)
        else:
            raise ValueError("Invalid approximation method specified. Choose from 'greedy', 'approximate', 'Hausdorff', or 'Bipartite'.")

    def distance(self, graph1, graph2):
        """
        Compute the Graph Edit Distance between two graphs.
        """
        
        distance=self.ged.compare([graph1, graph2],None)
        
        return distance[0][1]
    def similarity(self, graph1, graph2):
        """
        Compute the similarity between two graphs based on their Graph Edit Distance.
        """
        distance = self.ged.compare([graph1, graph2],None)
        return self.ged.similarity(distance)[0][1]
    def gram_matrix(self, graphs):
        """
        Compute the Gram matrix for a list of graphs.
        """
        gram_matrix =self.ged.compare(graphs, None)
        gram_similarity =self.ged.similarity(gram_matrix)
        return gram_similarity
    def similarity_matrix(self, graphs1, graphs2=None):

        if graphs2 is None:
            if DEBUG:
                print(f"Calculating similarity matrix for {len(graphs1)} graphs against themselves.")
            distance_matrix = self.ged.compare(graphs1, None)
            similarity_matrix = self.ged.similarity(distance_matrix)
        else:
            if DEBUG:
                print(f"Calculating similarity matrix for {len(graphs1)} graphs against {len(graphs2)} graphs.")
            similarity_matrix =np.zeros((len(graphs1), len(graphs2)))
            for i, g1 in enumerate(graphs1):
                for j, g2 in enumerate(graphs2):
                    similarity_matrix[i, j] = self.similarity(g1, g2)
        return similarity_matrix
    def distance_matrix(self, graphs1, graphs2=None):
        """
        Compute the distance matrix for a list of graphs.
        """
        if graphs2 is None:
            if DEBUG:
                print(f"Calculating distance matrix for {len(graphs1)} graphs against themselves.")
            distance_matrix = self.ged.compare(graphs1, None)
        else:
            if DEBUG:
                print(f"Calculating distance matrix for {len(graphs1)} graphs against {len(graphs2)} graphs.")
            distance_matrix =np.zeros((len(graphs1), len(graphs2)))
            for i, g1 in enumerate(graphs1):
                for j, g2 in enumerate(graphs2):
                    distance_matrix[i, j] = self.distance(g1, g2)
        return distance_matrix
    # def get_edit_path(self, graph1, graph2):
    #     """
    #     Get the edit path between two graphs.
    #     """
    #     edit_path = self.ged.edit_path(graph1, graph2)
    #     return edit_path
    
# g1=nx.complete_bipartite_graph(5,4) 
# g2=nx.complete_bipartite_graph(6,4)
# g3=nx.complete_bipartite_graph(5,5)
# g4=nx.complete_bipartite_graph(5,6)
# g5=nx.complete_bipartite_graph(3,6)
# g6=nx.complete_bipartite_graph(5,4)
# claclulator = GraphEditDistanceCalculator(
#     node_deletion_cost=1.0,
#     node_insertion_cost=1.0,
#     edge_deletion_cost=1.0,
#     edge_insertion_cost=1.0,
#     approximation=None
# )
# print(claclulator.distance_matrix([g1,g2,g3,g4]))
# print(claclulator.distance_matrix([g1,g2,g3,g4],None))
# print(claclulator.distance_matrix([g1,g2,g3,g4,g5,g6],None))
# print(claclulator.distance_matrix([g1,g2,g3,g4],[g5,g6]))

# print(claclulator.distance(g1,g2))
# print(claclulator.gram_matrix([g1,g2]))
