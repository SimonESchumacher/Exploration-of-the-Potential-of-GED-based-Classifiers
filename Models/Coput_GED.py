# Computer the Graph Edit Distance (GED) between two graphs
import sys
import networkx as nx
import networkx as nx
import numpy as np
import gmatch4py as gm
class GraphEditDistanceCalculator:
    def __init__(self, graph1, graph2,node_deletion_cost=1.0, node_insertion_cost=1.0, node_substituion_cost=1.0, edge_deletion_cost=0.5, edge_insertion_cost=0.5, edge_substitution_cost=0.1):
        self.graph1 = graph1
        self.graph2 = graph2
        self.node_deletion_cost = node_deletion_cost
        self.node_insertion_cost = node_insertion_cost
        self.node_substituion_cost = node_substituion_cost
        self.edge_deletion_cost = edge_deletion_cost
        self.edge_insertion_cost = edge_insertion_cost
        self.edge_substitution_cost = edge_substitution_cost


    def auto_compute_ged(self):
        """Compute the Graph Edit Distance between two graphs."""
        ged = graph_edit_distance(self.graph1, self.graph2)
        return ged
    
    def implemented_compute_GED(self):
        # Define custom costs (optional, but good practice for real-world scenarios)
        def node_match(n1, n2):
            return n1['label'] == n2['label']

        def node_del_cost(n):
            return 1.0

        def node_ins_cost(n):
            return 1.0

        def node_subst_cost(n1, n2):
            return 0.5 if n1['label'] != n2['label'] else 0.0

        def edge_del_cost(u, v):
            return 0.5

        def edge_ins_cost(u, v):
            return 0.5

        def edge_subst_cost(u1, v1, u2, v2):
            # For simplicity, let's say edge substitution costs 0.1 if both exist
            return 0.1

        # Calculate optimal edit paths
        # This function returns a list of (node_edit_path, edge_edit_path) tuples
        # and the minimum cost. There can be multiple optimal paths.
        paths, cost = nx.optimal_edit_paths(
            G1, G2,
            node_match=node_match,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            node_subst_cost=node_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost,
            edge_subst_cost=edge_subst_cost
        )

        print(f"GED cost: {cost}")

        # Print the first optimal edit path
        print("\nFirst optimal edit path:")
        node_edit_path = paths[0][0]
        edge_edit_path = paths[0][1]

        print("Node Edits:")
        for u, v in node_edit_path:
            if u is None:
                print(f"  Insert node {v} (label: {G2.nodes[v].get('label')})")
            elif v is None:
                print(f"  Delete node {u} (label: {G1.nodes[u].get('label')})")
            else:
                if G1.nodes[u].get('label') != G2.nodes[v].get('label'):
                    print(f"  Substitute node {u} (label: {G1.nodes[u].get('label')}) with node {v} (label: {G2.nodes[v].get('label')})")
                else:
                    print(f"  Match node {u} with node {v} (label: {G1.nodes[u].get('label')})")

        print("\nEdge Edits:")
        for (u1, v1), (u2, v2) in edge_edit_path:
            if u1 is None:
                print(f"  Insert edge ({u2}, {v2}) in G2")
            elif u2 is None:
                print(f"  Delete edge ({u1}, {v1}) from G1")
            else:
                print(f"  Match/Substitute edge ({u1}, {v1}) from G1 with ({u2}, {v2}) in G2")



# Example graphs
G1 = nx.Graph()
G1.add_nodes_from([
    (1, {'label': 'A'}),
    (2, {'label': 'B'})
])
G1.add_edges_from([(1, 2)])

G2 = nx.Graph()
G2.add_nodes_from([
    (3, {'label': 'A'}),
    (4, {'label': 'C'}),
    (5, {'label': 'D'})
])
G2.add_edges_from([(3, 4), (4, 5)])