from unittest import TestCase
from torch_geometric.data import Data
import torch

from graph_mixup.mixup_methods.if_mixup.mixup import mixup


class IfMixupTestCase(TestCase):
    def setUp(self):
        self.edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        self.x1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.edge_attr1 = torch.tensor([[0, 1], [1, 0]])
        # 0 -- 1 -- 2

        self.edge_index2 = torch.tensor([[0, 1], [1, 0]])
        self.x2 = torch.tensor([[0, 0, 1], [0, 1, 0]])
        self.x2_padded = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        self.edge_attr2 = torch.tensor([[1, 0]])
        self.edge_attr2_padded = torch.tensor([[1, 0], [0, 0]])
        # 0 -- 1

        self.g1 = Data(
            edge_index=self.edge_index1,
            x=self.x1,
            y=torch.tensor([[1, 0]]),
            num_nodes=self.x1.size(0),
            edge_attr=self.edge_attr1,
        )
        self.g2 = Data(
            edge_index=self.edge_index2,
            x=self.x2,
            y=torch.tensor([[0, 1]]),
            num_nodes=self.x2.size(0),
            edge_attr=self.edge_attr2,
        )

        self.mixup_graph = mixup(self.g1, self.g2, 2)

    def test_if_mixup_adjacency_matrix(self):
        self.assertTrue(self.mixup_graph.edge_index.equal(self.edge_index1))

    def test_reversed_roles(self):
        graph = mixup(self.g2, self.g1, 2)
        self.assertTrue(graph.validate())

    def test_reversed_edge_attr(self):
        edge_index1 = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        edge_attr1 = torch.tensor([[0, 1], [1, 0]])
        g1 = Data(
            edge_index=edge_index1,
            y=torch.tensor([[1, 0]]),
            edge_attr=edge_attr1,
            num_nodes=4,
        )
        # 0 -- 1    2 -- 3

        edge_index2 = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        edge_attr2 = torch.tensor([[1, 0], [1, 0], [1, 0]])
        g2 = Data(
            edge_index=edge_index2,
            y=torch.tensor([[0, 1]]),
            edge_attr=edge_attr2,
            num_nodes=3,
        )
        # 0 -- 1
        #  \  /
        #   2

        graph = mixup(g1, g2, 2)
        graph.validate()
