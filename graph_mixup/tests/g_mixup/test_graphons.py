import torch
from unittest import TestCase
from torch_geometric.datasets import FakeDataset
from torch_geometric.seed import seed_everything
from torch_geometric.data import Data

from graph_mixup.mixup_methods.g_mixup.graphons import *
from graph_mixup.transforms.one_hot_label_transform import OneHotLabel


class GMixupTestCase(TestCase):
    def setUp(self):
        seed_everything(0)

    def test_compute_median_node_number(self):
        dataset = FakeDataset(10, 20, num_classes=3, transform=OneHotLabel(3))
        self.assertEqual(21.0, compute_median_node_number(dataset))

    def test_split_dataset_into_classes_requires_one_hot(self):
        dataset = FakeDataset(10, 20, num_classes=3)
        with self.assertRaises(AssertionError):
            split_dataset_into_classes(dataset)

    def test_split_dataset_into_classes(self):
        dataset = FakeDataset(10, 20, num_classes=3, transform=OneHotLabel(3))
        splits = split_dataset_into_classes(dataset)

        item = dataset[0]
        self.assertTrue(item.edge_index.equal(splits[1][0].edge_index))

        self.assertTrue(len(splits[0]) == 3)
        self.assertTrue(len(splits[1]) == 5)
        self.assertTrue(len(splits[2]) == 2)

    def test_align_graphs(self):
        edge_index1 = torch.tensor([[0, 1], [1, 0]])
        g1 = Data(edge_index=edge_index1, x=torch.ones(2, 3))
        # G1: 0 -- 1

        edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        g2 = Data(edge_index=edge_index2, x=torch.ones(3, 3))
        # G2: 0 -- 1 -- 2

        edge_index3 = torch.tensor([[0, 1, 1, 1, 2, 3], [1, 0, 2, 3, 1, 1]])
        g3 = Data(edge_index=edge_index3, x=torch.ones(4, 3))
        # G3: 0 -- 1 -- 2
        #          |
        #          3
        adj_matrices, feat_matrices = align_graphs([g1, g2, g3], 3)

        sm = (adj_matrices[0] + adj_matrices[1] + adj_matrices[2]).to_dense()
        adj_expected = torch.Tensor([[0, 3, 2], [3, 0, 0], [2, 0, 0]])

        self.assertTrue(sm.equal(adj_expected))

        feat_expected = torch.ones(3, 3, 3)
        feat_expected[0, 2] = torch.zeros(3)  # zero-padding for G1

        self.assertTrue(torch.stack(feat_matrices).equal(feat_expected))

    def test_universal_singular_value_thresholding(self):

        edge_index1 = torch.tensor([[0, 1], [1, 0]])
        g1 = Data(edge_index=edge_index1, x=torch.randn(2, 3))
        # G1: 0 -- 1

        edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        g2 = Data(edge_index=edge_index2, x=torch.randn(3, 3))
        # G2: 0 -- 1 -- 2

        edge_index3 = torch.tensor([[0, 1, 1, 1, 2, 3], [1, 0, 2, 3, 1, 1]])
        g3 = Data(edge_index=edge_index3, x=torch.randn(4, 3))
        # G3: 0 -- 1 -- 2
        #          |
        #          3
        adj_matrices, _ = align_graphs([g1, g2, g3], 3)

        graphon = universal_singular_value_thresholding(adj_matrices, threshold=0.2)

        expected = torch.tensor(
            [
                [0.0000e00, 1.0000e00, 6.6667e-01],
                [1.0000e00, 7.9167e-08, 5.2778e-08],
                [6.6667e-01, 7.2646e-08, 4.8431e-08],
            ]
        )

        self.assertAlmostEqual(torch.abs(graphon - expected).sum().item(), 0.0, 4)

    def test_compute_class_graphons(self):
        dataset = FakeDataset(10, 3, num_classes=3, transform=OneHotLabel(3))
        compute_class_graphons_and_features(dataset)

    def test_sample_adj_matrix(self):

        graphon = torch.tensor([[1.0, 0.67, 0.33], [0.67, 0.0, 0.0], [0.33, 0.0, 0.0]])

        adj_matrix = sample_adj_matrix(graphon)
        expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Symmetric
        self.assertTrue(adj_matrix.equal(expected))

    def test_sample_graph(self):
        dataset = FakeDataset(10, 3, num_classes=3, transform=OneHotLabel(3))
        classes = compute_class_graphons_and_features(dataset)

        class_pair = (classes[0], classes[2])
        graph = sample_graph(class_pair, 0.7)
        self.assertTrue(graph.validate())
