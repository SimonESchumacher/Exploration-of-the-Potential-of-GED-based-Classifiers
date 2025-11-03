import torch
from graph_mixup.mixup_methods.s_mixup.utils.triple_set import TripleSet
from tests.base import BaseTest
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant


class TestTripleSet(BaseTest):
    def test_triple_set(self):
        dataset = TUDataset("data", "PROTEINS")
        transform = Constant()
        triple_dataset = TripleSet(dataset, transform)

        assert len(triple_dataset) == len(dataset)
        assert triple_dataset[0] is not None
        assert len(triple_dataset[0]) == 3
        assert torch.equal(triple_dataset[0][0].edge_index, dataset[0].edge_index)
