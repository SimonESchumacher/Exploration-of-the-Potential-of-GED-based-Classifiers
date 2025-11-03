from unittest import TestCase
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
from torch_geometric.seed import seed_everything
import torch
import numpy as np


from graph_mixup.transforms.one_hot_label_transform import OneHotLabel


class OneHotLabelTestCase(TestCase):
    def test_one_hot_label_transform(self):
        seed_everything(1)
        dataset = FakeDataset(3, 20, num_classes=3, transform=OneHotLabel(3))
        one_hot_labels = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 0]]).view(3, 1, 3)

        labels = []
        for data in dataset:
            labels.append(data.y)

        labels = torch.from_numpy(np.array(labels))

        self.assertTrue(one_hot_labels.equal(labels))
