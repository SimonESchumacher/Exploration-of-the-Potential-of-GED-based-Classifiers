import pytest
from lightning import seed_everything
from torch_geometric.datasets import TUDataset
from graph_mixup.mixup_methods.g_mixup.graphons import (
    compute_class_graphons_and_features,
)
from graph_mixup.mixup_methods.g_mixup.mixup import mixup
from graph_mixup.transforms.one_hot_label_transform import OneHotLabel


@pytest.fixture(autouse=True)
def seed():
    seed_everything(0)


def test_mixup():
    dataset = TUDataset("data", "PROTEINS", transform=OneHotLabel(2))
    class_graphons_and_features = compute_class_graphons_and_features(dataset)
    g = mixup(dataset[0], dataset[-1], 0.1, class_graphons_and_features)
    g.validate()
