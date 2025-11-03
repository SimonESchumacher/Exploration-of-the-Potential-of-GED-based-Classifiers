from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from pytest import fixture

from graph_mixup.mixup_methods.geomix.dataset import sample_vanilla_pair_close_in_size
from graph_mixup.transforms.one_hot_label_transform import OneHotLabel


@fixture
def dataset() -> Dataset:
    return TUDataset(root="data", name="PROTEINS", transform=OneHotLabel(2))


def test_sample_vanilla_pair_close_in_size(dataset: Dataset):
    g0, g1 = sample_vanilla_pair_close_in_size(dataset)
    assert not g0.y.equal(g1.y)
