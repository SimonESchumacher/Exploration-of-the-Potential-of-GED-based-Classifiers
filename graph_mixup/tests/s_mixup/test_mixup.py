import torch
from typing import Literal
import pytest
from lightning import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

from graph_mixup.mixup_methods.s_mixup.mixup import mixup
from graph_mixup.mixup_methods.s_mixup.gmnet import LitGraphMatchingNet


@pytest.fixture(autouse=True)
def setup():
    seed_everything(0)


@pytest.fixture
def proteins_graph_pair() -> tuple[Data, Data]:
    dataset = TUDataset("data", "PROTEINS")
    return dataset[0], dataset[-1]


@pytest.fixture
def gmnet() -> LitGraphMatchingNet:
    return LitGraphMatchingNet(
        node_feat_dim=3,
        num_layers=4,
        hidden=32,
    )


@pytest.mark.parametrize("sim_method", ["cos", "abs_diff"])
@pytest.mark.parametrize("normalize_method", ["softmax", "sinkhorn"])
def test_mixup(
    gmnet: LitGraphMatchingNet,
    proteins_graph_pair: tuple[Data, Data],
    sim_method: Literal["cos", "abs_diff"],
    normalize_method: Literal["softmax", "sinkhorn"],
):
    mixup_item = mixup(
        *proteins_graph_pair,
        0.4,
        gmnet,
        sim_method=sim_method,
        normalize_method=normalize_method,
    )

    assert mixup_item.x.shape == proteins_graph_pair[0].x.shape
    assert mixup_item.edge_index.shape == proteins_graph_pair[0].edge_index.shape
    assert mixup_item.edge_weight.size() == torch.Size(
        [proteins_graph_pair[0].edge_index.shape[1]]
    )
