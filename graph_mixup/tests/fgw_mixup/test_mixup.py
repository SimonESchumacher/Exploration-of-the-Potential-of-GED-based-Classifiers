import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from lightning import seed_everything
from graph_mixup.mixup_methods.fgw_mixup.mixup import mixup
import pytest


@pytest.fixture(autouse=True)
def seed():
    seed_everything(0)


@pytest.mark.parametrize(
    "case",
    [
        dict(
            config=dict(
                structure_matrix_style="adj",
                node_weight_init="degree",
                loss_fn="square_loss",
                relaxed=False,
            ),
            indices=(0, -1),
            expected=dict(
                x=torch.Size([40, 3]),
                x_sum=40.0,
                edge_index=torch.Size([2, 148]),
                edge_index_sum=5714,
                y=torch.Size([1]),
            ),
        ),
        dict(
            config=dict(
                structure_matrix_style="adj",
                node_weight_init="degree",
                loss_fn="kl_loss",
                relaxed=False,
            ),
            indices=(0, -1),
            expected=dict(
                x=torch.Size([40, 3]),
                x_sum=40.0,
                edge_index=torch.Size([2, 50]),
                edge_index_sum=1498,
                y=torch.Size([1]),
            ),
        ),
        dict(
            config=dict(
                structure_matrix_style="adj",
                node_weight_init="uniform",
                loss_fn="square_loss",
                relaxed=False,
            ),
            indices=(0, -1),
            expected=dict(
                x=torch.Size([40, 3]),
                x_sum=40.0,
                edge_index=torch.Size([2, 148]),
                edge_index_sum=5644,
                y=torch.Size([1]),
            ),
        ),
        dict(
            config=dict(
                structure_matrix_style="adj",
                node_weight_init="degree",
                loss_fn="square_loss",
                relaxed=True,
            ),
            # FIXME yields graphs without any edges for many index pairs, e.g., (5, -2), (15, -20), (150, -200)
            # /home/user/code/graph_mixup/graph_mixup/fgw_mixup/FGW_barycenter.py:34: RuntimeWarning: overflow encountered in exp
            #     X = np.exp(grad / rho) * X
            # /home/user/code/graph_mixup/graph_mixup/fgw_mixup/FGW_barycenter.py:35: RuntimeWarning: invalid value encountered in multiply
            #     X = X * (a / (X @ np.ones_like(b)))
            indices=(0, -1),
            expected=dict(
                x=torch.Size([40, 3]),
                x_sum=40.0,
                edge_index=torch.Size([2, 149]),
                edge_index_sum=4870,
                y=torch.Size([1]),
            ),
        ),
        dict(
            config=dict(
                structure_matrix_style="sp",
                node_weight_init="degree",
                loss_fn="square_loss",
                relaxed=False,
            ),
            indices=(10, -12),
            expected=dict(
                x=torch.Size([9, 3]),
                x_sum=9.0,
                edge_index=torch.Size([2, 28]),
                edge_index_sum=226,
                y=torch.Size([1]),
            ),
        ),
        dict(
            config=dict(
                structure_matrix_style="sp",
                node_weight_init="degree",
                loss_fn="square_loss",
                relaxed=True,
            ),
            # FIXME Yields graphs with too many edges, e.g., for (5, -2):
            # g1:    Data(edge_index=[2, 1632], x=[336, 3], y=[1])
            # g2:    Data(edge_index=[2, 18], x=[5, 3], y=[1])
            # mixup: Data(x=[74, 3], edge_index=[2, 5476], y=[1])
            # FIXME Generally has too many edges
            # FIXME All node features are 'nan'
            indices=(10, -12),
            expected=dict(
                x=torch.Size([9, 3]),
                x_sum=np.nan,
                edge_index=torch.Size([2, 81]),
                edge_index_sum=648,
                y=torch.Size([1]),
            ),
        ),
    ],
)
def test_fgw_mixup(case):
    dataset = TUDataset(root="data", name="PROTEINS")

    # Get two graphs from PROTEINS
    g0 = dataset[case["indices"][0]]
    g1 = dataset[case["indices"][1]]

    # Compute mixup
    data = mixup(g0=g0, g1=g1, **case["config"])

    print(case["config"])
    print(g0)
    print(g1)
    print(data)

    # Validate and test
    data.validate()
    assert data.x.size() == case["expected"]["x"]
    assert (
        case["expected"]["x_sum"] is np.nan
        or torch.abs(data.x.sum() - case["expected"]["x_sum"]) < 1e-3
    )
    assert data.edge_index.size() == case["expected"]["edge_index"]
    assert data.edge_index.sum().item() == case["expected"]["edge_index_sum"]
    assert data.y.size() == case["expected"]["y"]
