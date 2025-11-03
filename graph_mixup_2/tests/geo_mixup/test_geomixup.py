import pytest
import torch
from torch_geometric.data import Data
from graph_mixup.mixup_methods.geo_mixup.mixup import mixup
from tests.base import BaseTest


class TestGeomixup(BaseTest):
    @pytest.fixture
    def graph_pair(self) -> tuple[Data, Data]:
        data0 = Data(
            x=torch.ones(4, 1),
            y=torch.tensor([0]),
            num_nodes=4,
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        )
        data1 = Data(
            x=torch.ones(5, 1),
            y=torch.tensor([2]),
            num_nodes=5,
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]]
            ),
        )
        return data0, data1

    @pytest.mark.parametrize(
        "mixup_alpha, fgw_alpha",
        [
            (0.4, 1.0),
            (0.6, 0.5),
        ],
    )
    def test_mixup_method(self, graph_pair, mixup_alpha, fgw_alpha):
        g0, g1 = graph_pair
        mixed_graph = mixup(g0, g1, mixup_alpha, g0.num_nodes, fgw_alpha=fgw_alpha)
        print(g0)
        print(g1)
        print(mixed_graph)
        mixed_graph.validate()

        assert mixed_graph.x.shape == g0.x.shape
        assert mixed_graph.num_nodes == g0.num_nodes
