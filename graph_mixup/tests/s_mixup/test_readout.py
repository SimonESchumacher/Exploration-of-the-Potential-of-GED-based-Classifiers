import pytest
from typing import Literal
import torch
from graph_mixup.mixup_methods.s_mixup.gmnet.readout import Readout
from torch_geometric.data import Data, Batch

from tests.base import BaseTest


class TestReadout(BaseTest):
    @pytest.mark.parametrize("use_gate", [True, False])
    @pytest.mark.parametrize("pool_type", ["mean", "sum", "max"])
    def test_readout(
        self,
        use_gate: bool,
        pool_type: Literal["mean", "sum", "max"],
    ):
        node_feat_dim = 1
        node_hiddens = [4, 2]
        graph_hiddens = [4, 2]

        # Initialize the Readout module
        readout = Readout(
            node_feat_dim,
            node_hiddens,
            graph_hiddens,
            use_gate,
            pool_type,
        )

        # Initialize toy graph
        data = Data(
            x=torch.ones(4, node_feat_dim),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ]
            ),
        )
        batch = Batch.from_data_list([data, data])

        # Perform forward pass
        output = readout(batch.x, batch.batch)

        # Check the output shape
        assert output.shape == (2, graph_hiddens[-1])
