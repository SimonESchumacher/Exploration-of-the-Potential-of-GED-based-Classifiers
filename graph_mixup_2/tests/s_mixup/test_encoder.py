import torch
from torch_geometric.data import Data, Batch
import pytest
from graph_mixup.mixup_methods.s_mixup.gmnet.encoder import Encoder
from tests.base import BaseTest


class TestEncoder(BaseTest):
    @pytest.mark.parametrize("readout", [True, False])
    @pytest.mark.parametrize("pool_type", ["mean", "sum", "max"])
    @pytest.mark.parametrize("use_gate", [True, False])
    @pytest.mark.parametrize("node_update_type", ["mlp", "residual", "gru"])
    @pytest.mark.parametrize("layer_norm", [False, True])
    def test_encoder(
        self,
        pool_type,
        use_gate,
        node_update_type,
        layer_norm,
        readout,
    ):
        node_feat_dim = 1
        num_layers = 3
        hidden = 4

        # Initialize model
        encoder = Encoder(
            node_feat_dim,
            num_layers,
            hidden,
            pool_type,
            use_gate,
            node_update_type,
            layer_norm,
        )

        # Initialize toy graphs
        data0 = Data(
            x=torch.ones(4, node_feat_dim),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ]
            ),
        )
        data1 = Data(
            x=torch.ones(5, node_feat_dim),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4],
                    [1, 0, 2, 1, 3, 2, 4, 3],
                ]
            ),
        )
        batch0 = Batch.from_data_list([data0, data0])
        batch1 = Batch.from_data_list([data1, data1])

        out0, out1 = encoder(batch0, batch1, readout=readout)

        if readout:
            assert out0.shape == torch.Size([2, hidden])
            assert out1.shape == torch.Size([2, hidden])
        else:
            assert out0.shape == torch.Size([2 * 4, hidden])
            assert out1.shape == torch.Size([2 * 5, hidden])
