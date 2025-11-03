import torch
from torch_geometric.data import Data, Batch
import pytest
from graph_mixup.mixup_methods.s_mixup.gmnet.conv import GMNConv
from tests.base import BaseTest


class TestConv(BaseTest):
    @pytest.mark.parametrize("node_update_type", ["mlp", "residual", "gru"])
    @pytest.mark.parametrize("layer_norm", [False, True])
    def test_conv(self, node_update_type, layer_norm):

        node_feat_dim = 1
        message_net_hiddens = [4, 2]
        update_net_hiddens = [4, 2]

        conv = GMNConv(
            node_feat_dim,
            message_net_hiddens,
            update_net_hiddens,
            node_update_type,
            layer_norm,
        )

        # Initialize toy graphs
        data0 = Data(
            x=torch.ones(4, node_feat_dim),
            num_nodes=4,
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ]
            ),
        )
        data1 = Data(
            x=torch.ones(5, node_feat_dim),
            num_nodes=5,
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4],
                    [1, 0, 2, 1, 3, 2, 4, 3],
                ]
            ),
        )
        batch0 = Batch.from_data_list([data0, data0])
        batch1 = Batch.from_data_list([data1, data1])

        # Perform forward pass
        out0, out1 = conv(
            batch0.x,
            batch0.edge_index,
            batch0.batch,
            batch1.x,
            batch1.edge_index,
            batch1.batch,
        )

        assert out0.shape == torch.Size([2 * data0.num_nodes, node_feat_dim])
        assert out1.shape == torch.Size([2 * data1.num_nodes, node_feat_dim])
