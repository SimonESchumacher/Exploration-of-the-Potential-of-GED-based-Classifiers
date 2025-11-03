import torch
from torch_geometric.data import Data, Batch
import pytest
from torch_geometric.data import Batch

from graph_mixup.mixup_methods.s_mixup.gmnet import LitGraphMatchingNet
from tests.base import BaseTest


class TestLitGraphMatchingNet(BaseTest):
    node_feat_dim = 1
    num_layers = 3
    hidden = 4

    @pytest.fixture
    def toy_graphs(self):
        data0 = Data(
            x=torch.ones(4, self.node_feat_dim),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3],
                    [1, 0, 2, 1, 3, 2],
                ]
            ),
        )
        data1 = Data(
            x=torch.ones(5, self.node_feat_dim),
            edge_index=torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4],
                    [1, 0, 2, 1, 3, 2, 4, 3],
                ]
            ),
        )
        return data0, data1

    @pytest.mark.parametrize("pool_type", ["mean", "sum", "max"])
    @pytest.mark.parametrize(
        "fuse_type", ["abs_diff", "add", "multiply", "concat", "cos"]
    )
    @pytest.mark.parametrize("pred_head", [True, False])
    def test_graph_matching_net_forward(
        self,
        toy_graphs: tuple[Data, Data],
        fuse_type,
        pred_head,
        pool_type,
    ):
        model = LitGraphMatchingNet(
            node_feat_dim=self.node_feat_dim,
            num_layers=self.num_layers,
            hidden=self.hidden,
            pool_type=pool_type,
            fuse_type=fuse_type,
        )

        data0, data1 = toy_graphs
        batch0 = Batch.from_data_list([data0, data0])
        batch1 = Batch.from_data_list([data1, data1])

        pos_out = model(batch0, batch0, pred_head=pred_head)
        neg_out = model(batch1, batch1, pred_head=pred_head)

        if pred_head:
            if fuse_type == "cos":
                assert pos_out.shape == torch.Size([2, 1])
                assert neg_out.shape == torch.Size([2, 1])
            else:

                pass
        else:
            assert pos_out[0].shape == torch.Size([2, self.hidden])
            assert neg_out[0].shape == torch.Size([2, self.hidden])
            assert pos_out[1].shape == torch.Size([2, self.hidden])
            assert neg_out[1].shape == torch.Size([2, self.hidden])

    @pytest.mark.parametrize("pool_type", ["mean", "sum", "max"])
    @pytest.mark.parametrize(
        "fuse_type", ["abs_diff", "add", "multiply", "concat", "cos"]
    )
    @pytest.mark.parametrize("loss_type", ["margin", "hamming"])
    def test_graph_matching_net_training_step(
        self,
        toy_graphs: tuple[Data, Data],
        fuse_type,
        pool_type,
        loss_type,
    ):
        model = LitGraphMatchingNet(
            node_feat_dim=self.node_feat_dim,
            num_layers=self.num_layers,
            hidden=self.hidden,
            pool_type=pool_type,
            fuse_type=fuse_type,
            loss_type=loss_type,
        )

        batch0 = Batch.from_data_list([toy_graphs[0], toy_graphs[0]])
        batch1 = Batch.from_data_list([toy_graphs[1], toy_graphs[1]])
        model.configure_optimizers()
        out = model.training_step((batch0, batch0, batch1), 0)

        assert out.dim() == 0
