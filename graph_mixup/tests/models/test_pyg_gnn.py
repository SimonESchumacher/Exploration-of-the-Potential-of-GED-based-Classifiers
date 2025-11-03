import torch
import pytest
from graph_mixup.models.pyg_gnn import GCNClassification, GINClassification, PygGNN
from graph_mixup.transforms.one_hot_label_transform import OneHotLabel
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.loader import DataLoader
from tests.base import BaseTest


class TestPygGNN(BaseTest):
    @pytest.fixture
    def batch(self) -> Batch:
        proteins = TUDataset("data", "PROTEINS", transform=OneHotLabel(2))
        loader = DataLoader(proteins, batch_size=4, shuffle=True)
        return next(iter(loader))

    @pytest.fixture(params=["gcn", "gin"])
    def vanilla_model(self, request) -> PygGNN:
        params = dict(
            in_channels=3,
            num_conv_layers=4,
            hidden_channels=4,
            out_channels=2,
            aggr_fn=global_add_pool,
            dropout=0.5,
            norm="BatchNorm",
            num_post_processing_layers=2,
        )

        return (
            GCNClassification(**params)
            if request.param == "gcn"
            else GINClassification(**params)
        )

    @pytest.fixture(params=["gcn", "gin"])
    def mixup_model(self, request) -> PygGNN:
        params = dict(
            in_channels=3,
            num_conv_layers=4,
            hidden_channels=4,
            out_channels=2,
            aggr_fn=global_add_pool,
            dropout=0.5,
            norm="BatchNorm",
            num_post_processing_layers=2,
            mixup_method="emb_mixup",
            mixup_method_params=dict(
                mixup_alpha=0.2,
                vanilla_ratio=0.75,
                augmented_ratio=0.5,
            ),
        )

        return (
            GCNClassification(**params)
            if request.param == "gcn"
            else GINClassification(**params)
        )

    @pytest.mark.parametrize("use_eval", [False, True])
    def test_vanilla(self, vanilla_model, batch, use_eval):
        if use_eval:
            vanilla_model.eval()
        out, labels = vanilla_model(batch)

        assert out.size() == torch.Size([4, 2])
        assert labels.size() == torch.Size([4, 2])

    def test_mixup_train(self, mixup_model, batch):
        out, labels = mixup_model(batch)

        assert out.size() == torch.Size([5, 2])
        assert labels.size() == torch.Size([5, 2])
        assert not batch.y.equal(labels)

    def test_mixup_eval(self, mixup_model, batch):
        mixup_model.eval()
        out, labels = mixup_model(batch)

        assert out.size() == torch.Size([4, 2])
        assert labels.size() == torch.Size([4, 2])
        assert batch.y.equal(labels)
