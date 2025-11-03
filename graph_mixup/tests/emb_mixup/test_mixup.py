import torch
import pytest
from lightning import seed_everything
from graph_mixup.mixup_methods.emb_mixup.mixup import mixup


class TestEmbMixup:
    @pytest.fixture(autouse=True)
    def setup(self):
        seed_everything(0)

    def test_mixup_fully_augmented(self):
        items = torch.randn(4, 16)
        targets = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1]])

        mixup_items, mixup_targets = mixup(
            items, targets, mixup_alpha=0.2, vanilla_ratio=0.0, augmented_ratio=1.0
        )

        assert mixup_items.shape == items.shape
        assert mixup_targets.shape == targets.shape

    def test_mixup_partially_augmented(self):
        items = torch.randn(4, 16)
        targets = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1]])

        torch.manual_seed(1234)

        mixup_items, mixup_targets = mixup(
            items, targets, mixup_alpha=0.5, vanilla_ratio=0.75, augmented_ratio=0.75
        )
        print(mixup_items)
        print(mixup_targets)

        assert mixup_items.shape == torch.Size([6, 16])
        assert mixup_targets.shape == torch.Size([6, 2])
