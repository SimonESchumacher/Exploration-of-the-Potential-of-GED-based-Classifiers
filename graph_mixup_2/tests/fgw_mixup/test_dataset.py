import pytest
from torch_geometric.datasets import TUDataset
from torch.utils.data import Dataset
from graph_mixup.mixup_methods.fgw_mixup.dataset import FGWMixupDataset
from tests.dataset_base import MixupDatasetBaseTest


class TestFGWMixupDataset(MixupDatasetBaseTest):
    @pytest.fixture
    def mixup_dataset(self, vanilla_dataset: Dataset) -> Dataset:
        return FGWMixupDataset(vanilla_dataset, structure_matrix_style="sp")

    def test_no_one_hot_encoding_raises_error(self):
        with pytest.raises(AssertionError):
            FGWMixupDataset(TUDataset("data", "PROTEINS"))

    def test_len(self, vanilla_dataset):
        mixup_dataset = FGWMixupDataset(vanilla_dataset, augmented_ratio=0.0)
        assert len(mixup_dataset) == len(vanilla_dataset)

        mixup_dataset = FGWMixupDataset(vanilla_dataset, augmented_ratio=0.5)
        assert len(mixup_dataset) == int(1.5 * len(vanilla_dataset))

        mixup_dataset = FGWMixupDataset(
            vanilla_dataset, vanilla_ratio=0.5, augmented_ratio=0.5
        )
        assert abs(len(mixup_dataset) - len(vanilla_dataset)) <= 1
