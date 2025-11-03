import pytest
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from graph_mixup.transforms.one_hot_label_transform import OneHotLabel
from tests.base import BaseTest
from abc import ABC, abstractmethod
from tqdm import tqdm


class MixupDatasetBaseTest(ABC, BaseTest):
    @pytest.fixture
    def vanilla_dataset(self) -> Dataset:
        return TUDataset("data", "PROTEINS", transform=OneHotLabel(2))

    @abstractmethod
    @pytest.fixture
    def mixup_dataset(self, vanilla_dataset) -> Dataset:
        """Returns the mixup dataset generated from the given vanilla dataset."""

    @abstractmethod
    def test_no_one_hot_encoding_raises_error(self):
        """Test that initializing a dataset without one-hot encoding raises an error."""

    @abstractmethod
    def test_len(self, vanilla_dataset):
        """Test that the length of the dataset is correct."""

    def test_get_vanilla_item(self, mixup_dataset):
        item = mixup_dataset[0]
        assert isinstance(item, Data)
        item.validate()

    def test_get_mixup_item(self, mixup_dataset):
        item = mixup_dataset[-1]
        assert isinstance(item, Data)
        item.validate()

    def test_vanilla_and_mixup_item_have_similar_attributes(
        self, mixup_dataset: Dataset
    ):
        vanilla_item: Data = mixup_dataset[0]
        mixup_item: Data = mixup_dataset[-1]
        assert vanilla_item.to_dict().keys() == mixup_item.to_dict().keys()

    def test_iterate_through_dataset(self, mixup_dataset):
        for item in tqdm(mixup_dataset):
            assert isinstance(item, Data)
            item.validate()

    def test_iterate_through_data_loader(self, mixup_dataset):
        loader = DataLoader(mixup_dataset, batch_size=4)
        for batch in tqdm(loader):
            assert isinstance(batch, Batch)
            assert batch.x is not None
            assert batch.y is not None
            assert batch.edge_index is not None
