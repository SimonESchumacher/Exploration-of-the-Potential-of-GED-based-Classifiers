import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from graph_mixup.transforms.normalized_degree_transform import NormalizedDegree


def test_normalized_degree_transform():
    # Create a sample graph
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ]
    )
    x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.validate()

    # Apply the NormalizedDegree transform
    mean = torch.mean(degree(data.edge_index[0], dtype=torch.float))
    std = torch.std(degree(data.edge_index[0], dtype=torch.float))
    transform = NormalizedDegree(mean, std)
    transformed_data = transform(data)

    # Check if the transformation is applied correctly
    transformed_data.validate()
    expected_deg = (degree(data.edge_index[0], dtype=torch.float) - mean) / std
    assert torch.allclose(transformed_data.x, expected_deg.view(-1, 1))

    # Check if the original data is not modified
    assert torch.allclose(data.x, x)
    assert torch.allclose(data.edge_index, edge_index)
