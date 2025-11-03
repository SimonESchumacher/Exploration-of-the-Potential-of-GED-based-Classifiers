# Test cases for the graph_mixup.compute_ged module.

import os
import subprocess
import sys

import pytest

from torch_geometric.data import Data
import torch

from graph_mixup.compute_ged.__main__ import GEDComputer
from graph_mixup.compute_ged.typing import (
    ComputationMode,
    GEDResult,
    MissingGEDException,
    MissingMappingException,
)
from graph_mixup.compute_ged.parser import Args, parse_args

from graph_mixup.ged_database.models import Graph, Node


# --------------------------
# Fake subprocess.CompletedProcess for exact mode.
# --------------------------
class FakeCompletedProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# --------------------------
# Helper to create a toy Graph (ORM) for testing.
# --------------------------
def create_test_graph(
    graph_id: int, num_nodes: int, node_attrs: dict[int, tuple[float, ...]]
) -> Graph:
    g = Graph()
    g.graph_id = graph_id
    g.dataset_id = 0

    g.nodes = []
    for i in range(num_nodes):
        node = Node()
        node.node_id = graph_id * 1000 + i
        node.graph_id = graph_id
        node.index = i
        node.attributes = list(node_attrs.get(i, (0.0,)))
        g.nodes.append(node)

    g.edges = []
    return g


@pytest.fixture
def graphs():
    graph1 = create_test_graph(
        graph_id=1, num_nodes=3, node_attrs={0: (1.0,), 1: (2.0,), 2: (3.0,)}
    )
    graph2 = create_test_graph(
        graph_id=2, num_nodes=3, node_attrs={0: (1.0,), 1: (2.0,), 2: (3.0,)}
    )
    graph3 = create_test_graph(
        graph_id=3,
        num_nodes=5,
        node_attrs={0: (1.0,), 1: (2.0,), 2: (3.0,), 3: (4.0,), 4: (5.0,)},
    )
    return graph1, graph2, graph3


@pytest.fixture
def timeout_lb():
    return 5, 1000


def test_lower_bound_identical(graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb
    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy",
        ged_approx_method="IPFP",
    )
    lb = computer._lower_bound(graph1, graph2)
    assert lb == 0


def test_lower_bound_different(graphs, timeout_lb):
    graph1, _, graph3 = graphs
    timeout, lb_threshold = timeout_lb
    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy",
        ged_approx_method="IPFP",
    )
    diff_attrs = {0: (1.0,), 1: (2.0,), 2: (999.0,), 3: (4.0,), 4: (5.0,)}
    graph3._node_attrs = diff_attrs
    expected_lb = 2
    lb = computer._lower_bound(graph1, graph3)
    assert lb == expected_lb


def test_process_exact_success(monkeypatch, graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb
    fake_output = (
        "GED: 42\n"
        "Mapping: 0 -> 0, 1 -> 1, 2 -> 2\n"
        "Total time: 1,000 (microseconds)\n"
    ).encode()

    def mock_run(*args, **kwargs):
        return FakeCompletedProcess(stdout=fake_output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    result: GEDResult = computer.process(graph1, graph2)
    assert result.value == 42
    assert result.mapping == {0: 0, 1: 1, 2: 2}
    assert result.time == 1000
    assert result.lb == 0


def test_process_exact_timeout(monkeypatch, graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    monkeypatch.setattr(subprocess, "run", mock_run)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    result: GEDResult = computer.process(graph1, graph2)
    assert result.value == -1
    assert result.time == timeout * 1_000_000


def test_process_exact_missing_ged(monkeypatch, graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb
    fake_output = b"Mapping: 0 -> 0\nTotal time: 100 (microseconds)\n"

    def mock_run(*args, **kwargs):
        return FakeCompletedProcess(stdout=fake_output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    with pytest.raises(MissingGEDException):
        computer.process(graph1, graph2)


def test_process_exact_missing_mapping(monkeypatch, graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb
    fake_output = b"GED: 7\nTotal time: 200 (microseconds)\n"

    def mock_run(*args, **kwargs):
        return FakeCompletedProcess(stdout=fake_output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    with pytest.raises(MissingMappingException):
        computer.process(graph1, graph2)


def test_process_swaps_graph_ordering(monkeypatch, graphs, timeout_lb):
    graph1, _, graph3 = graphs
    timeout, lb_threshold = timeout_lb
    fake_output = (
        "GED: 13\n"
        "Mapping: 0 -> 0, 1 -> 1\n"
        "Total time: 300 (microseconds)\n"
    ).encode()

    def mock_run(*args, **kwargs):
        return FakeCompletedProcess(stdout=fake_output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", mock_run)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.EXACT,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    result: GEDResult = computer.process(graph3, graph1)
    assert result.graph_0_id == graph1.graph_id
    assert result.graph_1_id == graph3.graph_id
    assert result.value == 13


def test_process_approximate_mode(monkeypatch, graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb

    def compute_ged_mock(*args, **kwargs):
        return (9, {0: 2}, 12345)

    monkeypatch.setattr(
        "graph_mixup.compute_ged.__main__.compute_ged_with_gedlib",
        compute_ged_mock,
    )

    dummy = Data(
        x=torch.tensor([[1]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    def get_pyg_data_mock(*args, **kwargs):
        return dummy

    monkeypatch.setattr(Graph, "get_pyg_data", get_pyg_data_mock)

    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode=ComputationMode.APPROXIMATE,
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    result: GEDResult = computer.process(graph1, graph2)
    assert result.value == 9
    assert result.mapping == {0: 2}
    assert result.time == 12345
    assert result.lb == 0


def test_unknown_mode_raises(graphs, timeout_lb):
    graph1, graph2, _ = graphs
    timeout, lb_threshold = timeout_lb
    computer = GEDComputer(
        timeout=timeout,
        lb_threshold=lb_threshold,
        mode="invalid_mode",
        exec_path="dummy_exec",
        ged_approx_method="IPFP",
    )
    with pytest.raises((ValueError, AssertionError)):
        computer.process(graph1, graph2)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset_name", "MUTAG"])
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    args: Args = parse_args()
    assert args.dataset_name == "MUTAG"
    assert args.n_cpus == 6
    assert args.timeout == 20
    assert args.lb_threshold == 1000
    assert args.batch_size == 64
    assert args.method_name is None
    assert args.approx_method == "IPFP"
    assert args.exec == "bin/edit_path_exec"
    assert args.mode == ComputationMode.EXACT.value


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset_name",
            "MUTAG",
            "--n_cpus",
            "4",
            "--timeout",
            "10",
            "--lb_threshold",
            "5",
            "--batch_size",
            "8",
            "--method_name",
            "",
            "--method",
            "BRANCH_FAST",
            "--exec",
            "/tmp/my_exec",
            "--mode",
            "approximate",
        ],
    )
    args: Args = parse_args()
    assert args.dataset_name == "MUTAG"
    assert args.n_cpus == 4
    assert args.timeout == 10
    assert args.lb_threshold == 5
    assert args.batch_size == 8
    assert args.method_name == ""
    assert args.approx_method == "BRANCH_FAST"
    assert args.exec == "/tmp/my_exec"
    assert args.mode == ComputationMode.APPROXIMATE.value


def test_parse_args_invalid_dataset_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--dataset_name", "INVALID_DATASET", "--method", "IPFP"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_invalid_method(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--dataset_name", "MUTAG", "--method", "INVALID_METHOD"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_invalid_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--dataset_name", "MUTAG", "--mode", "unsupported"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_missing_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        parse_args()
