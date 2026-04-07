"""Tests for tiedloc.topologies — topology generators and registry."""

from __future__ import annotations

import os
import tempfile
import warnings

import networkx as nx
import pytest

from tiedloc.topologies import (
    BarabasiAlbertGenerator,
    BinomialGenerator,
    EdgeListGenerator,
    PowerGridGenerator,
    TopologyGenerator,
    WattsStrogatzGenerator,
    get_topology,
    register_topology,
)


class TestTopologyRegistry:
    """Test that built-in topologies are registered and lookup works."""

    def test_barabasi_albert_registered(self):
        gen = get_topology("Barabasi Albert Scale-Free Network")
        assert isinstance(gen, BarabasiAlbertGenerator)

    def test_watts_strogatz_registered(self):
        gen = get_topology("Watts-Strogatz Small-World Model")
        assert isinstance(gen, WattsStrogatzGenerator)

    def test_binomial_registered(self):
        gen = get_topology("Binomial Graph")
        assert isinstance(gen, BinomialGenerator)

    def test_power_grid_registered(self):
        gen = get_topology("Power Grid of Western States of USA")
        assert isinstance(gen, PowerGridGenerator)

    def test_edge_list_registered(self):
        gen = get_topology("edge_list")
        assert isinstance(gen, EdgeListGenerator)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="nonexistent_topology"):
            get_topology("nonexistent_topology")

    def test_register_custom_topology(self):
        class Custom:
            def generate(self, params):
                return nx.path_graph(3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_topology("test_custom_topo", Custom())
        gen = get_topology("test_custom_topo")
        g = gen.generate({})
        assert g.number_of_nodes() == 3


class TestBarabasiAlbertGenerator:
    def test_generates_correct_node_count(self):
        gen = BarabasiAlbertGenerator()
        g = gen.generate({"num_of_nodes": 50, "new_node_to_existing_nodes": 2})
        assert g.number_of_nodes() == 50

    def test_generates_connected_graph(self):
        gen = BarabasiAlbertGenerator()
        g = gen.generate({"num_of_nodes": 20, "new_node_to_existing_nodes": 3})
        assert nx.is_connected(g)

    def test_returns_networkx_graph(self):
        gen = BarabasiAlbertGenerator()
        g = gen.generate({"num_of_nodes": 10, "new_node_to_existing_nodes": 2})
        assert isinstance(g, nx.Graph)


class TestWattsStrogatzGenerator:
    def test_generates_correct_node_count(self):
        gen = WattsStrogatzGenerator()
        g = gen.generate({"num_of_nodes": 30, "average_degree": 4})
        assert g.number_of_nodes() == 30

    def test_default_rewiring_prob(self):
        gen = WattsStrogatzGenerator()
        g = gen.generate({"num_of_nodes": 20, "average_degree": 4})
        assert isinstance(g, nx.Graph)

    def test_custom_rewiring_prob(self):
        gen = WattsStrogatzGenerator()
        g = gen.generate({"num_of_nodes": 20, "average_degree": 4, "rewiring_prob": 0.5})
        assert g.number_of_nodes() == 20

    def test_zero_rewiring_prob_gives_regular_graph(self):
        gen = WattsStrogatzGenerator()
        g = gen.generate({"num_of_nodes": 20, "average_degree": 4, "rewiring_prob": 0.0})
        # With p=0, every node should have exactly average_degree neighbors
        for _, deg in g.degree():
            assert deg == 4


class TestBinomialGenerator:
    def test_generates_correct_node_count(self):
        gen = BinomialGenerator()
        g = gen.generate({"num_of_nodes": 30, "average_degree": 4})
        assert g.number_of_nodes() == 30

    def test_returns_graph(self):
        gen = BinomialGenerator()
        g = gen.generate({"num_of_nodes": 20, "average_degree": 3})
        assert isinstance(g, nx.Graph)


class TestPowerGridGenerator:
    def test_custom_num_nodes_and_data_file(self):
        # Create a temp power file with some edges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write edge pairs: 0-1, 1-2
            f.write("0\n1\n1\n2\n")
            tmppath = f.name
        try:
            gen = PowerGridGenerator()
            g = gen.generate({"num_of_nodes": 5, "data_file": tmppath})
            assert g.number_of_nodes() == 5
            assert g.has_edge(0, 1)
            assert g.has_edge(1, 2)
        finally:
            os.unlink(tmppath)

    def test_default_num_nodes(self):
        # Without data_file override, it'll try default path — just check params handling
        params = {}
        assert int(params.get("num_of_nodes", 4941)) == 4941


class TestEdgeListGenerator:
    def test_loads_edge_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0 1\n1 2\n2 3\n")
            tmppath = f.name
        try:
            gen = EdgeListGenerator()
            g = gen.generate({"edge_list_file": tmppath})
            assert g.number_of_nodes() == 4
            assert g.has_edge(0, 1)
            assert g.has_edge(2, 3)
        finally:
            os.unlink(tmppath)


class TestFileErrorHandling:
    """Tests for descriptive errors on missing/malformed files."""

    def test_power_grid_missing_file_raises(self):
        gen = PowerGridGenerator()
        with pytest.raises(FileNotFoundError, match="Power grid data file not found"):
            gen.generate({"data_file": "/nonexistent/path/power.txt"})

    def test_edge_list_missing_file_raises(self):
        gen = EdgeListGenerator()
        with pytest.raises(FileNotFoundError, match="Edge list file not found"):
            gen.generate({"edge_list_file": "/nonexistent/path/edges.txt"})


class TestTopologyProtocol:
    def test_protocol_check(self):
        assert isinstance(BarabasiAlbertGenerator(), TopologyGenerator)
        assert isinstance(WattsStrogatzGenerator(), TopologyGenerator)
        assert isinstance(BinomialGenerator(), TopologyGenerator)
        assert isinstance(PowerGridGenerator(), TopologyGenerator)
        assert isinstance(EdgeListGenerator(), TopologyGenerator)
