"""Tests for tiedloc.team_topologies — team topology generators and registry."""

from __future__ import annotations

import warnings

import networkx as nx
import pytest

from tiedloc.team_topologies import (
    CompleteGraphTeam,
    RegularGraphTeam,
    StarTeam,
    TeamTopologyGenerator,
    get_team_topology,
    register_team_topology,
)


class TestTeamTopologyRegistry:
    def test_regular_graph_registered(self):
        gen = get_team_topology("regular graph")
        assert isinstance(gen, RegularGraphTeam)

    def test_complete_registered(self):
        gen = get_team_topology("complete")
        assert isinstance(gen, CompleteGraphTeam)

    def test_star_registered(self):
        gen = get_team_topology("star")
        assert isinstance(gen, StarTeam)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="nonexistent_team_topo"):
            get_team_topology("nonexistent_team_topo")

    def test_register_custom(self):
        class CustomTeam:
            def generate(self, params):
                return nx.cycle_graph(params["team_members"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_team_topology("test_custom_team", CustomTeam())
        gen = get_team_topology("test_custom_team")
        g = gen.generate({"team_members": 5})
        assert g.number_of_nodes() == 5


class TestRegularGraphTeam:
    def test_correct_node_count(self):
        gen = RegularGraphTeam()
        g = gen.generate({"team_degree": 2, "team_members": 6})
        assert g.number_of_nodes() == 6

    def test_all_nodes_have_correct_degree(self):
        gen = RegularGraphTeam()
        g = gen.generate({"team_degree": 4, "team_members": 10})
        for _, deg in g.degree():
            assert deg == 4

    def test_returns_graph(self):
        gen = RegularGraphTeam()
        g = gen.generate({"team_degree": 2, "team_members": 6})
        assert isinstance(g, nx.Graph)


class TestCompleteGraphTeam:
    def test_correct_node_count(self):
        gen = CompleteGraphTeam()
        g = gen.generate({"team_members": 5})
        assert g.number_of_nodes() == 5

    def test_is_complete(self):
        gen = CompleteGraphTeam()
        g = gen.generate({"team_members": 4})
        # In a complete graph with n nodes, each node has degree n-1
        for _, deg in g.degree():
            assert deg == 3

    def test_edge_count(self):
        gen = CompleteGraphTeam()
        g = gen.generate({"team_members": 5})
        assert g.number_of_edges() == 10  # C(5,2) = 10


class TestStarTeam:
    def test_correct_node_count(self):
        gen = StarTeam()
        g = gen.generate({"team_members": 5})
        assert g.number_of_nodes() == 5

    def test_center_has_max_degree(self):
        gen = StarTeam()
        g = gen.generate({"team_members": 6})
        degrees = [d for _, d in g.degree()]
        assert max(degrees) == 5

    def test_edge_count(self):
        gen = StarTeam()
        g = gen.generate({"team_members": 4})
        assert g.number_of_edges() == 3


class TestTeamTopologyProtocol:
    def test_protocol_check(self):
        assert isinstance(RegularGraphTeam(), TeamTopologyGenerator)
        assert isinstance(CompleteGraphTeam(), TeamTopologyGenerator)
        assert isinstance(StarTeam(), TeamTopologyGenerator)
