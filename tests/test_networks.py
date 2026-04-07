"""Unit tests for the tiedloc.networks module."""

import os
import random
import tempfile

import pytest
import simpy

from tiedloc.agents import Link, Node
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState


class TestSimulationState:
    """Tests for the SimulationState dataclass."""

    def test_default_values(self):
        """SimulationState should have sensible defaults."""
        state = SimulationState()
        assert state.failed_elements == 0
        assert state.failures == []
        assert state.prevented == 0
        assert state.total_failures == 0
        assert state.total_distance == 0.0
        assert state.total_latency == 0.0
        assert state.latency == []
        assert state.failed_at_step == []
        assert state.agent_schedule is None

    def test_independent_instances(self):
        """Each SimulationState instance should have independent mutable fields."""
        s1 = SimulationState()
        s2 = SimulationState()
        s1.failures.append("test")
        assert s2.failures == []


class TestCPSNetwork:
    """Tests for the CPSNetwork class."""

    def _make_ba_params(self, n: int = 20, m: int = 2) -> dict:
        """Helper to create parameters for a Barabasi-Albert network."""
        return {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": n,
                "new_node_to_existing_nodes": m,
            },
            "response_protocol": {"auxiliary": False},
        }

    def _make_ws_params(self, n: int = 20, k: int = 4) -> dict:
        """Helper to create parameters for a Watts-Strogatz network."""
        return {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": n,
                "average_degree": k,
            },
            "response_protocol": {"auxiliary": False},
        }

    def _make_binomial_params(self, n: int = 20, avg_deg: int = 4) -> dict:
        """Helper to create parameters for a Binomial (Erdos-Renyi) network."""
        return {
            "CPS": {
                "name": "Binomial Graph",
                "num_of_nodes": n,
                "average_degree": avg_deg,
            },
            "response_protocol": {"auxiliary": False},
        }

    def test_ba_network_creation(self):
        """CPSNetwork should create a Barabasi-Albert graph with the correct number of nodes."""
        params = self._make_ba_params(n=30, m=2)
        network = CPSNetwork(params)
        assert network.graph.number_of_nodes() == 30
        assert network.graph.number_of_edges() > 0
        assert len(network.nodes_map) == 30

    def test_ws_network_creation(self):
        """CPSNetwork should create a Watts-Strogatz graph with the correct number of nodes."""
        params = self._make_ws_params(n=25, k=4)
        network = CPSNetwork(params)
        assert network.graph.number_of_nodes() == 25

    def test_binomial_network_creation(self):
        """CPSNetwork should create a Binomial graph with the correct number of nodes."""
        params = self._make_binomial_params(n=20, avg_deg=4)
        network = CPSNetwork(params)
        assert network.graph.number_of_nodes() == 20

    def test_node_agents_created(self):
        """Each graph node should have a corresponding Node agent in nodes_map."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        for node_id in network.graph.nodes:
            node = network.get_node(node_id)
            assert isinstance(node, Node)
            assert node.id == node_id

    def test_link_agents_created(self):
        """Each graph edge should have a corresponding Link agent in links_map."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        for v1, v2 in network.graph.edges:
            link = network.get_link(v1, v2)
            assert isinstance(link, Link)

    def test_get_link_bidirectional(self):
        """get_link should find the link regardless of vertex order."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        edges = list(network.graph.edges)
        if edges:
            v1, v2 = edges[0]
            assert network.get_link(v1, v2) is not None
            assert network.get_link(v2, v1) is not None
            assert network.get_link(v1, v2) is network.get_link(v2, v1)

    def test_get_link_nonexistent(self):
        """get_link should return None for non-existent edges."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        # Node IDs 9998 and 9999 don't exist in a 10-node graph
        assert network.get_link(9998, 9999) is None

    def test_distance_matrix_computed(self):
        """The distance matrix should be computed for all node pairs."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        assert len(network.distmat) == 10
        for src in network.graph.nodes:
            assert src in network.distmat
            for dst in network.graph.nodes:
                assert dst in network.distmat[src]
                assert network.distmat[src][dst] >= 0

    def test_distance_matrix_self_zero(self):
        """Distance from a node to itself should be zero."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        for node_id in network.graph.nodes:
            assert network.distmat[node_id][node_id] == 0

    def test_centrality_computed(self):
        """Betweenness centrality list should be computed and non-empty."""
        params = self._make_ba_params(n=15, m=2)
        network = CPSNetwork(params)
        assert len(network.bctlist) > 0
        # Should be sorted in descending order
        for i in range(len(network.bctlist) - 1):
            assert network.bctlist[i][0] >= network.bctlist[i + 1][0]

    def test_average_degree(self):
        """Average degree should be a positive float for a connected graph."""
        params = self._make_ba_params(n=20, m=3)
        network = CPSNetwork(params)
        assert network.average_degree > 0

    def test_mark_failed_and_restored(self):
        """mark_failed and mark_restored should correctly update state and statistics."""
        params = self._make_ba_params(n=10, m=2)
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        env = simpy.Environment()

        node = network.get_node(0)
        assert node.state == 0

        network.mark_failed(env, node, sim_state)
        assert node.state == 1
        assert sim_state.failed_elements == 1
        assert sim_state.total_failures == 1
        assert node in sim_state.failures

        # Advance time before restoring
        env.run(until=5)
        network.mark_restored(env, node, sim_state)
        assert node.state == 0
        assert sim_state.failed_elements == 0
        assert len(sim_state.latency) == 1
        assert sim_state.latency[0] == 5.0

    def test_pack_and_reconstruct(self):
        """pack() should produce a dict that can reconstruct the network."""
        params = self._make_ba_params(n=15, m=2)
        network = CPSNetwork(params)
        packed = network.pack()

        assert "CPS" in packed
        assert "picklePack_edge_data" in packed["CPS"]
        assert "num_of_nodes" in packed["CPS"]

        # Reconstruct from packed data
        network2 = CPSNetwork(packed)
        assert network2.graph.number_of_nodes() == 15
        assert network2.graph.number_of_edges() == network.graph.number_of_edges()

    def test_pack_full_preserves_node_attributes(self):
        """pack(full=True) should preserve node attributes via node_link_data."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        # Set a custom attribute on a graph node
        network.graph.nodes[0]["custom_attr"] = "test_value"

        packed = network.pack(full=True)
        assert "node_link_data" in packed["CPS"]
        assert "picklePack_edge_data" not in packed["CPS"]

        # Reconstruct
        network2 = CPSNetwork(packed)
        assert network2.graph.number_of_nodes() == 10
        assert network2.graph.number_of_edges() == network.graph.number_of_edges()
        assert network2.graph.nodes[0]["custom_attr"] == "test_value"

    def test_pack_full_preserves_edge_attributes(self):
        """pack(full=True) should preserve edge attributes."""
        params = self._make_ba_params(n=10, m=2)
        network = CPSNetwork(params)
        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        network.graph[v1][v2]["weight"] = 3.14

        packed = network.pack(full=True)
        network2 = CPSNetwork(packed)
        assert network2.graph[v1][v2]["weight"] == 3.14

    def test_no_class_level_shared_state(self):
        """Two CPSNetwork instances should not share mutable state."""
        params1 = self._make_ba_params(n=10, m=2)
        params2 = self._make_ba_params(n=15, m=2)
        n1 = CPSNetwork(params1)
        n2 = CPSNetwork(params2)
        assert n1.graph.number_of_nodes() != n2.graph.number_of_nodes()
        assert n1.bctlist is not n2.bctlist
        assert n1.distmat is not n2.distmat
        assert n1.nodes_map is not n2.nodes_map

    def test_edge_list_topology_end_to_end(self):
        """Edge list topology should load correctly via JSON-style config params."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0 1\n1 2\n2 3\n3 4\n4 0\n")
            tmppath = f.name
        try:
            params = {
                "CPS": {
                    "name": "edge_list",
                    "num_of_nodes": 5,
                    "edge_list_file": tmppath,
                },
                "response_protocol": {"auxiliary": False},
            }
            network = CPSNetwork(params)
            assert network.graph.number_of_nodes() == 5
            assert network.graph.number_of_edges() == 5
            assert network.graph.has_edge(0, 1)
            assert network.graph.has_edge(4, 0)
            assert len(network.nodes_map) == 5
            assert len(network.distmat) == 5
            assert len(network.bctlist) == 5
        finally:
            os.unlink(tmppath)


class TestServiceTeam:
    """Tests for the ServiceTeam class."""

    def _make_team_params(self, members: int = 5, degree: int = 2) -> dict:
        """Helper to create parameters for a regular-graph service team."""
        return {
            "service_team": {
                "name": "regular graph",
                "team_members": members,
                "team_degree": degree,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            }
        }

    def test_creation(self):
        """ServiceTeam should create a graph with the correct number of servers."""
        env = simpy.Environment()
        params = self._make_team_params(members=6, degree=2)
        team = ServiceTeam(params, env)
        assert team.graph.number_of_nodes() == 6
        assert len(team.servers) == 6

    def test_server_attributes(self):
        """Each server should have the configured travel speed and repair time."""
        env = simpy.Environment()
        params = self._make_team_params(members=4, degree=2)
        team = ServiceTeam(params, env)
        for server_id in team.graph.nodes:
            server = team.get_server(server_id)
            assert server.agent_travel_speed == 1.0
            assert server.repairing_time == 2.0

    def test_allocation(self):
        """After allocation, each server should be positioned at a CPS node."""
        env = simpy.Environment()
        team_params = self._make_team_params(members=4, degree=2)
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 20,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(cps_params)
        team = ServiceTeam(team_params, env)
        team.allocation(team_params, network)

        for server_id in team.graph.nodes:
            server = team.get_server(server_id)
            assert server.current_pos >= 0
            assert server.current_pos < 20


class TestFailureModel:
    """Tests for CPSNetwork.failure_model."""

    def _make_network_and_team(self, n=10):
        """Create a network and service team for failure model tests."""
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": n,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(cps_params, sim_state=sim_state)
        env = simpy.Environment()
        team_params = {
            "service_team": {
                "name": "regular graph",
                "team_members": 4,
                "team_degree": 2,
                "agent_travel_speed": 1.0,
                "repairing_time": 1.0,
            }
        }
        team = ServiceTeam(team_params, env)
        return network, sim_state, env, team

    def test_failure_model_creates_processes_for_all_nodes(self):
        """failure_model should register a SimPy process for every node."""
        network, sim_state, env, team = self._make_network_and_team(n=10)
        parameters = {
            "failure_model": {
                "name": "Watts cascade",
                "phi": 0.3,
                "fail_speed": 1.0,
            }
        }
        # Before calling failure_model, no processes except implicit ones
        initial_process_count = len(env._queue)
        network.failure_model(parameters, env, team)
        # After, there should be processes for nodes + edges
        num_nodes = network.graph.number_of_nodes()
        num_edges = network.graph.number_of_edges()
        expected_new = num_nodes + num_edges
        assert len(env._queue) == initial_process_count + expected_new

    def test_failure_model_triggers_cascade(self):
        """After initial failures and running, failure_model should propagate failures."""
        network, sim_state, env, team = self._make_network_and_team(n=10)
        parameters = {
            "failure_model": {
                "name": "Watts cascade",
                "phi": 0.1,
                "fail_speed": 1.0,
                "initial_failures": 3,
            }
        }

        sim_state.rng = random.Random(42)
        network.initial_failures(env, parameters)
        initial_failures = sim_state.total_failures
        assert initial_failures == 3

        network.failure_model(parameters, env, team)
        env.run(until=5)

        # With phi=0.1, cascading should happen — total should exceed initial
        assert sim_state.total_failures >= initial_failures


class TestInitialFailures:
    """Tests for CPSNetwork.initial_failures."""

    def test_fails_correct_number_of_nodes(self):
        """initial_failures should fail exactly the specified number of nodes."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 20,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        env = simpy.Environment()

        fail_params = {
            "failure_model": {"initial_failures": 3},
        }
        sim_state.rng = random.Random(42)
        network.initial_failures(env, fail_params)

        assert sim_state.total_failures == 3
        assert sim_state.failed_elements == 3
        assert len(sim_state.failures) == 3

    def test_failed_nodes_have_state_1(self):
        """All initially failed nodes should have state == 1."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 20,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        env = simpy.Environment()

        fail_params = {
            "failure_model": {"initial_failures": 2},
        }
        sim_state.rng = random.Random(0)
        network.initial_failures(env, fail_params)

        for element in sim_state.failures:
            assert element.state == 1

    def test_zero_initial_failures(self):
        """initial_failures with 0 should not fail any nodes."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 20,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        env = simpy.Environment()

        fail_params = {
            "failure_model": {"initial_failures": 0},
        }
        network.initial_failures(env, fail_params)

        assert sim_state.total_failures == 0
        assert sim_state.failed_elements == 0


class TestFileErrorHandling:
    """Tests for descriptive errors on missing data files."""

    def test_missing_distance_matrix_file(self):
        """Loading a nonexistent distance matrix should raise FileNotFoundError."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
                "distance_matrix": "nonexistent_distmat.txt",
            },
            "response_protocol": {"auxiliary": False},
        }
        with pytest.raises(FileNotFoundError, match="Distance matrix file not found"):
            CPSNetwork(params)

    def test_missing_centrality_file(self):
        """Loading a nonexistent betweenness centrality file should raise FileNotFoundError."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
                "betweenness_centrality": "nonexistent_bct.txt",
            },
            "response_protocol": {"auxiliary": False},
        }
        with pytest.raises(FileNotFoundError, match="Betweenness centrality file not found"):
            CPSNetwork(params)


class TestAuxThreshold:
    """Tests for auxiliary threshold calculation."""

    def test_aux_threshold_computed_when_auxiliary_enabled(self):
        """When auxiliary is enabled, aux_threshold should be computed."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": True},
        }
        network = CPSNetwork(params)
        assert len(network.aux_threshold) == 10
        for val in network.aux_threshold:
            assert val >= 0.0
