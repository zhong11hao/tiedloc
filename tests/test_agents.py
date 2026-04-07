"""Unit tests for the tiedloc.agents module."""

import pytest
import simpy

from tiedloc.agents import Link, Node, Server
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState
from tiedloc.responsestrategies import JobRequest


class TestNode:
    """Tests for the Node agent class."""

    def test_init_defaults(self):
        """A new Node should have state 0, empty hosting, and correct ID."""
        node = Node(42)
        assert node.id == 42
        assert node.state == 0
        assert node.hosting == []
        assert node.failed_time == 0.0
        assert node.activity is None
        assert node.nearness is None

    def test_hosting_is_per_instance(self):
        """Each Node instance should have its own hosting list (no shared mutable default)."""
        n1 = Node(1)
        n2 = Node(2)
        n1.hosting.append(99)
        assert 99 not in n2.hosting
        assert n2.hosting == []

    def test_state_transitions(self):
        """Node state should be settable to 0 (operational) or 1 (failed)."""
        node = Node(0)
        assert node.state == 0
        node.state = 1
        assert node.state == 1
        node.state = 0
        assert node.state == 0


class TestLink:
    """Tests for the Link agent class."""

    def test_init_defaults(self):
        """A new Link should have state 0 and correct vertex IDs."""
        link = Link(3, 7)
        assert link.vertex1 == 3
        assert link.vertex2 == 7
        assert link.state == 0
        assert link.failed_time == 0.0
        assert link.aux_just_in is False

    def test_aux_just_in_flag(self):
        """The aux_just_in flag should be settable."""
        link = Link(0, 1)
        link.aux_just_in = True
        assert link.aux_just_in is True
        link.aux_just_in = False
        assert link.aux_just_in is False


class TestServer:
    """Tests for the Server agent class."""

    def test_init(self):
        """A new Server should have correct attributes and an idle state."""
        env = simpy.Environment()
        server = Server(5, env, travel_speed=2.0, repair_time=3.0)
        assert server.id == 5
        assert server.agent_travel_speed == 2.0
        assert server.repairing_time == 3.0
        assert server.state == 0
        assert server.current_pos == -1
        assert server.resource is not None

    def test_allocate_to(self):
        """allocate_to should set the server's position and register it on the node."""
        env = simpy.Environment()
        server = Server(0, env, travel_speed=1.0, repair_time=1.0)
        node = Node(10)

        server.allocate_to(node)
        assert server.current_pos == 10
        assert server.id in node.hosting

    def test_allocate_to_multiple_servers(self):
        """Multiple servers can be allocated to the same node."""
        env = simpy.Environment()
        s1 = Server(0, env, travel_speed=1.0, repair_time=1.0)
        s2 = Server(1, env, travel_speed=1.0, repair_time=1.0)
        node = Node(5)

        s1.allocate_to(node)
        s2.allocate_to(node)
        assert len(node.hosting) == 2
        assert 0 in node.hosting
        assert 1 in node.hosting

    def test_bidding_basic(self):
        """bidding should return a cost based on distance, travel speed, and repair time."""
        env = simpy.Environment()
        server = Server(0, env, travel_speed=2.0, repair_time=5.0)
        server.current_pos = 0

        # Create a mock network with a simple distance matrix
        class MockNetwork:
            distmat = {0: {0: 0, 1: 10}, 1: {0: 10, 1: 0}}

        network = MockNetwork()
        cost, dep_time, arr_time = server.bidding(env, network, target=1)

        # cost = distance/speed + repair_time + departing_time
        # = 10/2.0 + 5.0 + 0.0 = 10.0
        assert cost == 10.0
        assert dep_time == 0.0
        assert arr_time == 5.0  # cost - repair_time

    def test_bidding_with_schedule(self):
        """bidding with an existing schedule should account for the last job's position and time."""
        env = simpy.Environment()
        server = Server(0, env, travel_speed=1.0, repair_time=2.0)
        server.current_pos = 0

        class MockNetwork:
            distmat = {0: {0: 0, 1: 5, 2: 8}, 1: {0: 5, 1: 0, 2: 3}, 2: {0: 8, 1: 3, 2: 0}}

        network = MockNetwork()

        # Schedule says agent will be at node 1 finishing at time 10
        schedule = {0: [{"node": 1, "finishing_time": 10.0}]}
        cost, dep_time, arr_time = server.bidding(env, network, target=2, agent_schedule=schedule)

        # departing from node 1 at time 10, distance to node 2 is 3
        # cost = 3/1.0 + 2.0 + 10.0 = 15.0
        assert cost == 15.0
        assert dep_time == 10.0
        assert arr_time == 13.0  # cost - repair_time

    def test_resource_capacity(self):
        """Server resource should have capacity 1."""
        env = simpy.Environment()
        server = Server(0, env, travel_speed=1.0, repair_time=1.0)
        assert server.resource.capacity == 1


class TestNodeFailWatts:
    """Tests for the Node.node_fail_watts SimPy process."""

    def _make_small_network(self):
        """Create a small network with known topology for failure testing."""
        params = {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": 6,
                "average_degree": 4,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        return network, sim_state

    def _make_service_team(self, env):
        """Create a minimal service team."""
        team_params = {
            "service_team": {
                "name": "regular graph",
                "team_members": 4,
                "team_degree": 2,
                "agent_travel_speed": 1.0,
                "repairing_time": 1.0,
            }
        }
        return ServiceTeam(team_params, env)

    def test_node_fails_when_threshold_exceeded(self):
        """A node should fail when the fraction of failed neighbor edges exceeds phi."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        # Pick a node with neighbors
        target_id = 0
        target_node = network.get_node(target_id)
        neighbors = list(network.graph.neighbors(target_id))
        assert len(neighbors) > 0

        # Fail all neighbor links to guarantee threshold is exceeded
        for neighbor in neighbors:
            link = network.get_link(target_id, neighbor)
            if link is not None:
                link.state = 1

        phi = 0.0  # Very low threshold — any failed edge triggers failure
        env.process(target_node.node_fail_watts(env, phi, 1.0, network, service_team))
        env.run(until=2)

        assert target_node.state == 1

    def test_node_does_not_fail_below_threshold(self):
        """A node should remain operational when failed edges are below phi."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        target_id = 0
        target_node = network.get_node(target_id)

        # Don't fail any edges — ratio is 0
        phi = 0.5
        env.process(target_node.node_fail_watts(env, phi, 1.0, network, service_team))
        env.run(until=3)

        assert target_node.state == 0

    def test_already_failed_node_does_not_fail_again(self):
        """A node that is already failed should not increment total_failures again."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        target_id = 0
        target_node = network.get_node(target_id)
        # Pre-fail the node
        network.mark_failed(env, target_node, sim_state)
        initial_total = sim_state.total_failures

        phi = 0.0
        env.process(target_node.node_fail_watts(env, phi, 1.0, network, service_team))
        env.run(until=3)

        # Should not have incremented because node was already failed
        assert sim_state.total_failures == initial_total


class TestEdgeFailWatts:
    """Tests for the Link.edge_fail_watts SimPy process."""

    def _make_small_network(self):
        """Create a small network for edge failure testing."""
        params = {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": 6,
                "average_degree": 4,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        network = CPSNetwork(params, sim_state=sim_state)
        return network, sim_state

    def _make_service_team(self, env):
        """Create a minimal service team."""
        team_params = {
            "service_team": {
                "name": "regular graph",
                "team_members": 4,
                "team_degree": 2,
                "agent_travel_speed": 1.0,
                "repairing_time": 1.0,
            }
        }
        return ServiceTeam(team_params, env)

    def test_edge_fails_when_adjacent_node_failed(self):
        """An edge should fail when one adjacent node is failed and not hosting a server."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)

        # Fail node v1 (without hosting any server)
        node1 = network.get_node(v1)
        node1.state = 1
        node1.failed_time = 0.0

        env.process(link.edge_fail_watts(env, 1.0, network, service_team))
        env.run(until=2)

        assert link.state == 1

    def test_edge_prevented_when_hosting_server(self):
        """An edge should not fail if the failed node is hosting a server."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)

        # Fail node v1 but give it a hosting server
        node1 = network.get_node(v1)
        node1.state = 1
        node1.failed_time = 0.0
        node1.hosting = [0]  # Server 0 is hosting

        env.process(link.edge_fail_watts(env, 1.0, network, service_team))
        env.run(until=2)

        assert link.state == 0
        assert sim_state.prevented >= 1

    def test_edge_fails_when_both_nodes_failed(self):
        """An edge should fail when both adjacent nodes are failed."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)

        # Fail both nodes
        network.get_node(v1).state = 1
        network.get_node(v2).state = 1

        env.process(link.edge_fail_watts(env, 1.0, network, service_team))
        env.run(until=2)

        assert link.state == 1

    def test_already_failed_edge_stays_failed(self):
        """An edge that is already failed should not trigger additional failures."""
        network, sim_state = self._make_small_network()
        env = simpy.Environment()
        service_team = self._make_service_team(env)

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)

        # Pre-fail the link
        link.state = 1
        initial_total = sim_state.total_failures

        env.process(link.edge_fail_watts(env, 1.0, network, service_team))
        env.run(until=3)

        # total_failures should not increase because the link was already failed
        assert sim_state.total_failures == initial_total


class TestProvideService:
    """Tests for the Server.provide_service SimPy process."""

    def _make_network_and_team(self):
        """Create a small network and service team for provide_service tests."""
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        sim_state = SimulationState()
        env = simpy.Environment()
        network = CPSNetwork(cps_params, sim_state=sim_state)

        team_params = {
            "service_team": {
                "name": "regular graph",
                "team_members": 4,
                "team_degree": 2,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            }
        }
        team = ServiceTeam(team_params, env)
        team.allocation(team_params, network)
        return env, network, team, sim_state

    def test_server_restores_failed_node(self):
        """A server should travel to and restore a failed node."""
        env, network, team, sim_state = self._make_network_and_team()

        # Fail a node
        target_node = network.get_node(0)
        network.mark_failed(env, target_node, sim_state)
        assert target_node.state == 1

        # Pick a server
        server = team.get_server(0)
        cost, dep_time, arr_time = server.bidding(env, network, 0)

        job = {
            "node": 0,
            "failure": target_node,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        req = server.resource.request()
        job_req = JobRequest(req, job)
        env.process(server.provide_service(env, network, job_req))
        env.run(until=cost + 1)

        assert target_node.state == 0
        assert server.current_pos == 0

    def test_server_handles_link_failure(self):
        """A server at vertex1 of a failed link should not restore the link (needs collaborator)."""
        env, network, team, sim_state = self._make_network_and_team()

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)
        network.mark_failed(env, link, sim_state)
        assert link.state == 1

        server = team.get_server(0)
        cost, dep_time, arr_time = server.bidding(env, network, v1)

        job = {
            "node": v1,
            "failure": link,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        req = server.resource.request()
        job_req = JobRequest(req, job)
        env.process(server.provide_service(env, network, job_req))
        env.run(until=cost + 1)

        # Server at vertex1 of a link should NOT mark it restored (early return in provide_service)
        assert link.state == 1
        assert server.current_pos == v1
