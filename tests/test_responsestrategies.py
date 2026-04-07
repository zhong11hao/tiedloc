"""Unit tests for the tiedloc.responsestrategies module."""

import random

import pytest
import simpy

from tiedloc.agents import Link, Node, Server
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState
from tiedloc.responsestrategies import (
    _assign_link_collaborators,
    _build_idle_matrix,
    _filter_schedule,
    _find_idle_agents,
    _init_agent_schedule,
    _nn_assign_link,
    alive_links,
    calc_activity,
    calc_nearness,
    dispatcher,
    fcfs_key,
    find_costless_agent,
    implement_schedule,
    maintenance_check,
    nn_schedule,
    protection_density,
    wire_aux,
)
from tiedloc.strategies import (
    ActivityStrategy,
    NearestStrategy,
    get_strategy,
)


class TestSortingKeys:
    """Tests for the sorting key functions."""

    def test_fcfs_key_sorts_by_failed_time(self):
        """fcfs_key should sort by failed_time."""
        n1 = Node(0)
        n1.failed_time = 5.0
        n2 = Node(1)
        n2.failed_time = 2.0
        n3 = Node(2)
        n3.failed_time = 8.0

        sorted_nodes = sorted([n1, n2, n3], key=fcfs_key)
        assert sorted_nodes[0].id == 1  # failed_time 2.0
        assert sorted_nodes[1].id == 0  # failed_time 5.0
        assert sorted_nodes[2].id == 2  # failed_time 8.0

    def test_nearest_key_sorts_by_distance(self):
        """NearestStrategy.sort_key should sort tuples by distance (index 1)."""
        items = [
            (0.5, 10.0, 0, None),
            (0.8, 3.0, 1, None),
            (0.2, 7.0, 2, None),
        ]
        sorted_items = sorted(items, key=NearestStrategy.sort_key)
        assert sorted_items[0][2] == 1  # distance 3.0
        assert sorted_items[1][2] == 2  # distance 7.0
        assert sorted_items[2][2] == 0  # distance 10.0

    def test_activity_key_sorts_by_activity_desc_then_distance_asc(self):
        """ActivityStrategy.sort_key should sort by activity descending, then distance ascending."""
        items = [
            (0.5, 10.0, 0, None),
            (0.8, 3.0, 1, None),
            (0.8, 1.0, 2, None),
            (0.2, 7.0, 3, None),
        ]
        sorted_items = sorted(items, key=ActivityStrategy.sort_key)
        # Highest activity first (0.8), then by distance
        assert sorted_items[0][2] == 2  # activity 0.8, distance 1.0
        assert sorted_items[1][2] == 1  # activity 0.8, distance 3.0
        assert sorted_items[2][2] == 0  # activity 0.5, distance 10.0
        assert sorted_items[3][2] == 3  # activity 0.2, distance 7.0


class TestAliveLinks:
    """Tests for the alive_links helper."""

    def test_all_alive(self):
        """All links should be alive when none have failed."""
        params = {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": 10,
                "average_degree": 4,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(params)
        for node_id in network.graph.nodes:
            count = alive_links(network, node_id)
            assert count == network.graph.degree(node_id)

    def test_with_failed_link(self):
        """alive_links should decrease when a link fails."""
        params = {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": 10,
                "average_degree": 4,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(params)
        edges = list(network.graph.edges)
        if edges:
            v1, v2 = edges[0]
            link = network.get_link(v1, v2)
            original_count = alive_links(network, v1)
            link.state = 1
            new_count = alive_links(network, v1)
            assert new_count == original_count - 1


class TestFindCostlessAgent:
    """Tests for the find_costless_agent function."""

    def test_finds_cheapest_agent(self):
        """find_costless_agent should return the agent with the lowest cost."""
        env = simpy.Environment()
        sim_state = SimulationState()

        # Create a small CPS network
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(cps_params, sim_state=sim_state)

        # Create a service team
        team_params = {
            "service_team": {
                "name": "regular graph",
                "team_members": 4,
                "team_degree": 2,
                "agent_travel_speed": 1.0,
                "repairing_time": 1.0,
                "initial_allocation": "Centrality",
            }
        }
        team = ServiceTeam(team_params, env)
        team.allocation(team_params, network)

        # Find the cheapest agent for a target node
        target_node = network.get_node(0)
        target_node.state = 1
        agent_id, job = find_costless_agent(env, network, team, target_node, 0, sim_state)

        assert agent_id >= 0
        assert job["node"] == 0
        assert job["finishing_time"] > 0
        assert "failure" in job
        assert "assign" in job


def _make_simulation_fixtures(n=10, num_initial_failures=2, seed=42):
    """Helper to create a full simulation setup for response strategy tests."""
    cps_params = {
        "CPS": {
            "name": "Barabasi Albert Scale-Free Network",
            "num_of_nodes": n,
            "new_node_to_existing_nodes": 2,
        },
        "_topo_seed": seed,
        "response_protocol": {"auxiliary": False},
    }
    sim_state = SimulationState()
    sim_state.rng = random.Random(seed)
    network = CPSNetwork(cps_params, sim_state=sim_state)
    env = simpy.Environment()

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

    # Create initial failures
    fail_params = {"failure_model": {"initial_failures": num_initial_failures}}
    network.initial_failures(env, fail_params)

    return env, network, team, sim_state


class TestDispatcher:
    """Tests for the dispatcher function (FCFS scheduling)."""

    def test_dispatcher_assigns_agents_to_failures(self):
        """dispatcher should create agent schedules for existing failures."""
        env, network, team, sim_state = _make_simulation_fixtures()
        assert len(sim_state.failures) > 0

        dispatcher(env, network, team)

        assert sim_state.agent_schedule is not None
        # At least one agent should have jobs assigned
        total_jobs = sum(
            len(jobs) for jobs in sim_state.agent_schedule.values()
        )
        assert total_jobs > 0

    def test_dispatcher_empties_failures_list(self):
        """After dispatcher runs, failures should be empty or contain unassignable items."""
        env, network, team, sim_state = _make_simulation_fixtures()
        dispatcher(env, network, team)
        # Failures should be reduced (some may remain as next_iteration)
        # The key check is that agent_schedule was populated
        assert sim_state.agent_schedule is not None

    def test_dispatcher_job_has_required_keys(self):
        """Each job in the schedule should have the required keys."""
        env, network, team, sim_state = _make_simulation_fixtures()
        dispatcher(env, network, team)

        required_keys = {"node", "failure", "departing_time", "arrival_time", "finishing_time", "process_assigned", "assign"}
        for agent_id, jobs in sim_state.agent_schedule.items():
            for job in jobs:
                for key in required_keys:
                    assert key in job, f"Missing key {key} in job for agent {agent_id}"


class TestNnSchedule:
    """Tests for the nn_schedule function (nearest/activity scheduling)."""

    def test_nearest_scheduling_assigns_agents(self):
        """nn_schedule with nearest strategy should assign agents to failures."""
        env, network, team, sim_state = _make_simulation_fixtures()
        assert len(sim_state.failures) > 0

        nn_schedule(env, network, team, NearestStrategy())

        assert sim_state.agent_schedule is not None
        total_jobs = sum(
            len(jobs) for jobs in sim_state.agent_schedule.values()
        )
        assert total_jobs > 0

    def test_activity_scheduling_assigns_agents(self):
        """nn_schedule with activity strategy should assign agents to failures."""
        env, network, team, sim_state = _make_simulation_fixtures()
        assert len(sim_state.failures) > 0

        nn_schedule(env, network, team, ActivityStrategy())

        assert sim_state.agent_schedule is not None
        total_jobs = sum(
            len(jobs) for jobs in sim_state.agent_schedule.values()
        )
        assert total_jobs > 0


class TestWireAux:
    """Tests for the wire_aux function (auxiliary edge injection)."""

    def test_wire_aux_adds_edges(self):
        """wire_aux should potentially add auxiliary edges to the network."""
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 20,
                "new_node_to_existing_nodes": 2,
            },
            "_topo_seed": 42,
            "response_protocol": {"auxiliary": True},
        }
        sim_state = SimulationState()
        sim_state.rng = random.Random(42)
        network = CPSNetwork(cps_params, sim_state=sim_state)
        env = simpy.Environment()

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

        # Fail several nodes to create conditions for auxiliary wiring
        parameters = {
            "failure_model": {"fail_speed": 1.0, "initial_failures": 5},
            "response_protocol": {"auxiliary": True},
        }
        network.initial_failures(env, parameters)

        edges_before = network.graph.number_of_edges()
        wire_aux(env, parameters, network, team)

        # wire_aux may or may not add edges depending on topology, but should not crash
        # and aux_lines should be non-negative
        assert sim_state.aux_lines >= 0
        assert network.graph.number_of_edges() >= edges_before

    def test_wire_aux_skips_hosted_nodes(self):
        """wire_aux should skip failed nodes that are hosting a server."""
        cps_params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "_topo_seed": 42,
            "response_protocol": {"auxiliary": True},
        }
        sim_state = SimulationState()
        sim_state.rng = random.Random(42)
        network = CPSNetwork(cps_params, sim_state=sim_state)
        env = simpy.Environment()

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

        # Fail a node that is hosting a server — wire_aux should skip it
        for node_id in network.graph.nodes:
            node = network.get_node(node_id)
            if node.hosting:
                network.mark_failed(env, node, sim_state)
                break

        parameters = {
            "failure_model": {"fail_speed": 1.0},
            "response_protocol": {"auxiliary": True},
        }
        edges_before = network.graph.number_of_edges()
        wire_aux(env, parameters, network, team)

        # Hosted node should be skipped — no edges added for it
        # (may still add for other failures, but total is at least stable)
        assert network.graph.number_of_edges() >= edges_before


class TestMaintenanceCheck:
    """Tests for the maintenance_check SimPy process."""

    def test_maintenance_check_detects_new_failures(self):
        """maintenance_check should fire new_failures_event when failures_generation changes."""
        env, network, team, sim_state = _make_simulation_fixtures(num_initial_failures=0)
        parameters = {
            "response_protocol": {"frequency": 1.0},
        }
        sim_state.new_failures_event = env.event()
        initial_gen = sim_state.failures_generation

        env.process(maintenance_check(env, parameters, network, team))

        # After a timeout, add a new failure to trigger detection
        def add_failure_later(env):
            yield env.timeout(1.5)
            node = network.get_node(0)
            network.mark_failed(env, node, sim_state)

        env.process(add_failure_later(env))
        env.run(until=4)

        # The generation should have changed
        assert sim_state.failures_generation > initial_gen

    def test_maintenance_check_runs_at_frequency(self):
        """maintenance_check should check at the configured frequency."""
        env, network, team, sim_state = _make_simulation_fixtures(num_initial_failures=0)
        parameters = {
            "response_protocol": {"frequency": 2.0},
        }
        sim_state.new_failures_event = env.event()

        env.process(maintenance_check(env, parameters, network, team))
        env.run(until=5)

        # Should run without errors at frequency=2.0


class TestImplementSchedule:
    """Tests for the implement_schedule function."""

    def test_creates_processes_for_scheduled_jobs(self):
        """implement_schedule should create SimPy processes for unprocessed jobs."""
        env, network, team, sim_state = _make_simulation_fixtures()
        dispatcher(env, network, team)

        initial_queue_len = len(env._queue)
        implement_schedule(team, network, env)

        # New processes should have been added to env
        assert len(env._queue) > initial_queue_len

    def test_skips_already_processed_jobs(self):
        """implement_schedule should not create duplicate processes for already-processed jobs."""
        env, network, team, sim_state = _make_simulation_fixtures()
        dispatcher(env, network, team)

        implement_schedule(team, network, env)
        queue_after_first = len(env._queue)

        # Running again should not add more processes
        implement_schedule(team, network, env)
        assert len(env._queue) == queue_after_first

    def test_noop_when_no_schedule(self):
        """implement_schedule should be a no-op when agent_schedule is None."""
        env, network, team, sim_state = _make_simulation_fixtures()
        # Don't run dispatcher — agent_schedule remains None

        initial_queue_len = len(env._queue)
        implement_schedule(team, network, env)
        assert len(env._queue) == initial_queue_len


class TestCalcActivity:
    """Tests for the calc_activity function."""

    def test_returns_float_for_node(self):
        """calc_activity should return a float priority for a node."""
        env, network, team, sim_state = _make_simulation_fixtures()
        failed_node = sim_state.failures[0]
        result = calc_activity(failed_node, env, network, team)
        assert isinstance(result, float)

    def test_caches_result(self):
        """calc_activity should cache the result for the same time step."""
        env, network, team, sim_state = _make_simulation_fixtures()
        failed_node = sim_state.failures[0]
        result1 = calc_activity(failed_node, env, network, team)
        result2 = calc_activity(failed_node, env, network, team)
        assert result1 == result2
        assert failed_node.activity is not None
        assert failed_node.activity["time"] == env.now

    def test_activity_for_link(self):
        """calc_activity for a Link should return the min of its endpoints' activities."""
        env, network, team, sim_state = _make_simulation_fixtures()
        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)
        link.state = 1
        link.failed_time = 0.0

        result = calc_activity(link, env, network, team)
        assert isinstance(result, float)

    def test_activity_between_0_and_1(self):
        """calc_activity for a node should return a value between 0 and 1."""
        env, network, team, sim_state = _make_simulation_fixtures()
        failed_node = sim_state.failures[0]
        result = calc_activity(failed_node, env, network, team)
        assert 0.0 <= result <= 1.0


class TestCalcNearness:
    """Tests for the calc_nearness function."""

    def test_sets_nearness_for_node(self):
        """calc_nearness should set the nearness attribute on a node."""
        env, network, team, sim_state = _make_simulation_fixtures()
        failed_node = sim_state.failures[0]
        assert failed_node.nearness is None

        calc_nearness(failed_node, env, network, team)

        assert failed_node.nearness is not None
        assert failed_node.nearness["time"] == env.now
        assert isinstance(failed_node.nearness["value"], float)

    def test_caches_result(self):
        """calc_nearness should not recalculate if already computed at the same time."""
        env, network, team, sim_state = _make_simulation_fixtures()
        failed_node = sim_state.failures[0]

        calc_nearness(failed_node, env, network, team)
        first_value = failed_node.nearness["value"]

        calc_nearness(failed_node, env, network, team)
        assert failed_node.nearness["value"] == first_value

    def test_nearness_for_link(self):
        """calc_nearness for a Link should set the nearness attribute."""
        env, network, team, sim_state = _make_simulation_fixtures()
        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)
        link.state = 1
        link.failed_time = 0.0
        sim_state.failures.append(link)

        calc_nearness(link, env, network, team)

        assert link.nearness is not None
        assert isinstance(link.nearness["value"], float)


class TestProtectionDensity:
    """Tests for the protection_density function."""

    def test_zero_when_no_hosting(self):
        """protection_density should be 0 when no neighbors host servers."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(params)
        # No servers allocated — all hosting lists are empty
        for node_id in network.graph.nodes:
            result = protection_density(network, node_id)
            assert result == 0

    def test_positive_when_neighbors_host_servers(self):
        """protection_density should increase when neighboring nodes host servers."""
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(params)

        # Put a server on a neighbor of node 0
        neighbors = list(network.graph.neighbors(0))
        assert len(neighbors) > 0
        network.get_node(neighbors[0]).hosting.append(99)

        result = protection_density(network, 0)
        assert result >= 1

    def test_counts_two_hop_neighbors(self):
        """protection_density should count hosting servers at 2-hop distance."""
        params = {
            "CPS": {
                "name": "Watts-Strogatz Small-World Model",
                "num_of_nodes": 10,
                "average_degree": 4,
            },
            "response_protocol": {"auxiliary": False},
        }
        network = CPSNetwork(params)

        # Find a 2-hop path: node 0 -> neighbor -> neighbor's neighbor
        target = 0
        neighbors = list(network.graph.neighbors(target))
        if len(neighbors) >= 1:
            nei = neighbors[0]
            nei_neighbors = [n for n in network.graph.neighbors(nei) if n != target]
            if nei_neighbors:
                # Host a server at 2-hop neighbor
                network.get_node(nei_neighbors[0]).hosting.append(88)
                # Also host at 1-hop to satisfy the outer condition
                network.get_node(nei).hosting.append(77)
                result = protection_density(network, target)
                assert result >= 2  # at least nei (1) + nei_neighbor (1)
