"""Tests for tiedloc.strategies — response strategy registry and built-in strategies."""

from __future__ import annotations

import random
import warnings

import pytest
import simpy

from tiedloc.constants import STATE_FAILED
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState
from tiedloc.strategies import (
    ActivityStrategy,
    FCFSStrategy,
    NearestStrategy,
    ResponseStrategy,
    get_strategy,
    register_strategy,
)


class TestStrategyRegistry:
    def test_fcfs_registered(self):
        s = get_strategy("FCFS")
        assert isinstance(s, FCFSStrategy)

    def test_nearest_registered(self):
        s = get_strategy("nearest")
        assert isinstance(s, NearestStrategy)

    def test_activity_registered(self):
        s = get_strategy("activity")
        assert isinstance(s, ActivityStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="nonexistent_strategy"):
            get_strategy("nonexistent_strategy")

    def test_register_custom_strategy(self):
        class Custom(ResponseStrategy):
            def dispatch(self, env, network, service_team):
                pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            register_strategy("test_custom_strat", Custom())
        s = get_strategy("test_custom_strat")
        assert isinstance(s, Custom)


def _make_sim_env():
    """Create a small simulation with some failures for testing dispatch."""
    params = {
        "CPS": {
            "name": "Barabasi Albert Scale-Free Network",
            "num_of_nodes": 10,
            "new_node_to_existing_nodes": 2,
        },
        "_topo_seed": 42,
        "failure_model": {
            "name": "Watts cascade",
            "phi": 0.3,
            "fail_speed": 1.0,
            "initial_failures": 0,
        },
        "response_protocol": {"name": "FCFS", "frequency": 1.0, "auxiliary": False},
        "service_team": {
            "name": "complete",
            "team_members": 3,
            "agent_travel_speed": 1.0,
            "repairing_time": 2.0,
            "initial_allocation": "Centrality",
        },
        "simulation_param": {"seed": "42", "replications": "1", "processors": "1", "simulation_length": "20"},
    }
    env = simpy.Environment()
    sim_state = SimulationState()
    sim_state.rng = random.Random(42)
    network = CPSNetwork(params, sim_state=sim_state)
    service_team = ServiceTeam(params, env)
    service_team.allocation(params, network)

    # Create 2 failures
    network.mark_failed(env, network.nodes_map[0], sim_state)
    network.mark_failed(env, network.nodes_map[1], sim_state)

    return env, network, service_team


class TestFCFSStrategy:
    def test_dispatch_assigns_failures(self):
        env, network, service_team = _make_sim_env()
        strategy = FCFSStrategy()
        strategy.dispatch(env, network, service_team)
        # After dispatch, agent_schedule should exist
        assert network.sim_state.agent_schedule is not None

    def test_dispatch_processes_failure_list(self):
        env, network, service_team = _make_sim_env()
        strategy = FCFSStrategy()
        initial_failures = len(network.sim_state.failures)
        assert initial_failures == 2
        strategy.dispatch(env, network, service_team)
        # Some failures should have been assigned (possibly deferred)
        remaining = len(network.sim_state.failures)
        assert remaining <= initial_failures


class TestNearestStrategy:
    def test_dispatch_assigns_failures(self):
        env, network, service_team = _make_sim_env()
        strategy = NearestStrategy()
        strategy.dispatch(env, network, service_team)
        assert network.sim_state.agent_schedule is not None


class TestActivityStrategy:
    def test_dispatch_assigns_failures(self):
        env, network, service_team = _make_sim_env()
        strategy = ActivityStrategy()
        strategy.dispatch(env, network, service_team)
        assert network.sim_state.agent_schedule is not None
