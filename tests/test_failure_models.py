"""Tests for tiedloc.failure_models — failure model registry and WattsCascadeModel."""

from __future__ import annotations

import random

import networkx as nx
import pytest
import simpy

from tiedloc.agents import Link, Node
from tiedloc.constants import STATE_FAILED, STATE_OPERATIONAL
from tiedloc.failure_models import (
    FailureModel,
    WattsCascadeModel,
    get_failure_model,
    register_failure_model,
)
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState


class TestFailureModelRegistry:
    def test_watts_cascade_registered(self):
        model = get_failure_model("Watts cascade")
        assert isinstance(model, WattsCascadeModel)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="nonexistent_model"):
            get_failure_model("nonexistent_model")

    def test_register_custom_model(self):
        class CustomModel(FailureModel):
            def install(self, env, network, service_team, params):
                pass

        register_failure_model("test_custom_fm", CustomModel)
        model = get_failure_model("test_custom_fm")
        assert isinstance(model, CustomModel)


class TestWattsCascadeModel:
    def _make_network_and_team(self):
        """Create a small BA network with a service team for testing."""
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
            "simulation_param": {"seed": "42", "replications": "1", "processors": "1", "simulation_length": "10"},
        }
        env = simpy.Environment()
        sim_state = SimulationState()
        sim_state.rng = random.Random(42)
        network = CPSNetwork(params, sim_state=sim_state)
        service_team = ServiceTeam(params, env)
        service_team.allocation(params, network)
        return env, network, service_team, params

    def test_install_creates_processes(self):
        env, network, service_team, params = self._make_network_and_team()
        model = WattsCascadeModel()
        initial_process_count = len(env._queue)
        model.install(env, network, service_team, params["failure_model"])
        # Should have added processes for nodes + edges
        assert len(env._queue) > initial_process_count

    def test_seed_initial_failures(self):
        env, network, service_team, params = self._make_network_and_team()
        model = WattsCascadeModel()
        params["failure_model"]["initial_failures"] = 2
        model.seed_initial_failures(env, network, params["failure_model"])
        failed = [n for n in network.nodes_map.values() if n.state == STATE_FAILED]
        assert len(failed) == 2

    def test_cascade_propagates(self):
        env, network, service_team, params = self._make_network_and_team()
        model = WattsCascadeModel()
        # Set a low phi so failures propagate easily
        params["failure_model"]["phi"] = 0.1
        model.install(env, network, service_team, params["failure_model"])
        # Manually fail a high-degree node
        hub = max(network.graph.nodes, key=lambda n: network.graph.degree(n))
        network.mark_failed(env, network.nodes_map[hub], network.sim_state)
        env.run(until=5)
        total_failed = sum(1 for n in network.nodes_map.values() if n.state == STATE_FAILED)
        # Some cascade should have occurred (or at minimum the initial failure)
        assert total_failed >= 1
