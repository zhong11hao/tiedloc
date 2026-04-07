"""Tests for tiedloc.repair_models — repair model registry and FixedRepairModel."""

from __future__ import annotations

import pytest
import simpy

from tiedloc.agents import Node, Server
from tiedloc.constants import STATE_FAILED, STATE_OPERATIONAL
from tiedloc.networks import CPSNetwork, SimulationState
from tiedloc.repair_models import (
    FixedRepairModel,
    RepairModel,
    get_repair_model,
    register_repair_model,
)


class TestRepairModelRegistry:
    def test_fixed_registered(self):
        model = get_repair_model("fixed")
        assert isinstance(model, FixedRepairModel)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="nonexistent_repair"):
            get_repair_model("nonexistent_repair")

    def test_register_custom(self):
        class CustomRepair(RepairModel):
            def estimate_repair_time(self, server, element, params):
                return 5.0

            def execute_repair(self, env, server, network, element):
                yield env.timeout(5.0)

        register_repair_model("test_custom_repair", CustomRepair)
        model = get_repair_model("test_custom_repair")
        assert isinstance(model, CustomRepair)


class TestFixedRepairModel:
    def test_estimate_repair_time(self):
        env = simpy.Environment()
        model = FixedRepairModel()
        server = Server(0, env, travel_speed=1.0, repair_time=3.5)
        node = Node(0)
        est = model.estimate_repair_time(server, node, {})
        assert est == 3.5

    def test_execute_repair_restores_element(self):
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "_topo_seed": 42,
            "failure_model": {"name": "Watts cascade", "phi": 0.3, "fail_speed": 1.0, "initial_failures": 0},
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
        network = CPSNetwork(params, sim_state=sim_state)

        model = FixedRepairModel()
        node = network.nodes_map[0]
        network.mark_failed(env, node, sim_state)
        assert node.state == STATE_FAILED

        server = Server(0, env, travel_speed=1.0, repair_time=2.0)

        def repair_process():
            yield from model.execute_repair(env, server, network, node)

        env.process(repair_process())
        env.run()
        assert node.state == STATE_OPERATIONAL

    def test_execute_repair_takes_correct_time(self):
        params = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "_topo_seed": 42,
            "failure_model": {"name": "Watts cascade", "phi": 0.3, "fail_speed": 1.0, "initial_failures": 0},
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
        network = CPSNetwork(params, sim_state=sim_state)
        node = network.nodes_map[0]
        network.mark_failed(env, node, sim_state)

        server = Server(0, env, travel_speed=1.0, repair_time=3.0)
        model = FixedRepairModel()

        def repair_process():
            yield from model.execute_repair(env, server, network, node)

        env.process(repair_process())
        env.run()
        assert env.now == pytest.approx(3.0)
