"""Tests for tiedloc.agent_behaviors — DefaultAgentBehavior in SimPy simulations."""

from __future__ import annotations

import random

import pytest
import simpy

from tiedloc.agent_behaviors import AgentBehavior, DefaultAgentBehavior
from tiedloc.agents import Link, Node, Server
from tiedloc.constants import STATE_BUSY, STATE_FAILED, STATE_IDLE, STATE_OPERATIONAL
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState
from tiedloc.responsestrategies import JobRequest


def _make_test_env():
    """Build a minimal simulation environment for agent behavior tests."""
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
        "simulation_param": {"seed": "42", "replications": "1", "processors": "1", "simulation_length": "50"},
    }
    env = simpy.Environment()
    sim_state = SimulationState()
    sim_state.rng = random.Random(42)
    network = CPSNetwork(params, sim_state=sim_state)
    service_team = ServiceTeam(params, env)
    service_team.allocation(params, network)
    return env, network, service_team


class TestDefaultAgentBehaviorNodeRepair:
    """Test DefaultAgentBehavior.execute() for node repair."""

    def test_node_repair_restores_state(self):
        env, network, service_team = _make_test_env()
        sim_state = network.sim_state

        # Fail a node
        node = network.nodes_map[0]
        network.mark_failed(env, node, sim_state)
        assert node.state == STATE_FAILED

        # Pick an idle server
        server = service_team.get_server(0)
        # Ensure server is at a known position
        if server.current_pos == -1:
            server.current_pos = 0
            node_at = network.get_node(0)
            if server.id not in node_at.hosting:
                node_at.hosting.append(server.id)

        # Build job params
        target = node.id
        cost, dep_time, arr_time = server.bidding(env, network, target)
        job_params = {
            "node": target,
            "failure": node,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        req = server.resource.request()
        job_req = JobRequest(req, job_params)
        env.process(server.provide_service(env, network, job_req))
        env.run()

        assert node.state == STATE_OPERATIONAL
        assert server.state == STATE_IDLE

    def test_server_becomes_busy_during_repair(self):
        env, network, service_team = _make_test_env()
        sim_state = network.sim_state

        node = network.nodes_map[1]
        network.mark_failed(env, node, sim_state)

        server = service_team.get_server(0)
        if server.current_pos == -1:
            server.current_pos = 0
            network.get_node(0).hosting.append(server.id)

        target = node.id
        cost, dep_time, arr_time = server.bidding(env, network, target)
        job_params = {
            "node": target,
            "failure": node,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        busy_observed = []

        def observer():
            yield env.timeout(0.5)
            busy_observed.append(server.state)

        req = server.resource.request()
        job_req = JobRequest(req, job_params)
        env.process(server.provide_service(env, network, job_req))
        env.process(observer())
        env.run()

        # Server should have been busy at t=0.5
        assert busy_observed[0] == STATE_BUSY


class TestDefaultAgentBehaviorLinkRepair:
    """Test DefaultAgentBehavior.execute() for link repair."""

    def test_link_at_vertex1_skips_restore(self):
        env, network, service_team = _make_test_env()
        sim_state = network.sim_state

        # Find a link
        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)
        network.mark_failed(env, link, sim_state)
        assert link.state == STATE_FAILED

        server = service_team.get_server(0)
        server.current_pos = v1
        target_node = network.get_node(v1)
        if server.id not in target_node.hosting:
            target_node.hosting.append(server.id)

        # Job targeting vertex1 of the link — should skip restore for links
        cost, dep_time, arr_time = server.bidding(env, network, v1)
        job_params = {
            "node": v1,
            "failure": link,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        req = server.resource.request()
        job_req = JobRequest(req, job_params)
        env.process(server.provide_service(env, network, job_req))
        env.run()

        # Link at vertex1 should NOT be restored (only vertex2 side restores)
        assert link.state == STATE_FAILED

    def test_link_at_vertex2_restores(self):
        env, network, service_team = _make_test_env()
        sim_state = network.sim_state

        edges = list(network.graph.edges)
        v1, v2 = edges[0]
        link = network.get_link(v1, v2)
        network.mark_failed(env, link, sim_state)

        server = service_team.get_server(0)
        # Position at v2 so the repair targets vertex2 side
        server.current_pos = v2
        target_node = network.get_node(v2)
        if server.id not in target_node.hosting:
            target_node.hosting.append(server.id)

        cost, dep_time, arr_time = server.bidding(env, network, v2)
        job_params = {
            "node": v2,
            "failure": link,
            "departing_time": dep_time,
            "arrival_time": arr_time,
            "finishing_time": cost,
            "process_assigned": True,
            "assign": True,
        }

        req = server.resource.request()
        job_req = JobRequest(req, job_params)
        env.process(server.provide_service(env, network, job_req))
        env.run()

        # Link should be restored when serviced from vertex2 side
        assert link.state == STATE_OPERATIONAL


class TestAgentBehaviorABC:
    def test_default_is_subclass(self):
        assert issubclass(DefaultAgentBehavior, AgentBehavior)

    def test_default_instance(self):
        behavior = DefaultAgentBehavior()
        assert isinstance(behavior, AgentBehavior)
