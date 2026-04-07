"""Response strategy dispatch for the tiedloc simulator.

This module implements the centralized scheduling and dispatch logic that
assigns repair agents to failed network elements. It supports multiple
response protocols (FCFS, nearest, activity) using a Strategy Pattern
instead of fragile dynamic dispatch.
"""

from __future__ import annotations

from collections.abc import Generator

import networkx as nx
import simpy

from tiedloc.agents import Link, Node, Server
from tiedloc.constants import STATE_FAILED, STATE_OPERATIONAL
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState
from tiedloc.strategies import get_strategy


class JobRequest:
    """Wraps a SimPy resource request with associated job parameters.

    This avoids monkey-patching ``params`` onto the SimPy ``Request`` object.
    """

    __slots__ = ("request", "params")

    def __init__(self, request: simpy.resources.resource.Request, params: dict) -> None:
        self.request = request
        self.params = params


def fcfs_key(item: Node | Link) -> float:
    """Sort failures by their failed time (first-come, first-served)."""
    return item.failed_time


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------

def init(
    env: simpy.Environment,
    parameters: dict,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> Generator[simpy.Event, None, None]:
    """Main response strategy event loop (SimPy process).

    Waits for freed-agent or new-failure events, then dispatches repair
    agents according to the configured response protocol.

    Args:
        env: The SimPy simulation environment.
        parameters: Simulation parameters dictionary.
        network: The CPS network instance.
        service_team: The service team instance.
    """
    sim_state = network.sim_state
    sim_state.new_failures_event = env.event()
    env.process(maintenance_check(env, parameters, network, service_team))

    while True:
        if service_team.last_dispatch > -1:
            events = [
                service_team.get_server(a).freed_agent_event
                for a in service_team.graph.nodes
            ]
            events.append(sim_state.new_failures_event)
            yield env.any_of(events)

        if service_team.last_dispatch >= env.now:
            continue
        service_team.last_dispatch = env.now

        # Optional auxiliary edge injection
        if parameters["response_protocol"].get("auxiliary", False):
            aux_trigger_time = float(
                parameters["response_protocol"].get("aux_trigger_time", 5)
            )
            if env.now == aux_trigger_time:
                wire_aux(env, parameters, network, service_team)

        response_protocol = parameters["response_protocol"]["name"]
        strategy = get_strategy(response_protocol)
        strategy.dispatch(env, network, service_team)

        implement_schedule(service_team, network, env)


def maintenance_check(
    env: simpy.Environment,
    parameters: dict,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> Generator[simpy.Event, None, None]:
    """Periodic check for new failures and available agents (SimPy process).

    Args:
        env: The SimPy simulation environment.
        parameters: Simulation parameters dictionary.
        network: The CPS network instance.
        service_team: The service team instance.
    """
    sim_state = network.sim_state
    check_base = sim_state.failures_generation
    frequency = float(parameters["response_protocol"]["frequency"])

    while True:
        checker = sim_state.failures_generation
        if checker != check_base:
            sim_state.new_failures_event.succeed()
            sim_state.new_failures_event = env.event()
        check_base = checker
        yield env.timeout(frequency)


# ---------------------------------------------------------------------------
# Priority calculation helpers
# ---------------------------------------------------------------------------

def calc_activity(
    f1: Node | Link,
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> float:
    """Calculate the activity priority for a failed element.

    Args:
        f1: The failed Node or Link.
        env: The SimPy simulation environment.
        network: The CPS network instance.
        service_team: The service team instance.

    Returns:
        A float representing the activity priority (lower = more urgent).
    """
    if f1.activity is not None and f1.activity["time"] == env.now:
        return f1.activity["value"]

    if isinstance(f1, Link):
        val = min(
            calc_activity(network.get_node(f1.vertex1), env, network, service_team),
            calc_activity(network.get_node(f1.vertex2), env, network, service_team),
        )
        f1.activity = {"time": env.now, "value": val}
        return val

    graph = network.graph
    node_id = f1.id
    failed_edge = 0
    aux_adjuster = 0

    for neighbor in graph.neighbors(node_id):
        link = network.get_link(node_id, neighbor)
        if link is None:
            continue

        if link.state == STATE_FAILED:
            if hasattr(link, "aux_just_in") and link.aux_just_in:
                aux_adjuster += 1
                continue
            failed_edge += 1
            break_flag = False
            for ai in f1.hosting:
                for a_nei in service_team.graph.neighbors(ai):
                    server = service_team.get_server(a_nei)
                    if server.current_pos == neighbor:
                        failed_edge -= 1
                        break_flag = True
                        break
                if break_flag:
                    break

    degree = graph.degree(node_id)
    effective = degree - aux_adjuster
    if effective == 0:
        phyi = 0.0
    else:
        phyi = failed_edge / effective

    prio = 1 - phyi
    f1.activity = {"time": env.now, "value": prio}
    return prio


def calc_nearness(
    f1: Node | Link,
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> None:
    """Calculate the nearness priority for a failed element.

    Args:
        f1: The failed Node or Link.
        env: The SimPy simulation environment.
        network: The CPS network instance.
        service_team: The service team instance.
    """
    if f1.nearness is not None and f1.nearness["time"] == env.now:
        return

    sim_state = network.sim_state
    if isinstance(f1, Link):
        _, job = find_costless_agent(env, network, service_team, f1, f1.vertex1, sim_state)
        _, job2 = find_costless_agent(env, network, service_team, f1, f1.vertex2, sim_state)
        close_to_finish = max(job["finishing_time"], job2["finishing_time"])
    else:
        _, job = find_costless_agent(env, network, service_team, f1, f1.id, sim_state)
        close_to_finish = job["finishing_time"]

    f1.nearness = {"time": env.now, "value": close_to_finish}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def alive_links(network: CPSNetwork, node_id: int) -> int:
    """Count the number of operational links connected to a node."""
    count = 0
    for neighbor in network.graph.neighbors(node_id):
        link = network.get_link(node_id, neighbor)
        if link is not None:
            count += 1 - link.state
    return count


def protection_density(network: CPSNetwork, node_id: int) -> int:
    """Calculate the protection density around a node."""
    protected = 0
    for neighbor in network.graph.neighbors(node_id):
        node = network.get_node(neighbor)
        if node.hosting:
            protected += 1
            for nei2 in network.graph.neighbors(neighbor):
                if nei2 != node_id:
                    nei2_node = network.get_node(nei2)
                    if nei2_node.hosting:
                        protected += 1
    return protected


def wire_aux(
    env: simpy.Environment,
    parameters: dict,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> None:
    """Inject auxiliary edges to improve network resilience.

    Args:
        env: The SimPy simulation environment.
        parameters: Simulation parameters dictionary.
        network: The CPS network instance.
        service_team: The service team instance.
    """
    sim_state = network.sim_state
    graph = network.graph
    fail_speed = float(parameters["failure_model"]["fail_speed"])

    for fdummy in list(sim_state.failures):
        if not isinstance(fdummy, Node):
            continue
        target = fdummy
        if network.get_node(target.id).hosting:
            continue

        for neighbor in list(graph.neighbors(target.id)):
            break_flag = False
            neighbor_node = network.get_node(neighbor)
            if neighbor_node.state == STATE_FAILED:
                continue
            if graph.degree(neighbor) >= network.average_degree:
                continue
            if not neighbor_node.hosting:
                for nei2 in list(graph.neighbors(neighbor)):
                    if target.id == nei2:
                        continue
                    nei2_node = network.get_node(nei2)
                    if nei2_node.hosting:
                        continue
                    if nei2_node.state == STATE_FAILED:
                        continue
                    if not graph.has_edge(target.id, nei2):
                        q = 0.0
                        for nei_one in graph.neighbors(target.id):
                            if graph.has_edge(nei_one, nei2):
                                q += 1.0
                        if graph.degree(nei2) > (network.average_degree * q):
                            continue
                        if q > network.aux_threshold[target.id]:
                            graph.add_edge(target.id, nei2)
                            new_link = Link(target.id, nei2)
                            network.links_map[(target.id, nei2)] = new_link
                            env.process(
                                new_link.edge_fail_watts(env, fail_speed, network, service_team)
                            )
                            network.mark_failed(env, new_link, sim_state)
                            new_link.aux_just_in = True
                            sim_state.aux_lines += 1
                            sim_state.aux_failures += 1
                            sim_state.total_failures -= 1
                            break_flag = True
                            break
            if break_flag:
                break


# ---------------------------------------------------------------------------
# Centralized scheduling — helpers
# ---------------------------------------------------------------------------

def _init_agent_schedule(
    service_team: ServiceTeam,
    sim_state: SimulationState,
) -> None:
    """Initialize agent_schedule from current server resource queues.

    Populates sim_state.agent_schedule if it is None.
    """
    if sim_state.agent_schedule is not None:
        return
    sim_state.agent_schedule = {}
    for nid in service_team.graph.nodes:
        server = service_team.get_server(nid)
        sim_state.agent_schedule[nid] = [
            server._active_jobs[req]
            for req in list(server.resource.users) + list(server.resource.queue)
            if req in server._active_jobs
        ]


def _assign_link_collaborators(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
    f1: Link,
    primary_agent: int,
    primary_job: dict,
    sim_state: SimulationState,
) -> tuple[int, dict]:
    """Find and assign a collaborator agent for the other end of a failed Link.

    Synchronizes finishing_time between both jobs and sets collaborator fields.

    Returns:
        A tuple of (collaborator_agent_id, collaborator_job).
    """
    v2 = f1.vertex2 if primary_job["node"] == f1.vertex1 else f1.vertex1
    v2_agent, job2 = find_costless_agent(
        env, network, service_team, f1, v2, sim_state, collaborator=primary_agent
    )

    # Synchronize finishing times to the slower of the two
    if primary_job["finishing_time"] < job2["finishing_time"]:
        primary_job["finishing_time"] = job2["finishing_time"]
    else:
        job2["finishing_time"] = primary_job["finishing_time"]

    primary_job["collaborator"] = v2_agent
    job2["collaborator"] = primary_agent
    sim_state.agent_schedule[v2_agent].append(job2)

    return v2_agent, job2


def _filter_schedule(
    service_team: ServiceTeam,
    sim_state: SimulationState,
) -> None:
    """Remove non-assigned jobs from agent_schedule."""
    for agent_id in service_team.graph.nodes:
        if agent_id in sim_state.agent_schedule:
            sim_state.agent_schedule[agent_id] = [
                job for job in sim_state.agent_schedule[agent_id] if job["assign"]
            ]


# ---------------------------------------------------------------------------
# Centralized scheduling
# ---------------------------------------------------------------------------

def dispatcher(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
) -> None:
    """Centralized FCFS dispatcher that assigns agents to failures.

    Args:
        env: The SimPy simulation environment.
        network: The CPS network instance.
        service_team: The service team instance.
    """
    sim_state = network.sim_state
    _init_agent_schedule(service_team, sim_state)

    re_assign: list[Link] = []
    next_iteration: list[Node | Link] = []

    while sim_state.failures:
        f1 = sim_state.failures.pop(0)
        v1 = f1.vertex1 if isinstance(f1, Link) else f1.id

        least_agent, job = find_costless_agent(env, network, service_team, f1, v1, sim_state)
        sim_state.agent_schedule[least_agent].append(job)

        if isinstance(f1, Link):
            _, job2 = _assign_link_collaborators(
                env, network, service_team, f1, least_agent, job, sim_state
            )
            if job["assign"] != job2["assign"]:
                re_assign.append(f1)
                job["assign"] = False
                job2["assign"] = False
            elif not job["assign"]:
                next_iteration.append(f1)
        elif not job["assign"]:
            next_iteration.append(f1)

    _filter_schedule(service_team, sim_state)

    # Re-assign collaboration for links where one side wasn't ready
    while re_assign:
        f1 = re_assign.pop(0)
        v1 = f1.vertex1
        least_agent, job = find_costless_agent(env, network, service_team, f1, v1, sim_state)
        sim_state.agent_schedule[least_agent].append(job)
        _, job2 = _assign_link_collaborators(
            env, network, service_team, f1, least_agent, job, sim_state
        )
        job["assign"] = True
        job2["assign"] = True

    sim_state.failures = next_iteration


def find_costless_agent(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
    f1: Node | Link,
    target: int,
    sim_state: SimulationState,
    collaborator: int | None = None,
) -> tuple[int, dict]:
    """Find the agent with the lowest cost to service a target node.

    Args:
        env: The SimPy simulation environment.
        network: The CPS network instance.
        service_team: The service team instance.
        f1: The failed element.
        target: The target node ID.
        sim_state: The simulation state.
        collaborator: Optional agent ID to restrict search to neighbors.

    Returns:
        A tuple of (agent_id, job_dict).
    """
    least_cost = -1.0
    least_agent = -1
    departing_time = -1.0
    arrival_time = -1.0

    if collaborator is not None:
        eligible = service_team.graph.neighbors(collaborator)
    else:
        eligible = service_team.graph.nodes

    schedule_dict = None
    if sim_state.agent_schedule is not None:
        schedule_dict = sim_state.agent_schedule

    for agent_id in eligible:
        server = service_team.get_server(agent_id)
        cost, ag_dep, ag_arr = server.bidding(env, network, target, schedule_dict)
        if cost < least_cost or least_cost < 0:
            least_cost = cost
            least_agent = agent_id
            departing_time = ag_dep
            arrival_time = ag_arr

    job = {
        "node": target,
        "failure": f1,
        "departing_time": departing_time,
        "arrival_time": arrival_time,
        "finishing_time": least_cost,
        "process_assigned": False,
        "assign": (departing_time == env.now),
    }
    return (least_agent, job)


def _find_idle_agents(
    env: simpy.Environment,
    service_team: ServiceTeam,
    sim_state: SimulationState,
) -> list[dict]:
    """Return a list of idle agent descriptors (agent_id, pos).

    An agent is idle if it has no scheduled work or its last scheduled job
    finishes at or before the current time.
    """
    idle_agents: list[dict] = []
    for agent_id in service_team.graph.nodes:
        server = service_team.get_server(agent_id)
        if sim_state.agent_schedule is not None and agent_id in sim_state.agent_schedule:
            schedule = sim_state.agent_schedule[agent_id]
            if schedule and env.now < schedule[-1]["finishing_time"]:
                continue
        idle_agents.append({"agent_id": agent_id, "pos": server.current_pos})
    return idle_agents


def _build_idle_matrix(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
    idle_agents: list[dict],
    failures: list[Node | Link],
) -> list[tuple]:
    """Build a distance/activity matrix between idle agents and failures.

    Each entry is (activity, distance, agent_id, failure).
    """
    idle_mat: list[tuple] = []
    for f1 in failures:
        act = calc_activity(f1, env, network, service_team)
        for idler in idle_agents:
            if isinstance(f1, Link):
                dist = min(
                    network.distmat[idler["pos"]][f1.vertex1],
                    network.distmat[idler["pos"]][f1.vertex2],
                )
            else:
                dist = network.distmat[idler["pos"]][f1.id]
            idle_mat.append((act, dist, idler["agent_id"], f1))
    return idle_mat


def _nn_assign_link(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
    sim_state: SimulationState,
    agent_id: int,
    f1: Link,
    job: dict,
    v2: int,
) -> tuple[bool, int]:
    """Handle link assignment in nn_schedule: find a neighbor to service v2.

    Returns:
        (success, v2_agent_id) — success is True if a collaborator was found.
    """
    least_cost = -1.0
    v2_agent = -1
    dep_time = -1.0
    arr_time = -1.0

    for ag_id in service_team.graph.neighbors(agent_id):
        s = service_team.get_server(ag_id)
        c, ad, aa = s.bidding(env, network, v2, sim_state.agent_schedule)
        if ad > env.now:
            continue
        if c < least_cost or least_cost < 0:
            least_cost = c
            v2_agent = ag_id
            dep_time = ad
            arr_time = aa

    if least_cost == -1:
        return False, -1

    job2 = {
        "node": v2,
        "failure": f1,
        "departing_time": dep_time,
        "arrival_time": arr_time,
        "finishing_time": least_cost,
        "process_assigned": False,
        "assign": (dep_time == env.now),
    }

    # Synchronize finishing times
    if job["finishing_time"] < job2["finishing_time"]:
        job["finishing_time"] = job2["finishing_time"]
    else:
        job2["finishing_time"] = job["finishing_time"]

    job["collaborator"] = v2_agent
    job2["collaborator"] = agent_id
    sim_state.agent_schedule[v2_agent].append(job2)
    job["assign"] = True
    job2["assign"] = True

    return True, v2_agent


def nn_schedule(
    env: simpy.Environment,
    network: CPSNetwork,
    service_team: ServiceTeam,
    strategy,
) -> None:
    """Nearest-neighbor and activity-based scheduling.

    Args:
        env: The SimPy simulation environment.
        network: The CPS network instance.
        service_team: The service team instance.
        strategy: A ResponseStrategy instance with a sort_key attribute.
    """
    sim_state = network.sim_state

    idle_agents = _find_idle_agents(env, service_team, sim_state)
    idle_mat = _build_idle_matrix(env, network, service_team, idle_agents, sim_state.failures)

    key_func = strategy.sort_key if strategy.sort_key is not None else lambda item: (item[1],)
    idle_mat.sort(key=key_func)

    _init_agent_schedule(service_team, sim_state)

    while idle_mat:
        act, dist, agent_id, f1 = idle_mat.pop(0)
        if f1 not in sim_state.failures:
            continue
        sim_state.failures.remove(f1)

        if isinstance(f1, Link):
            cur_dist = network.distmat[service_team.get_server(agent_id).current_pos][f1.vertex1]
            if cur_dist <= dist:
                v1, v2 = f1.vertex1, f1.vertex2
            else:
                v1, v2 = f1.vertex2, f1.vertex1
        else:
            v1 = f1.id
            v2 = -1

        server = service_team.get_server(agent_id)
        cost, ag_dep, ag_arr = server.bidding(env, network, v1, sim_state.agent_schedule)

        job = {
            "node": v1,
            "failure": f1,
            "departing_time": ag_dep,
            "arrival_time": ag_arr,
            "finishing_time": cost,
            "process_assigned": False,
            "assign": (ag_dep == env.now),
        }
        sim_state.agent_schedule[agent_id].append(job)

        if isinstance(f1, Link):
            success, v2_agent = _nn_assign_link(
                env, network, service_team, sim_state, agent_id, f1, job, v2
            )
            if success:
                idle_mat = [
                    (a, d, aid, fx)
                    for (a, d, aid, fx) in idle_mat
                    if aid != agent_id and aid != v2_agent and fx != f1
                ]
            else:
                job["assign"] = False
                sim_state.agent_schedule[agent_id].pop()
                sim_state.failures.append(f1)
        else:
            job["assign"] = True
            idle_mat = [
                (a, d, aid, fx)
                for (a, d, aid, fx) in idle_mat
                if aid != agent_id and fx != f1
            ]

    _filter_schedule(service_team, sim_state)


def implement_schedule(
    service_team: ServiceTeam,
    network: CPSNetwork,
    env: simpy.Environment,
) -> None:
    """Assign SimPy processes to all scheduled and unprocessed jobs.

    Args:
        service_team: The service team instance.
        network: The CPS network instance.
        env: The SimPy simulation environment.
    """
    sim_state = network.sim_state
    if sim_state.agent_schedule is None:
        return

    for agent_id in service_team.graph.nodes:
        if agent_id not in sim_state.agent_schedule:
            continue
        server = service_team.get_server(agent_id)
        for job in sim_state.agent_schedule[agent_id]:
            if job["process_assigned"]:
                continue
            if job["assign"]:
                job["process_assigned"] = True
                req = server.resource.request()
                job_req = JobRequest(req, job)
                env.process(server.provide_service(env, network, job_req))
