"""Agent models for the tiedloc simulator.

This module defines the core agent types used in the simulation:
- Node: Represents a network node subject to cascading failures (Watts model).
- Link: Represents a network edge with failure propagation.
- Server: Represents a repair agent that can travel to and restore failed elements.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import simpy

from tiedloc.constants import STATE_BUSY, STATE_FAILED, STATE_IDLE, STATE_OPERATIONAL

if TYPE_CHECKING:
    from tiedloc.agent_behaviors import AgentBehavior
    from tiedloc.networks import CPSNetwork, ServiceTeam
    from tiedloc.repair_models import RepairModel
    from tiedloc.responsestrategies import JobRequest


class Node:
    """A network node that can fail according to the Watts cascade model.

    Attributes:
        id: Unique identifier for this node.
        state: 0 = operational, 1 = failed.
        hosting: List of server agent IDs currently located at this node.
        failed_time: Simulation time at which the node failed.
        activity: Cached activity priority for scheduling.
        nearness: Cached nearness priority for scheduling.
    """

    def __init__(self, node_id: int) -> None:
        self.id: int = node_id
        self.state: int = STATE_OPERATIONAL
        self.hosting: list[int] = []
        self.failed_time: float = 0.0
        self.activity: dict[str, float] | None = None
        self.nearness: dict[str, float] | None = None

    def node_fail_watts(
        self,
        env: simpy.Environment,
        phi: float,
        tstep: float,
        network: CPSNetwork,
        service_team: ServiceTeam,
    ) -> Generator[simpy.Event, None, None]:
        """SimPy process: periodically check and propagate Watts cascade failure.

        Args:
            env: The SimPy simulation environment.
            phi: Failure threshold for the Watts cascade model.
            tstep: Time step between failure checks.
            network: The CPS network instance (uses composition interface).
            service_team: The service team graph for preventability checks.
        """
        graph = network.graph
        sim_state = network.sim_state

        while True:
            if self.state == STATE_OPERATIONAL:
                failed_edge = 0
                aux_adjuster = 0

                for neighbor in graph.neighbors(self.id):
                    link = network.get_link(self.id, neighbor)
                    if link is None:
                        continue

                    if link.state == STATE_FAILED:
                        if hasattr(link, "aux_just_in") and link.aux_just_in:
                            aux_adjuster += 1
                            continue
                        failed_edge += 1

                        # Check preventability
                        break_flag = False
                        for ai in self.hosting:
                            for a_nei in service_team.graph.neighbors(ai):
                                server = service_team.get_server(a_nei)
                                if server.current_pos == neighbor:
                                    failed_edge -= 1
                                    break_flag = True
                                    break
                            if break_flag:
                                break

                degree = graph.degree(self.id)
                effective_degree = degree - aux_adjuster
                if effective_degree == 0:
                    phyi = 0.0
                else:
                    phyi = failed_edge / effective_degree

                if phyi > phi:
                    network.mark_failed(env, self, sim_state)

            yield env.timeout(tstep)


class Link:
    """A network edge that can fail during cascading failures.

    Attributes:
        vertex1: ID of the first endpoint node.
        vertex2: ID of the second endpoint node.
        state: 0 = operational, 1 = failed.
        failed_time: Simulation time at which the link failed.
        aux_just_in: Flag indicating this is a newly added auxiliary link.
        activity: Cached activity priority for scheduling.
        nearness: Cached nearness priority for scheduling.
    """

    def __init__(self, vertex1: int, vertex2: int) -> None:
        self.vertex1: int = vertex1
        self.vertex2: int = vertex2
        self.state: int = STATE_OPERATIONAL
        self.failed_time: float = 0.0
        self.aux_just_in: bool = False
        self.activity: dict[str, float] | None = None
        self.nearness: dict[str, float] | None = None

    def edge_fail_watts(
        self,
        env: simpy.Environment,
        tstep: float,
        network: CPSNetwork,
        service_team: ServiceTeam,
    ) -> Generator[simpy.Event, None, None]:
        """SimPy process: periodically check and propagate edge failure.

        Args:
            env: The SimPy simulation environment.
            tstep: Time step between failure checks.
            network: The CPS network instance.
            service_team: The service team graph for preventability checks.
        """
        graph = network.graph
        sim_state = network.sim_state
        node1 = network.get_node(self.vertex1)
        node2 = network.get_node(self.vertex2)

        while True:
            if self.state == STATE_OPERATIONAL:
                if node1.state == STATE_FAILED and node2.state == STATE_OPERATIONAL:
                    if not node1.hosting:
                        self.state = STATE_FAILED
                    else:
                        sim_state.prevented += 1

                elif node2.state == STATE_FAILED and node1.state == STATE_OPERATIONAL:
                    if not node2.hosting:
                        self.state = STATE_FAILED
                    else:
                        sim_state.prevented += 1

                elif node1.state == STATE_FAILED and node2.state == STATE_FAILED:
                    self.state = STATE_FAILED
                    # Check preventability
                    break_flag = False
                    for ai in node1.hosting:
                        for a_nei in service_team.graph.neighbors(ai):
                            server = service_team.get_server(a_nei)
                            if server.current_pos == self.vertex2:
                                self.state = STATE_OPERATIONAL
                                sim_state.prevented += 1
                                break_flag = True
                                break
                        if break_flag:
                            break

                if self.state == STATE_FAILED:
                    network.mark_failed(env, self, sim_state)

            yield env.timeout(tstep)


class Server:
    """A repair agent that can travel to failed elements and restore them.

    Attributes:
        id: Unique identifier for this server.
        agent_travel_speed: Speed at which the agent moves through the network.
        repairing_time: Time required to repair a failed element.
        waiting_list: Queue of pending repair jobs.
        state: 0 = idle, 1 = busy.
        current_pos: ID of the node where the agent is currently located.
        resource: SimPy Resource for managing job concurrency.
        freed_agent_event: SimPy Event triggered when the agent becomes free.
    """

    def __init__(
        self,
        server_id: int,
        env: simpy.Environment,
        travel_speed: float,
        repair_time: float,
        repair_model: RepairModel | None = None,
        behavior: AgentBehavior | None = None,
        capacity: int = 1,
    ) -> None:
        from tiedloc.agent_behaviors import DefaultAgentBehavior
        from tiedloc.repair_models import FixedRepairModel

        self.id: int = server_id
        self.agent_travel_speed: float = travel_speed
        self.repairing_time: float = repair_time
        self.repair_model: RepairModel = repair_model or FixedRepairModel()
        self.behavior: AgentBehavior = behavior or DefaultAgentBehavior()
        self.state: int = STATE_IDLE
        self.current_pos: int = -1
        self.resource: simpy.Resource = simpy.Resource(env, capacity=capacity)
        self.freed_agent_event: simpy.Event = env.event()
        self._active_jobs: dict[simpy.resources.resource.Request, dict] = {}

    def allocate_to(self, cps_node: Node) -> None:
        """Assign this server to a CPS network node.

        Args:
            cps_node: The network node to allocate this server to.
        """
        self.current_pos = cps_node.id
        cps_node.hosting.append(self.id)

    def bidding(
        self,
        env: simpy.Environment,
        network: CPSNetwork,
        target: int,
        agent_schedule: dict[int, list[dict]] | None = None,
    ) -> tuple[float, float, float]:
        """Calculate the cost for this agent to service a target node.

        Args:
            env: The SimPy simulation environment.
            network: The CPS network instance (for distance matrix).
            target: The ID of the target node to service.
            agent_schedule: Optional schedule dict for look-ahead.

        Returns:
            A tuple of (total_cost, departing_time, arrival_time).
        """
        ag_from = self.current_pos
        ag_departing_time = env.now

        if agent_schedule is not None and self.id in agent_schedule:
            schedule = agent_schedule[self.id]
            if schedule:
                ag_from = schedule[-1]["node"]
                ag_departing_time = max(ag_departing_time, schedule[-1]["finishing_time"])

        path_len = network.distmat[ag_from][target]
        cost = path_len / self.agent_travel_speed
        cost += self.repairing_time
        cost += ag_departing_time
        ag_arrival_time = cost - self.repairing_time
        return (cost, ag_departing_time, ag_arrival_time)

    def provide_service(
        self, env: simpy.Environment, network: CPSNetwork, job_req: JobRequest
    ) -> Generator[simpy.Event, None, None]:
        """SimPy process: travel to and repair a failed element.

        Delegates to self.behavior.execute() for the actual job handling.

        Args:
            env: The SimPy simulation environment.
            network: The CPS network instance.
            job_req: A JobRequest wrapping the SimPy resource request and job parameters.
        """
        yield from self.behavior.execute(env, self, network, job_req)
