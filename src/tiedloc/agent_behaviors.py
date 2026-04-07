"""Pluggable agent behavior models for the tiedloc simulator.

This module defines the AgentBehavior ABC for customizing how repair agents
handle job requests (travel, repair, resource management).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import simpy

from tiedloc.constants import STATE_BUSY, STATE_IDLE

if TYPE_CHECKING:
    from tiedloc.agents import Server
    from tiedloc.networks import CPSNetwork
    from tiedloc.responsestrategies import JobRequest


class AgentBehavior(ABC):
    """Defines how a repair agent handles a job request."""

    @abstractmethod
    def execute(
        self,
        env: simpy.Environment,
        server: Server,
        network: CPSNetwork,
        job_req: JobRequest,
    ) -> Generator[simpy.Event, None, None]:
        """SimPy process: handle a single job from dispatch to completion.

        This replaces Server.provide_service(). The implementation
        controls movement, repair, and resource management.
        """
        ...


class DefaultAgentBehavior(AgentBehavior):
    """Default behavior matching the original provide_service logic."""

    def execute(self, env, server, network, job_req):
        from tiedloc.agents import Link

        sim_state = network.sim_state
        req = job_req.request
        params = job_req.params
        server._active_jobs[req] = params
        yield req
        server.state = STATE_BUSY

        node_at_current = network.get_node(server.current_pos)
        if env.now < params["arrival_time"]:
            if server.id in node_at_current.hosting:
                node_at_current.hosting.remove(server.id)
            server.current_pos = -1
            move_time = params["arrival_time"] - env.now
            yield env.timeout(move_time)
            sim_state.total_distance += move_time * server.agent_travel_speed

        server.current_pos = params["node"]
        target_node = network.get_node(server.current_pos)
        if server.id not in target_node.hosting:
            target_node.hosting.append(server.id)

        # Repair phase: delegate to repair model or handle link special case
        if isinstance(params["failure"], Link) and server.current_pos == params["failure"].vertex1:
            # Link vertex1 side: wait for collaborator, don't restore
            if env.now < params["finishing_time"]:
                yield env.timeout(params["finishing_time"] - env.now)
            if hasattr(params["failure"], "aux_just_in"):
                params["failure"].aux_just_in = False
        else:
            # Delegate actual repair to the server's repair model
            yield from server.repair_model.execute_repair(env, server, network, params["failure"])

        server.state = STATE_IDLE
        server.resource.release(req)
        server._active_jobs.pop(req, None)

        if not server.resource.users and not server.resource.queue:
            server.freed_agent_event.succeed()
            server.freed_agent_event = env.event()
