"""Pluggable failure cascade models for the tiedloc simulator.

This module defines the FailureModel ABC and a registry of built-in models.
The default WattsCascadeModel implements the Watts threshold cascade, where
a node fails when the fraction of its failed neighbor edges exceeds phi.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import simpy

if TYPE_CHECKING:
    from tiedloc.networks import CPSNetwork, ServiceTeam


class FailureModel(ABC):
    """Abstract base class for cascade failure propagation models."""

    @abstractmethod
    def install(
        self,
        env: simpy.Environment,
        network: CPSNetwork,
        service_team: ServiceTeam,
        params: dict,
    ) -> None:
        """Register SimPy processes for failure propagation.

        Called once after initial failures are seeded.

        Args:
            env: SimPy environment.
            network: The CPS network.
            service_team: The service team.
            params: The "failure_model" section of the input config.
        """
        ...

    def seed_initial_failures(
        self,
        env: simpy.Environment,
        network: CPSNetwork,
        params: dict,
    ) -> None:
        """Seed initial failures. Default: random node selection."""
        total_nodes = network.graph.number_of_nodes()
        num_initial = int(params.get("initial_failures", 0))
        for node_id in network.sim_state.rng.sample(range(total_nodes), num_initial):
            network.mark_failed(env, network.nodes_map[node_id], network.sim_state)


# --- Registry ---

_FAILURE_MODEL_REGISTRY: dict[str, type[FailureModel]] = {}


def register_failure_model(name: str, cls: type[FailureModel]) -> None:
    """Register a failure model class by name."""
    _FAILURE_MODEL_REGISTRY[name] = cls


def get_failure_model(name: str) -> FailureModel:
    """Look up and instantiate a registered failure model."""
    if name not in _FAILURE_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown failure model '{name}'. "
            f"Available: {list(_FAILURE_MODEL_REGISTRY.keys())}"
        )
    return _FAILURE_MODEL_REGISTRY[name]()


# --- Built-in: Watts Cascade ---


class WattsCascadeModel(FailureModel):
    """Watts threshold cascade: a node fails when the fraction of
    its failed neighbor edges exceeds phi.

    Delegates to Node.node_fail_watts() and Link.edge_fail_watts()
    for the actual cascade SimPy processes.
    """

    def install(self, env, network, service_team, params):
        phi = float(params["phi"])
        fail_speed = float(params["fail_speed"])

        for node_id in network.graph.nodes:
            node = network.nodes_map[node_id]
            env.process(
                node.node_fail_watts(env, phi, fail_speed, network, service_team)
            )

        for v1, v2 in network.graph.edges:
            link = network.get_link(v1, v2)
            if link is not None:
                env.process(
                    link.edge_fail_watts(env, fail_speed, network, service_team)
                )


register_failure_model("Watts cascade", WattsCascadeModel)
