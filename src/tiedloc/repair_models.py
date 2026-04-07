"""Pluggable repair models for the tiedloc simulator.

This module defines the RepairModel ABC and a registry of built-in models.
Repair models control how repair agents estimate and execute repairs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import simpy

if TYPE_CHECKING:
    from tiedloc.agents import Link, Node, Server
    from tiedloc.networks import CPSNetwork


class RepairModel(ABC):
    """Controls how repair agents estimate and execute repairs."""

    @abstractmethod
    def estimate_repair_time(
        self, server: Server, element: Node | Link, params: dict
    ) -> float:
        """Estimate the repair duration for a given element.

        Args:
            server: The repair agent.
            element: The failed node or link.
            params: The "service_team" section config.

        Returns:
            Estimated repair time in simulation time units.
        """
        ...

    @abstractmethod
    def execute_repair(
        self,
        env: simpy.Environment,
        server: Server,
        network: CPSNetwork,
        element: Node | Link,
    ) -> Generator[simpy.Event, None, None]:
        """SimPy process: perform the actual repair.

        Yield SimPy events as needed.
        """
        ...


# --- Registry ---

_REPAIR_MODEL_REGISTRY: dict[str, type[RepairModel]] = {}


def register_repair_model(name: str, cls: type[RepairModel]) -> None:
    """Register a repair model class by name."""
    _REPAIR_MODEL_REGISTRY[name] = cls


def get_repair_model(name: str) -> RepairModel:
    """Look up and instantiate a registered repair model."""
    if name not in _REPAIR_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown repair model '{name}'. "
            f"Available: {list(_REPAIR_MODEL_REGISTRY.keys())}"
        )
    return _REPAIR_MODEL_REGISTRY[name]()


# --- Built-in: Fixed Repair Time ---


class FixedRepairModel(RepairModel):
    """Constant repair time (current behavior)."""

    def estimate_repair_time(self, server, element, params):
        return server.repairing_time

    def execute_repair(self, env, server, network, element):
        yield env.timeout(server.repairing_time)
        network.mark_restored(env, element, network.sim_state)


register_repair_model("fixed", FixedRepairModel)
