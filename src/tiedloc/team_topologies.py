"""Pluggable team topology generators for the tiedloc simulator.

This module defines the TeamTopologyGenerator protocol and a registry of
built-in generators for the service team communication graph.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import networkx as nx


@runtime_checkable
class TeamTopologyGenerator(Protocol):
    """Generates the communication graph for the service team."""

    def generate(self, params: dict) -> nx.Graph:
        """Create and return the team communication graph.

        Args:
            params: The "service_team" section of the input config.

        Returns:
            A networkx.Graph where each node is an agent ID.
        """
        ...


# --- Registry ---

_TEAM_TOPOLOGY_REGISTRY: dict[str, type[TeamTopologyGenerator]] = {}


def register_team_topology(name: str, gen: type | TeamTopologyGenerator) -> None:
    """Register a team topology generator class (or instance for backward compat) by name."""
    if isinstance(gen, type):
        _TEAM_TOPOLOGY_REGISTRY[name] = gen
    else:
        warnings.warn(
            "Passing an instance to register_team_topology() is deprecated. "
            "Pass the class instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _TEAM_TOPOLOGY_REGISTRY[name] = type(gen)


def get_team_topology(name: str) -> TeamTopologyGenerator:
    """Look up and instantiate a registered team topology generator."""
    if name not in _TEAM_TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown team topology '{name}'. "
            f"Available: {list(_TEAM_TOPOLOGY_REGISTRY.keys())}"
        )
    return _TEAM_TOPOLOGY_REGISTRY[name]()


# --- Built-in Generators ---


class RegularGraphTeam:
    """Random regular graph: each agent has the same degree."""

    def generate(self, params: dict) -> nx.Graph:
        return nx.random_regular_graph(
            params["team_degree"], params["team_members"],
            seed=params.get("seed"),
        )


class CompleteGraphTeam:
    """All agents can communicate with all others."""

    def generate(self, params: dict) -> nx.Graph:
        return nx.complete_graph(params["team_members"])


class StarTeam:
    """One coordinator, all others report to it."""

    def generate(self, params: dict) -> nx.Graph:
        return nx.star_graph(params["team_members"] - 1)


register_team_topology("regular graph", RegularGraphTeam)
register_team_topology("complete", CompleteGraphTeam)
register_team_topology("star", StarTeam)
