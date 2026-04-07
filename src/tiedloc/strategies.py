"""Pluggable response/dispatch strategies for the tiedloc simulator.

This module defines the ResponseStrategy ABC and a registry of built-in
strategies. Each strategy implements dispatch(), which assigns repair agents
to failures.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import simpy

if TYPE_CHECKING:
    from tiedloc.networks import CPSNetwork, ServiceTeam


class ResponseStrategy(ABC):
    """Abstract base class for agent dispatch strategies."""

    sort_key = None  # Optional: callable for sorting in nn_schedule

    @abstractmethod
    def dispatch(
        self,
        env: simpy.Environment,
        network: CPSNetwork,
        service_team: ServiceTeam,
    ) -> None:
        """Assign agents to failures.

        When called, network.sim_state.failures contains the current
        list of unassigned failures. The strategy should populate
        sim_state.agent_schedule with job assignments.
        """
        ...


# --- Registry ---

_STRATEGY_REGISTRY: dict[str, type[ResponseStrategy]] = {}


def register_strategy(name: str, strategy: type | ResponseStrategy) -> None:
    """Register a response strategy class (or instance for backward compat) by name."""
    if isinstance(strategy, type):
        _STRATEGY_REGISTRY[name] = strategy
    else:
        warnings.warn(
            "Passing an instance to register_strategy() is deprecated. "
            "Pass the class instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _STRATEGY_REGISTRY[name] = type(strategy)


def get_strategy(name: str) -> ResponseStrategy:
    """Look up and instantiate a registered response strategy."""
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown response strategy '{name}'. "
            f"Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[name]()


# --- Built-in Strategies ---
# These are thin wrappers that delegate to the existing dispatch functions
# in responsestrategies.py to avoid duplicating logic.


class FCFSStrategy(ResponseStrategy):
    """First-come, first-served: sort failures by time, assign greedily."""

    def dispatch(self, env, network, service_team):
        from tiedloc.responsestrategies import dispatcher, fcfs_key
        sim_state = network.sim_state
        sim_state.failures.sort(key=fcfs_key)
        dispatcher(env, network, service_team)


class NearestStrategy(ResponseStrategy):
    """Nearest-neighbor: assign each idle agent to the closest failure."""

    sort_key = staticmethod(lambda item: (item[1],))

    def dispatch(self, env, network, service_team):
        from tiedloc.responsestrategies import nn_schedule
        nn_schedule(env, network, service_team, self)


class ActivityStrategy(ResponseStrategy):
    """Activity-based: prioritize failures at highest cascade risk."""

    sort_key = staticmethod(lambda item: (-item[0], item[1]))

    def dispatch(self, env, network, service_team):
        from tiedloc.responsestrategies import nn_schedule
        nn_schedule(env, network, service_team, self)


register_strategy("FCFS", FCFSStrategy)
register_strategy("nearest", NearestStrategy)
register_strategy("activity", ActivityStrategy)
