"""Pluggable metric collectors for the tiedloc simulator.

This module defines the MetricCollector ABC and a registry of built-in
metrics. Users can register custom metrics to extend simulation output.
"""

from __future__ import annotations

import math
import statistics as pystats
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState


class MetricCollector(ABC):
    """Collects a single metric from simulation runs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name (used as key in results dict)."""
        ...

    @abstractmethod
    def collect(
        self,
        sim_state: SimulationState,
        network: CPSNetwork,
        service_team: ServiceTeam,
        env_now: float,
    ) -> Any:
        """Extract this metric's value from a single replication."""
        ...

    @abstractmethod
    def aggregate(self, values: list[Any]) -> dict | float:
        """Aggregate values across replications."""
        ...


# --- Registry ---

_METRIC_COLLECTORS: list[MetricCollector] = []


def register_metric(collector: MetricCollector) -> None:
    """Register a metric collector."""
    _METRIC_COLLECTORS.append(collector)


def unregister_metric(collector: MetricCollector) -> None:
    """Unregister a previously registered metric collector."""
    try:
        _METRIC_COLLECTORS.remove(collector)
    except ValueError:
        pass


def get_all_metrics() -> list[MetricCollector]:
    """Get all registered metric collectors."""
    return list(_METRIC_COLLECTORS)


def _reset_metric_registry(saved: list[MetricCollector]) -> None:
    """Reset the metric registry to a previous state. For testing only."""
    _METRIC_COLLECTORS.clear()
    _METRIC_COLLECTORS.extend(saved)


# --- Built-in Metrics ---

def _mean_sem(values: list[float]) -> dict[str, float]:
    """Compute mean and SEM for a list of floats, handling inf."""
    finite = [v for v in values if not math.isinf(v)]
    if not finite:
        return {"Mean": float("inf"), "SEM": 0.0}
    m = pystats.mean(finite)
    sd = pystats.pstdev(finite)
    return {
        "Mean": float(m),
        "SEM": float(sd / math.sqrt(len(finite))),
    }


class TotalFailuresMetric(MetricCollector):
    """Total number of elements that failed during the simulation."""

    @property
    def name(self) -> str:
        return "Total_Failures"

    def collect(self, sim_state, network, service_team, env_now):
        return sim_state.total_failures

    def aggregate(self, values):
        return _mean_sem(values)


class PreventabilityMetric(MetricCollector):
    """Fraction of potential cascade events prevented by agents."""

    @property
    def name(self) -> str:
        return "Preventability"

    def collect(self, sim_state, network, service_team, env_now):
        total = sim_state.prevented + sim_state.total_failures
        return sim_state.prevented / total if total > 0 else 0.0

    def aggregate(self, values):
        return _mean_sem(values)


class TotalDistanceMetric(MetricCollector):
    """Total distance traveled by all repair agents."""

    @property
    def name(self) -> str:
        return "Total_Distance_Traveled_by_Agent"

    def collect(self, sim_state, network, service_team, env_now):
        return sim_state.total_distance

    def aggregate(self, values):
        return _mean_sem(values)


class AuxLinesMetric(MetricCollector):
    """Number of auxiliary edges injected."""

    @property
    def name(self) -> str:
        return "Aux_Lines"

    def collect(self, sim_state, network, service_team, env_now):
        return sim_state.aux_lines

    def aggregate(self, values):
        return _mean_sem(values)


class TotalLatencyMetric(MetricCollector):
    """Sum of all individual failure-to-repair wait times."""

    @property
    def name(self) -> str:
        return "Total_Latency"

    def collect(self, sim_state, network, service_team, env_now):
        return sim_state.total_latency

    def aggregate(self, values):
        return _mean_sem(values)


class MeanLatencyQoSMetric(MetricCollector):
    """Average latency across individual failure events (QoS metric)."""

    @property
    def name(self) -> str:
        return "Mean_Latency_QoS"

    def collect(self, sim_state, network, service_team, env_now):
        return list(sim_state.latency)

    def aggregate(self, values):
        # values is a list of lists — flatten
        flat = []
        for v in values:
            flat.extend(v)
        return _mean_sem(flat) if flat else {"Mean": 0.0, "SEM": 0.0}


# Register all built-in metrics
for _cls in [
    TotalFailuresMetric,
    PreventabilityMetric,
    TotalDistanceMetric,
    AuxLinesMetric,
    TotalLatencyMetric,
    MeanLatencyQoSMetric,
]:
    register_metric(_cls())
