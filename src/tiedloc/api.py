"""Public API for the tiedloc simulator.

This module provides a programmatic interface to configure and run simulations.
It is the primary entry point for library usage.

Usage:
    import tiedloc

    config = tiedloc.SimulationConfig.from_json("examples/small_ba.json")
    results = tiedloc.run(config)
    print(results.stats["Recoverability"])
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

from tiedloc import networks, simulations
from tiedloc.failure_models import FailureModel
from tiedloc.metrics import MetricCollector, register_metric, unregister_metric
from tiedloc.strategies import ResponseStrategy
from tiedloc.team_topologies import TeamTopologyGenerator
from tiedloc.topologies import TopologyGenerator


@dataclass
class SimulationConfig:
    """Programmatic configuration for a tiedloc simulation.

    Each field maps to a section of the JSON input format. String values
    for topology, failure_model, strategy, and team_topology are looked up
    in the corresponding registry. Instance values are used directly.
    """

    # CPS Network
    topology: str | TopologyGenerator = "Barabasi Albert Scale-Free Network"
    topology_params: dict = field(default_factory=lambda: {
        "num_of_nodes": 100,
        "new_node_to_existing_nodes": 3,
    })

    # Failure model
    failure_model: str | FailureModel = "Watts cascade"
    failure_params: dict = field(default_factory=lambda: {
        "phi": 0.3,
        "fail_speed": 1.0,
        "initial_failures": 5,
    })

    # Response strategy
    strategy: str | ResponseStrategy = "FCFS"
    strategy_params: dict = field(default_factory=lambda: {
        "frequency": 1.0,
        "auxiliary": False,
    })

    # Service team
    team_topology: str | TeamTopologyGenerator = "regular graph"
    team_params: dict = field(default_factory=lambda: {
        "team_members": 10,
        "team_degree": 4,
        "agent_travel_speed": 1.0,
        "repairing_time": 2.0,
        "initial_allocation": "Centrality",
    })

    # Simulation parameters
    seed: int | None = 42
    replications: int = 10
    processors: int = 1
    simulation_length: int = 100

    # Custom metrics (in addition to built-ins)
    extra_metrics: list[MetricCollector] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> SimulationConfig:
        """Create config from a parsed JSON dictionary.

        Args:
            data: A dictionary matching the tiedloc JSON input format.

        Returns:
            A SimulationConfig populated from the dictionary.
        """
        return cls(
            topology=data["CPS"]["name"],
            topology_params=data["CPS"],
            failure_model=data["failure_model"]["name"],
            failure_params=data["failure_model"],
            strategy=data["response_protocol"]["name"],
            strategy_params=data["response_protocol"],
            team_topology=data["service_team"]["name"],
            team_params=data["service_team"],
            seed=int(seed_val) if (seed_val := data["simulation_param"].get("seed")) is not None else None,
            replications=int(data["simulation_param"]["replications"]),
            processors=int(data["simulation_param"]["processors"]),
            simulation_length=int(data["simulation_param"]["simulation_length"]),
        )

    @classmethod
    def from_json(cls, path: str) -> SimulationConfig:
        """Load config from a JSON file (backwards compatible).

        Args:
            path: Path to a tiedloc JSON input file.

        Returns:
            A SimulationConfig populated from the JSON file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file contains invalid JSON.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in configuration file {path}: {exc}"
            ) from exc
        return cls.from_dict(data)

    def to_parameters(self) -> dict:
        """Convert to the internal parameters dict format used by the simulator.

        Returns:
            A dictionary matching the JSON input format.
        """
        topology_name = self.topology if isinstance(self.topology, str) else self.topology.__class__.__name__
        failure_name = self.failure_model if isinstance(self.failure_model, str) else self.failure_model.__class__.__name__
        strategy_name = self.strategy if isinstance(self.strategy, str) else self.strategy.__class__.__name__
        team_name = self.team_topology if isinstance(self.team_topology, str) else self.team_topology.__class__.__name__

        cps = dict(self.topology_params)
        cps["name"] = topology_name

        fm = dict(self.failure_params)
        fm["name"] = failure_name

        rp = dict(self.strategy_params)
        rp["name"] = strategy_name

        st = dict(self.team_params)
        st["name"] = team_name

        return {
            "CPS": cps,
            "failure_model": fm,
            "response_protocol": rp,
            "service_team": st,
            "simulation_param": {
                "seed": str(self.seed) if self.seed is not None else None,
                "replications": str(self.replications),
                "processors": str(self.processors),
                "simulation_length": str(self.simulation_length),
            },
        }


@dataclass
class SimulationResults:
    """Results from a tiedloc simulation.

    Attributes:
        stats: Aggregated statistics.
        samples: Raw per-replication values for each metric.
        config: The configuration that produced these results.
    """

    stats: dict[str, Any]
    samples: dict[str, list]
    config: SimulationConfig

    def to_json(self, path: str) -> None:
        """Save aggregated stats to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4, sort_keys=True)


def run(config: SimulationConfig) -> SimulationResults:
    """Run a tiedloc simulation with the given configuration.

    This is the primary programmatic entry point. Any ``extra_metrics``
    specified in the config are temporarily registered before the run
    and unregistered afterward so they don't leak into other runs.

    Args:
        config: A SimulationConfig instance.

    Returns:
        SimulationResults with aggregated statistics and per-replication samples.
    """
    parameters = config.to_parameters()

    master_rng = random.Random(config.seed)
    parameters["_topo_seed"] = master_rng.getrandbits(64)
    parameters["_dispatch_seed"] = master_rng.getrandbits(64)

    cps_network = networks.CPSNetwork(parameters)

    # Temporarily register extra metrics for this run
    for metric in config.extra_metrics:
        register_metric(metric)

    try:
        results = simulations.parallel_dispatch(parameters, cps_network)
        stat_results, samples = simulations.statistics(results)
    finally:
        for metric in config.extra_metrics:
            unregister_metric(metric)

    return SimulationResults(
        stats=stat_results,
        samples=samples,
        config=config,
    )


def sweep(
    base_config: SimulationConfig,
    param_name: str,
    values: list,
) -> list[SimulationResults]:
    """Run a parameter sweep, varying one parameter across values.

    Args:
        base_config: Base configuration.
        param_name: Dot-separated parameter path (e.g., "failure_params.phi").
        values: List of values to sweep.

    Returns:
        List of SimulationResults, one per value.
    """
    parts = param_name.split(".")
    if len(parts) > 2:
        raise ValueError(
            f"param_name '{param_name}' has {len(parts)} levels; "
            f"only 1 or 2 levels are supported (e.g., 'seed' or 'failure_params.phi')"
        )

    # Validate that the attribute/key exists on the base config
    if len(parts) == 2:
        if not hasattr(base_config, parts[0]):
            raise ValueError(f"Config has no attribute '{parts[0]}'")
        target = getattr(base_config, parts[0])
        if not isinstance(target, dict) or parts[1] not in target:
            raise ValueError(f"'{parts[0]}' has no key '{parts[1]}'")
    elif len(parts) == 1:
        if not hasattr(base_config, parts[0]):
            raise ValueError(f"Config has no attribute '{parts[0]}'")

    results = []
    for val in values:
        config = copy.deepcopy(base_config)
        if len(parts) == 2:
            target = getattr(config, parts[0])
            target[parts[1]] = val
        elif len(parts) == 1:
            setattr(config, parts[0], val)
        results.append(run(config))
    return results
