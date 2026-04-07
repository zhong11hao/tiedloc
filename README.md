# TIE/DLOC v2.0

**TIE/DLOC** (Team Integration Evaluator - Dynamic Line of Collaboration) is a discrete-event simulator for analyzing cascading failures and response strategies in Cyber-Physical Systems (CPS) networks. TIE/DLOC can be used as a CLI tool or as a pluggable Python library.

## What's New in v2.0

- **Python 3.11+ / networkx 3.x** — Fully modernized from Python 2 / networkx 1.x
- **8 pluggable extension points** — Topology, failure model, response strategy, team topology, repair model, metrics, agent behavior, and library API
- **Programmatic API** — `SimulationConfig`, `run()`, and `sweep()` for library usage
- **Named constants** — Magic numbers replaced with self-documenting constants in `constants.py` (`STATE_OPERATIONAL`, `STATE_FAILED`, `STATE_IDLE`, `STATE_BUSY`)
- **Seed/randomness redesign** — `SimulationState.rng` provides per-worker RNG isolation for deterministic, reproducible simulations across parallel replications
- **`extra_metrics` integration** — Register custom metrics per-run via `SimulationConfig.extra_metrics` with automatic register/unregister and `try/finally` cleanup
- **`pack(full=True)` option** — Full graph metadata serialization via `nx.node_link_data` preserving node and edge attributes across process boundaries
- **Metrics registry with `unregister_metric()`** — Clean lifecycle management for custom metrics
- **Refactored dispatch helpers** — 6 helper functions extracted from `dispatcher`/`nn_schedule` in `responsestrategies.py` for testability and clarity
- **Error handling** — Descriptive `FileNotFoundError` and `ValueError` exceptions for file I/O in `networks.py` and `topologies.py`
- **Comprehensive test suite** — 230+ tests covering all modules

## Overview

TIE/DLOC simulates cascading failures in network topologies and evaluates the effectiveness of different repair agent dispatch strategies. It models a CPS network subject to the Watts cascade failure model, alongside a team of service agents that travel through the network to restore failed elements.

## Features

- **Multiple network topologies**: Barabasi-Albert, Watts-Strogatz, Binomial, edge list, and real-world power grid data.
- **Cascading failure model**: Watts cascade with configurable threshold and propagation speed.
- **Response strategies**: FCFS, nearest-neighbor, and activity-based dispatch protocols.
- **Auxiliary edge injection**: Dynamic network augmentation with configurable trigger time.
- **Parallel simulation**: Multi-process replication for statistical confidence.
- **Comprehensive statistics**: Recoverability, latency, preventability, and per-step failure tracking.
- **Pluggable library API**: Register custom topologies, failure models, strategies, team structures, repair models, and metrics via TIE/DLOC's registry system.
- **Per-run custom metrics**: Pass `extra_metrics` to `SimulationConfig` for temporary metric registration scoped to a single run.
- **Full graph serialization**: `pack(full=True)` preserves all node/edge attributes when transferring networks across process boundaries.
- **Deterministic reproducibility**: Seed-based RNG hierarchy (`master_rng` -> `_topo_seed` + `_dispatch_seed` -> per-worker `rng`) ensures identical results across runs.
- **Type-safe codebase**: Concrete type annotations throughout, minimal use of `Any`.

## Requirements

- Python >= 3.11
- networkx >= 3.0
- simpy >= 4.0

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
tiedloc path/to/input.json
```

Or run as a module:

```bash
python -m tiedloc path/to/input.json
```

### Library API

TIE/DLOC can be used programmatically as a Python library:

```python
import tiedloc

# From an existing JSON config
config = tiedloc.SimulationConfig.from_json("examples/small_ba.json")
results = tiedloc.run(config)
print(results.stats["Recoverability"])

# Fully programmatic
config = tiedloc.SimulationConfig(
    topology="Barabasi Albert Scale-Free Network",
    topology_params={"num_of_nodes": 50, "new_node_to_existing_nodes": 3},
    failure_model="Watts cascade",
    failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 5},
    strategy="FCFS",
    strategy_params={"frequency": 1.0, "auxiliary": False},
    team_topology="regular graph",
    team_params={
        "team_members": 10,
        "team_degree": 4,
        "agent_travel_speed": 1.0,
        "repairing_time": 2.0,
        "initial_allocation": "Centrality",
    },
    seed=42,
    replications=10,
    processors=1,
    simulation_length=100,
)
results = tiedloc.run(config)
print(results.stats["Total_Failures"])

# Save results
results.to_json("output.json")
```

### Parameter Sweep

```python
import tiedloc

config = tiedloc.SimulationConfig.from_json("examples/small_ba.json")
sweep_results = tiedloc.sweep(config, "failure_params.phi", [0.1, 0.2, 0.3, 0.4, 0.5])
for r in sweep_results:
    print(f"phi={r.config.failure_params['phi']}: "
          f"Recoverability={r.stats['Recoverability']:.2f}")
```

### Per-Run Custom Metrics

Custom metrics can be scoped to a single run using `extra_metrics`. They are automatically registered before the run and unregistered afterward via `try/finally` cleanup:

```python
import statistics
from tiedloc.api import SimulationConfig, run
from tiedloc.metrics import MetricCollector

class PeakFailureMetric(MetricCollector):
    @property
    def name(self):
        return "Peak_Failures"

    def collect(self, sim_state, network, service_team, env_now):
        return max(sim_state.failed_at_step) if sim_state.failed_at_step else 0

    def aggregate(self, values):
        m = statistics.mean(values)
        return {"Mean": m, "SEM": 0.0}

config = SimulationConfig(
    topology_params={"num_of_nodes": 30, "new_node_to_existing_nodes": 2},
    failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 2},
    team_params={"team_members": 4, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=42, replications=5, processors=1, simulation_length=50,
    extra_metrics=[PeakFailureMetric()],
)
results = run(config)
print(f"Peak Failures: {results.stats['Peak_Failures']}")
```

### Entropy Mode (Non-Deterministic Runs)

Set `seed=None` to use system entropy for each run:

```python
config = tiedloc.SimulationConfig(seed=None, replications=10, simulation_length=50)
results = tiedloc.run(config)
```

### Full Graph Serialization

Use `pack(full=True)` to preserve all graph metadata (node/edge attributes) when serializing networks for multiprocessing:

```python
from tiedloc.networks import CPSNetwork

params = {
    "CPS": {"name": "Barabasi Albert Scale-Free Network",
            "num_of_nodes": 20, "new_node_to_existing_nodes": 2},
    "response_protocol": {"auxiliary": False},
}
network = CPSNetwork(params)
network.graph.nodes[0]["custom_attr"] = "my_value"

packed = network.pack(full=True)   # uses nx.node_link_data
network2 = CPSNetwork(packed)
assert network2.graph.nodes[0]["custom_attr"] == "my_value"
```

The default `pack()` (without `full=True`) uses a lightweight edge-list path that is faster but only preserves topology.

## Extension Points

TIE/DLOC is designed as a pluggable library. Every major component can be replaced with a custom implementation by registering it in the corresponding registry.

### Custom Network Topologies

Implement a class with a `generate(params: dict) -> nx.Graph` method:

```python
from tiedloc.topologies import register_topology
import networkx as nx

class FatTreeGenerator:
    def generate(self, params: dict) -> nx.Graph:
        k = params.get("fat_tree_k", 4)
        return nx.balanced_tree(r=k, h=3)

register_topology("fat_tree", FatTreeGenerator)
```

Then use in JSON config (`"name": "fat_tree"`) or programmatically:

```python
config = tiedloc.SimulationConfig(
    topology="fat_tree",
    topology_params={"num_of_nodes": 85, "fat_tree_k": 4},
)
```

**Built-in topologies**: `"Barabasi Albert Scale-Free Network"`, `"Watts-Strogatz Small-World Model"`, `"Binomial Graph"`, `"Power Grid of Western States of USA"`, `"edge_list"`.

### Custom Failure Cascade Models

Subclass `FailureModel` with an `install()` method that registers SimPy processes:

```python
from tiedloc.failure_models import FailureModel, register_failure_model

class PercolationModel(FailureModel):
    def install(self, env, network, service_team, params):
        p = float(params["p"])
        tstep = float(params["fail_speed"])
        for node_id in network.graph.nodes:
            node = network.nodes_map[node_id]
            env.process(self._percolate(env, p, tstep, node, network))

    def _percolate(self, env, p, tstep, node, network):
        while True:
            if node.state == 0 and network.sim_state.rng.random() < p:
                network.mark_failed(env, node, network.sim_state)
            yield env.timeout(tstep)

register_failure_model("percolation", PercolationModel)
```

You can also override `seed_initial_failures()` to customize how initial failures are selected. The default implementation randomly selects `initial_failures` nodes using `sim_state.rng.sample()` for reproducible selection.

**Built-in failure models**: `"Watts cascade"`.

### Custom Response/Dispatch Strategies

Subclass `ResponseStrategy` with a `dispatch()` method:

```python
from tiedloc.strategies import ResponseStrategy, register_strategy

class CentralityPriorityStrategy(ResponseStrategy):
    def dispatch(self, env, network, service_team):
        sim_state = network.sim_state
        centrality = {int(row[1]): row[0] for row in network.bctlist}

        def centrality_key(item):
            if hasattr(item, 'id'):
                return -centrality.get(item.id, 0)
            return -max(centrality.get(item.vertex1, 0),
                        centrality.get(item.vertex2, 0))

        sim_state.failures.sort(key=centrality_key)
        from tiedloc.responsestrategies import dispatcher
        dispatcher(env, network, service_team)

register_strategy("centrality_priority", CentralityPriorityStrategy)
```

**Built-in strategies**: `"FCFS"`, `"nearest"`, `"activity"`.

### Custom Team Structures

Implement a class with a `generate(params: dict) -> nx.Graph` method:

```python
from tiedloc.team_topologies import register_team_topology
import networkx as nx

class HierarchicalTeam:
    def generate(self, params: dict) -> nx.Graph:
        n = params["team_members"]
        return nx.star_graph(n - 1)

register_team_topology("hierarchical", HierarchicalTeam)
```

**Built-in team topologies**: `"regular graph"`, `"complete"`, `"star"`.

### Custom Repair Models

Subclass `RepairModel` to control repair estimation and execution:

```python
from tiedloc.repair_models import RepairModel, register_repair_model

class StochasticRepairModel(RepairModel):
    def estimate_repair_time(self, server, element, params):
        return server.repairing_time * 1.5  # conservative estimate

    def execute_repair(self, env, server, network, element):
        import math
        actual_time = network.sim_state.rng.lognormvariate(
            mu=math.log(server.repairing_time), sigma=0.3
        )
        yield env.timeout(actual_time)
        network.mark_restored(env, element, network.sim_state)

register_repair_model("stochastic", StochasticRepairModel)
```

**Built-in repair models**: `"fixed"`.

### Custom Metrics

Subclass `MetricCollector` to add custom simulation outputs:

```python
from tiedloc.metrics import MetricCollector, register_metric

class MaxCascadeDepthMetric(MetricCollector):
    @property
    def name(self):
        return "Max_Cascade_Depth"

    def collect(self, sim_state, network, service_team, env_now):
        return max(sim_state.failed_at_step) if sim_state.failed_at_step else 0

    def aggregate(self, values):
        import statistics, math
        m = statistics.mean(values)
        sd = statistics.pstdev(values)
        return {"Mean": m, "SEM": sd / math.sqrt(len(values)) if values else 0.0}

register_metric(MaxCascadeDepthMetric())
```

To remove a previously registered metric, use `unregister_metric()`:

```python
from tiedloc.metrics import unregister_metric

unregister_metric(my_metric_instance)
```

**Built-in metrics**: `Total_Failures`, `Preventability`, `Total_Distance_Traveled_by_Agent`, `Aux_Lines`, `Total_Latency`, `Mean_Latency_QoS`.

### Custom Agent Behavior

Subclass `AgentBehavior` to customize how repair agents handle jobs:

```python
from tiedloc.agent_behaviors import AgentBehavior

class PreemptiveAgentBehavior(AgentBehavior):
    def execute(self, env, server, network, job_req):
        req = job_req.request
        params = job_req.params
        yield req  # acquire resource
        # Custom travel + repair logic
        yield env.timeout(params["finishing_time"] - env.now)
        network.mark_restored(env, params["failure"], network.sim_state)
        server.resource.release(req)
```

Agent behaviors are injectable programmatically via the `Server` constructor (not via JSON config):

```python
from tiedloc.agents import Server
server = Server(0, env, travel_speed=1.0, repair_time=2.0,
                behavior=PreemptiveAgentBehavior())
```

### Example Input Files

The `examples/` directory contains ready-to-use JSON configurations:

| File | Network | Strategy | Notes |
|------|---------|----------|-------|
| `small_ba.json` | Barabasi-Albert (20 nodes) | FCFS | Basic scale-free network |
| `small_ws.json` | Watts-Strogatz (20 nodes) | nearest | Small-world topology |
| `activity_ba.json` | Barabasi-Albert (20 nodes) | activity | Activity-based dispatch |
| `auxiliary_ba.json` | Barabasi-Albert (20 nodes) | FCFS | Auxiliary edge injection enabled |

Run any example with:

```bash
tiedloc examples/small_ba.json
```

## Custom Integration Guide

This guide shows how to integrate TIE/DLOC into your own Python project as a library. You'll learn how to import the API, customize every component, and run simulations programmatically.

### Quick Start

```python
import tiedloc

config = tiedloc.SimulationConfig(
    topology="Barabasi Albert Scale-Free Network",
    topology_params={"num_of_nodes": 30, "new_node_to_existing_nodes": 2},
    failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 2},
    team_params={"team_members": 4, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=42, replications=5, processors=1, simulation_length=50,
)
results = tiedloc.run(config)
print(f"Recoverability: {results.stats['Recoverability']}")
print(f"Total Failures: {results.stats['Total_Failures']}")
```

### What You Can Customize

| Extension Point | Import | Subclass / Implement | Register With | Methods to Implement |
|---|---|---|---|---|
| Network Topology | `from tiedloc.topologies import register_topology` | Class with `generate(self, params: dict) -> nx.Graph` | `register_topology("name", MyClass)` | `generate(params)` |
| Failure Model | `from tiedloc.failure_models import FailureModel, register_failure_model` | `FailureModel` (ABC) | `register_failure_model("name", MyClass)` | `install(env, network, service_team, params)`, optionally `seed_initial_failures(env, network, params)` |
| Response Strategy | `from tiedloc.strategies import ResponseStrategy, register_strategy` | `ResponseStrategy` (ABC) | `register_strategy("name", MyClass)` | `dispatch(env, network, service_team)` |
| Team Topology | `from tiedloc.team_topologies import register_team_topology` | Class with `generate(self, params: dict) -> nx.Graph` | `register_team_topology("name", MyClass)` | `generate(params)` |
| Repair Model | `from tiedloc.repair_models import RepairModel, register_repair_model` | `RepairModel` (ABC) | `register_repair_model("name", MyClass)` | `estimate_repair_time(server, element, params)`, `execute_repair(env, server, network, element)` |
| Metrics | `from tiedloc.metrics import MetricCollector, register_metric` | `MetricCollector` (ABC) | `register_metric(MyMetric())` | `name` (property), `collect(sim_state, network, service_team, env_now)`, `aggregate(values)` |
| Agent Behavior | `from tiedloc.agent_behaviors import AgentBehavior` | `AgentBehavior` (ABC) | Passed to `Server(behavior=...)` constructor | `execute(env, server, network, job_req)` |
| Initial Failures | Override `FailureModel.seed_initial_failures()` | `FailureModel` (ABC) | Via failure model registration | `seed_initial_failures(env, network, params)` |

**Key distinction**: Most extension points use **class-based registration** (pass the class, not an instance). The exception is `register_metric()`, which takes an **instance**.

### Step-by-Step: Custom Topology

This example defines a ring topology generator, registers it, and runs a simulation with it.

```python
import networkx as nx
from tiedloc.topologies import register_topology
import tiedloc

class RingTopology:
    """Generates a simple ring (cycle) graph."""
    def generate(self, params: dict) -> nx.Graph:
        n = int(params["num_of_nodes"])
        return nx.cycle_graph(n)

# Register the class (not an instance)
register_topology("ring", RingTopology)

# Use it in a simulation
config = tiedloc.SimulationConfig(
    topology="ring",
    topology_params={"num_of_nodes": 20},
    failure_params={"phi": 0.4, "fail_speed": 1.0, "initial_failures": 2},
    team_params={"team_members": 4, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=123, replications=5, processors=1, simulation_length=50,
)
results = tiedloc.run(config)
print(f"Ring topology — Recoverability: {results.stats['Recoverability']}")
print(f"Ring topology — Total Failures: {results.stats['Total_Failures']}")
```

### Step-by-Step: Custom Failure Model

This example defines a simple random failure model where each operational node has a probability `p` of failing at each time step, independent of neighbors.

```python
from collections.abc import Generator
import simpy
from tiedloc.failure_models import FailureModel, register_failure_model
import tiedloc

class RandomFailureModel(FailureModel):
    """Each node fails independently with probability p at each time step."""

    def install(self, env, network, service_team, params):
        p = float(params.get("p", 0.05))
        tstep = float(params.get("fail_speed", 1.0))
        for node_id in network.graph.nodes:
            node = network.nodes_map[node_id]
            env.process(self._check_fail(env, p, tstep, node, network))

    def _check_fail(self, env, p, tstep, node, network) -> Generator[simpy.Event, None, None]:
        while True:
            if node.state == 0 and network.sim_state.rng.random() < p:
                network.mark_failed(env, node, network.sim_state)
            yield env.timeout(tstep)

register_failure_model("random_independent", RandomFailureModel)

config = tiedloc.SimulationConfig(
    topology="Barabasi Albert Scale-Free Network",
    topology_params={"num_of_nodes": 30, "new_node_to_existing_nodes": 2},
    failure_model="random_independent",
    failure_params={"p": 0.03, "fail_speed": 1.0, "initial_failures": 1},
    team_params={"team_members": 6, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=99, replications=5, processors=1, simulation_length=50,
)
results = tiedloc.run(config)
print(f"Random failure model — Recoverability: {results.stats['Recoverability']}")
print(f"Random failure model — Total Failures: {results.stats['Total_Failures']}")
```

### Combining Multiple Customizations

This example registers a custom topology, failure model, response strategy, and metric, then uses all of them in a single simulation.

```python
import math
import statistics
from collections.abc import Generator

import networkx as nx
import simpy

import tiedloc
from tiedloc.topologies import register_topology
from tiedloc.failure_models import FailureModel, register_failure_model
from tiedloc.strategies import ResponseStrategy, register_strategy
from tiedloc.metrics import MetricCollector, register_metric
from tiedloc.responsestrategies import dispatcher

# 1. Custom topology: grid graph (converted to integer node labels)
class GridTopology:
    def generate(self, params: dict) -> nx.Graph:
        side = int(math.sqrt(int(params["num_of_nodes"])))
        g = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(g)

register_topology("grid", GridTopology)

# 2. Custom failure model: delayed random cascade
class DelayedRandomFailure(FailureModel):
    """Nodes only start failing after a delay period."""
    def install(self, env, network, service_team, params):
        delay = float(params.get("delay", 5.0))
        p = float(params.get("p", 0.05))
        tstep = float(params.get("fail_speed", 1.0))
        for node_id in network.graph.nodes:
            node = network.nodes_map[node_id]
            env.process(self._delayed_fail(env, delay, p, tstep, node, network))

    def _delayed_fail(self, env, delay, p, tstep, node, network) -> Generator[simpy.Event, None, None]:
        yield env.timeout(delay)
        while True:
            if node.state == 0 and network.sim_state.rng.random() < p:
                network.mark_failed(env, node, network.sim_state)
            yield env.timeout(tstep)

register_failure_model("delayed_random", DelayedRandomFailure)

# 3. Custom strategy: random dispatch
class RandomDispatchStrategy(ResponseStrategy):
    """Assign failures to agents in random order, then use FCFS dispatcher."""
    def dispatch(self, env, network, service_team):
        network.sim_state.rng.shuffle(network.sim_state.failures)
        dispatcher(env, network, service_team)

register_strategy("random_dispatch", RandomDispatchStrategy)

# 4. Custom metric: peak failure count
class PeakFailureMetric(MetricCollector):
    @property
    def name(self):
        return "Peak_Failures"

    def collect(self, sim_state, network, service_team, env_now):
        return max(sim_state.failed_at_step) if sim_state.failed_at_step else 0

    def aggregate(self, values):
        m = statistics.mean(values)
        sd = statistics.pstdev(values)
        return {"Mean": m, "SEM": sd / math.sqrt(len(values)) if values else 0.0}

register_metric(PeakFailureMetric())

# Run with all customizations
config = tiedloc.SimulationConfig(
    topology="grid",
    topology_params={"num_of_nodes": 25},  # 5x5 grid
    failure_model="delayed_random",
    failure_params={"delay": 3.0, "p": 0.04, "fail_speed": 1.0, "initial_failures": 1},
    strategy="random_dispatch",
    strategy_params={"frequency": 1.0, "auxiliary": False},
    team_topology="complete",
    team_params={"team_members": 4, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=7, replications=5, processors=1, simulation_length=50,
)
results = tiedloc.run(config)
print(f"Combined — Recoverability: {results.stats['Recoverability']}")
print(f"Combined — Total Failures: {results.stats['Total_Failures']}")
# Custom metric appears in results automatically
if "Peak_Failures" in results.stats:
    print(f"Combined — Peak Failures: {results.stats['Peak_Failures']}")
```

### Programmatic API Reference

#### `SimulationConfig`

A dataclass that configures all aspects of a simulation.

```python
@dataclass
class SimulationConfig:
    topology: str | TopologyGenerator = "Barabasi Albert Scale-Free Network"
    topology_params: dict = ...   # e.g. {"num_of_nodes": 100, "new_node_to_existing_nodes": 3}
    failure_model: str | FailureModel = "Watts cascade"
    failure_params: dict = ...    # e.g. {"phi": 0.3, "fail_speed": 1.0, "initial_failures": 5}
    strategy: str | ResponseStrategy = "FCFS"
    strategy_params: dict = ...   # e.g. {"frequency": 1.0, "auxiliary": False}
    team_topology: str | TeamTopologyGenerator = "regular graph"
    team_params: dict = ...       # e.g. {"team_members": 10, "team_degree": 4, ...}
    seed: int | None = 42        # None for entropy mode (non-deterministic)
    replications: int = 10
    processors: int = 1
    simulation_length: int = 100
    extra_metrics: list[MetricCollector] = []  # Per-run custom metrics
```

**Constructors:**

- `SimulationConfig(...)` — Create directly with keyword arguments.
- `SimulationConfig.from_json(path: str)` — Load from a JSON config file.
- `SimulationConfig.from_dict(data: dict)` — Create from a parsed JSON dictionary matching the TIE/DLOC input format.

**Methods:**

- `to_parameters() -> dict` — Convert to the internal parameters dict format used by the simulation engine. This is called automatically by `run()`.

#### `SimulationResults`

A dataclass returned by `run()` and `sweep()`.

```python
@dataclass
class SimulationResults:
    stats: dict[str, Any]        # Aggregated statistics (Mean/SEM)
    samples: dict[str, list]     # Raw per-replication values
    config: SimulationConfig     # The config that produced these results
```

**Methods:**

- `to_json(path: str)` — Save aggregated stats to a JSON file.

#### `run(config: SimulationConfig) -> SimulationResults`

Run a simulation with the given configuration. This is the primary entry point.

The `run()` function:
1. Converts the config to internal parameters via `to_parameters()`
2. Creates a master RNG from `config.seed` that derives `_topo_seed` and `_dispatch_seed`
3. Generates the CPS network topology
4. Temporarily registers any `extra_metrics` from the config
5. Dispatches replications (parallel if `processors > 1`)
6. Aggregates results via `statistics()`
7. Unregisters `extra_metrics` in a `finally` block to prevent leaks

#### `sweep(base_config, param_name, values) -> list[SimulationResults]`

Run a parameter sweep, varying one parameter across multiple values.

- `base_config: SimulationConfig` — Base configuration (deep-copied for each value).
- `param_name: str` — Dot-separated parameter path. Supports 1 or 2 levels:
  - Top-level: `"seed"`, `"replications"`, `"simulation_length"`
  - Nested: `"failure_params.phi"`, `"topology_params.num_of_nodes"`, `"team_params.team_members"`
- `values: list` — List of values to sweep.

### Running a Parameter Sweep

```python
import tiedloc

config = tiedloc.SimulationConfig(
    topology="Barabasi Albert Scale-Free Network",
    topology_params={"num_of_nodes": 30, "new_node_to_existing_nodes": 2},
    failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 2},
    team_params={"team_members": 4, "team_degree": 2, "agent_travel_speed": 1.0,
                 "repairing_time": 2.0, "initial_allocation": "Centrality"},
    seed=42, replications=5, processors=1, simulation_length=50,
)

# Sweep over failure threshold
sweep_results = tiedloc.sweep(config, "failure_params.phi", [0.1, 0.2, 0.3, 0.4, 0.5])
for r in sweep_results:
    phi = r.config.failure_params["phi"]
    print(f"phi={phi}: Recoverability={r.stats['Recoverability']:.2f}, "
          f"Total Failures={r.stats['Total_Failures']}")
```

### Tips & Gotchas

1. **Registries are global.** Once you call `register_topology("my_topo", MyClass)`, it's available for the rest of the process. Register at module load time for consistency.

2. **Register classes, not instances** (except for metrics). The `register_topology()`, `register_failure_model()`, `register_strategy()`, `register_team_topology()`, and `register_repair_model()` functions all expect a class. Passing an instance triggers a deprecation warning and uses `type(instance)`. The exception is `register_metric()`, which takes an instance.

3. **Custom RepairModel is wired through DefaultAgentBehavior.** The default agent behavior calls `server.repair_model.execute_repair(...)`. To use a custom repair model, register it and set it on `Server` instances — or create a custom `AgentBehavior` that handles repair differently. Currently, repair models are set on the `Server` constructor (`repair_model=...`), which is created internally by `ServiceTeam.__init__()`.

4. **Seed hierarchy for reproducibility.** The RNG hierarchy is: `config.seed` -> `master_rng` -> `_topo_seed` (for graph generation) + `_dispatch_seed` -> per-worker `dispatch_rng` -> `rep_seeds[i]` -> per-worker `sim_state.rng`. This ensures each replication has an isolated RNG (`SimulationState.rng`) and the topology is generated deterministically. Set `seed=None` for entropy mode (non-deterministic runs).

5. **Use `sim_state.rng` in custom failure models**, not the global `random` module, to maintain per-worker RNG isolation and reproducibility. The `rng` attribute is a `random.Random` instance seeded per-replication.

6. **Aggregated results vs. raw samples.** `results.stats` contains aggregated metrics (typically `{"Mean": ..., "SEM": ...}` dicts). `results.samples` contains raw per-replication lists for each metric. Special cases: `Recoverability` is a float (0.0-1.0), not a Mean/SEM dict. `Median_Recovery_Time` is also a bare float.

7. **`processors > 1` uses multiprocessing.** Custom components must be picklable. Avoid lambda functions or closures in registered classes if you plan to use parallel replications.

8. **`grid_2d_graph` produces tuple node IDs.** Networkx's `grid_2d_graph` returns nodes as `(row, col)` tuples, which may not work with tiedloc's integer-based node ID system. Use `nx.convert_node_labels_to_integers()` if your generator produces non-integer node labels.

9. **`extra_metrics` vs `register_metric()`**. Use `extra_metrics` on `SimulationConfig` for per-run custom metrics that automatically clean up. Use `register_metric()` for global metrics that persist for the process lifetime. The `extra_metrics` approach uses `try/finally` to guarantee `unregister_metric()` is called even if the simulation fails.

## Input Reference

The input JSON has 5 required sections. All fields within each section are required unless noted.

### `CPS` — Network Topology

Defines the network graph that will be subject to cascading failures.

- **`name`** — Topology type. One of:
  - `"Barabasi Albert Scale-Free Network"` — Scale-free network (preferential attachment). Requires `new_node_to_existing_nodes`.
  - `"Watts-Strogatz Small-World Model"` — Small-world network (high clustering, short paths). Requires `average_degree`.
  - `"Binomial Graph"` — Erdos-Renyi random graph. Requires `average_degree`.
  - `"Power Grid of Western States of USA"` — Real-world topology loaded from `data/power.txt` (4941 nodes). No extra params needed.
  - `"edge_list"` — Load from an edge list file. Requires `edge_list_file` path.
  - Any custom topology name registered via `register_topology()`.
- **`num_of_nodes`** — Number of nodes in the network. Integer.
- **`new_node_to_existing_nodes`** — *(Barabasi-Albert only)* Each new node connects to this many existing nodes. Controls edge density. Integer.
- **`average_degree`** — *(Watts-Strogatz and Binomial only)* Average number of connections per node. Integer.

### `failure_model` — Cascade Behavior

Controls how failures propagate through the network.

- **`name`** — Cascade model. Built-in: `"Watts cascade"`. Can be any registered failure model name.
- **`phi`** — Failure threshold (0.0-1.0). A node fails when the fraction of its failed neighbors reaches this value. Lower = more aggressive cascades. Float.
- **`fail_speed`** — Time units between cascade propagation steps. Float.
- **`initial_failures`** — Number of nodes randomly failed at time 0. Integer.

### `simulation_param` — Run Configuration

- **`seed`** — Random seed for reproducibility. Integer. Omit or set to `null` for entropy mode.
- **`replications`** — Number of independent simulation runs. Results are averaged with SEM. Integer.
- **`processors`** — Number of parallel workers. Use `1` for sequential. Integer.
- **`simulation_length`** — Maximum simulation time in time units. The simulation stops at this point even if failures remain. Integer.

### `response_protocol` — Agent Dispatch Strategy

Controls how repair agents are assigned to failed elements.

- **`name`** — Dispatch strategy. Built-in options:
  - `"FCFS"` — First-come, first-served. Agents are assigned to failures in chronological order.
  - `"nearest"` — Nearest-neighbor. Each idle agent is assigned the closest failure by graph distance.
  - `"activity"` — Activity-based. Prioritizes failures whose neighbors are most at risk of cascading, then by distance.
  - Any custom strategy name registered via `register_strategy()`.
- **`frequency`** — How often (in time units) the dispatcher checks for new failures and available agents. Float.
- **`auxiliary`** — Whether to inject backup edges during the simulation to improve resilience. Boolean.
- **`aux_trigger_time`** — *(Optional, only when `auxiliary` is true)* Simulation time at which auxiliary edges are injected. Defaults to `5`. Float.

### `service_team` — Repair Agents

Defines the team of agents that travel through the network to repair failures.

- **`name`** — Agent communication topology. Built-in: `"regular graph"`, `"complete"`, `"star"`. Can be any registered team topology name.
- **`team_members`** — Number of repair agents. Integer.
- **`team_degree`** — Each agent's connectivity to other agents. Must be even and less than `team_members`. Integer.
- **`team_members` and `team_degree` constraint** — `team_degree` must satisfy `team_degree < team_members` and `team_degree` must be even (requirement of `networkx.random_regular_graph`).
- **`agent_travel_speed`** — Speed at which agents traverse network edges (hops per time unit). Float.
- **`repairing_time`** — Time units required to repair one failed element. Float.
- **`initial_allocation`** — Where agents start. Currently only `"Centrality"` is supported — agents are placed at nodes with the highest betweenness centrality.

## Output Reference

The simulator prints the input JSON, a log line per replication, then a results JSON. It also writes two files alongside the input: `<name>OUTPUT.json` (aggregated stats) and `<name>SAMPLES.json` (raw per-replication data).

### Result Fields

Most metrics report `Mean` and `SEM` (Standard Error of Mean) across replications.

**Aggregate metrics:**

- **`Total_Failures`** — Total number of elements (nodes + edges) that failed during the simulation.
- **`Recoverability`** — Fraction of replications where all failures were repaired before `simulation_length`. Range: 0.0-1.0.
- **`Preventability`** — Fraction of potential cascade events that were prevented by agents arriving in time. Computed as `prevented / (prevented + total_failures)`.
- **`Mean_Recovery_Time`** — Average time from first failure to full recovery, excluding replications that never recovered.
- **`Mean_Recovery_Time_withINF`** — Same as above, but unrecovered replications are counted as infinity.
- **`Median_Recovery_Time`** — Median recovery time across all replications (including infinity for unrecovered runs).
- **`Mean_Latency_QoS`** — Average wait time between a failure occurring and an agent arriving to begin repair. Lower is better.
- **`Total_Latency`** — Sum of all individual failure-to-repair wait times.
- **`Total_Distance_Traveled_by_Agent`** — Total graph hops traveled by all repair agents across the simulation.
- **`Aux_Lines`** — Number of auxiliary (backup) edges injected. Always 0 when `auxiliary` is false.

**Time series:**

- **`Step_000`, `Step_001`, ... `Step_NNN`** — Number of currently-failed elements at each simulation time step. This traces the cascade curve: rising as failures spread, falling as agents repair, reaching 0 when fully recovered. The number of steps equals `simulation_length`.

### Output Files

Given an input file `examples/small_ba.json`, the simulator writes:

- **`examples/small_baOUTPUT.json`** — Full aggregated statistics (same as the printed results JSON).
- **`examples/small_baSAMPLES.json`** — Raw per-replication values for `Total_Latency`, `Preventability`, `Total_Distance_Traveled_by_Agent`, and `Total_Failures`. Useful for custom analysis or plotting.

## Running Tests

```bash
pytest
```

Or with verbose output:

```bash
python -m pytest tests/ -v
```

## Development

### Linting and Type Checking (optional)

The following tools are recommended but not required (not installed by default):

```bash
# Install ruff (optional)
pip install ruff

# Run linter
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### Code Quality

The codebase follows these quality practices:

- **Named constants** (`constants.py`) for `STATE_OPERATIONAL`, `STATE_FAILED`, `STATE_IDLE`, `STATE_BUSY` — no magic numbers.
- **Per-worker RNG isolation** via `SimulationState.rng` — each replication gets its own `random.Random` instance seeded from the master RNG hierarchy.
- **Explicit allowlists** for dynamic attribute setting (no unconstrained `setattr`).
- **Context managers** for all file I/O operations.
- **Descriptive error messages** for `FileNotFoundError` and `ValueError` on file I/O failures in `networks.py` and `topologies.py`.
- **Iterative SimPy processes** using `while True` loops instead of recursive re-scheduling.
- **Generation counters** (`failures_generation`) for change detection instead of fragile `hash(str(...))`.
- **Configurable parameters** instead of magic numbers (e.g., `aux_trigger_time`).
- **JobRequest wrapper** for associating job parameters with SimPy resource requests.
- **Concrete type annotations** with `Generator` return types for SimPy processes.
- **Registry pattern** for extensibility (topologies, failure models, strategies, etc.).
- **`try/finally` cleanup** for `extra_metrics` registration in `run()`.
- **Refactored dispatch helpers** — `_init_agent_schedule`, `_assign_link_collaborators`, `_filter_schedule`, `_find_idle_agents`, `_build_idle_matrix`, `_nn_assign_link` extracted from monolithic functions for testability.

## Module Reference

| Module | Purpose |
|--------|---------|
| `__init__.py` | Package exports (`SimulationConfig`, `SimulationResults`, `run`, `sweep`) |
| `__main__.py` | `python -m tiedloc` entry point |
| `main.py` | CLI entry point with argument parsing |
| `api.py` | Library API — `SimulationConfig`, `SimulationResults`, `run()`, `sweep()` |
| `agents.py` | Core agent models — `Node`, `Link`, `Server` |
| `agent_behaviors.py` | `AgentBehavior` ABC + `DefaultAgentBehavior` |
| `constants.py` | Named constants — `STATE_OPERATIONAL`, `STATE_FAILED`, `STATE_IDLE`, `STATE_BUSY` |
| `failure_models.py` | `FailureModel` ABC + `WattsCascadeModel` + registry |
| `metrics.py` | `MetricCollector` ABC + 6 built-in metrics + `register_metric()`/`unregister_metric()` |
| `networks.py` | `CPSNetwork`, `ServiceTeam`, `SimulationState` — network graph models with composition |
| `repair_models.py` | `RepairModel` ABC + `FixedRepairModel` + registry |
| `responsestrategies.py` | Dispatch logic — `dispatcher()`, `nn_schedule()`, `JobRequest`, priority helpers, auxiliary edge injection |
| `simulations.py` | Simulation engine — `parallel_dispatch()`, `worker()`, `statistics()`, `save_results()` |
| `strategies.py` | `ResponseStrategy` ABC + `FCFSStrategy`/`NearestStrategy`/`ActivityStrategy` + registry |
| `team_topologies.py` | `TeamTopologyGenerator` protocol + `RegularGraphTeam`/`CompleteGraphTeam`/`StarTeam` + registry |
| `topologies.py` | `TopologyGenerator` protocol + 5 built-in generators + registry |

## Project Structure

```
tiedloc/
├── pyproject.toml
├── LICENSE
├── README.md
├── examples/
│   ├── small_ba.json               # Barabasi-Albert, FCFS
│   ├── small_ws.json               # Watts-Strogatz, nearest
│   ├── activity_ba.json            # Barabasi-Albert, activity
│   └── auxiliary_ba.json           # Barabasi-Albert, auxiliary edges
├── src/
│   └── tiedloc/
│       ├── __init__.py              # Package exports (SimulationConfig, run, sweep)
│       ├── __main__.py              # python -m tiedloc entry point
│       ├── main.py                  # CLI entry point
│       ├── api.py                   # Library API (SimulationConfig, SimulationResults, run, sweep)
│       ├── agents.py                # Node, Link, and Server agent models
│       ├── agent_behaviors.py       # AgentBehavior ABC + default behavior
│       ├── constants.py             # Named constants (STATE_OPERATIONAL, STATE_FAILED, etc.)
│       ├── failure_models.py        # FailureModel ABC + Watts cascade
│       ├── metrics.py               # MetricCollector ABC + built-in metrics + unregister
│       ├── networks.py              # CPSNetwork, ServiceTeam, SimulationState (with rng)
│       ├── repair_models.py         # RepairModel ABC + fixed repair time
│       ├── responsestrategies.py    # Dispatch logic, helpers, and response protocols
│       ├── simulations.py           # Simulation engine and statistics
│       ├── strategies.py            # ResponseStrategy ABC + FCFS/nearest/activity
│       ├── team_topologies.py       # TeamTopologyGenerator protocol + built-ins
│       └── topologies.py            # TopologyGenerator protocol + built-in generators
└── tests/
    ├── __init__.py                  # Package marker
    ├── conftest.py                  # Shared test fixtures (metric registry preservation)
    ├── test_agents.py               # Agent model tests
    ├── test_agent_behaviors.py      # Agent behavior tests
    ├── test_api.py                  # API tests (config, run, sweep, extra_metrics)
    ├── test_failure_models.py       # Failure model registry tests
    ├── test_integration.py          # End-to-end simulation tests (all strategies + reproducibility)
    ├── test_metrics.py              # Metric collector tests
    ├── test_networks.py             # Network model tests (pack, pack(full=True), error handling)
    ├── test_repair_models.py        # Repair model tests
    ├── test_responsestrategies.py   # Strategy dispatch tests (helpers, dispatcher, nn_schedule)
    ├── test_simulations.py          # Simulation engine tests
    ├── test_strategies.py           # Response strategy registry tests
    ├── test_team_topologies.py      # Team topology tests
    └── test_topologies.py           # Network topology tests (including error handling)
```

## License

Apache-2.0
