"""Pluggable network topology generators for the tiedloc simulator.

This module defines the TopologyGenerator protocol and a registry of built-in
generators. Users can register custom generators by name for use in JSON configs,
or pass generator instances directly via the Python API.
"""

from __future__ import annotations

import os
import warnings
from typing import Protocol, runtime_checkable

import networkx as nx


@runtime_checkable
class TopologyGenerator(Protocol):
    """Generates a networkx Graph from configuration parameters."""

    def generate(self, params: dict) -> nx.Graph:
        """Create and return a network graph.

        Args:
            params: The "CPS" section of the input config dict.

        Returns:
            A networkx.Graph instance.
        """
        ...


# --- Registry ---

_TOPOLOGY_REGISTRY: dict[str, type[TopologyGenerator]] = {}


def register_topology(name: str, generator: type | TopologyGenerator) -> None:
    """Register a topology generator class (or instance for backward compat) by name."""
    if isinstance(generator, type):
        _TOPOLOGY_REGISTRY[name] = generator
    else:
        warnings.warn(
            "Passing an instance to register_topology() is deprecated. "
            "Pass the class instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _TOPOLOGY_REGISTRY[name] = type(generator)


def get_topology(name: str) -> TopologyGenerator:
    """Look up and instantiate a registered topology generator."""
    if name not in _TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology '{name}'. "
            f"Available: {list(_TOPOLOGY_REGISTRY.keys())}"
        )
    return _TOPOLOGY_REGISTRY[name]()


# --- Built-in Generators ---


class BarabasiAlbertGenerator:
    """Barabasi-Albert scale-free network via preferential attachment."""

    def generate(self, params: dict) -> nx.Graph:
        return nx.barabasi_albert_graph(
            int(params["num_of_nodes"]),
            int(params["new_node_to_existing_nodes"]),
            seed=params.get("_topo_seed"),
        )


class WattsStrogatzGenerator:
    """Watts-Strogatz small-world network."""

    def generate(self, params: dict) -> nx.Graph:
        return nx.watts_strogatz_graph(
            int(params["num_of_nodes"]),
            int(params["average_degree"]),
            float(params.get("rewiring_prob", 0.3)),
            seed=params.get("_topo_seed"),
        )


class BinomialGenerator:
    """Erdos-Renyi binomial random graph."""

    def generate(self, params: dict) -> nx.Graph:
        p = params["average_degree"] / params["num_of_nodes"]
        return nx.binomial_graph(params["num_of_nodes"], p, seed=params.get("_topo_seed"))


class PowerGridGenerator:
    """Load the Western US power grid from data/power.txt."""

    def generate(self, params: dict) -> nx.Graph:
        num_nodes = int(params.get("num_of_nodes", 4941))
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        link_flag = 0
        from_node = -1
        data_file = params.get("data_file", os.path.join("data", "power.txt"))
        try:
            with open(data_file, "r") as fobj:
                for line in fobj:
                    try:
                        line_val = int(line)
                    except ValueError:
                        continue
                    if link_flag == 0:
                        from_node = line_val
                        link_flag = 1
                    else:
                        link_flag = 0
                        graph.add_edge(from_node, line_val)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Power grid data file not found: {data_file}"
            )
        return graph


class EdgeListGenerator:
    """Load a graph from an edge list file."""

    def generate(self, params: dict) -> nx.Graph:
        path = params["edge_list_file"]
        try:
            return nx.read_edgelist(path, nodetype=int)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Edge list file not found: {path}"
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to parse edge list file {path}: {exc}"
            ) from exc


# Register defaults
register_topology("Barabasi Albert Scale-Free Network", BarabasiAlbertGenerator)
register_topology("Watts-Strogatz Small-World Model", WattsStrogatzGenerator)
register_topology("Binomial Graph", BinomialGenerator)
register_topology("Power Grid of Western States of USA", PowerGridGenerator)
register_topology("edge_list", EdgeListGenerator)
