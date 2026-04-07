"""Network models for the tiedloc simulator.

This module defines the network graph structures used in the simulation:
- CPSNetwork: The Cyber-Physical System network subject to cascading failures.
- ServiceTeam: The team of repair agents organized as a graph.
- SimulationState: Encapsulated state for a single simulation run.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import simpy

from tiedloc.agents import Link, Node, Server
from tiedloc.constants import STATE_FAILED, STATE_OPERATIONAL
from tiedloc.failure_models import get_failure_model
from tiedloc.team_topologies import get_team_topology
from tiedloc.topologies import get_topology


@dataclass
class SimulationState:
    """Encapsulates all mutable simulation state previously monkey-patched onto the SimPy env.

    Attributes:
        failed_elements: Current count of failed elements.
        failures: List of currently failed Node/Link objects.
        prevented: Count of prevented failures.
        aux_lines: Count of auxiliary lines added.
        total_failures: Cumulative failure count.
        failed_at_step: List tracking failures at each time step.
        total_distance: Total distance traveled by agents.
        total_latency: Cumulative latency of all failures.
        latency: List of individual latency values.
        new_failures_event: SimPy event for signaling new failures.
        agent_schedule: Per-agent job schedules.
    """

    failed_elements: int = 0
    failures: list[Node | Link] = field(default_factory=list)
    failures_generation: int = 0
    prevented: int = 0
    aux_lines: int = 0
    aux_failures: int = 0
    total_failures: int = 0
    failed_at_step: list[int] = field(default_factory=list)
    total_distance: float = 0.0
    total_latency: float = 0.0
    latency: list[float] = field(default_factory=list)
    new_failures_event: simpy.Event | None = None
    agent_schedule: dict[int, list[dict]] | None = None
    rng: random.Random = field(default_factory=lambda: random.Random())


_CPS_ALLOWED_KEYS = frozenset({
    "name", "num_of_nodes", "average_degree", "new_node_to_existing_nodes",
    "bctlist", "distmat", "aux_threshold", "betweenness_centrality",
    "distance_matrix", "edge_list_file",
})

_TEAM_ALLOWED_KEYS = frozenset({
    "name", "team_members", "team_degree",
    "agent_travel_speed", "repairing_time", "initial_allocation",
})


class CPSNetwork:
    """Cyber-Physical System network using composition over inheritance.

    Instead of inheriting from networkx.Graph, this class contains a graph
    instance and delegates graph operations to it. Domain logic (failure models,
    centrality, distance matrices) is kept separate from the graph data structure.

    Attributes:
        graph: The underlying networkx Graph instance.
        nodes_map: Mapping of node IDs to Node agent objects.
        links_map: Mapping of (v1, v2) tuples to Link agent objects.
        bctlist: Betweenness centrality list sorted in descending order.
        distmat: All-pairs shortest path length dictionary.
        name: Name/type of the network.
        num_of_nodes: Number of nodes in the network.
        average_degree: Average degree of the network.
        aux_threshold: Auxiliary threshold values per node.
        sim_state: Reference to the current simulation state.
    """

    def __init__(self, parameters: dict, sim_state: SimulationState | None = None) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.nodes_map: dict[int, Node] = {}
        self.links_map: dict[tuple[int, int], Link] = {}
        self.bctlist: list[list[float | int]] = []
        self.distmat: dict[int, dict[int, float]] = {}
        self.name: str = ""
        self.num_of_nodes: int = 0
        self.average_degree: float = 0.0
        self.aux_threshold: list[float] = []
        self.sim_state: SimulationState = sim_state or SimulationState()

        cps_params = parameters.get("CPS", parameters)
        for key, val in cps_params.items():
            if key in _CPS_ALLOWED_KEYS:
                setattr(self, key, val)

        # Capture topology seed from top-level parameters for graph generation
        self._topo_seed = parameters.get("_topo_seed")

        if "node_link_data" in cps_params:
            self.graph = nx.node_link_graph(cps_params["node_link_data"])
            self.num_of_nodes = self.graph.number_of_nodes()
        elif "picklePack_edge_data" in cps_params:
            self.graph.add_nodes_from(range(self.num_of_nodes))
            self.graph.add_edges_from(cps_params["picklePack_edge_data"])
        else:
            self._load_data()
            self._load_dist_mat()
            self._calc_centrality()
            if parameters.get("response_protocol", {}).get("auxiliary", False):
                self._calc_aux()
            degrees = [d for _, d in self.graph.degree()]
            if degrees:
                self.average_degree = sum(degrees) / len(degrees)

        # Create agent objects for each node and edge
        for node_id in self.graph.nodes:
            self.nodes_map[node_id] = Node(node_id)

        for v1, v2 in self.graph.edges:
            link = Link(v1, v2)
            self.links_map[(v1, v2)] = link

    def get_node(self, node_id: int) -> Node:
        """Retrieve the Node agent for a given node ID."""
        return self.nodes_map[node_id]

    def get_link(self, v1: int, v2: int) -> Link | None:
        """Retrieve the Link agent for a given edge, checking both directions."""
        if (v1, v2) in self.links_map:
            return self.links_map[(v1, v2)]
        if (v2, v1) in self.links_map:
            return self.links_map[(v2, v1)]
        return None

    def pack(self, *, full: bool = False) -> dict:
        """Serialize the network state for multiprocessing transfer.

        Args:
            full: When True, use ``nx.node_link_data`` to preserve all graph
                metadata (node attributes, edge attributes, etc.) across process
                boundaries.  The default (False) uses the lightweight edge-list
                path which is faster but only preserves topology.

        Returns:
            A dictionary containing the network parameters needed to reconstruct
            the network in a worker process.
        """
        pack_data: dict[str, Any] = {}

        if full:
            pack_data["node_link_data"] = nx.node_link_data(self.graph)
        else:
            pack_data["picklePack_edge_data"] = list(self.graph.edges)

        for attr in ("aux_threshold", "average_degree", "distmat", "name", "num_of_nodes", "bctlist"):
            val = getattr(self, attr, None)
            if val is not None:
                pack_data[attr] = val
        return {"CPS": pack_data}

    def _calc_centrality(self) -> None:
        """Calculate betweenness centrality for all nodes."""
        if not self.bctlist:
            if hasattr(self, "betweenness_centrality"):
                bct_path = os.path.join("data", self.betweenness_centrality)
                try:
                    with open(bct_path, "r") as infile:
                        for line in infile:
                            nums = line.split(" ")
                            self.bctlist.append([float(nums[0]), int(nums[1])])
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Betweenness centrality file not found: {bct_path}"
                    )
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Malformed betweenness centrality file {bct_path}: {exc}"
                    ) from exc
            else:
                bct = nx.betweenness_centrality(self.graph)
                self.bctlist = [[nval, nid] for nid, nval in bct.items()]
                self.bctlist.sort(reverse=True)

    def _calc_aux(self) -> None:
        """Calculate auxiliary thresholds for each node."""
        clustering = nx.clustering(self.graph)
        degrees = dict(self.graph.degree())
        self.aux_threshold = [
            clustering.get(n, 0) * degrees.get(n, 0)
            for n in self.graph.nodes
        ]

    def failure_model(
        self,
        parameters: dict,
        env: simpy.Environment,
        service_team: ServiceTeam,
    ) -> None:
        """Initialize the failure cascade model for all nodes and edges.

        Args:
            parameters: Simulation parameters dictionary.
            env: The SimPy simulation environment.
            service_team: The service team for preventability checks.
        """
        fm = get_failure_model(parameters["failure_model"]["name"])
        fm.install(env, self, service_team, parameters["failure_model"])

    def initial_failures(
        self, env: simpy.Environment, parameters: dict
    ) -> None:
        """Seed initial failures, delegating to the failure model when available.

        If the configured failure model overrides ``seed_initial_failures()``,
        that override is used.  Otherwise the default random-node selection
        from ``FailureModel.seed_initial_failures()`` runs.

        Args:
            env: The SimPy simulation environment.
            parameters: Simulation parameters dictionary.
        """
        fm_params = parameters.get("failure_model", {})
        fm_name = fm_params.get("name")
        if fm_name is not None:
            fm = get_failure_model(fm_name)
            fm.seed_initial_failures(env, self, fm_params)
        else:
            # Fallback: default random-node selection
            total_nodes = self.graph.number_of_nodes()
            num_initial = int(fm_params.get("initial_failures", 0))
            for node_id in self.sim_state.rng.sample(range(total_nodes), num_initial):
                self.mark_failed(env, self.nodes_map[node_id], self.sim_state)

    def mark_failed(
        self, env: simpy.Environment, element: Node | Link, sim_state: SimulationState
    ) -> None:
        """Mark an element (node or link) as failed.

        Args:
            env: The SimPy simulation environment.
            element: The Node or Link that has failed.
            sim_state: The simulation state to update.
        """
        element.state = STATE_FAILED
        element.failed_time = env.now
        sim_state.failed_elements += 1
        sim_state.failures.append(element)
        sim_state.total_failures += 1
        sim_state.failures_generation += 1

    def mark_restored(
        self, env: simpy.Environment, element: Node | Link, sim_state: SimulationState
    ) -> None:
        """Mark an element as restored after repair.

        Args:
            env: The SimPy simulation environment.
            element: The Node or Link that has been restored.
            sim_state: The simulation state to update.
        """
        element.state = STATE_OPERATIONAL
        latency = env.now - element.failed_time
        sim_state.latency.append(latency)
        sim_state.total_latency += latency
        sim_state.failed_elements -= 1

    def _load_data(self) -> None:
        """Load or generate the network graph based on the configured name."""
        generator = get_topology(self.name)
        cps_params = {"num_of_nodes": self.num_of_nodes}
        # Pass through all attributes the generator might need
        for attr in ("new_node_to_existing_nodes", "average_degree", "edge_list_file", "_topo_seed"):
            if hasattr(self, attr):
                cps_params[attr] = getattr(self, attr)
        saved_name = self.name
        self.graph = generator.generate(cps_params)
        # Always update num_of_nodes from the generated graph
        self.num_of_nodes = self.graph.number_of_nodes()
        self.name = saved_name

    def _load_dist_mat(self) -> None:
        """Load or calculate the all-pairs shortest path distance matrix."""
        if hasattr(self, "distance_matrix"):
            dist_path = os.path.join("data", self.distance_matrix)
            try:
                with open(dist_path, "r") as infile:
                    line_id = 0
                    for line in infile:
                        self.distmat[line_id] = {}
                        nums = line.split(" ")
                        for key, x in enumerate(nums):
                            try:
                                self.distmat[line_id][key] = float(x)
                            except ValueError:
                                pass
                        line_id += 1
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Distance matrix file not found: {dist_path}"
                )
        else:
            raw_dist = dict(nx.all_pairs_shortest_path_length(self.graph))
            # Convert to nested dict and fill missing entries
            max_length = 0
            for src in raw_dist:
                for dst in raw_dist[src]:
                    if raw_dist[src][dst] > max_length:
                        max_length = raw_dist[src][dst]

            self.distmat = {}
            for a in self.graph.nodes:
                self.distmat[a] = {}
                for b in self.graph.nodes:
                    if a in raw_dist and b in raw_dist[a]:
                        self.distmat[a][b] = raw_dist[a][b]
                    else:
                        self.distmat[a][b] = max_length


class ServiceTeam:
    """A team of repair agents organized as a graph using composition.

    Attributes:
        graph: The underlying networkx Graph for agent connectivity.
        servers: Mapping of agent IDs to Server objects.
        last_dispatch: Timestamp of the last dispatch event.
        name: Name/type of the service team graph.
    """

    def __init__(self, parameters: dict, env: simpy.Environment) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.servers: dict[int, Server] = {}
        self.last_dispatch: float = -1.0
        self.name: str = ""

        team_params = parameters.get("service_team", {})
        for key, val in team_params.items():
            if key in _TEAM_ALLOWED_KEYS:
                setattr(self, key, val)

        travel_speed = team_params.get("agent_travel_speed", 1.0)
        repair_time = team_params.get("repairing_time", 1.0)
        agent_capacity = int(team_params.get("agent_capacity", 1))

        generator = get_team_topology(self.name)
        self.graph = generator.generate(team_params)

        for node_id in self.graph.nodes:
            self.servers[node_id] = Server(
                node_id, env, travel_speed, repair_time, capacity=agent_capacity
            )

    def get_server(self, server_id: int) -> Server:
        """Retrieve the Server agent for a given ID."""
        return self.servers[server_id]

    def allocation(self, parameters: dict, cps_network: CPSNetwork) -> None:
        """Allocate servers to CPS network nodes based on the configured strategy.

        Args:
            parameters: Simulation parameters dictionary.
            cps_network: The CPS network to allocate servers to.
        """
        method = parameters["service_team"].get("initial_allocation", "")
        if method == "Centrality":
            self._centrality_allocation(parameters, cps_network)

    def _centrality_allocation(
        self, parameters: dict, cps_network: CPSNetwork
    ) -> None:
        """Allocate servers based on betweenness centrality ranking."""
        cps_network._calc_centrality()
        num_cps_nodes = cps_network.graph.number_of_nodes()
        for node_id in self.graph.nodes:
            target_idx = node_id % num_cps_nodes
            target_node_id = int(cps_network.bctlist[target_idx][1])
            target_node = cps_network.get_node(target_node_id)
            self.servers[node_id].allocate_to(target_node)
