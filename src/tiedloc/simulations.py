"""Simulation engine for the tiedloc simulator.

This module manages the discrete-event simulation lifecycle, including
parallel dispatch of replications, statistical aggregation, and result
output. The simulation state is encapsulated in SimulationState rather
than being monkey-patched onto the SimPy environment.
"""

from __future__ import annotations

import copy
import math
import json
import multiprocessing
import os
import random
import statistics as pystats
from collections.abc import Generator
from typing import Any

import simpy

from tiedloc import responsestrategies
from tiedloc.constants import STATE_FAILED
from tiedloc.metrics import get_all_metrics
from tiedloc.networks import CPSNetwork, ServiceTeam, SimulationState

_BUILTIN_STAT_NAMES = frozenset({
    "Total_Latency", "Mean_Latency_QoS", "Mean_Recovery_Time",
    "Mean_Recovery_Time_withINF", "Total_Failures", "Preventability",
    "Total_Distance_Traveled_by_Agent", "Aux_Lines",
})


def create_simulation_state() -> SimulationState:
    """Create a fresh SimulationState for a new simulation run.

    Returns:
        A new SimulationState instance.
    """
    return SimulationState()


def sim_each_time_unit(env: simpy.Environment, sim_state: SimulationState) -> Generator[simpy.Event, None, None]:
    """SimPy process: record the number of failed elements at each time step.

    Args:
        env: The SimPy simulation environment.
        sim_state: The simulation state to record into.
    """
    while True:
        sim_state.failed_at_step.append(sim_state.failed_elements)
        yield env.timeout(1)


def start(env: simpy.Environment, parameters: dict, rep_num: int) -> None:
    """Run a single simulation replication.

    Args:
        env: The SimPy simulation environment.
        parameters: Simulation parameters dictionary.
        rep_num: The replication number (for logging).
    """
    print(f"Start of Simulation replication {rep_num}")
    env.run(until=float(parameters["simulation_param"]["simulation_length"]))


def parallel_dispatch(parameters: dict, network: CPSNetwork) -> list[dict]:
    """Dispatch simulation replications in parallel.

    Args:
        parameters: Simulation parameters dictionary.
        network: The CPS network instance (will be serialized for workers).

    Returns:
        A list of result dictionaries, one per replication.
    """
    num_reps = int(parameters["simulation_param"]["replications"])
    dispatch_seed = parameters.get("_dispatch_seed")
    dispatch_rng = random.Random(dispatch_seed)
    rep_seeds = [dispatch_rng.getrandbits(64) for _ in range(num_reps)]
    parameters["gpack"] = network.pack()

    num_procs = int(parameters["simulation_param"]["processors"])
    args_list = [(parameters, rep_seed, repnum) for repnum, rep_seed in enumerate(rep_seeds)]

    if num_procs > 1:
        with multiprocessing.Pool(processes=num_procs) as pool:
            return pool.map(worker, args_list)
    else:
        return list(map(worker, args_list))


def worker(args: tuple[dict, int, int]) -> dict:
    """Execute a single simulation replication.

    Args:
        args: A tuple of (parameters, random_seed, replication_number).

    Returns:
        A dictionary containing the simulation results for this replication.
    """
    parameters, rep_seed, repnum = args
    parameters = copy.deepcopy(parameters)

    env = simpy.Environment()
    sim_state = create_simulation_state()

    worker_rng = random.Random(rep_seed)
    sim_state.rng = random.Random(worker_rng.getrandbits(64))

    cps_network = CPSNetwork(parameters["gpack"], sim_state=sim_state)
    cps_network.initial_failures(env, parameters)

    parameters["service_team"]["seed"] = worker_rng.getrandbits(64)
    service_team = ServiceTeam(parameters, env)
    service_team.allocation(parameters, cps_network)

    cps_network.failure_model(parameters, env, service_team)

    env.process(responsestrategies.init(env, parameters, cps_network, service_team))
    env.process(sim_each_time_unit(env, sim_state))

    start(env, parameters, repnum)

    # Account for failures that were never repaired
    for node_id in cps_network.graph.nodes:
        node = cps_network.get_node(node_id)
        if node.state == STATE_FAILED:
            sim_state.total_latency += env.now - node.failed_time

    for v1, v2 in cps_network.graph.edges:
        link = cps_network.get_link(v1, v2)
        if link is not None and link.state == STATE_FAILED:
            sim_state.total_latency += env.now - link.failed_time

    env_pack = {
        "total_latency": sim_state.total_latency,
        "latency": sim_state.latency,
        "total_failures": sim_state.total_failures,
        "prevented": sim_state.prevented,
        "total_distance": sim_state.total_distance,
        "aux_lines": sim_state.aux_lines,
        "failed_at_step": sim_state.failed_at_step,
    }

    # Collect custom metrics from registered collectors (skip built-ins)
    custom_metrics = {}
    for collector in get_all_metrics():
        if collector.name not in _BUILTIN_STAT_NAMES:
            custom_metrics[collector.name] = collector.collect(
                sim_state, cps_network, service_team, env.now
            )
    env_pack["custom_metrics"] = custom_metrics

    del cps_network, service_team
    return env_pack


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

def statistics(results: list[dict]) -> tuple[dict, dict]:
    """Aggregate results across all replications.

    Args:
        results: A list of result dictionaries from worker processes.

    Returns:
        A tuple of (stat_results, samples) where stat_results contains
        aggregated statistics and samples contains raw per-replication data.
    """
    stats: dict[str, list] = {
        "Total_Latency": [],
        "Mean_Latency_QoS": [],
        "Mean_Recovery_Time": [],
        "Mean_Recovery_Time_withINF": [],
        "Total_Failures": [],
        "Preventability": [],
        "Total_Distance_Traveled_by_Agent": [],
        "Aux_Lines": [],
    }

    if results and results[0]["failed_at_step"]:
        for i in range(len(results[0]["failed_at_step"])):
            stats[f"Step_{str(i).zfill(3)}"] = []

    recoverability = 0

    for env_pack in results:
        stats["Total_Latency"].append(env_pack["total_latency"])
        stats["Mean_Latency_QoS"] += env_pack["latency"]
        stats["Total_Failures"].append(env_pack["total_failures"])

        total_events = env_pack["prevented"] + env_pack["total_failures"]
        if total_events > 0:
            stats["Preventability"].append(env_pack["prevented"] / total_events)
        else:
            stats["Preventability"].append(0.0)

        stats["Total_Distance_Traveled_by_Agent"].append(env_pack["total_distance"])
        stats["Aux_Lines"].append(env_pack["aux_lines"])

        for i, x in enumerate(env_pack["failed_at_step"]):
            key = f"Step_{str(i).zfill(3)}"
            if key in stats:
                stats[key].append(x)

        recovery_flag = find_recovery_time(env_pack["failed_at_step"])
        stats["Mean_Recovery_Time_withINF"].append(recovery_flag)
        if not math.isinf(recovery_flag):
            stats["Mean_Recovery_Time"].append(recovery_flag)
            recoverability += 1

    num_reps = len(stats["Total_Latency"]) if stats["Total_Latency"] else 1

    stat_results: dict[str, Any] = {
        "Recoverability": recoverability / num_reps,
        "Median_Recovery_Time": float(pystats.median(stats["Mean_Recovery_Time_withINF"]))
        if stats["Mean_Recovery_Time_withINF"]
        else 0.0,
    }

    for key, val in stats.items():
        if val:
            finite_val = [v for v in val if not math.isinf(v)]
            if finite_val:
                m = pystats.mean(finite_val)
                sd = pystats.pstdev(finite_val)
                stat_results[key] = {
                    "Mean": float(m),
                    "SEM": float(sd / math.sqrt(len(finite_val))),
                }
            else:
                stat_results[key] = {"Mean": float("inf"), "SEM": 0.0}
        else:
            stat_results[key] = {"Mean": 0.0, "SEM": 0.0}

    # Aggregate custom metrics from registered collectors
    for collector in get_all_metrics():
        if collector.name not in stat_results:
            per_rep_values = [
                ep["custom_metrics"][collector.name]
                for ep in results
                if "custom_metrics" in ep and collector.name in ep["custom_metrics"]
            ]
            if per_rep_values:
                stat_results[collector.name] = collector.aggregate(per_rep_values)

    return (stat_results, stats)


def find_recovery_time(failed_list: list[int]) -> float:
    """Estimate the recovery time from a time series of failure counts.

    Args:
        failed_list: A list of failure counts at each time step.

    Returns:
        The estimated recovery time, or math.inf if recovery is not achieved.
    """
    if not failed_list:
        return 0.0
    k = next((i for i, val in enumerate(failed_list) if val == 0), -1)
    if k == -1:
        max_val = max(failed_list)
        max_list = [i for i, x in enumerate(failed_list) if x == max_val]
        if max_list and max_list[-1] == len(failed_list) - 1:
            k = math.inf
        else:
            if max_list:
                segment_x = list(range(max_list[-1], len(failed_list)))
                segment_y = failed_list[max_list[-1] : len(failed_list)]
                n = len(segment_x)
                if n < 3:
                    return math.inf
                # Linear regression (degree 1): y = a*x + b
                sx = sum(segment_x)
                sy = sum(segment_y)
                sxy = sum(x * y for x, y in zip(segment_x, segment_y))
                sx2 = sum(x * x for x in segment_x)
                denom = n * sx2 - sx * sx
                if denom == 0:
                    k = math.inf
                else:
                    a = (n * sxy - sx * sy) / denom
                    b = (sy - a * sx) / n
                    if a >= 0:
                        k = math.inf
                    else:
                        k = -b / a
            else:
                k = math.inf
    return float(k)


def save_results(
    path_to_input_file: str, stat_results: dict, samples: dict
) -> None:
    """Save simulation results to JSON files.

    Args:
        path_to_input_file: Path to the original input file (used to derive output paths).
        stat_results: Aggregated statistics dictionary.
        samples: Raw per-replication samples dictionary.
    """
    base, ext = os.path.splitext(path_to_input_file)

    stat_path = f"{base}OUTPUT{ext}"
    with open(stat_path, "w") as outfile:
        outfile.write(json.dumps(stat_results, sort_keys=True, indent=4, separators=(",", ": ")))

    sample_path = f"{base}SAMPLES{ext}"
    out_samples = {
        key: val
        for key, val in samples.items()
        if key
        in (
            "Total_Latency",
            "Preventability",
            "Total_Distance_Traveled_by_Agent",
            "Total_Failures",
        )
    }
    with open(sample_path, "w") as outfile:
        outfile.write(json.dumps(out_samples, sort_keys=True, indent=4, separators=(",", ": ")))
