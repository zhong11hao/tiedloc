"""Tests for tiedloc.api — SimulationConfig, run, and sweep."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from tiedloc.api import SimulationConfig, SimulationResults, run, sweep
from tiedloc.failure_models import WattsCascadeModel
from tiedloc.metrics import MetricCollector, get_all_metrics
from tiedloc.strategies import FCFSStrategy
from tiedloc.team_topologies import CompleteGraphTeam
from tiedloc.topologies import BarabasiAlbertGenerator


class TestSimulationConfigDefaults:
    def test_default_topology(self):
        cfg = SimulationConfig()
        assert cfg.topology == "Barabasi Albert Scale-Free Network"

    def test_default_failure_model(self):
        cfg = SimulationConfig()
        assert cfg.failure_model == "Watts cascade"

    def test_default_strategy(self):
        cfg = SimulationConfig()
        assert cfg.strategy == "FCFS"

    def test_default_team_topology(self):
        cfg = SimulationConfig()
        assert cfg.team_topology == "regular graph"

    def test_default_replications(self):
        cfg = SimulationConfig()
        assert cfg.replications == 10

    def test_default_seed(self):
        cfg = SimulationConfig()
        assert cfg.seed == 42

    def test_default_simulation_length(self):
        cfg = SimulationConfig()
        assert cfg.simulation_length == 100


class TestSimulationConfigFromDict:
    def _sample_dict(self):
        return {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 15,
                "new_node_to_existing_nodes": 2,
            },
            "failure_model": {
                "name": "Watts cascade",
                "phi": 0.25,
                "fail_speed": 1.0,
                "initial_failures": 1,
            },
            "response_protocol": {
                "name": "FCFS",
                "frequency": 1.0,
                "auxiliary": False,
            },
            "service_team": {
                "name": "complete",
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            "simulation_param": {
                "seed": "99",
                "replications": "2",
                "processors": "1",
                "simulation_length": "30",
            },
        }

    def test_from_dict_topology(self):
        cfg = SimulationConfig.from_dict(self._sample_dict())
        assert cfg.topology == "Barabasi Albert Scale-Free Network"

    def test_from_dict_replications(self):
        cfg = SimulationConfig.from_dict(self._sample_dict())
        assert cfg.replications == 2

    def test_from_dict_seed(self):
        cfg = SimulationConfig.from_dict(self._sample_dict())
        assert cfg.seed == 99

    def test_from_dict_simulation_length(self):
        cfg = SimulationConfig.from_dict(self._sample_dict())
        assert cfg.simulation_length == 30


class TestSimulationConfigFromJson:
    def test_roundtrip_via_file(self):
        data = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "failure_model": {"name": "Watts cascade", "phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            "response_protocol": {"name": "FCFS", "frequency": 1.0, "auxiliary": False},
            "service_team": {
                "name": "complete",
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            "simulation_param": {"seed": "42", "replications": "1", "processors": "1", "simulation_length": "20"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmppath = f.name
        try:
            cfg = SimulationConfig.from_json(tmppath)
            assert cfg.topology == "Barabasi Albert Scale-Free Network"
            assert cfg.seed == 42
        finally:
            os.unlink(tmppath)


class TestSimulationConfigToParameters:
    def test_produces_valid_structure(self):
        cfg = SimulationConfig()
        params = cfg.to_parameters()
        assert "CPS" in params
        assert "failure_model" in params
        assert "response_protocol" in params
        assert "service_team" in params
        assert "simulation_param" in params

    def test_name_fields_present(self):
        cfg = SimulationConfig()
        params = cfg.to_parameters()
        assert params["CPS"]["name"] == "Barabasi Albert Scale-Free Network"
        assert params["failure_model"]["name"] == "Watts cascade"
        assert params["response_protocol"]["name"] == "FCFS"
        assert params["service_team"]["name"] == "regular graph"

    def test_simulation_param_are_strings(self):
        cfg = SimulationConfig()
        params = cfg.to_parameters()
        for key in ("seed", "replications", "processors", "simulation_length"):
            assert isinstance(params["simulation_param"][key], str)

    def test_with_instance_topology(self):
        gen = BarabasiAlbertGenerator()
        cfg = SimulationConfig(topology=gen)
        params = cfg.to_parameters()
        assert params["CPS"]["name"] == "BarabasiAlbertGenerator"

    def test_roundtrip_dict(self):
        data = {
            "CPS": {
                "name": "Barabasi Albert Scale-Free Network",
                "num_of_nodes": 10,
                "new_node_to_existing_nodes": 2,
            },
            "failure_model": {"name": "Watts cascade", "phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            "response_protocol": {"name": "FCFS", "frequency": 1.0, "auxiliary": False},
            "service_team": {
                "name": "complete",
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            "simulation_param": {"seed": "42", "replications": "1", "processors": "1", "simulation_length": "20"},
        }
        cfg = SimulationConfig.from_dict(data)
        params = cfg.to_parameters()
        assert params["CPS"]["name"] == data["CPS"]["name"]
        assert params["simulation_param"]["seed"] == data["simulation_param"]["seed"]


class TestRun:
    def test_produces_simulation_results(self):
        cfg = SimulationConfig(
            topology="Barabasi Albert Scale-Free Network",
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = run(cfg)
        assert isinstance(results, SimulationResults)

    def test_results_has_stats(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = run(cfg)
        assert "Recoverability" in results.stats
        assert "Total_Failures" in results.stats

    def test_results_has_samples(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=2,
            simulation_length=20,
        )
        results = run(cfg)
        assert "Total_Failures" in results.samples
        assert len(results.samples["Total_Failures"]) == 2

    def test_results_config_preserved(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = run(cfg)
        assert results.config is cfg


class TestSweep:
    def test_produces_multiple_results(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = sweep(cfg, "failure_params.phi", [0.2, 0.4])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, SimulationResults)

    def test_sweep_top_level_param(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = sweep(cfg, "seed", [1, 2])
        assert len(results) == 2

    def test_sweep_rejects_three_level_param(self):
        """sweep() should reject param_name with more than 2 levels."""
        cfg = SimulationConfig()
        with pytest.raises(ValueError, match="3 levels"):
            sweep(cfg, "a.b.c", [1])

    def test_sweep_rejects_missing_key(self):
        """sweep() should reject a nested key that doesn't exist."""
        cfg = SimulationConfig()
        with pytest.raises(ValueError, match="no key"):
            sweep(cfg, "failure_params.nonexistent_key", [1])

    def test_sweep_rejects_missing_attribute(self):
        """sweep() should reject a top-level attribute that doesn't exist."""
        cfg = SimulationConfig()
        with pytest.raises(ValueError, match="no attribute"):
            sweep(cfg, "nonexistent_attribute", [1])


class TestSimulationResultsToJson:
    def test_to_json_creates_file(self):
        cfg = SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )
        results = run(cfg)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = f.name
        try:
            results.to_json(tmppath)
            with open(tmppath) as f:
                data = json.load(f)
            assert "Recoverability" in data
        finally:
            os.unlink(tmppath)


class _NodeCountMetric(MetricCollector):
    """Test metric that counts the number of nodes."""

    @property
    def name(self):
        return "Node_Count"

    def collect(self, sim_state, network, service_team, env_now):
        return network.graph.number_of_nodes()

    def aggregate(self, values):
        import statistics as pystats
        return {"Mean": pystats.mean(values), "SEM": 0.0}


class TestExtraMetrics:
    """Tests that extra_metrics in SimulationConfig are wired into results."""

    def _base_config(self):
        return SimulationConfig(
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
            failure_params={"phi": 0.3, "fail_speed": 1.0, "initial_failures": 1},
            team_topology="complete",
            team_params={
                "team_members": 3,
                "agent_travel_speed": 1.0,
                "repairing_time": 2.0,
                "initial_allocation": "Centrality",
            },
            replications=1,
            simulation_length=20,
        )

    def test_extra_metric_appears_in_results(self):
        """extra_metrics should appear in the simulation results."""
        metric = _NodeCountMetric()
        cfg = self._base_config()
        cfg.extra_metrics = [metric]
        results = run(cfg)
        assert "Node_Count" in results.stats
        assert results.stats["Node_Count"]["Mean"] == 10.0

    def test_extra_metric_unregistered_after_run(self):
        """extra_metrics should be unregistered after run completes."""
        metric = _NodeCountMetric()
        cfg = self._base_config()
        cfg.extra_metrics = [metric]
        all_before = get_all_metrics()
        run(cfg)
        all_after = get_all_metrics()
        assert len(all_after) == len(all_before)
        assert metric not in all_after
