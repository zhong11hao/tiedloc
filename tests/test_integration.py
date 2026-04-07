"""Integration tests for end-to-end tiedloc simulation runs."""

import math
import os

import pytest

from tiedloc.api import SimulationConfig, run


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load_and_run(filename):
    """Load an example JSON input and run a full simulation, returning (stat_results, samples)."""
    path = os.path.join(EXAMPLES_DIR, filename)
    config = SimulationConfig.from_json(path)
    results = run(config)
    return results.stats, results.samples


class TestEndToEndSmallBA:
    """End-to-end simulation with a small Barabasi-Albert network (FCFS)."""

    def test_stat_results_has_expected_keys(self):
        """stat_results should contain all standard output keys."""
        stat_results, _ = _load_and_run("small_ba.json")
        expected_keys = {
            "Recoverability",
            "Median_Recovery_Time",
            "Total_Latency",
            "Mean_Latency_QoS",
            "Mean_Recovery_Time_withINF",
            "Total_Failures",
            "Preventability",
            "Total_Distance_Traveled_by_Agent",
            "Aux_Lines",
        }
        for key in expected_keys:
            assert key in stat_results, f"Missing key: {key}"

    def test_recoverability_in_valid_range(self):
        """Recoverability should be between 0 and 1."""
        stat_results, _ = _load_and_run("small_ba.json")
        assert 0.0 <= stat_results["Recoverability"] <= 1.0

    def test_total_latency_non_negative(self):
        """Total latency mean should be non-negative."""
        stat_results, _ = _load_and_run("small_ba.json")
        mean = stat_results["Total_Latency"]["Mean"]
        if not math.isinf(mean):
            assert mean >= 0.0

    def test_total_failures_positive(self):
        """There should be at least some failures (we seed initial failures)."""
        stat_results, _ = _load_and_run("small_ba.json")
        assert stat_results["Total_Failures"]["Mean"] > 0

    def test_samples_has_expected_keys(self):
        """samples dict should contain per-replication data."""
        _, samples = _load_and_run("small_ba.json")
        assert "Total_Latency" in samples
        assert "Total_Failures" in samples
        assert "Preventability" in samples

    def test_samples_length_matches_replications(self):
        """Each sample list should have length equal to number of replications."""
        _, samples = _load_and_run("small_ba.json")
        # small_ba.json has 10 replications
        assert len(samples["Total_Latency"]) == 10
        assert len(samples["Total_Failures"]) == 10

    def test_preventability_in_valid_range(self):
        """Preventability should be between 0 and 1 for each replication."""
        _, samples = _load_and_run("small_ba.json")
        for p in samples["Preventability"]:
            assert 0.0 <= p <= 1.0

    def test_step_keys_present(self):
        """Step_NNN keys should be present in stat_results."""
        stat_results, _ = _load_and_run("small_ba.json")
        assert "Step_000" in stat_results

    def test_sem_non_negative(self):
        """SEM values should be non-negative."""
        stat_results, _ = _load_and_run("small_ba.json")
        for key, val in stat_results.items():
            if isinstance(val, dict) and "SEM" in val:
                assert val["SEM"] >= 0.0, f"Negative SEM for {key}"


class TestEndToEndSmallWS:
    """End-to-end simulation with a Watts-Strogatz network (nearest strategy)."""

    def test_runs_successfully(self):
        """Simulation with nearest protocol should complete without errors."""
        stat_results, samples = _load_and_run("small_ws.json")
        assert "Recoverability" in stat_results
        assert 0.0 <= stat_results["Recoverability"] <= 1.0

    def test_total_distance_non_negative(self):
        """Total distance traveled by agents should be non-negative."""
        stat_results, _ = _load_and_run("small_ws.json")
        mean = stat_results["Total_Distance_Traveled_by_Agent"]["Mean"]
        if not math.isinf(mean):
            assert mean >= 0.0


class TestEndToEndActivityBA:
    """End-to-end simulation with activity-based dispatch."""

    def test_runs_successfully(self):
        """Simulation with activity protocol should complete without errors."""
        stat_results, samples = _load_and_run("activity_ba.json")
        assert "Recoverability" in stat_results
        assert 0.0 <= stat_results["Recoverability"] <= 1.0
        assert stat_results["Total_Failures"]["Mean"] > 0

    def test_median_recovery_time_valid(self):
        """Median recovery time should be non-negative (may be inf)."""
        stat_results, _ = _load_and_run("activity_ba.json")
        mrt = stat_results["Median_Recovery_Time"]
        assert mrt >= 0.0 or math.isinf(mrt)


class TestEndToEndAuxiliaryBA:
    """End-to-end simulation with auxiliary edge injection enabled."""

    def test_runs_successfully(self):
        """Simulation with auxiliary edges should complete without errors."""
        stat_results, samples = _load_and_run("auxiliary_ba.json")
        assert "Recoverability" in stat_results
        assert 0.0 <= stat_results["Recoverability"] <= 1.0

    def test_aux_lines_non_negative(self):
        """Aux_Lines should be non-negative (may be zero if no injection triggered)."""
        stat_results, _ = _load_and_run("auxiliary_ba.json")
        mean = stat_results["Aux_Lines"]["Mean"]
        if not math.isinf(mean):
            assert mean >= 0.0


class TestDeterministicWithSeed:
    """Verify that using the same seed produces identical results."""

    def test_reproducibility(self):
        """Two runs with the same seed should produce identical stat_results."""
        stat1, _ = _load_and_run("small_ba.json")
        stat2, _ = _load_and_run("small_ba.json")
        # Compare all non-step keys
        for key in ("Recoverability", "Total_Failures", "Total_Latency", "Preventability"):
            assert stat1[key] == stat2[key], f"Non-deterministic result for {key}"


class TestEntropyMode:
    """Verify that seed=None produces different results across runs."""

    def test_different_results_without_seed(self):
        """Two runs with seed=None should (almost certainly) differ."""
        cfg = SimulationConfig(
            seed=None,
            replications=5,
            simulation_length=20,
            topology_params={"num_of_nodes": 10, "new_node_to_existing_nodes": 2},
        )
        r1 = run(cfg)
        r2 = run(cfg)
        any_different = any(
            r1.stats[k] != r2.stats[k]
            for k in ("Recoverability", "Total_Failures")
        )
        assert any_different, "Two unseeded runs produced identical results"


class TestReproducibility:
    """Stronger reproducibility test with more replications."""

    def test_same_seed_same_results_multi_rep(self):
        """Seed=42 with 20 reps must be fully deterministic."""
        cfg = SimulationConfig(
            seed=42,
            replications=20,
            simulation_length=30,
            topology_params={"num_of_nodes": 15, "new_node_to_existing_nodes": 2},
        )
        r1 = run(cfg)
        r2 = run(cfg)
        for key in r1.stats:
            assert r1.stats[key] == r2.stats[key], f"Non-deterministic: {key}"
