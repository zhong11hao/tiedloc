"""Unit tests for the tiedloc.simulations module."""

import json
import math
import os
import tempfile

import pytest
import simpy

from tiedloc.networks import SimulationState
from tiedloc.simulations import (
    create_simulation_state,
    find_recovery_time,
    save_results,
    sim_each_time_unit,
    statistics,
)


class TestCreateSimulationState:
    """Tests for the create_simulation_state factory."""

    def test_returns_fresh_state(self):
        """create_simulation_state should return a new SimulationState with defaults."""
        state = create_simulation_state()
        assert isinstance(state, SimulationState)
        assert state.failed_elements == 0
        assert state.failures == []

    def test_independent_states(self):
        """Multiple calls should return independent instances."""
        s1 = create_simulation_state()
        s2 = create_simulation_state()
        s1.failed_elements = 10
        assert s2.failed_elements == 0


class TestSimEachTimeUnit:
    """Tests for the sim_each_time_unit process."""

    def test_records_failures_at_each_step(self):
        """sim_each_time_unit should record failed_elements at each integer time step."""
        env = simpy.Environment()
        state = SimulationState()
        state.failed_elements = 3

        env.process(sim_each_time_unit(env, state))
        env.run(until=5)

        assert len(state.failed_at_step) == 5
        assert all(v == 3 for v in state.failed_at_step)

    def test_tracks_changing_failures(self):
        """sim_each_time_unit should reflect changes in failed_elements over time."""
        env = simpy.Environment()
        state = SimulationState()

        def change_failures(env, state):
            state.failed_elements = 0
            yield env.timeout(2)
            state.failed_elements = 5
            yield env.timeout(3)

        env.process(sim_each_time_unit(env, state))
        env.process(change_failures(env, state))
        env.run(until=5)

        # Steps 0, 1 should have 0 failures; steps 2, 3, 4 should have 5
        assert state.failed_at_step[0] == 0
        assert state.failed_at_step[1] == 0
        assert state.failed_at_step[2] == 5
        assert state.failed_at_step[3] == 5
        assert state.failed_at_step[4] == 5


class TestFindRecoveryTime:
    """Tests for the find_recovery_time function."""

    def test_immediate_recovery(self):
        """If failures reach 0 at step 0, recovery time should be 0."""
        assert find_recovery_time([0, 0, 0]) == 0.0

    def test_recovery_at_step_3(self):
        """If failures reach 0 at step 3, recovery time should be 3."""
        assert find_recovery_time([5, 3, 1, 0, 0]) == 3.0

    def test_no_recovery_constant(self):
        """If failures never decrease, recovery time should be inf."""
        result = find_recovery_time([5, 5, 5, 5, 5])
        assert math.isinf(result)

    def test_no_recovery_increasing(self):
        """If failures keep increasing, recovery time should be inf."""
        result = find_recovery_time([1, 2, 3, 4, 5])
        assert math.isinf(result)

    def test_empty_list(self):
        """An empty list should return 0.0 without crashing."""
        result = find_recovery_time([])
        assert result == 0.0


class TestStatistics:
    """Tests for the statistics aggregation function."""

    def _make_result(
        self,
        total_latency: float = 10.0,
        latency: list | None = None,
        total_failures: int = 5,
        prevented: int = 2,
        total_distance: float = 20.0,
        aux_lines: int = 0,
        failed_at_step: list | None = None,
    ) -> dict:
        """Helper to create a mock worker result."""
        return {
            "total_latency": total_latency,
            "latency": latency or [2.0, 3.0],
            "total_failures": total_failures,
            "prevented": prevented,
            "total_distance": total_distance,
            "aux_lines": aux_lines,
            "failed_at_step": failed_at_step or [3, 2, 1, 0, 0],
        }

    def test_single_replication(self):
        """Statistics from a single replication should have SEM of 0."""
        results = [self._make_result()]
        stat_results, samples = statistics(results)

        assert "Recoverability" in stat_results
        assert stat_results["Total_Latency"]["Mean"] == 10.0
        assert stat_results["Total_Latency"]["SEM"] == 0.0
        assert stat_results["Total_Failures"]["Mean"] == 5.0

    def test_multiple_replications(self):
        """Statistics from multiple replications should aggregate correctly."""
        results = [
            self._make_result(total_latency=10.0, total_failures=5, prevented=2),
            self._make_result(total_latency=20.0, total_failures=8, prevented=4),
        ]
        stat_results, samples = statistics(results)

        assert stat_results["Total_Latency"]["Mean"] == 15.0
        assert stat_results["Total_Failures"]["Mean"] == 6.5
        assert len(samples["Total_Latency"]) == 2

    def test_preventability_calculation(self):
        """Preventability should be prevented / (prevented + total_failures)."""
        results = [self._make_result(total_failures=6, prevented=4)]
        stat_results, _ = statistics(results)

        # 4 / (4 + 6) = 0.4
        assert stat_results["Preventability"]["Mean"] == 0.4

    def test_recoverability(self):
        """Recoverability should count replications that recovered."""
        results = [
            self._make_result(failed_at_step=[3, 2, 1, 0, 0]),  # recovers at step 3
            self._make_result(failed_at_step=[3, 3, 3, 3, 3]),  # never recovers
        ]
        stat_results, _ = statistics(results)
        assert stat_results["Recoverability"] == 0.5


class TestSaveResults:
    """Tests for the save_results function."""

    def test_creates_output_files(self):
        """save_results should create OUTPUT and SAMPLES JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "test.json")
            with open(input_path, "w") as f:
                f.write("{}")

            stat_results = {"Recoverability": 0.5, "Total_Failures": {"Mean": 5.0, "SEM": 1.0}}
            samples = {
                "Total_Latency": [10.0, 20.0],
                "Preventability": [0.3, 0.5],
                "Total_Distance_Traveled_by_Agent": [100.0],
                "Total_Failures": [5, 8],
                "Step_000": [3, 4],  # should not be in samples output
            }

            save_results(input_path, stat_results, samples)

            output_path = os.path.join(tmpdir, "testOUTPUT.json")
            samples_path = os.path.join(tmpdir, "testSAMPLES.json")

            assert os.path.exists(output_path)
            assert os.path.exists(samples_path)

            with open(output_path) as f:
                loaded = json.load(f)
                assert loaded["Recoverability"] == 0.5

            with open(samples_path) as f:
                loaded = json.load(f)
                assert "Total_Latency" in loaded
                assert "Step_000" not in loaded
