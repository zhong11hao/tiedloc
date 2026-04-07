"""Tests for tiedloc.metrics — metric collectors and registry."""

from __future__ import annotations

import math

import pytest

from tiedloc.metrics import (
    AuxLinesMetric,
    MeanLatencyQoSMetric,
    MetricCollector,
    PreventabilityMetric,
    TotalDistanceMetric,
    TotalFailuresMetric,
    TotalLatencyMetric,
    _mean_sem,
    get_all_metrics,
    register_metric,
)
from tiedloc.networks import SimulationState


class _FakeNetwork:
    """Minimal stub for CPSNetwork needed by metric collectors."""
    pass


class _FakeServiceTeam:
    """Minimal stub for ServiceTeam needed by metric collectors."""
    pass


class TestMeanSem:
    def test_normal_values(self):
        result = _mean_sem([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["Mean"] == pytest.approx(3.0)
        assert result["SEM"] > 0.0

    def test_single_value(self):
        result = _mean_sem([42.0])
        assert result["Mean"] == pytest.approx(42.0)
        assert result["SEM"] == pytest.approx(0.0)

    def test_inf_values(self):
        result = _mean_sem([float("inf"), float("inf")])
        assert math.isinf(result["Mean"])
        assert result["SEM"] == pytest.approx(0.0)

    def test_mixed_inf_and_finite(self):
        result = _mean_sem([1.0, float("inf"), 3.0])
        # mean computed on finite values only: [1.0, 3.0]
        assert result["Mean"] == pytest.approx(2.0)

    def test_empty_list(self):
        result = _mean_sem([])
        assert math.isinf(result["Mean"])
        assert result["SEM"] == pytest.approx(0.0)


class TestMetricRegistry:
    def test_built_in_metrics_registered(self):
        metrics = get_all_metrics()
        names = [m.name for m in metrics]
        assert "Total_Failures" in names
        assert "Preventability" in names
        assert "Total_Distance_Traveled_by_Agent" in names
        assert "Aux_Lines" in names
        assert "Total_Latency" in names
        assert "Mean_Latency_QoS" in names

    def test_register_custom_metric(self):
        class CustomMetric(MetricCollector):
            @property
            def name(self):
                return "test_custom_metric_reg"

            def collect(self, sim_state, network, service_team, env_now):
                return 99

            def aggregate(self, values):
                return sum(values)

        initial_count = len(get_all_metrics())
        register_metric(CustomMetric())
        assert len(get_all_metrics()) == initial_count + 1

    def test_get_all_metrics_returns_list(self):
        metrics = get_all_metrics()
        assert isinstance(metrics, list)
        for m in metrics:
            assert isinstance(m, MetricCollector)


class TestTotalFailuresMetric:
    def test_collect(self):
        m = TotalFailuresMetric()
        sim_state = SimulationState()
        sim_state.total_failures = 7
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == 7

    def test_aggregate(self):
        m = TotalFailuresMetric()
        result = m.aggregate([3.0, 5.0, 7.0])
        assert "Mean" in result
        assert "SEM" in result
        assert result["Mean"] == pytest.approx(5.0)

    def test_name(self):
        assert TotalFailuresMetric().name == "Total_Failures"


class TestPreventabilityMetric:
    def test_collect_with_events(self):
        m = PreventabilityMetric()
        sim_state = SimulationState()
        sim_state.prevented = 3
        sim_state.total_failures = 7
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == pytest.approx(0.3)

    def test_collect_zero_events(self):
        m = PreventabilityMetric()
        sim_state = SimulationState()
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == pytest.approx(0.0)

    def test_name(self):
        assert PreventabilityMetric().name == "Preventability"


class TestTotalDistanceMetric:
    def test_collect(self):
        m = TotalDistanceMetric()
        sim_state = SimulationState()
        sim_state.total_distance = 42.5
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == pytest.approx(42.5)

    def test_name(self):
        assert TotalDistanceMetric().name == "Total_Distance_Traveled_by_Agent"


class TestAuxLinesMetric:
    def test_collect(self):
        m = AuxLinesMetric()
        sim_state = SimulationState()
        sim_state.aux_lines = 3
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == 3

    def test_name(self):
        assert AuxLinesMetric().name == "Aux_Lines"


class TestTotalLatencyMetric:
    def test_collect(self):
        m = TotalLatencyMetric()
        sim_state = SimulationState()
        sim_state.total_latency = 15.5
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == pytest.approx(15.5)

    def test_name(self):
        assert TotalLatencyMetric().name == "Total_Latency"


class TestMeanLatencyQoSMetric:
    def test_collect(self):
        m = MeanLatencyQoSMetric()
        sim_state = SimulationState()
        sim_state.latency = [1.0, 2.0, 3.0]
        val = m.collect(sim_state, _FakeNetwork(), _FakeServiceTeam(), 10.0)
        assert val == [1.0, 2.0, 3.0]

    def test_aggregate_flattens(self):
        m = MeanLatencyQoSMetric()
        result = m.aggregate([[1.0, 2.0], [3.0, 4.0]])
        assert "Mean" in result
        assert result["Mean"] == pytest.approx(2.5)

    def test_aggregate_empty(self):
        m = MeanLatencyQoSMetric()
        result = m.aggregate([[], []])
        assert result["Mean"] == pytest.approx(0.0)

    def test_name(self):
        assert MeanLatencyQoSMetric().name == "Mean_Latency_QoS"
