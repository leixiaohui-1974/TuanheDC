"""
V3.10 Performance Benchmark Tests for TAOS

Benchmarks performance of all V3.10 modules:
- Sensor simulation throughput
- Actuator simulation throughput
- Data assimilation speed
- Prediction performance
- Evaluation speed
- Memory usage

Author: TAOS Development Team
Version: 3.10
"""

import time
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    name: str
    iterations: int
    total_time: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # ops/second


class V310Benchmark:
    """Performance benchmark suite for V3.10 modules."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark(self, name: str, func, iterations: int = 100,
                  warmup: int = 10) -> BenchmarkResult:
        """
        Run benchmark on a function.

        Args:
            name: Benchmark name
            func: Function to benchmark (no arguments)
            iterations: Number of iterations
            warmup: Warmup iterations

        Returns:
            BenchmarkResult
        """
        # Warmup
        for _ in range(warmup):
            func()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times)
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=np.sum(times),
            avg_time_ms=np.mean(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=iterations / np.sum(times)
        )

        self.results.append(result)
        return result

    def print_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Name':<40} {'Avg (ms)':<12} {'Min (ms)':<12} {'Throughput':<15}")
        print("-" * 80)

        for r in self.results:
            print(f"{r.name:<40} {r.avg_time_ms:<12.3f} {r.min_time_ms:<12.3f} "
                  f"{r.throughput:<15.1f} ops/s")

        print("=" * 80)


def run_sensor_benchmarks(bench: V310Benchmark):
    """Run sensor simulation benchmarks."""
    from sensor_simulation import SensorSimulationEngine

    engine = SensorSimulationEngine()
    true_state = {
        'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
        'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
        'Q_in': 80.0, 'Q_out': 80.0
    }

    def measure():
        engine.measure(true_state, dt=0.1)

    bench.benchmark("Sensor: Single measurement", measure, iterations=1000)

    # Batch measurement benchmark
    def batch_measure():
        for _ in range(10):
            engine.measure(true_state, dt=0.1)

    bench.benchmark("Sensor: 10-measurement batch", batch_measure, iterations=100)

    # Status retrieval
    def get_status():
        engine.get_full_status()

    bench.benchmark("Sensor: Get status", get_status, iterations=500)


def run_actuator_benchmarks(bench: V310Benchmark):
    """Run actuator simulation benchmarks."""
    from actuator_simulation import ActuatorSimulationEngine

    engine = ActuatorSimulationEngine()
    engine.set_water_levels(h_upstream=5.0, h_downstream=2.0)
    engine.command_flows(Q_in=80.0, Q_out=80.0)

    def step():
        engine.step(dt=0.1)

    bench.benchmark("Actuator: Single step", step, iterations=1000)

    def step_and_command():
        engine.command_flows(Q_in=80.0 + np.random.randn() * 5,
                            Q_out=80.0 + np.random.randn() * 5)
        engine.step(dt=0.1)

    bench.benchmark("Actuator: Command + step", step_and_command, iterations=500)


def run_assimilation_benchmarks(bench: V310Benchmark):
    """Run data assimilation benchmarks."""
    from data_assimilation import (
        DataAssimilationEngine, AssimilationMethod, AssimilationConfig
    )

    # EKF benchmark
    config = AssimilationConfig(method=AssimilationMethod.EXTENDED_KALMAN)
    engine = DataAssimilationEngine(config)
    engine.initialize({'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0})

    def ekf_cycle():
        engine.predict({'Q_in': 80, 'Q_out': 80}, dt=0.1)
        engine.assimilate({
            'h': 4.0 + np.random.randn() * 0.1,
            'v': 2.0 + np.random.randn() * 0.05,
            'T_sun': 25.0 + np.random.randn() * 0.5,
            'T_shade': 22.0 + np.random.randn() * 0.3
        })

    bench.benchmark("Assimilation: EKF cycle", ekf_cycle, iterations=500)

    # EnKF benchmark
    config = AssimilationConfig(
        method=AssimilationMethod.ENSEMBLE_KALMAN,
        ensemble_size=30
    )
    engine = DataAssimilationEngine(config)
    engine.initialize({'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0})

    def enkf_cycle():
        engine.predict({'Q_in': 80, 'Q_out': 80}, dt=0.1)
        engine.assimilate({
            'h': 4.0 + np.random.randn() * 0.1,
            'v': 2.0 + np.random.randn() * 0.05,
            'T_sun': 25.0 + np.random.randn() * 0.5,
            'T_shade': 22.0 + np.random.randn() * 0.3
        })

    bench.benchmark("Assimilation: EnKF (30) cycle", enkf_cycle, iterations=200)

    # Particle filter benchmark
    config = AssimilationConfig(method=AssimilationMethod.PARTICLE_FILTER)
    engine = DataAssimilationEngine(config)
    engine.initialize({'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0})

    def pf_cycle():
        engine.predict({'Q_in': 80, 'Q_out': 80}, dt=0.1)
        engine.assimilate({
            'h': 4.0 + np.random.randn() * 0.1,
            'v': 2.0 + np.random.randn() * 0.05,
            'T_sun': 25.0 + np.random.randn() * 0.5,
            'T_shade': 22.0 + np.random.randn() * 0.3
        })

    bench.benchmark("Assimilation: Particle Filter cycle", pf_cycle, iterations=100)


def run_idz_benchmarks(bench: V310Benchmark):
    """Run IDZ model adapter benchmarks."""
    from idz_model_adapter import IDZModelAdapter, IDZModel

    # IDZ model step
    model = IDZModel()
    control = {'Q_in': 80.0, 'Q_out': 80.0}
    env = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

    def idz_step():
        model.step(control, env, dt=0.1)

    bench.benchmark("IDZ: Model step", idz_step, iterations=2000)

    # IDZ adapter update
    adapter = IDZModelAdapter()
    hifi_state = {
        'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
        'joint_gap': 20.0, 'vib_amp': 0.0
    }

    def adapter_update():
        adapter.update_from_hifi(hifi_state, control, env, dt=0.1)

    bench.benchmark("IDZ: Adapter update", adapter_update, iterations=200)

    # IDZ prediction
    def idz_predict():
        adapter.predict(control, env, horizon=20, dt=0.1)

    bench.benchmark("IDZ: 20-step prediction", idz_predict, iterations=200)


def run_evaluation_benchmarks(bench: V310Benchmark):
    """Run state evaluation benchmarks."""
    from state_evaluation import StateEvaluator, MultiObjectiveEvaluator

    evaluator = StateEvaluator()
    state = {
        'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
        'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
        'Q_in': 80.0, 'Q_out': 80.0
    }
    control = {'Q_in': 80.0, 'Q_out': 80.0}

    def evaluate():
        evaluator.evaluate(state, control)

    bench.benchmark("Evaluation: Full evaluation", evaluate, iterations=500)

    def deviation():
        evaluator.evaluate_deviation(state)

    bench.benchmark("Evaluation: Deviation only", deviation, iterations=1000)

    def risk():
        evaluator.assess_risk(state)

    bench.benchmark("Evaluation: Risk assessment", risk, iterations=1000)

    # Multi-objective
    mo_evaluator = MultiObjectiveEvaluator()

    def multi_objective():
        mo_evaluator.evaluate(state, control)

    bench.benchmark("Evaluation: Multi-objective", multi_objective, iterations=500)


def run_prediction_benchmarks(bench: V310Benchmark):
    """Run state prediction benchmarks."""
    from state_prediction import (
        StatePredictionEngine, PredictionMethod,
        PhysicsPredictor, EnsemblePredictor
    )

    state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
             'joint_gap': 20.0, 'vib_amp': 0.0}
    control = {'Q_in': 80.0, 'Q_out': 80.0}
    environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

    # Physics predictor
    physics = PhysicsPredictor()

    def physics_step():
        physics.predict_step(state, control, environment, dt=1.0)

    bench.benchmark("Prediction: Physics step", physics_step, iterations=2000)

    control_seq = [control] * 60
    env_seq = [environment] * 60

    def physics_trajectory():
        physics.predict_trajectory(state, control_seq, env_seq, dt=1.0)

    bench.benchmark("Prediction: Physics 60-step", physics_trajectory, iterations=100)

    # Ensemble predictor
    ensemble = EnsemblePredictor(num_members=20)

    def ensemble_predict():
        ensemble.predict_ensemble(state, control, environment, horizon=30, dt=1.0)

    bench.benchmark("Prediction: Ensemble (20) 30-step", ensemble_predict, iterations=50)

    # Full engine
    engine = StatePredictionEngine()

    def engine_short():
        engine.predict(state, control, environment, 'short',
                      PredictionMethod.PHYSICS_BASED)

    bench.benchmark("Prediction: Engine short horizon", engine_short, iterations=100)


def run_governance_benchmarks(bench: V310Benchmark):
    """Run data governance benchmarks."""
    from data_governance import DataGovernanceEngine, DataQualityValidator

    validator = DataQualityValidator()
    data = {
        'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
        'Q_in': 80.0, 'Q_out': 80.0
    }

    def validate():
        validator.validate(data)

    bench.benchmark("Governance: Validation", validate, iterations=1000)

    engine = DataGovernanceEngine()

    def process():
        engine.process_data(data, source='test')

    bench.benchmark("Governance: Full process", process, iterations=500)


def run_full_pipeline_benchmark(bench: V310Benchmark):
    """Run full pipeline benchmark."""
    from sensor_simulation import SensorSimulationEngine
    from actuator_simulation import ActuatorSimulationEngine
    from data_assimilation import DataAssimilationEngine
    from state_evaluation import StateEvaluator
    from state_prediction import StatePredictionEngine, PredictionMethod

    sensor = SensorSimulationEngine()
    actuator = ActuatorSimulationEngine()
    assimilation = DataAssimilationEngine()
    evaluator = StateEvaluator()
    prediction = StatePredictionEngine()

    actuator.set_water_levels(h_upstream=5.0, h_downstream=2.0)
    assimilation.initialize({'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0})

    true_state = {
        'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
        'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
        'Q_in': 80.0, 'Q_out': 80.0
    }
    control = {'Q_in': 80.0, 'Q_out': 80.0}
    environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

    def full_cycle():
        # Sensor
        measured = sensor.measure(true_state, dt=0.1)

        # Actuator
        actuator.command_flows(Q_in=80.0, Q_out=80.0)
        actuator.step(dt=0.1)

        # Assimilation
        assimilation.predict(control, dt=0.1)
        assimilation.assimilate({
            'h': measured['measured_state'].get('h', 4.0),
            'v': measured['measured_state'].get('v', 2.0),
            'T_sun': measured['measured_state'].get('T_sun', 25.0),
            'T_shade': measured['measured_state'].get('T_shade', 22.0)
        })

        # Evaluation
        evaluator.evaluate(true_state, control)

    bench.benchmark("Pipeline: Full sensor-actuator-assim-eval", full_cycle, iterations=100)


def run_memory_benchmark():
    """Run memory usage benchmark."""
    import tracemalloc

    print("\n" + "=" * 80)
    print("MEMORY USAGE BENCHMARK")
    print("=" * 80)

    modules = [
        ("SensorSimulationEngine", "sensor_simulation", "SensorSimulationEngine"),
        ("ActuatorSimulationEngine", "actuator_simulation", "ActuatorSimulationEngine"),
        ("DataAssimilationEngine", "data_assimilation", "DataAssimilationEngine"),
        ("IDZModelAdapter", "idz_model_adapter", "IDZModelAdapter"),
        ("StateEvaluator", "state_evaluation", "StateEvaluator"),
        ("StatePredictionEngine", "state_prediction", "StatePredictionEngine"),
        ("DataGovernanceEngine", "data_governance", "DataGovernanceEngine"),
    ]

    for name, module_name, class_name in modules:
        tracemalloc.start()

        module = __import__(module_name)
        cls = getattr(module, class_name)
        instance = cls()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{name:<35} Current: {current/1024:>8.1f} KB  Peak: {peak/1024:>8.1f} KB")

    print("=" * 80)


# =============================================================================
# Phase 1-3 Module Benchmarks
# =============================================================================

def run_realtime_data_benchmarks(bench: V310Benchmark):
    """Benchmark real-time data interface operations."""
    from realtime_data_interface import RealtimeDataManager, DataBuffer

    manager = RealtimeDataManager()
    buffer = DataBuffer(max_size=1000)

    # Benchmark data buffer operations
    def buffer_add():
        from datetime import datetime
        buffer.add("test_tag", 25.5, datetime.now())

    bench.benchmark("Realtime: Buffer add", buffer_add, iterations=1000)

    # Benchmark buffer statistics
    for i in range(100):
        from datetime import datetime
        buffer.add(f"stat_tag_{i % 10}", float(i), datetime.now())

    def buffer_stats():
        buffer.get_statistics("stat_tag_0")

    bench.benchmark("Realtime: Buffer stats", buffer_stats, iterations=500)

    # Benchmark manager status
    def manager_status():
        manager.get_status()

    bench.benchmark("Realtime: Manager status", manager_status, iterations=500)


def run_alarm_benchmarks(bench: V310Benchmark):
    """Benchmark alarm event management operations."""
    from alarm_event_management import AlarmManager, AlarmSeverity, AlarmCategory

    manager = AlarmManager()

    # Add some rules
    from alarm_event_management import AlarmRule, AlarmCondition, ComparisonOperator
    rule = AlarmRule(
        rule_id="bench_rule",
        name="Benchmark Rule",
        condition=AlarmCondition(
            tag="temperature",
            operator=ComparisonOperator.GREATER_THAN,
            value=100
        ),
        severity=AlarmSeverity.HIGH,
        category=AlarmCategory.PROCESS
    )
    manager.add_rule(rule)

    # Benchmark rule evaluation
    def evaluate_data():
        manager.evaluate_data({"temperature": 50.0, "pressure": 101.3})

    bench.benchmark("Alarm: Evaluate data", evaluate_data, iterations=500)

    # Benchmark get active alarms
    def get_active():
        manager.get_active_alarms()

    bench.benchmark("Alarm: Get active", get_active, iterations=500)

    # Benchmark alarm history
    def get_history():
        manager.get_alarm_history(limit=10)

    bench.benchmark("Alarm: Get history", get_history, iterations=500)


def run_reporting_benchmarks(bench: V310Benchmark):
    """Benchmark reporting and visualization operations."""
    from reporting_visualization import (
        ReportingManager, DataAggregator, ChartGenerator,
        AggregationType, ChartType
    )

    manager = ReportingManager()
    aggregator = DataAggregator()
    chart_gen = ChartGenerator()

    # Add test data
    from datetime import datetime, timedelta
    base_time = datetime.now()
    test_data = []
    for i in range(100):
        test_data.append({
            'timestamp': base_time - timedelta(hours=i),
            'value': 50 + (i % 20)
        })

    for d in test_data:
        aggregator.add_data_point("test_metric", d['value'], d['timestamp'])

    # Benchmark aggregation
    def aggregate_hourly():
        aggregator.get_aggregated("test_metric", AggregationType.HOURLY)

    bench.benchmark("Report: Hourly aggregation", aggregate_hourly, iterations=200)

    # Benchmark chart generation
    def generate_line_chart():
        chart_gen.generate_line_chart(
            "Test Chart",
            {"test": [1, 2, 3, 4, 5]},
            ["A", "B", "C", "D", "E"]
        )

    bench.benchmark("Report: Line chart gen", generate_line_chart, iterations=200)

    # Benchmark gauge chart
    def generate_gauge():
        chart_gen.generate_gauge_chart("Test Gauge", 75, 0, 100)

    bench.benchmark("Report: Gauge chart gen", generate_gauge, iterations=200)


def run_knowledge_graph_benchmarks(bench: V310Benchmark):
    """Benchmark knowledge graph operations."""
    from knowledge_graph import KnowledgeGraphManager, EntityType, RelationType

    manager = KnowledgeGraphManager()

    # Build a test graph
    entities = []
    for i in range(50):
        eid = manager.add_entity(
            EntityType.EQUIPMENT,
            f"Equipment_{i}",
            {"location": f"Section_{i % 5}"}
        )
        entities.append(eid)

    # Add relations
    for i in range(len(entities) - 1):
        manager.add_relation(
            entities[i], entities[i + 1],
            RelationType.CONNECTS_TO
        )

    # Benchmark entity query
    def query_entities():
        manager.get_entities(EntityType.EQUIPMENT)

    bench.benchmark("KG: Query entities", query_entities, iterations=200)

    # Benchmark neighbor query
    def query_neighbors():
        manager.get_neighbors(entities[25])

    bench.benchmark("KG: Query neighbors", query_neighbors, iterations=200)

    # Benchmark path finding
    def find_path():
        manager.find_path(entities[0], entities[10], max_depth=5)

    bench.benchmark("KG: Find path", find_path, iterations=100)

    # Benchmark graph statistics
    def get_stats():
        manager.get_statistics()

    bench.benchmark("KG: Get statistics", get_stats, iterations=200)


def run_aiops_benchmarks(bench: V310Benchmark):
    """Benchmark AIOps operations."""
    from aiops import AIOpsManager, AnomalyDetector, IntelligentDiagnostics

    manager = AIOpsManager()
    detector = AnomalyDetector()
    diagnostics = IntelligentDiagnostics()

    # Add baseline data
    for i in range(100):
        detector.add_data_point("entity_001", "temperature", 25 + (i % 5) * 0.1)

    # Benchmark anomaly detection
    def detect_anomaly():
        detector.detect("entity_001", "temperature", 25.5)

    bench.benchmark("AIOps: Anomaly detect", detect_anomaly, iterations=500)

    # Benchmark metric processing
    def process_metric():
        manager.process_metric("entity_001", "temperature", 25.5)

    bench.benchmark("AIOps: Process metric", process_metric, iterations=300)

    # Benchmark diagnosis
    def run_diagnosis():
        diagnostics.diagnose("entity_001", {"temperature": 25.5}, [])

    bench.benchmark("AIOps: Diagnosis", run_diagnosis, iterations=200)

    # Benchmark statistics
    def get_statistics():
        manager.get_statistics()

    bench.benchmark("AIOps: Get statistics", get_statistics, iterations=500)


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("TAOS V3.10 PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print()

    bench = V310Benchmark()

    print("Running sensor benchmarks...")
    run_sensor_benchmarks(bench)

    print("Running actuator benchmarks...")
    run_actuator_benchmarks(bench)

    print("Running assimilation benchmarks...")
    run_assimilation_benchmarks(bench)

    print("Running IDZ benchmarks...")
    run_idz_benchmarks(bench)

    print("Running evaluation benchmarks...")
    run_evaluation_benchmarks(bench)

    print("Running prediction benchmarks...")
    run_prediction_benchmarks(bench)

    print("Running governance benchmarks...")
    run_governance_benchmarks(bench)

    print("Running full pipeline benchmark...")
    run_full_pipeline_benchmark(bench)

    # Phase 1-3 benchmarks
    print("\nRunning Phase 1-3 benchmarks...")
    print("-" * 40)

    print("Running real-time data benchmarks...")
    run_realtime_data_benchmarks(bench)

    print("Running alarm benchmarks...")
    run_alarm_benchmarks(bench)

    print("Running reporting benchmarks...")
    run_reporting_benchmarks(bench)

    print("Running knowledge graph benchmarks...")
    run_knowledge_graph_benchmarks(bench)

    print("Running AIOps benchmarks...")
    run_aiops_benchmarks(bench)

    bench.print_results()

    run_memory_benchmark()

    # Summary
    print("\nSUMMARY")
    print("-" * 40)
    total_throughput = sum(r.throughput for r in bench.results)
    avg_latency = np.mean([r.avg_time_ms for r in bench.results])
    print(f"Total benchmarks: {len(bench.results)}")
    print(f"Average latency: {avg_latency:.3f} ms")
    print(f"Combined throughput: {total_throughput:.0f} ops/s")

    # Real-time capability assessment
    print("\nREAL-TIME CAPABILITY ASSESSMENT")
    print("-" * 40)
    critical_ops = [
        ("Sensor measurement", 100),  # 100 Hz requirement
        ("Actuator step", 100),
        ("Assimilation (EKF)", 50),
        ("Evaluation", 50),
    ]

    for op_name, required_hz in critical_ops:
        matching = [r for r in bench.results if op_name.lower() in r.name.lower()]
        if matching:
            actual_hz = matching[0].throughput
            status = "PASS" if actual_hz >= required_hz else "FAIL"
            print(f"{op_name:<30} Required: {required_hz:>6} Hz  "
                  f"Actual: {actual_hz:>8.0f} Hz  [{status}]")


if __name__ == '__main__':
    main()
