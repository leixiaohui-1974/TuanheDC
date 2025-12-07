"""
Hardware-in-the-Loop (HIL) Simulation Test Framework for TAOS

This module provides comprehensive HIL testing capabilities for the
Tuanhe Aqueduct Autonomous Operation System, enabling:

1. Full scenario simulation with realistic timing
2. Performance metrics collection and analysis
3. Safety validation under all operating conditions
4. Autonomous operation verification
5. Stress testing and failure mode analysis
"""

import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from simulation import AqueductSimulation
from control import AutonomousController, ControlMode, PerceptionSystem, RiskLevel


@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    passed: bool
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run."""
    total_steps: int = 0
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    safety_violations: int = 0
    scenario_detections: Dict[str, int] = field(default_factory=dict)
    control_actions: Dict[str, List[float]] = field(default_factory=dict)


class HILTestRunner:
    """
    Hardware-in-the-Loop Test Runner for TAOS.

    Simulates real-time operation with realistic timing and
    comprehensive metrics collection.
    """

    def __init__(self, real_time_factor: float = 10.0):
        """
        Initialize the HIL test runner.

        Args:
            real_time_factor: Speed multiplier (10.0 = 10x real-time)
        """
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()
        self.real_time_factor = real_time_factor
        self.results: List[TestResult] = []
        self.current_metrics = PerformanceMetrics()

    def reset(self):
        """Reset simulation and controller state."""
        self.sim.reset()
        self.ctrl.reset()
        self.current_metrics = PerformanceMetrics()

    def run_simulation_step(self, dt: float = 0.5) -> Dict[str, Any]:
        """
        Run a single simulation step with timing.

        Returns:
            Dictionary containing state and actions
        """
        start_time = time.perf_counter()

        state = self.sim.get_state()
        actions = self.ctrl.decide(state)
        self.sim.step(dt, actions)

        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000

        return {
            'state': state,
            'actions': actions,
            'response_time_ms': response_time_ms
        }

    def collect_metrics(self, step_result: Dict[str, Any]):
        """Collect metrics from a simulation step."""
        self.current_metrics.total_steps += 1

        # Response time tracking
        response_times = getattr(self, '_response_times', [])
        response_times.append(step_result['response_time_ms'])
        self._response_times = response_times

        # Scenario detection tracking
        for scenario in step_result['actions'].get('active_scenarios', []):
            self.current_metrics.scenario_detections[scenario] = \
                self.current_metrics.scenario_detections.get(scenario, 0) + 1

        # Control action tracking
        for key in ['Q_in', 'Q_out']:
            if key not in self.current_metrics.control_actions:
                self.current_metrics.control_actions[key] = []
            self.current_metrics.control_actions[key].append(
                step_result['actions'].get(key, 0)
            )

        # Safety check
        if not self.sim.is_safe_state():
            self.current_metrics.safety_violations += 1

    def finalize_metrics(self):
        """Calculate final metrics after test completion."""
        if hasattr(self, '_response_times') and self._response_times:
            self.current_metrics.avg_response_time_ms = \
                statistics.mean(self._response_times)
            self.current_metrics.max_response_time_ms = max(self._response_times)
            self.current_metrics.min_response_time_ms = min(self._response_times)
        self._response_times = []


class FullScenarioHILTest:
    """Full scenario HIL test suite."""

    def __init__(self):
        self.runner = HILTestRunner()
        self.test_results: List[TestResult] = []

    def run_scenario_test(
        self,
        scenario_id: str,
        duration_steps: int = 100,
        extra_setup: callable = None
    ) -> TestResult:
        """
        Run a single scenario test.

        Args:
            scenario_id: Scenario identifier (e.g., 'S1.1')
            duration_steps: Number of simulation steps
            extra_setup: Optional function for additional setup

        Returns:
            TestResult object with test outcomes
        """
        test_name = f"HIL_{scenario_id}"
        errors = []
        warnings = []
        start_time = time.time()

        try:
            self.runner.reset()
            self.runner.sim.inject_scenario(scenario_id)

            if extra_setup:
                extra_setup(self.runner.sim)

            # Run simulation
            for _ in range(duration_steps):
                result = self.runner.run_simulation_step(dt=0.5)
                self.runner.collect_metrics(result)

                # Check for critical failures
                state = result['state']
                if state['h'] <= 0.1 or state['h'] >= 7.9:
                    warnings.append(f"Water level at boundary: {state['h']:.2f}m")

            self.runner.finalize_metrics()

            # Validate test outcomes
            passed = True
            final_state = self.runner.sim.get_state()

            # Check water level is valid
            if final_state['h'] <= 0.1 or final_state['h'] >= 7.9:
                passed = False
                errors.append(f"Final water level invalid: {final_state['h']:.2f}m")

            # Check for excessive safety violations
            if self.runner.current_metrics.safety_violations > duration_steps * 0.5:
                warnings.append(
                    f"High safety violation rate: "
                    f"{self.runner.current_metrics.safety_violations}/{duration_steps}"
                )

            duration = time.time() - start_time

            return TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=duration,
                metrics=asdict(self.runner.current_metrics),
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=time.time() - start_time,
                errors=[str(e)]
            )

    def run_all_scenarios(self) -> Dict[str, TestResult]:
        """Run all scenario tests."""
        scenarios = {
            'NORMAL': None,
            'S1.1': None,
            'S3.1': None,
            'S3.3': None,
            'S4.1': None,
            'S5.1': lambda sim: setattr(sim, 'ground_accel', 0.8),
        }

        results = {}
        for scenario_id, extra_setup in scenarios.items():
            print(f"\nRunning HIL test for {scenario_id}...")
            result = self.run_scenario_test(scenario_id, extra_setup=extra_setup)
            results[scenario_id] = result
            status = "PASSED" if result.passed else "FAILED"
            print(f"  {status} ({result.duration_seconds:.2f}s)")
            if result.errors:
                for error in result.errors:
                    print(f"    ERROR: {error}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    WARNING: {warning}")

        return results


class StressTestRunner:
    """Stress testing for extreme conditions."""

    def __init__(self):
        self.runner = HILTestRunner()

    def run_rapid_scenario_switching(
        self,
        cycles: int = 10,
        steps_per_scenario: int = 20
    ) -> TestResult:
        """Test rapid switching between scenarios."""
        test_name = "StressTest_RapidScenarioSwitch"
        scenarios = ['NORMAL', 'S1.1', 'S3.1', 'S3.3', 'S4.1', 'S5.1']
        errors = []
        warnings = []
        start_time = time.time()

        try:
            self.runner.reset()

            for cycle in range(cycles):
                for scenario in scenarios:
                    self.runner.sim.inject_scenario(scenario)
                    if scenario == 'S5.1':
                        self.runner.sim.ground_accel = 0.8

                    for _ in range(steps_per_scenario):
                        result = self.runner.run_simulation_step()
                        self.runner.collect_metrics(result)

            self.runner.finalize_metrics()

            # Check system stability
            final_state = self.runner.sim.get_state()
            passed = 0.1 < final_state['h'] < 7.9

            return TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=time.time() - start_time,
                metrics=asdict(self.runner.current_metrics),
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=time.time() - start_time,
                errors=[str(e)]
            )

    def run_extreme_conditions(self, duration_steps: int = 200) -> TestResult:
        """Test under extreme environmental conditions."""
        test_name = "StressTest_ExtremeConditions"
        errors = []
        warnings = []
        start_time = time.time()

        try:
            self.runner.reset()

            # Extreme heat
            self.runner.sim.T_ambient = 50.0
            self.runner.sim.T_sun = 70.0
            self.runner.sim.solar_rad = 1.0

            for _ in range(duration_steps // 2):
                result = self.runner.run_simulation_step()
                self.runner.collect_metrics(result)

            # Extreme cold
            self.runner.sim.T_ambient = -30.0
            self.runner.sim.T_sun = -20.0
            self.runner.sim.T_shade = -25.0
            self.runner.sim.solar_rad = 0.0

            for _ in range(duration_steps // 2):
                result = self.runner.run_simulation_step()
                self.runner.collect_metrics(result)

            self.runner.finalize_metrics()

            final_state = self.runner.sim.get_state()
            passed = 0.1 < final_state['h'] < 7.9

            return TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=time.time() - start_time,
                metrics=asdict(self.runner.current_metrics),
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            return TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=time.time() - start_time,
                errors=[str(e)]
            )


class AutonomyValidator:
    """Validates autonomous operation capabilities."""

    def __init__(self):
        self.runner = HILTestRunner()

    def validate_scenario_detection(self) -> TestResult:
        """Validate that all scenarios are correctly detected upon injection."""
        test_name = "Autonomy_ScenarioDetection"
        errors = []
        start_time = time.time()

        scenario_tests = [
            ('S1.1', lambda s: s.inject_scenario('S1.1')),
            ('S3.1', lambda s: s.inject_scenario('S3.1')),
            ('S3.3', lambda s: s.inject_scenario('S3.3')),
            ('S4.1', lambda s: s.inject_scenario('S4.1')),
            ('S5.1', lambda s: (s.inject_scenario('S5.1'), setattr(s, 'ground_accel', 0.8))),
        ]

        for expected_scenario, setup_fn in scenario_tests:
            self.runner.reset()
            setup_fn(self.runner.sim)

            # For S5.1, we need more steps for vibration to build up
            steps_to_run = 15 if expected_scenario == 'S5.1' else 2

            detected_at_any_point = False
            for _ in range(steps_to_run):
                self.runner.run_simulation_step()
                state = self.runner.sim.get_state()
                actions = self.runner.ctrl.decide(state)

                if expected_scenario in actions.get('active_scenarios', []):
                    detected_at_any_point = True
                    break

            # Also check immediately after injection for non-seismic scenarios
            if not detected_at_any_point and expected_scenario != 'S5.1':
                # Re-inject and check immediately
                self.runner.reset()
                setup_fn(self.runner.sim)
                state = self.runner.sim.get_state()
                actions = self.runner.ctrl.decide(state)
                if expected_scenario in actions.get('active_scenarios', []):
                    detected_at_any_point = True

            if not detected_at_any_point:
                errors.append(
                    f"Failed to detect {expected_scenario} at any point. "
                    f"Final detected: {actions.get('active_scenarios', [])}"
                )

        return TestResult(
            test_name=test_name,
            passed=len(errors) == 0,
            duration_seconds=time.time() - start_time,
            errors=errors
        )

    def validate_emergency_response(self) -> TestResult:
        """Validate emergency response for S5.1 + S3.3."""
        test_name = "Autonomy_EmergencyResponse"
        errors = []
        start_time = time.time()

        self.runner.reset()
        self.runner.sim.inject_scenario('S5.1')
        self.runner.sim.bearing_locked = True
        self.runner.sim.ground_accel = 0.8

        # Run until vibration builds up
        for _ in range(15):
            self.runner.run_simulation_step()

        state = self.runner.sim.get_state()
        actions = self.runner.ctrl.decide(state)

        if 'EMERGENCY' not in actions.get('status', ''):
            errors.append(f"Emergency not triggered. Status: {actions.get('status')}")

        if actions.get('Q_out') != 200.0:
            errors.append(f"Dump valve not fully open. Q_out: {actions.get('Q_out')}")

        if actions.get('Q_in') != 0.0:
            errors.append(f"Inlet not closed. Q_in: {actions.get('Q_in')}")

        return TestResult(
            test_name=test_name,
            passed=len(errors) == 0,
            duration_seconds=time.time() - start_time,
            errors=errors
        )

    def validate_recovery(self) -> TestResult:
        """Validate system recovery after scenario resolution."""
        test_name = "Autonomy_Recovery"
        errors = []
        start_time = time.time()

        self.runner.reset()

        # Inject S1.1
        self.runner.sim.inject_scenario('S1.1')
        for _ in range(30):
            self.runner.run_simulation_step()

        # Return to normal
        self.runner.sim.inject_scenario('NORMAL')
        for _ in range(50):
            self.runner.run_simulation_step()

        final_state = self.runner.sim.get_state()

        # Should recover to near-normal operation
        if final_state['fr'] > 0.9:
            errors.append(f"Froude number still high: {final_state['fr']:.2f}")

        if abs(final_state['h'] - 4.0) > 1.0:
            errors.append(f"Water level not recovered: {final_state['h']:.2f}")

        return TestResult(
            test_name=test_name,
            passed=len(errors) == 0,
            duration_seconds=time.time() - start_time,
            errors=errors
        )


def run_full_hil_test_suite():
    """Run the complete HIL test suite."""
    print("=" * 60)
    print("TAOS Hardware-in-the-Loop Test Suite")
    print("Tuanhe Aqueduct Autonomous Operation System")
    print("=" * 60)

    all_results = []

    # 1. Full Scenario Tests
    print("\n[1/3] Running Full Scenario HIL Tests...")
    scenario_tester = FullScenarioHILTest()
    scenario_results = scenario_tester.run_all_scenarios()
    all_results.extend(scenario_results.values())

    # 2. Stress Tests
    print("\n[2/3] Running Stress Tests...")
    stress_tester = StressTestRunner()

    print("  Running rapid scenario switching...")
    rapid_result = stress_tester.run_rapid_scenario_switching()
    all_results.append(rapid_result)
    print(f"    {'PASSED' if rapid_result.passed else 'FAILED'}")

    print("  Running extreme conditions test...")
    extreme_result = stress_tester.run_extreme_conditions()
    all_results.append(extreme_result)
    print(f"    {'PASSED' if extreme_result.passed else 'FAILED'}")

    # 3. Autonomy Validation
    print("\n[3/3] Running Autonomy Validation...")
    autonomy_validator = AutonomyValidator()

    print("  Validating scenario detection...")
    detection_result = autonomy_validator.validate_scenario_detection()
    all_results.append(detection_result)
    print(f"    {'PASSED' if detection_result.passed else 'FAILED'}")

    print("  Validating emergency response...")
    emergency_result = autonomy_validator.validate_emergency_response()
    all_results.append(emergency_result)
    print(f"    {'PASSED' if emergency_result.passed else 'FAILED'}")

    print("  Validating recovery...")
    recovery_result = autonomy_validator.validate_recovery()
    all_results.append(recovery_result)
    print(f"    {'PASSED' if recovery_result.passed else 'FAILED'}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed

    for result in all_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_name} ({result.duration_seconds:.2f}s)")

    print(f"\nTotal: {len(all_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(all_results)*100:.1f}%")

    return all_results


if __name__ == '__main__':
    results = run_full_hil_test_suite()
    exit(0 if all(r.passed for r in results) else 1)
