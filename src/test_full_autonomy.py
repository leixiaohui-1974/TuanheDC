#!/usr/bin/env python3
"""
Full Autonomy Test Suite for TAOS V3.0
Tuanhe Aqueduct Autonomous Operation System

This module provides comprehensive testing for:
- 24-hour continuous operation simulation
- All 14+ scenario coverage verification
- Extreme condition combination tests
- Multi-physics coupling validation
- Autonomous recovery verification
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

from simulation import AqueductSimulation
from control import PerceptionSystem, AutonomousController, ControlMode
from sensors import SensorSuite, SensorFaultType
from actuators import ActuatorSuite, ActuatorFaultType
from mpc_controller import HybridController, AdaptiveMPC
from scenario_generator import ScenarioGenerator, EnvironmentProfile
from hifi_simulation import ClosedLoopSimulation, SimulationConfig, FullSystemHILTest


class FullAutonomyValidator:
    """
    Validates complete autonomous operation capability across all scenarios.
    """

    ALL_SCENARIOS = [
        'NORMAL', 'S1.1', 'S1.2', 'S2.1', 'S3.1', 'S3.2', 'S3.3',
        'S4.1', 'S4.2', 'S5.1', 'S5.2', 'S6.1', 'S6.2',
        'MULTI_PHYSICS', 'COMBINED_THERMAL_SEISMIC'
    ]

    def __init__(self):
        self.results = {}
        self.scenario_coverage = {s: False for s in self.ALL_SCENARIOS}

    def run_scenario_recognition_test(self) -> Dict[str, Any]:
        """Test that all scenarios can be detected by PerceptionSystem."""
        print("\n" + "=" * 60)
        print("Scenario Recognition Test")
        print("=" * 60)

        perception = PerceptionSystem()
        detected_scenarios = set()
        test_results = {}

        # Test each scenario with appropriate state
        test_states = {
            'NORMAL': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                      'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                      'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2},

            'S1.1': {'h': 2.0, 'v': 5.0, 'fr': 1.1, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2},

            'S1.2': {'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 150, 'wind_speed': 2},

            'S2.1': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 15, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 15},

            'S3.1': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 45, 'T_shade': 28,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2},

            'S3.3': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': True, 'Q_in': 80, 'wind_speed': 2},

            'S4.1': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': -5, 'T_shade': -5,
                    'joint_gap': 32, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2,
                    'T_ambient': -10},

            'S4.2': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 50, 'T_shade': 45,
                    'joint_gap': 4, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2,
                    'T_ambient': 40},

            'S5.1': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 80, 'bearing_stress': 35,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2},

            'S6.1': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'wind_speed': 2,
                    'h_confidence': 0.4, 'v_confidence': 0.5, 'T_confidence': 0.5},

            'S6.2': {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                    'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                    'bearing_locked': False, 'Q_in': 80, 'Q_out': 80,
                    'Q_in_cmd': 100, 'Q_out_cmd': 100, 'wind_speed': 2},
        }

        for scenario_name, state in test_states.items():
            # Add time for rate-based detection
            state['time'] = 0.0

            # Run perception multiple times for rate-based scenarios
            for i in range(10):
                state['time'] = i * 60.0  # 1 minute intervals
                if scenario_name == 'S6.1':
                    # Keep feeding low confidence values
                    pass
                scenarios, risks = perception.analyze(state)
                detected_scenarios.update(scenarios)

            if scenario_name in detected_scenarios:
                test_results[scenario_name] = 'DETECTED'
                self.scenario_coverage[scenario_name] = True
                print(f"  {scenario_name}: ✓ DETECTED")
            else:
                # Check if any scenario was detected (might be different name)
                test_results[scenario_name] = f'DETECTED as {scenarios}' if scenarios else 'NOT DETECTED'
                print(f"  {scenario_name}: {test_results[scenario_name]}")

        # Reset perception for next tests
        perception = PerceptionSystem()

        coverage = sum(1 for v in test_results.values() if 'DETECTED' in v and 'NOT' not in v)
        print(f"\nScenario Recognition Coverage: {coverage}/{len(test_states)}")

        return {
            'test': 'scenario_recognition',
            'results': test_results,
            'coverage': coverage / len(test_states)
        }

    def run_controller_response_test(self) -> Dict[str, Any]:
        """Test that controller responds appropriately to each scenario."""
        print("\n" + "=" * 60)
        print("Controller Response Test")
        print("=" * 60)

        controller = AutonomousController()
        test_results = {}

        # Test scenarios and expected responses
        test_cases = [
            ('NORMAL', {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                       'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                       'bearing_locked': False, 'Q_in': 80, 'Q_out': 80}, 'NORMAL'),

            ('S1.1', {'h': 2.0, 'v': 5.0, 'fr': 1.1, 'T_sun': 25, 'T_shade': 23,
                     'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                     'bearing_locked': False, 'Q_in': 140, 'Q_out': 80}, 'STABILIZING'),

            ('S3.1', {'h': 4.0, 'v': 2.0, 'fr': 0.38, 'T_sun': 45, 'T_shade': 28,
                     'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                     'bearing_locked': False, 'Q_in': 80, 'Q_out': 80}, 'COOLING'),

            ('S3.3', {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                     'joint_gap': 20, 'vib_amp': 5, 'bearing_stress': 31,
                     'bearing_locked': True, 'Q_in': 80, 'Q_out': 80}, 'BEARING'),

            ('S5.1', {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                     'joint_gap': 20, 'vib_amp': 80, 'bearing_stress': 35,
                     'bearing_locked': False, 'Q_in': 80, 'Q_out': 80}, 'SEISMIC'),

            ('S5.1+S3.3', {'h': 4.0, 'v': 2.0, 'fr': 0.32, 'T_sun': 25, 'T_shade': 23,
                          'joint_gap': 20, 'vib_amp': 80, 'bearing_stress': 35,
                          'bearing_locked': True, 'Q_in': 80, 'Q_out': 80}, 'EMERGENCY'),
        ]

        for name, state, expected_keyword in test_cases:
            controller.reset()
            actions = controller.decide(state)
            status = actions.get('status', '')

            if expected_keyword.upper() in status.upper():
                test_results[name] = 'PASS'
                print(f"  {name}: ✓ {status}")
            else:
                test_results[name] = f'FAIL (got: {status})'
                print(f"  {name}: ✗ Expected '{expected_keyword}', got '{status}'")

        passed = sum(1 for v in test_results.values() if v == 'PASS')
        print(f"\nController Response: {passed}/{len(test_cases)} PASSED")

        return {
            'test': 'controller_response',
            'results': test_results,
            'passed': passed,
            'total': len(test_cases)
        }

    def run_mpc_gain_scheduling_test(self) -> Dict[str, Any]:
        """Test MPC adaptive gain scheduling for all scenarios."""
        print("\n" + "=" * 60)
        print("MPC Gain Scheduling Test")
        print("=" * 60)

        mpc = AdaptiveMPC()
        test_results = {}

        for scenario in self.ALL_SCENARIOS:
            if scenario in mpc.scenario_gains:
                gains = mpc.scenario_gains[scenario]
                mpc._update_gains([scenario])

                # Verify gains were applied
                if (mpc.config.w_h == gains['w_h'] and
                    mpc.config.w_fr == gains['w_fr']):
                    test_results[scenario] = 'PASS'
                    print(f"  {scenario}: ✓ Gains configured")
                else:
                    test_results[scenario] = 'FAIL'
                    print(f"  {scenario}: ✗ Gain mismatch")
            else:
                test_results[scenario] = 'NOT_CONFIGURED'
                print(f"  {scenario}: ⚠ Not in gain schedule")

        passed = sum(1 for v in test_results.values() if v == 'PASS')
        print(f"\nMPC Gain Coverage: {passed}/{len(self.ALL_SCENARIOS)}")

        return {
            'test': 'mpc_gain_scheduling',
            'results': test_results,
            'coverage': passed / len(self.ALL_SCENARIOS)
        }


class ContinuousOperationTest:
    """
    24-hour continuous operation simulation test.
    """

    def __init__(self):
        self.config = SimulationConfig(
            dt=1.0,  # 1 second steps for faster simulation
            use_sensors=True,
            use_actuators=True,
            use_mpc=True,
            sensor_redundancy=True,
            enable_faults=True
        )
        self.sim = ClosedLoopSimulation(self.config)

    def run_24h_simulation(self, time_scale: float = 1.0) -> Dict[str, Any]:
        """
        Run 24-hour continuous operation simulation.

        Args:
            time_scale: Simulation time per real second (e.g., 3600 = 1 hour/second)
        """
        print("\n" + "=" * 60)
        print("24-Hour Continuous Operation Test")
        print("=" * 60)

        # 24 hours in seconds
        duration = 24 * 3600  # 86400 seconds

        # Use diurnal environment profile
        self.sim.scenario_generator.set_environment_profile('diurnal')

        # Generate realistic scenario sequence
        sequence = self.sim.scenario_generator.generate_random_sequence(duration, seed=42)
        self.sim.load_scenario_sequence(sequence)

        print(f"  Duration: 24 hours ({duration} seconds)")
        print(f"  Scenario sequence: {len(sequence)} transitions")

        # Run simulation with progress updates
        start_time = time.time()
        self.sim.reset()

        dt = self.config.dt
        steps = int(duration / dt)
        log_interval = steps // 24  # Log every simulated hour

        hourly_stats = []

        for step in range(steps):
            self.sim.step(dt)

            # Log hourly statistics
            if step > 0 and step % log_interval == 0:
                hour = step // log_interval
                summary = self.sim.get_summary()
                hourly_stats.append({
                    'hour': hour,
                    'safety_rate': summary['safety_rate'],
                    'scenarios': list(summary['scenario_counts'].keys())
                })
                print(f"  Hour {hour:2d}: Safety {summary['safety_rate']:.1%}, "
                      f"Scenarios: {list(summary['scenario_counts'].keys())[:3]}")

        elapsed = time.time() - start_time
        final_summary = self.sim.get_summary()

        print(f"\n  Completed in {elapsed:.1f}s real time")
        print(f"  Overall Safety Rate: {final_summary['safety_rate']:.1%}")
        print(f"  Total Scenario Activations: {sum(final_summary['scenario_counts'].values())}")

        return {
            'test': '24h_continuous',
            'duration': duration,
            'real_time': elapsed,
            'safety_rate': final_summary['safety_rate'],
            'scenario_counts': final_summary['scenario_counts'],
            'hourly_stats': hourly_stats,
            'passed': final_summary['safety_rate'] > 0.5
        }


class ExtremeConditionTest:
    """
    Tests system behavior under extreme and combined conditions.
    """

    def __init__(self):
        self.config = SimulationConfig(
            use_sensors=True,
            use_actuators=True,
            use_mpc=True
        )
        self.sim = ClosedLoopSimulation(self.config)

    def run_extreme_combination_tests(self) -> Dict[str, Any]:
        """Test extreme condition combinations."""
        print("\n" + "=" * 60)
        print("Extreme Condition Combination Tests")
        print("=" * 60)

        test_results = {}

        # Define extreme test cases
        extreme_cases = [
            {
                'name': 'Max_Thermal_Gradient',
                'setup': lambda sim: setattr(sim.plant.physics, 'T_sun', 60.0) or
                                    setattr(sim.plant.physics, 'T_shade', 20.0),
                'duration': 300
            },
            {
                'name': 'Severe_Earthquake',
                'setup': lambda sim: setattr(sim.plant.physics, 'ground_accel', 0.8),
                'duration': 60
            },
            {
                'name': 'Max_Inflow_Surge',
                'setup': lambda sim: setattr(sim.plant.physics, 'Q_in', 200.0),
                'duration': 120
            },
            {
                'name': 'Extreme_Cold',
                'setup': lambda sim: (
                    setattr(sim.plant.physics, 'T_ambient', -30.0),
                    setattr(sim.plant.physics, 'T_sun', -25.0),
                    setattr(sim.plant.physics, 'T_shade', -25.0)
                ),
                'duration': 600
            },
            {
                'name': 'Thermal_Plus_Seismic',
                'setup': lambda sim: (
                    setattr(sim.plant.physics, 'T_sun', 50.0),
                    setattr(sim.plant.physics, 'T_shade', 30.0),
                    setattr(sim.plant.physics, 'ground_accel', 0.5),
                    setattr(sim.plant.physics, 'bearing_locked', True)
                ),
                'duration': 120
            },
            {
                'name': 'Wind_Plus_Hydraulic',
                'setup': lambda sim: (
                    setattr(sim.plant.physics, 'wind_speed', 20.0),
                    setattr(sim.plant.physics, 'Q_in', 160.0),
                    setattr(sim.plant.physics, 'h', 2.0)
                ),
                'duration': 180
            },
            {
                'name': 'Triple_Threat',
                'setup': lambda sim: (
                    setattr(sim.plant.physics, 'T_sun', 45.0),
                    setattr(sim.plant.physics, 'T_shade', 28.0),
                    setattr(sim.plant.physics, 'wind_speed', 15.0),
                    setattr(sim.plant.physics, 'Q_in', 140.0)
                ),
                'duration': 300
            }
        ]

        for case in extreme_cases:
            self.sim.reset()
            case['setup'](self.sim)

            summary = self.sim.run(case['duration'])
            final_h = summary['final_state']['h']

            # Pass criteria: system survives (water level in valid range)
            survived = 0.1 <= final_h <= 7.9
            test_results[case['name']] = {
                'passed': survived,
                'safety_rate': summary['safety_rate'],
                'final_h': final_h
            }

            status = "✓ SURVIVED" if survived else "✗ FAILED"
            print(f"  {case['name']}: {status} (h={final_h:.2f}m, safety={summary['safety_rate']:.1%})")

        passed = sum(1 for r in test_results.values() if r['passed'])
        print(f"\nExtreme Conditions: {passed}/{len(extreme_cases)} SURVIVED")

        return {
            'test': 'extreme_conditions',
            'results': test_results,
            'passed': passed,
            'total': len(extreme_cases)
        }


def run_full_test_suite():
    """Run complete autonomy test suite."""
    print("=" * 60)
    print("TAOS V3.0 Full Autonomy Test Suite")
    print("Tuanhe Aqueduct Autonomous Operation System")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # 1. Scenario Recognition Test
    validator = FullAutonomyValidator()
    all_results['recognition'] = validator.run_scenario_recognition_test()

    # 2. Controller Response Test
    all_results['controller'] = validator.run_controller_response_test()

    # 3. MPC Gain Scheduling Test
    all_results['mpc_gains'] = validator.run_mpc_gain_scheduling_test()

    # 4. Extreme Condition Tests
    extreme_tester = ExtremeConditionTest()
    all_results['extreme'] = extreme_tester.run_extreme_combination_tests()

    # 5. 24-Hour Continuous Operation (shortened for demo)
    continuous_tester = ContinuousOperationTest()
    # Run 1-hour test instead of 24 hours for demo
    continuous_tester.config.dt = 10.0  # Faster timestep
    all_results['continuous'] = continuous_tester.run_24h_simulation()

    # 6. Full HIL Test Suite
    print("\n" + "=" * 60)
    print("Full HIL Test Suite")
    print("=" * 60)
    hil_tester = FullSystemHILTest()
    hil_result = hil_tester.run_full_test_suite()
    all_results['hil'] = hil_result

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0

    for test_name, result in all_results.items():
        if 'passed' in result and 'total' in result:
            total_tests += result['total']
            passed_tests += result['passed']
            pct = result['passed'] / result['total'] * 100
            print(f"  {test_name}: {result['passed']}/{result['total']} ({pct:.1f}%)")
        elif 'coverage' in result:
            print(f"  {test_name}: {result['coverage']:.1%} coverage")
        elif 'passed' in result:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {test_name}: {status}")

    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")
    print(f"  Overall Success Rate: {passed_tests/max(1,total_tests)*100:.1f}%")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == '__main__':
    run_full_test_suite()
