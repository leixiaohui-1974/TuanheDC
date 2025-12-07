"""
High-Fidelity Simulation System for TAOS

This module provides:
- Complete plant model with sensors and actuators
- Integrated scenario generation
- Real-time and accelerated simulation modes
- Comprehensive state logging and analysis
- Full closed-loop simulation capability
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from simulation import AqueductSimulation
from sensors import SensorSuite, SensorFaultType
from actuators import ActuatorSuite, ActuatorController, ActuatorFaultType
from mpc_controller import HybridController
from control import PerceptionSystem, ControlMode
from scenario_generator import ScenarioGenerator, ScenarioSequencer


@dataclass
class SimulationConfig:
    """Configuration for high-fidelity simulation."""
    dt: float = 0.5                    # Base time step (seconds)
    real_time_factor: float = 1.0      # 1.0 = real-time, 10.0 = 10x faster
    use_sensors: bool = True
    use_actuators: bool = True
    use_mpc: bool = True
    sensor_redundancy: bool = True
    enable_faults: bool = False
    log_interval: int = 1              # Log every N steps
    max_history: int = 10000


class PlantModel:
    """
    Complete plant model integrating physics, sensors, and actuators.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

        # Core simulation
        self.physics = AqueductSimulation()

        # Sensors
        if self.config.use_sensors:
            self.sensors = SensorSuite(
                use_redundancy=self.config.sensor_redundancy,
                use_kalman=True
            )
        else:
            self.sensors = None

        # Actuators
        if self.config.use_actuators:
            self.actuators = ActuatorSuite()
            self.actuator_controller = ActuatorController(self.actuators)
        else:
            self.actuators = None
            self.actuator_controller = None

        # State
        self.time = 0.0
        self.step_count = 0

    def reset(self):
        """Reset plant to initial conditions."""
        self.physics.reset()
        if self.sensors:
            self.sensors.reset()
        if self.actuators:
            self.actuators.reset()
        self.time = 0.0
        self.step_count = 0

    def get_true_state(self) -> Dict[str, Any]:
        """Get true physical state (for reference)."""
        return self.physics.get_state()

    def get_measured_state(self, dt: float = 0.1) -> Dict[str, Any]:
        """Get measured state through sensors."""
        true_state = self.get_true_state()

        if self.sensors:
            return self.sensors.measure(true_state, dt)
        else:
            return true_state

    def apply_control(self, Q_in: float, Q_out: float, dt: float,
                     emergency: bool = False) -> Dict[str, float]:
        """Apply control commands through actuators."""
        if self.actuator_controller:
            return self.actuator_controller.execute_command(Q_in, Q_out, dt, emergency)
        else:
            return {'Q_in_actual': Q_in, 'Q_out_actual': Q_out}

    def step(self, control_actions: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """
        Advance plant by one time step.

        Args:
            control_actions: Control commands
            dt: Time step

        Returns:
            Step result with state and diagnostics
        """
        # Apply control through actuators
        Q_in_cmd = control_actions.get('Q_in', self.physics.Q_in)
        Q_out_cmd = control_actions.get('Q_out', self.physics.Q_out)
        emergency = control_actions.get('emergency_dump', False)

        actuator_result = self.apply_control(Q_in_cmd, Q_out_cmd, dt, emergency)

        # Update physics with actual actuator outputs
        physics_actions = {
            'Q_in': actuator_result.get('Q_in_actual', Q_in_cmd),
            'Q_out': actuator_result.get('Q_out_actual', Q_out_cmd)
        }

        self.physics.step(dt, physics_actions)
        self.time += dt
        self.step_count += 1

        # Get measured state
        measured = self.get_measured_state(dt)

        return {
            'time': self.time,
            'step': self.step_count,
            'true_state': self.get_true_state(),
            'measured_state': measured,
            'actuator_state': actuator_result,
            'control_applied': physics_actions
        }

    def inject_sensor_fault(self, sensor_name: str, fault_type: SensorFaultType,
                           magnitude: float = 0.0):
        """Inject sensor fault."""
        if self.sensors:
            self.sensors.inject_sensor_fault(sensor_name, fault_type, magnitude)

    def inject_actuator_fault(self, actuator_name: str, fault_type: ActuatorFaultType,
                             magnitude: float = 0.0):
        """Inject actuator fault."""
        if self.actuators:
            self.actuators.inject_fault(actuator_name, fault_type, magnitude)


class ClosedLoopSimulation:
    """
    Complete closed-loop simulation system.
    Integrates plant, controller, and scenario generation.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

        # Plant
        self.plant = PlantModel(self.config)

        # Controller
        if self.config.use_mpc:
            self.controller = HybridController()
        else:
            from control import AutonomousController
            self.controller = AutonomousController()

        # Perception
        self.perception = PerceptionSystem()

        # Scenario generation
        self.scenario_generator = ScenarioGenerator()
        self.scenario_sequencer = ScenarioSequencer(self.scenario_generator)

        # Logging
        self.history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_steps': 0,
            'safe_steps': 0,
            'scenario_counts': {},
            'control_effort': 0.0
        }

        # Callbacks
        self.step_callbacks: List[Callable] = []

    def reset(self):
        """Reset entire simulation."""
        self.plant.reset()
        if hasattr(self.controller, 'reset'):
            self.controller.reset()
        self.scenario_generator.reset()
        self.history = []
        self.performance_metrics = {
            'total_steps': 0,
            'safe_steps': 0,
            'scenario_counts': {},
            'control_effort': 0.0
        }

    def add_callback(self, callback: Callable):
        """Add step callback function."""
        self.step_callbacks.append(callback)

    def load_scenario_sequence(self, sequence: List[tuple]):
        """Load predefined scenario sequence."""
        self.scenario_sequencer.load_sequence(sequence)

    def step(self, dt: float = None) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Returns:
            Step result with all state information
        """
        dt = dt or self.config.dt

        # Update scenarios
        new_scenario = self.scenario_sequencer.step(
            self.plant.time, self.plant.physics
        )

        # Update environment
        env_state = self.scenario_generator.update(dt, self.plant.physics)

        # Get measured state
        measured_state = self.plant.get_measured_state(dt)

        # Detect scenarios through perception
        scenarios, risks = self.perception.analyze(measured_state)

        # Compute control
        if hasattr(self.controller, 'decide'):
            control_result = self.controller.decide(measured_state, scenarios)
        else:
            control_result = self.controller.decide(measured_state)

        # Apply control
        control_actions = {
            'Q_in': control_result.get('Q_in', 80.0),
            'Q_out': control_result.get('Q_out', 80.0),
            'emergency_dump': 'EMERGENCY' in control_result.get('status', '')
        }

        # Step plant
        plant_result = self.plant.step(control_actions, dt)

        # Build step result
        step_result = {
            'time': self.plant.time,
            'step': self.plant.step_count,
            'true_state': plant_result['true_state'],
            'measured_state': measured_state,
            'detected_scenarios': scenarios,
            'risks': risks,
            'control_status': control_result.get('status', 'UNKNOWN'),
            'control_actions': control_actions,
            'actuator_state': plant_result['actuator_state'],
            'environment': env_state,
            'new_scenario': new_scenario
        }

        # Update metrics
        self._update_metrics(step_result)

        # Log if needed
        if self.plant.step_count % self.config.log_interval == 0:
            self._log_step(step_result)

        # Execute callbacks
        for callback in self.step_callbacks:
            callback(step_result)

        return step_result

    def run(self, duration: float, dt: float = None) -> Dict[str, Any]:
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration in seconds
            dt: Time step (uses config default if not specified)

        Returns:
            Simulation summary
        """
        dt = dt or self.config.dt
        start_time = self.plant.time
        end_time = start_time + duration

        while self.plant.time < end_time:
            self.step(dt)

        return self.get_summary()

    def _update_metrics(self, step_result: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics['total_steps'] += 1

        if self.plant.physics.is_safe_state():
            self.performance_metrics['safe_steps'] += 1

        for scenario in step_result['detected_scenarios']:
            counts = self.performance_metrics['scenario_counts']
            counts[scenario] = counts.get(scenario, 0) + 1

        # Control effort (sum of absolute rate changes)
        Q_in = step_result['control_actions']['Q_in']
        Q_out = step_result['control_actions']['Q_out']
        self.performance_metrics['control_effort'] += abs(Q_in - 80) + abs(Q_out - 80)

    def _log_step(self, step_result: Dict[str, Any]):
        """Log step to history."""
        if len(self.history) >= self.config.max_history:
            self.history.pop(0)

        # Create compact log entry
        entry = {
            'time': step_result['time'],
            'h': step_result['true_state']['h'],
            'v': step_result['true_state']['v'],
            'fr': step_result['true_state']['fr'],
            'T_sun': step_result['true_state']['T_sun'],
            'T_shade': step_result['true_state']['T_shade'],
            'vib_amp': step_result['true_state']['vib_amp'],
            'Q_in': step_result['control_actions']['Q_in'],
            'Q_out': step_result['control_actions']['Q_out'],
            'status': step_result['control_status'],
            'scenarios': step_result['detected_scenarios']
        }
        self.history.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        metrics = self.performance_metrics

        if metrics['total_steps'] == 0:
            safety_rate = 1.0
        else:
            safety_rate = metrics['safe_steps'] / metrics['total_steps']

        return {
            'duration': self.plant.time,
            'total_steps': metrics['total_steps'],
            'safety_rate': safety_rate,
            'scenario_counts': metrics['scenario_counts'],
            'avg_control_effort': metrics['control_effort'] / max(1, metrics['total_steps']),
            'final_state': self.plant.get_true_state()
        }

    def get_history_dataframe(self):
        """Get history as a list of dicts (can be converted to pandas DataFrame)."""
        return self.history.copy()


class FullSystemHILTest:
    """
    Full-system Hardware-in-the-Loop test suite.
    """

    def __init__(self):
        self.config = SimulationConfig(
            use_sensors=True,
            use_actuators=True,
            use_mpc=True,
            sensor_redundancy=True,
            enable_faults=True
        )
        self.sim = ClosedLoopSimulation(self.config)
        self.results: List[Dict[str, Any]] = []

    def run_scenario_test(self, scenario_id: str, duration: float = 300.0) -> Dict[str, Any]:
        """Run single scenario test."""
        self.sim.reset()

        # Inject scenario
        self.sim.scenario_generator.inject_scenario(scenario_id, self.sim.plant.physics)

        # Run simulation
        summary = self.sim.run(duration)

        result = {
            'scenario': scenario_id,
            'duration': duration,
            'passed': summary['safety_rate'] > 0.5,
            **summary
        }
        self.results.append(result)
        return result

    def run_multi_scenario_test(self, duration: float = 1800.0) -> Dict[str, Any]:
        """Run test with random scenario sequence."""
        self.sim.reset()

        # Generate random sequence
        sequence = self.sim.scenario_generator.generate_random_sequence(duration, seed=42)
        self.sim.load_scenario_sequence(sequence)

        # Run simulation
        summary = self.sim.run(duration)

        result = {
            'scenario': 'RANDOM_SEQUENCE',
            'duration': duration,
            'passed': summary['safety_rate'] > 0.3,
            'sequence': sequence,
            **summary
        }
        self.results.append(result)
        return result

    def run_fault_injection_test(self) -> Dict[str, Any]:
        """Test with sensor and actuator faults."""
        self.sim.reset()

        # Run normal for a bit
        self.sim.run(60.0)

        # Inject sensor fault
        self.sim.plant.inject_sensor_fault('h', SensorFaultType.DRIFT, 0.01)

        # Run with fault
        self.sim.run(120.0)

        # Inject actuator fault
        self.sim.plant.inject_actuator_fault('outlet', ActuatorFaultType.SLOW, 0.5)

        # Run with both faults
        self.sim.run(120.0)

        summary = self.sim.get_summary()

        result = {
            'scenario': 'FAULT_INJECTION',
            'duration': 300.0,
            'passed': summary['safety_rate'] > 0.4,
            **summary
        }
        self.results.append(result)
        return result

    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete HIL test suite."""
        print("=" * 60)
        print("Full System HIL Test Suite")
        print("=" * 60)

        # Individual scenarios
        scenarios = ['NORMAL', 'S1.1', 'S3.1', 'S3.3', 'S4.1', 'S5.1']

        for scenario in scenarios:
            print(f"\nTesting {scenario}...")
            result = self.run_scenario_test(scenario, duration=120.0)
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {status} - Safety: {result['safety_rate']:.1%}")

        # Multi-scenario test
        print("\nRunning multi-scenario sequence...")
        result = self.run_multi_scenario_test(duration=600.0)
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {status} - Safety: {result['safety_rate']:.1%}")

        # Fault injection test
        print("\nRunning fault injection test...")
        result = self.run_fault_injection_test()
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {status} - Safety: {result['safety_rate']:.1%}")

        # Summary
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")

        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total,
            'results': self.results
        }


if __name__ == '__main__':
    tester = FullSystemHILTest()
    tester.run_full_test_suite()
