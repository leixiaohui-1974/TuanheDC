"""
Advanced Scenario Generator for TAOS

This module provides:
- Multi-physics coupling scenario generation
- Time-varying environmental profiles
- Probabilistic scenario sequences
- Realistic disturbance patterns
- Scenario combination and transition logic
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class ScenarioSeverity(Enum):
    """Scenario severity levels."""
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4


@dataclass
class ScenarioProfile:
    """Definition of a scenario profile."""
    id: str
    name: str
    description: str
    severity: ScenarioSeverity
    duration_range: Tuple[float, float]  # seconds
    parameters: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)


class EnvironmentProfile:
    """Time-varying environmental conditions."""

    def __init__(self, profile_type: str = 'diurnal'):
        self.profile_type = profile_type
        self.time = 0.0

    def get_conditions(self, t: float) -> Dict[str, float]:
        """Get environmental conditions at time t (seconds)."""
        if self.profile_type == 'diurnal':
            return self._diurnal_profile(t)
        elif self.profile_type == 'storm':
            return self._storm_profile(t)
        elif self.profile_type == 'winter':
            return self._winter_profile(t)
        elif self.profile_type == 'summer':
            return self._summer_profile(t)
        else:
            return self._constant_profile()

    def _diurnal_profile(self, t: float) -> Dict[str, float]:
        """24-hour diurnal cycle."""
        hour = (t / 3600.0) % 24.0

        # Temperature variation
        T_base = 20.0
        T_amp = 10.0
        T_ambient = T_base + T_amp * np.sin((hour - 6) * np.pi / 12.0)

        # Solar radiation (0 at night, peaks at noon)
        if 6 <= hour <= 18:
            solar_rad = np.sin((hour - 6) * np.pi / 12.0)
        else:
            solar_rad = 0.0

        # Wind (typically stronger in afternoon)
        wind_base = 2.0
        wind_var = 3.0 * np.sin((hour - 12) * np.pi / 12.0) if 8 <= hour <= 20 else 0
        wind_speed = max(0, wind_base + wind_var + np.random.normal(0, 0.5))

        return {
            'T_ambient': T_ambient,
            'solar_rad': solar_rad,
            'wind_speed': wind_speed
        }

    def _storm_profile(self, t: float) -> Dict[str, float]:
        """Storm conditions with high wind and rain."""
        # Storm intensity varies over time
        storm_phase = (t % 7200) / 7200.0  # 2-hour cycle

        T_ambient = 15.0 - 5.0 * np.sin(storm_phase * 2 * np.pi)
        solar_rad = 0.1 * (1 - np.sin(storm_phase * np.pi))
        wind_speed = 8.0 + 7.0 * np.sin(storm_phase * 2 * np.pi) + np.random.normal(0, 2)

        return {
            'T_ambient': T_ambient,
            'solar_rad': max(0, solar_rad),
            'wind_speed': max(0, wind_speed)
        }

    def _winter_profile(self, t: float) -> Dict[str, float]:
        """Cold winter conditions."""
        hour = (t / 3600.0) % 24.0

        T_base = -5.0
        T_amp = 8.0
        T_ambient = T_base + T_amp * np.sin((hour - 6) * np.pi / 12.0)

        solar_rad = 0.3 * max(0, np.sin((hour - 8) * np.pi / 8.0)) if 8 <= hour <= 16 else 0

        return {
            'T_ambient': T_ambient,
            'solar_rad': solar_rad,
            'wind_speed': 3.0 + np.random.normal(0, 1.5)
        }

    def _summer_profile(self, t: float) -> Dict[str, float]:
        """Hot summer conditions."""
        hour = (t / 3600.0) % 24.0

        T_base = 30.0
        T_amp = 12.0
        T_ambient = T_base + T_amp * np.sin((hour - 6) * np.pi / 12.0)

        solar_rad = max(0, np.sin((hour - 5) * np.pi / 14.0)) if 5 <= hour <= 19 else 0

        return {
            'T_ambient': T_ambient,
            'solar_rad': solar_rad,
            'wind_speed': 2.0 + np.random.normal(0, 1.0)
        }

    def _constant_profile(self) -> Dict[str, float]:
        """Constant conditions for testing."""
        return {
            'T_ambient': 25.0,
            'solar_rad': 0.5,
            'wind_speed': 2.0
        }


class ScenarioGenerator:
    """
    Advanced scenario generator with multi-physics coupling.
    """

    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.active_scenarios: List[str] = []
        self.environment = EnvironmentProfile('diurnal')
        self.time = 0.0
        self.scenario_start_times: Dict[str, float] = {}
        self.transition_matrix = self._build_transition_matrix()

    def _define_scenarios(self) -> Dict[str, ScenarioProfile]:
        """Define all available scenarios."""
        return {
            'NORMAL': ScenarioProfile(
                id='NORMAL',
                name='Normal Operation',
                description='Standard operating conditions',
                severity=ScenarioSeverity.MILD,
                duration_range=(3600, 86400),
                parameters={'Q_in': 80.0}
            ),

            'S1.1': ScenarioProfile(
                id='S1.1',
                name='Hydraulic Jump',
                description='High Froude number causing flow instability',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(60, 600),
                parameters={'Q_in': 140.0, 'h_initial': 2.0},
                triggers=['high_inflow', 'gate_malfunction'],
                incompatible_with=['S5.1']
            ),

            'S1.2': ScenarioProfile(
                id='S1.2',
                name='Surge Wave',
                description='Transient surge from upstream',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(30, 300),
                parameters={'Q_surge': 50.0, 'surge_duration': 30.0}
            ),

            'S2.1': ScenarioProfile(
                id='S2.1',
                name='Vortex-Induced Vibration',
                description='Wind-induced structural vibration',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(300, 3600),
                parameters={'wind_critical': 12.0},
                triggers=['high_wind']
            ),

            'S3.1': ScenarioProfile(
                id='S3.1',
                name='Thermal Bending',
                description='Sun/shade temperature differential',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(1800, 7200),
                parameters={'T_sun': 45.0, 'T_shade': 28.0, 'solar_rad': 1.0},
                triggers=['summer_noon', 'clear_sky']
            ),

            'S3.2': ScenarioProfile(
                id='S3.2',
                name='Rapid Cooling',
                description='Sudden temperature drop',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(600, 1800),
                parameters={'cooling_rate': 5.0}  # degrees per hour
            ),

            'S3.3': ScenarioProfile(
                id='S3.3',
                name='Bearing Lock',
                description='Thermal expansion causing bearing lockup',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(600, 3600),
                parameters={'bearing_locked': True, 'T_ambient': -10.0}
            ),

            'S4.1': ScenarioProfile(
                id='S4.1',
                name='Joint Gap Expansion',
                description='Cold weather joint opening',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(3600, 14400),
                parameters={'T_ambient': -15.0, 'T_sun': -10.0, 'T_shade': -10.0}
            ),

            'S4.2': ScenarioProfile(
                id='S4.2',
                name='Joint Compression',
                description='Hot weather joint closing',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(3600, 14400),
                parameters={'T_ambient': 40.0, 'T_sun': 55.0}
            ),

            'S5.1': ScenarioProfile(
                id='S5.1',
                name='Seismic Event',
                description='Earthquake ground acceleration',
                severity=ScenarioSeverity.EXTREME,
                duration_range=(10, 120),
                parameters={'ground_accel': 0.5, 'frequency': 2.0},
                incompatible_with=['S1.1', 'S2.1']
            ),

            'S5.2': ScenarioProfile(
                id='S5.2',
                name='Aftershock Sequence',
                description='Multiple smaller seismic events',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(300, 1800),
                parameters={'num_shocks': 5, 'max_accel': 0.3}
            ),

            'S6.1': ScenarioProfile(
                id='S6.1',
                name='Sensor Degradation',
                description='Progressive sensor failure',
                severity=ScenarioSeverity.MODERATE,
                duration_range=(1800, 7200),
                parameters={'affected_sensors': ['h', 'v'], 'degradation_rate': 0.1}
            ),

            'S6.2': ScenarioProfile(
                id='S6.2',
                name='Actuator Fault',
                description='Gate actuator malfunction',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(300, 1800),
                parameters={'affected_actuator': 'outlet', 'fault_type': 'slow'}
            ),

            'COMBINED_THERMAL_SEISMIC': ScenarioProfile(
                id='COMBINED_THERMAL_SEISMIC',
                name='Thermal + Seismic',
                description='Combined thermal stress and earthquake',
                severity=ScenarioSeverity.EXTREME,
                duration_range=(30, 120),
                parameters={'ground_accel': 0.5, 'bearing_locked': True}
            ),

            'MULTI_PHYSICS': ScenarioProfile(
                id='MULTI_PHYSICS',
                name='Multi-Physics Coupling',
                description='Wind, thermal, and hydraulic combined',
                severity=ScenarioSeverity.SEVERE,
                duration_range=(1800, 7200),
                parameters={
                    'wind_speed': 12.0,
                    'T_sun': 40.0,
                    'T_shade': 25.0,
                    'Q_in': 120.0
                }
            )
        }

    def _build_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build scenario transition probability matrix."""
        return {
            'NORMAL': {'S1.1': 0.02, 'S3.1': 0.05, 'S4.1': 0.03, 'S5.1': 0.001, 'NORMAL': 0.9},
            'S1.1': {'NORMAL': 0.3, 'S1.1': 0.6, 'S5.1': 0.0, 'S3.1': 0.1},
            'S3.1': {'NORMAL': 0.2, 'S3.1': 0.7, 'S3.3': 0.1},
            'S3.3': {'NORMAL': 0.1, 'S3.3': 0.8, 'S5.1': 0.02, 'COMBINED_THERMAL_SEISMIC': 0.08},
            'S4.1': {'NORMAL': 0.3, 'S4.1': 0.6, 'S3.3': 0.1},
            'S5.1': {'NORMAL': 0.5, 'S5.2': 0.3, 'COMBINED_THERMAL_SEISMIC': 0.2},
        }

    def set_environment_profile(self, profile_type: str):
        """Set environmental profile type."""
        self.environment = EnvironmentProfile(profile_type)

    def inject_scenario(self, scenario_id: str, sim: Any):
        """
        Inject scenario into simulation.

        Args:
            scenario_id: Scenario identifier
            sim: Simulation object to modify
        """
        if scenario_id not in self.scenarios:
            return

        profile = self.scenarios[scenario_id]
        params = profile.parameters

        # Check compatibility
        for active in self.active_scenarios:
            if active in profile.incompatible_with:
                return

        # Apply parameters
        if scenario_id == 'NORMAL':
            sim.Q_in = params.get('Q_in', 80.0)
            sim.bearing_locked = False
            sim.ground_accel = 0.0

        elif scenario_id == 'S1.1':
            sim.Q_in = params.get('Q_in', 140.0)
            sim.h = params.get('h_initial', 2.0)
            sim.v = sim.Q_in / (sim.Width * sim.h)

        elif scenario_id == 'S1.2':
            # Surge wave - temporary flow increase
            sim.Q_in += params.get('Q_surge', 50.0)

        elif scenario_id == 'S2.1':
            sim.wind_speed = params.get('wind_critical', 12.0)

        elif scenario_id == 'S3.1':
            sim.T_sun = params.get('T_sun', 45.0)
            sim.T_shade = params.get('T_shade', 28.0)
            sim.solar_rad = params.get('solar_rad', 1.0)
            sim.T_ambient = 35.0

        elif scenario_id == 'S3.3':
            sim.bearing_locked = params.get('bearing_locked', True)
            sim.T_ambient = params.get('T_ambient', -10.0)

        elif scenario_id == 'S4.1':
            sim.T_ambient = params.get('T_ambient', -15.0)
            sim.T_sun = params.get('T_sun', -10.0)
            sim.T_shade = params.get('T_shade', -10.0)
            sim.solar_rad = 0.0

        elif scenario_id == 'S4.2':
            sim.T_ambient = params.get('T_ambient', 40.0)
            sim.T_sun = params.get('T_sun', 55.0)
            sim.solar_rad = 1.0

        elif scenario_id == 'S5.1':
            sim.ground_accel = params.get('ground_accel', 0.5)

        elif scenario_id == 'COMBINED_THERMAL_SEISMIC':
            sim.ground_accel = params.get('ground_accel', 0.5)
            sim.bearing_locked = params.get('bearing_locked', True)

        elif scenario_id == 'MULTI_PHYSICS':
            sim.wind_speed = params.get('wind_speed', 12.0)
            sim.T_sun = params.get('T_sun', 40.0)
            sim.T_shade = params.get('T_shade', 25.0)
            sim.Q_in = params.get('Q_in', 120.0)

        # Track active scenario
        if scenario_id not in self.active_scenarios:
            self.active_scenarios.append(scenario_id)
            self.scenario_start_times[scenario_id] = self.time

    def update(self, dt: float, sim: Any) -> Dict[str, Any]:
        """
        Update environment and check for scenario transitions.

        Args:
            dt: Time step
            sim: Simulation object

        Returns:
            Current environment and scenario state
        """
        self.time += dt

        # Get environmental conditions
        env = self.environment.get_conditions(self.time)

        # Apply environment to simulation
        sim.T_ambient = env['T_ambient']
        sim.solar_rad = env['solar_rad']
        sim.wind_speed = env['wind_speed']

        # Check scenario durations
        expired = []
        for scenario_id in self.active_scenarios:
            if scenario_id in self.scenario_start_times:
                elapsed = self.time - self.scenario_start_times[scenario_id]
                profile = self.scenarios.get(scenario_id)
                if profile and elapsed > profile.duration_range[1]:
                    expired.append(scenario_id)

        # Remove expired scenarios
        for scenario_id in expired:
            self.active_scenarios.remove(scenario_id)
            del self.scenario_start_times[scenario_id]

        return {
            'time': self.time,
            'environment': env,
            'active_scenarios': self.active_scenarios.copy()
        }

    def generate_random_sequence(self, duration: float, seed: int = None) -> List[Tuple[float, str]]:
        """
        Generate random scenario sequence for testing.

        Args:
            duration: Total duration in seconds
            seed: Random seed for reproducibility

        Returns:
            List of (time, scenario_id) tuples
        """
        if seed is not None:
            np.random.seed(seed)

        sequence = [(0.0, 'NORMAL')]
        t = 0.0
        current = 'NORMAL'

        while t < duration:
            # Get transition probabilities
            transitions = self.transition_matrix.get(current, {'NORMAL': 1.0})

            # Sample next scenario
            scenarios = list(transitions.keys())
            probs = list(transitions.values())
            probs = np.array(probs) / sum(probs)

            next_scenario = np.random.choice(scenarios, p=probs)

            # Sample duration
            profile = self.scenarios.get(next_scenario)
            if profile:
                dur_min, dur_max = profile.duration_range
                duration_s = np.random.uniform(dur_min, dur_max)
            else:
                duration_s = 600.0

            t += duration_s
            if t < duration:
                sequence.append((t, next_scenario))
                current = next_scenario

        return sequence

    def get_scenario_info(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a scenario."""
        if scenario_id not in self.scenarios:
            return None

        profile = self.scenarios[scenario_id]
        return {
            'id': profile.id,
            'name': profile.name,
            'description': profile.description,
            'severity': profile.severity.name,
            'duration_range': profile.duration_range,
            'parameters': profile.parameters
        }

    def reset(self):
        """Reset generator state."""
        self.active_scenarios = []
        self.scenario_start_times = {}
        self.time = 0.0


class ScenarioSequencer:
    """
    Sequences and executes scenario tests.
    """

    def __init__(self, generator: ScenarioGenerator):
        self.generator = generator
        self.sequence: List[Tuple[float, str]] = []
        self.current_index = 0

    def load_sequence(self, sequence: List[Tuple[float, str]]):
        """Load a scenario sequence."""
        self.sequence = sorted(sequence, key=lambda x: x[0])
        self.current_index = 0

    def load_from_json(self, json_str: str):
        """Load sequence from JSON."""
        data = json.loads(json_str)
        self.sequence = [(item['time'], item['scenario']) for item in data]
        self.current_index = 0

    def step(self, current_time: float, sim: Any) -> Optional[str]:
        """
        Execute scenario changes based on current time.

        Returns:
            New scenario if one was activated, None otherwise
        """
        if self.current_index >= len(self.sequence):
            return None

        next_time, next_scenario = self.sequence[self.current_index]

        if current_time >= next_time:
            self.generator.inject_scenario(next_scenario, sim)
            self.current_index += 1
            return next_scenario

        return None

    def get_progress(self) -> float:
        """Get sequence completion progress (0-1)."""
        if not self.sequence:
            return 1.0
        return self.current_index / len(self.sequence)
