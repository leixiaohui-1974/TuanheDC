"""
High-Fidelity Sensor Simulation Module for TAOS V3.10

This module provides comprehensive sensor simulation with:
- Physics-based sensor models (optical, ultrasonic, thermal, strain gauge, etc.)
- Environmental interference modeling
- Cross-sensor correlation and coupling effects
- Degradation and aging models
- Calibration drift simulation
- Multi-physics sensor fusion

Author: TAOS Development Team
Version: 3.10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class SensorType(Enum):
    """Physical sensor types with different characteristics."""
    ULTRASONIC_LEVEL = "ultrasonic_level"       # Water level measurement
    RADAR_LEVEL = "radar_level"                  # Radar water level
    PRESSURE_LEVEL = "pressure_level"            # Pressure-based level
    DOPPLER_VELOCITY = "doppler_velocity"        # Acoustic Doppler
    ELECTROMAGNETIC_FLOW = "em_flow"             # EM flow meter
    RTD_TEMPERATURE = "rtd_temp"                 # RTD temperature sensor
    THERMOCOUPLE = "thermocouple"                # Thermocouple
    INFRARED_TEMP = "ir_temp"                    # Non-contact IR
    STRAIN_GAUGE = "strain_gauge"                # Strain measurement
    LVDT_DISPLACEMENT = "lvdt"                   # Linear displacement
    ACCELEROMETER = "accelerometer"              # Vibration/seismic
    FIBER_OPTIC = "fiber_optic"                  # Distributed sensing


class SensorDegradationMode(Enum):
    """Sensor degradation modes."""
    NONE = "none"
    LINEAR_DRIFT = "linear_drift"               # Gradual linear drift
    EXPONENTIAL_DECAY = "exponential_decay"     # Sensitivity decay
    STEP_CHANGE = "step_change"                 # Sudden calibration shift
    INTERMITTENT = "intermittent"               # Random dropouts
    FOULING = "fouling"                         # Sensor fouling (buildup)
    CORROSION = "corrosion"                     # Corrosion effects


@dataclass
class SensorPhysicsModel:
    """Physics-based sensor model parameters."""
    sensor_type: SensorType

    # Measurement range and resolution
    range_min: float = 0.0
    range_max: float = 100.0
    resolution: float = 0.01

    # Accuracy specifications
    accuracy_percent: float = 0.5
    repeatability: float = 0.1
    hysteresis: float = 0.05

    # Dynamic response
    response_time_ms: float = 100.0
    bandwidth_hz: float = 10.0

    # Environmental sensitivity
    temp_coefficient: float = 0.01          # %/°C
    pressure_sensitivity: float = 0.0       # %/kPa
    humidity_sensitivity: float = 0.0       # %/%RH

    # Noise characteristics
    white_noise_std: float = 0.01
    pink_noise_amplitude: float = 0.005

    # Power and installation
    supply_voltage: float = 24.0            # V
    power_consumption: float = 5.0          # W
    cable_length_m: float = 50.0
    cable_resistance_per_m: float = 0.05    # Ohm/m


@dataclass
class SensorCalibration:
    """Sensor calibration parameters."""
    zero_offset: float = 0.0
    span_factor: float = 1.0
    linearity_coefficients: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    last_calibration_time: float = 0.0
    calibration_interval_days: float = 90.0
    drift_rate_per_day: float = 0.001


class HighFidelitySensor:
    """
    High-fidelity sensor simulation with physics-based modeling.

    Implements realistic sensor behavior including:
    - Physical measurement principles
    - Environmental interference
    - Degradation and aging
    - Calibration drift
    """

    def __init__(self, name: str, physics: SensorPhysicsModel,
                 calibration: Optional[SensorCalibration] = None):
        self.name = name
        self.physics = physics
        self.calibration = calibration or SensorCalibration()

        # Internal state
        self._current_value = 0.0
        self._raw_value = 0.0
        self._filtered_value = 0.0
        self._filter_state = 0.0

        # History for dynamics
        self._history = deque(maxlen=100)
        self._time_since_start = 0.0

        # Degradation state
        self.degradation_mode = SensorDegradationMode.NONE
        self.degradation_factor = 0.0
        self._accumulated_drift = 0.0

        # Pink noise state (1/f noise)
        self._pink_noise_state = np.zeros(5)

        # Fouling accumulation
        self._fouling_level = 0.0

        # Health metrics
        self.is_healthy = True
        self.health_score = 1.0
        self.fault_code = 0

    def reset(self):
        """Reset sensor to initial state."""
        self._current_value = 0.0
        self._raw_value = 0.0
        self._filtered_value = 0.0
        self._filter_state = 0.0
        self._history.clear()
        self._time_since_start = 0.0
        self._accumulated_drift = 0.0
        self._pink_noise_state = np.zeros(5)
        self._fouling_level = 0.0
        self.is_healthy = True
        self.health_score = 1.0
        self.fault_code = 0

    def _generate_pink_noise(self) -> float:
        """Generate 1/f pink noise using Voss-McCartney algorithm."""
        # Update each octave with probability 1/2^n
        for i in range(5):
            if np.random.random() < 1.0 / (2 ** i):
                self._pink_noise_state[i] = np.random.normal(0, 1)
        return np.sum(self._pink_noise_state) * self.physics.pink_noise_amplitude / 5.0

    def _apply_environmental_effects(self, true_value: float,
                                      environment: Dict[str, float]) -> float:
        """Apply environmental interference to measurement."""
        value = true_value

        # Temperature effect
        temp = environment.get('temperature', 25.0)
        temp_error = self.physics.temp_coefficient * (temp - 25.0) / 100.0
        value *= (1.0 + temp_error)

        # Pressure effect (for level sensors)
        if self.physics.sensor_type in [SensorType.PRESSURE_LEVEL, SensorType.ULTRASONIC_LEVEL]:
            pressure = environment.get('pressure', 101.325)  # kPa
            pressure_error = self.physics.pressure_sensitivity * (pressure - 101.325) / 100.0
            value *= (1.0 + pressure_error)

        # Humidity effect
        humidity = environment.get('humidity', 50.0)
        humidity_error = self.physics.humidity_sensitivity * (humidity - 50.0) / 100.0
        value *= (1.0 + humidity_error)

        # Wind interference for ultrasonic sensors
        if self.physics.sensor_type == SensorType.ULTRASONIC_LEVEL:
            wind_speed = environment.get('wind_speed', 0.0)
            if wind_speed > 5.0:
                wind_noise = np.random.normal(0, 0.01 * wind_speed)
                value += wind_noise

        # Solar interference for IR sensors
        if self.physics.sensor_type == SensorType.INFRARED_TEMP:
            solar_rad = environment.get('solar_radiation', 0.0)
            solar_error = solar_rad * 0.05  # Up to 5°C error at max solar
            value += solar_error

        return value

    def _apply_calibration(self, value: float) -> float:
        """Apply calibration corrections and drift."""
        # Apply linearity correction (polynomial)
        coeffs = self.calibration.linearity_coefficients
        corrected = sum(c * (value ** i) for i, c in enumerate(coeffs))

        # Apply zero offset
        corrected += self.calibration.zero_offset

        # Apply span factor
        corrected *= self.calibration.span_factor

        # Apply accumulated drift
        corrected += self._accumulated_drift

        return corrected

    def _apply_degradation(self, value: float, dt: float) -> float:
        """Apply sensor degradation effects."""
        if self.degradation_mode == SensorDegradationMode.NONE:
            return value

        elif self.degradation_mode == SensorDegradationMode.LINEAR_DRIFT:
            self._accumulated_drift += self.degradation_factor * dt
            return value + self._accumulated_drift

        elif self.degradation_mode == SensorDegradationMode.EXPONENTIAL_DECAY:
            decay = np.exp(-self.degradation_factor * self._time_since_start)
            return value * decay

        elif self.degradation_mode == SensorDegradationMode.FOULING:
            # Fouling reduces sensitivity over time
            self._fouling_level += self.degradation_factor * dt * 0.001
            self._fouling_level = min(self._fouling_level, 0.5)  # Max 50% reduction
            return value * (1.0 - self._fouling_level)

        elif self.degradation_mode == SensorDegradationMode.INTERMITTENT:
            if np.random.random() < self.degradation_factor:
                return np.nan
            return value

        return value

    def _apply_dynamic_response(self, value: float, dt: float) -> float:
        """Apply first-order dynamic response."""
        tau = self.physics.response_time_ms / 1000.0
        if tau > 0:
            alpha = dt / (tau + dt)
            self._filter_state = alpha * value + (1 - alpha) * self._filter_state
            return self._filter_state
        return value

    def measure(self, true_value: float, dt: float = 0.1,
                environment: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Take a measurement with full physics modeling.

        Args:
            true_value: Actual physical value
            dt: Time step in seconds
            environment: Environmental conditions dict

        Returns:
            Tuple of (measured_value, metadata_dict)
        """
        self._time_since_start += dt
        environment = environment or {}

        # Store raw value
        self._raw_value = true_value

        # Apply environmental effects
        value = self._apply_environmental_effects(true_value, environment)

        # Apply calibration
        value = self._apply_calibration(value)

        # Apply degradation
        value = self._apply_degradation(value, dt)

        # Check for NaN from intermittent failure
        if np.isnan(value):
            self._history.append((self._time_since_start, np.nan))
            return np.nan, {'valid': False, 'fault_code': 1}

        # Add white noise
        white_noise = np.random.normal(0, self.physics.white_noise_std)
        value += white_noise

        # Add pink noise
        pink_noise = self._generate_pink_noise()
        value += pink_noise

        # Apply dynamic response (low-pass filtering)
        value = self._apply_dynamic_response(value, dt)

        # Apply resolution quantization
        if self.physics.resolution > 0:
            value = round(value / self.physics.resolution) * self.physics.resolution

        # Apply range limits
        value = np.clip(value, self.physics.range_min, self.physics.range_max)

        self._current_value = value
        self._filtered_value = value
        self._history.append((self._time_since_start, value))

        # Update health score
        error = abs(value - true_value)
        expected_error = self.physics.accuracy_percent * abs(true_value) / 100.0
        if expected_error > 0:
            self.health_score = max(0, 1.0 - error / (3 * expected_error))

        metadata = {
            'valid': True,
            'raw': self._raw_value,
            'filtered': self._filtered_value,
            'health_score': self.health_score,
            'degradation': self.degradation_mode.value,
            'time': self._time_since_start
        }

        return value, metadata

    def get_statistics(self) -> Dict[str, float]:
        """Get sensor statistics from history."""
        if len(self._history) < 2:
            return {}

        values = [v for _, v in self._history if not np.isnan(v)]
        if not values:
            return {}

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'samples': len(values),
            'dropout_rate': 1.0 - len(values) / len(self._history)
        }


class SensorNetwork:
    """
    Network of sensors with cross-correlation modeling.

    Simulates realistic sensor networks including:
    - Sensor redundancy and voting
    - Cross-sensor correlation
    - Network delays and synchronization
    - Sensor fusion algorithms
    """

    def __init__(self, name: str):
        self.name = name
        self.sensors: Dict[str, HighFidelitySensor] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self._measurement_buffer: Dict[str, deque] = {}

        # Network timing
        self.network_delay_ms = 10.0
        self.sync_jitter_ms = 2.0

        # Fusion weights
        self.fusion_weights: Dict[str, float] = {}

    def add_sensor(self, sensor: HighFidelitySensor, weight: float = 1.0):
        """Add a sensor to the network."""
        self.sensors[sensor.name] = sensor
        self._measurement_buffer[sensor.name] = deque(maxlen=50)
        self.fusion_weights[sensor.name] = weight

    def remove_sensor(self, name: str):
        """Remove a sensor from the network."""
        if name in self.sensors:
            del self.sensors[name]
            del self._measurement_buffer[name]
            del self.fusion_weights[name]

    def set_correlation_matrix(self, matrix: np.ndarray):
        """Set the cross-correlation matrix for sensors."""
        n = len(self.sensors)
        if matrix.shape != (n, n):
            raise ValueError(f"Correlation matrix must be {n}x{n}")
        self.correlation_matrix = matrix

    def measure_all(self, true_values: Dict[str, float], dt: float = 0.1,
                    environment: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Take measurements from all sensors in the network.

        Returns:
            Dict with individual measurements and fused values
        """
        measurements = {}
        metadata = {}

        # Individual measurements
        for name, sensor in self.sensors.items():
            if name in true_values:
                value, meta = sensor.measure(true_values[name], dt, environment)
                measurements[name] = value
                metadata[name] = meta
                self._measurement_buffer[name].append(value)

        # Apply correlation if matrix is set
        if self.correlation_matrix is not None:
            self._apply_correlation(measurements)

        # Calculate fused values using weighted average
        fused = self._fuse_measurements(measurements)

        return {
            'individual': measurements,
            'metadata': metadata,
            'fused': fused,
            'timestamp': time.time(),
            'network_health': self._calculate_network_health()
        }

    def _apply_correlation(self, measurements: Dict[str, float]):
        """Apply cross-correlation between sensors."""
        names = list(measurements.keys())
        n = len(names)

        if n < 2 or self.correlation_matrix is None:
            return

        # Generate correlated noise
        uncorrelated_noise = np.random.normal(0, 0.01, n)

        try:
            L = np.linalg.cholesky(self.correlation_matrix[:n, :n])
            correlated_noise = L @ uncorrelated_noise

            for i, name in enumerate(names):
                if not np.isnan(measurements[name]):
                    measurements[name] += correlated_noise[i]
        except np.linalg.LinAlgError:
            pass  # Skip if matrix is not positive definite

    def _fuse_measurements(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Fuse measurements using weighted average."""
        fused = {}

        # Group sensors by type (assuming naming convention)
        groups = {}
        for name, value in measurements.items():
            base_name = name.rsplit('_', 1)[0]
            if base_name not in groups:
                groups[base_name] = []
            if not np.isnan(value):
                weight = self.fusion_weights.get(name, 1.0)
                groups[base_name].append((value, weight))

        for base_name, values_weights in groups.items():
            if values_weights:
                total_weight = sum(w for _, w in values_weights)
                if total_weight > 0:
                    weighted_sum = sum(v * w for v, w in values_weights)
                    fused[base_name] = weighted_sum / total_weight

        return fused

    def _calculate_network_health(self) -> float:
        """Calculate overall network health."""
        if not self.sensors:
            return 0.0

        health_scores = [s.health_score for s in self.sensors.values()]
        return np.mean(health_scores)

    def get_sensor_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sensors in the network."""
        status = {}
        for name, sensor in self.sensors.items():
            status[name] = {
                'type': sensor.physics.sensor_type.value,
                'health_score': sensor.health_score,
                'is_healthy': sensor.is_healthy,
                'degradation': sensor.degradation_mode.value,
                'statistics': sensor.get_statistics()
            }
        return status


class SensorSimulationEngine:
    """
    Complete sensor simulation engine for the aqueduct system.

    Provides:
    - Full sensor network simulation
    - Environmental modeling
    - Fault injection capabilities
    - Real-time monitoring
    """

    def __init__(self):
        self.networks: Dict[str, SensorNetwork] = {}
        self.environment: Dict[str, float] = {
            'temperature': 25.0,
            'pressure': 101.325,
            'humidity': 50.0,
            'wind_speed': 0.0,
            'solar_radiation': 0.0
        }

        # Initialize default sensor networks
        self._initialize_default_networks()

        # Simulation state
        self.simulation_time = 0.0
        self.measurement_count = 0

    def _initialize_default_networks(self):
        """Initialize default sensor networks for the aqueduct."""

        # Water level sensors network (redundant)
        level_network = SensorNetwork("water_level")
        for i in range(3):
            physics = SensorPhysicsModel(
                sensor_type=SensorType.ULTRASONIC_LEVEL if i < 2 else SensorType.RADAR_LEVEL,
                range_min=0.0,
                range_max=10.0,
                resolution=0.001,
                accuracy_percent=0.25,
                response_time_ms=200.0,
                white_noise_std=0.02,
                temp_coefficient=0.02
            )
            sensor = HighFidelitySensor(f"level_{i+1}", physics)
            level_network.add_sensor(sensor, weight=1.0 if i < 2 else 1.2)
        self.networks["water_level"] = level_network

        # Velocity sensors network
        velocity_network = SensorNetwork("velocity")
        for i in range(2):
            physics = SensorPhysicsModel(
                sensor_type=SensorType.DOPPLER_VELOCITY,
                range_min=0.0,
                range_max=15.0,
                resolution=0.01,
                accuracy_percent=1.0,
                response_time_ms=500.0,
                white_noise_std=0.05
            )
            sensor = HighFidelitySensor(f"velocity_{i+1}", physics)
            velocity_network.add_sensor(sensor)
        self.networks["velocity"] = velocity_network

        # Temperature sensors network
        temp_network = SensorNetwork("temperature")

        # Sun-side temperature sensors
        for i in range(2):
            physics = SensorPhysicsModel(
                sensor_type=SensorType.RTD_TEMPERATURE,
                range_min=-50.0,
                range_max=80.0,
                resolution=0.01,
                accuracy_percent=0.1,
                response_time_ms=3000.0,
                white_noise_std=0.1
            )
            sensor = HighFidelitySensor(f"temp_sun_{i+1}", physics)
            temp_network.add_sensor(sensor)

        # Shade-side temperature sensors
        for i in range(2):
            physics = SensorPhysicsModel(
                sensor_type=SensorType.RTD_TEMPERATURE,
                range_min=-50.0,
                range_max=80.0,
                resolution=0.01,
                accuracy_percent=0.1,
                response_time_ms=3000.0,
                white_noise_std=0.1
            )
            sensor = HighFidelitySensor(f"temp_shade_{i+1}", physics)
            temp_network.add_sensor(sensor)

        self.networks["temperature"] = temp_network

        # Structural sensors network
        structural_network = SensorNetwork("structural")

        # Joint gap sensor
        physics = SensorPhysicsModel(
            sensor_type=SensorType.LVDT_DISPLACEMENT,
            range_min=0.0,
            range_max=50.0,
            resolution=0.01,
            accuracy_percent=0.1,
            response_time_ms=50.0,
            white_noise_std=0.05,
            temp_coefficient=0.05
        )
        sensor = HighFidelitySensor("joint_gap", physics)
        structural_network.add_sensor(sensor)

        # Vibration sensor
        physics = SensorPhysicsModel(
            sensor_type=SensorType.ACCELEROMETER,
            range_min=0.0,
            range_max=200.0,
            resolution=0.1,
            accuracy_percent=2.0,
            response_time_ms=10.0,
            bandwidth_hz=100.0,
            white_noise_std=0.5
        )
        sensor = HighFidelitySensor("vibration", physics)
        structural_network.add_sensor(sensor)

        # Bearing stress sensor
        physics = SensorPhysicsModel(
            sensor_type=SensorType.STRAIN_GAUGE,
            range_min=0.0,
            range_max=100.0,
            resolution=0.01,
            accuracy_percent=0.5,
            response_time_ms=100.0,
            white_noise_std=0.2,
            temp_coefficient=0.03
        )
        sensor = HighFidelitySensor("bearing_stress", physics)
        structural_network.add_sensor(sensor)

        self.networks["structural"] = structural_network

        # Flow sensors network
        flow_network = SensorNetwork("flow")

        # Inlet flow
        physics = SensorPhysicsModel(
            sensor_type=SensorType.ELECTROMAGNETIC_FLOW,
            range_min=0.0,
            range_max=500.0,
            resolution=0.1,
            accuracy_percent=0.5,
            response_time_ms=200.0,
            white_noise_std=1.0
        )
        sensor = HighFidelitySensor("flow_in", physics)
        flow_network.add_sensor(sensor)

        # Outlet flow
        sensor = HighFidelitySensor("flow_out", physics)
        flow_network.add_sensor(sensor)

        self.networks["flow"] = flow_network

    def update_environment(self, **kwargs):
        """Update environmental conditions."""
        self.environment.update(kwargs)

    def measure(self, true_state: Dict[str, Any], dt: float = 0.1) -> Dict[str, Any]:
        """
        Take measurements from all sensor networks.

        Args:
            true_state: Dictionary of true physical state values
            dt: Time step

        Returns:
            Comprehensive measurement dictionary
        """
        self.simulation_time += dt
        self.measurement_count += 1

        results = {
            'timestamp': self.simulation_time,
            'networks': {},
            'fused_state': {},
            'health_report': {}
        }

        # Map true state to sensor networks
        network_inputs = {
            'water_level': {'level_1': true_state.get('h', 0),
                           'level_2': true_state.get('h', 0),
                           'level_3': true_state.get('h', 0)},
            'velocity': {'velocity_1': true_state.get('v', 0),
                        'velocity_2': true_state.get('v', 0)},
            'temperature': {
                'temp_sun_1': true_state.get('T_sun', 25),
                'temp_sun_2': true_state.get('T_sun', 25),
                'temp_shade_1': true_state.get('T_shade', 25),
                'temp_shade_2': true_state.get('T_shade', 25)
            },
            'structural': {
                'joint_gap': true_state.get('joint_gap', 20),
                'vibration': true_state.get('vib_amp', 0),
                'bearing_stress': true_state.get('bearing_stress', 30)
            },
            'flow': {
                'flow_in': true_state.get('Q_in', 80),
                'flow_out': true_state.get('Q_out', 80)
            }
        }

        # Measure from each network
        for network_name, network in self.networks.items():
            if network_name in network_inputs:
                measurement = network.measure_all(
                    network_inputs[network_name],
                    dt,
                    self.environment
                )
                results['networks'][network_name] = measurement
                results['fused_state'].update(measurement.get('fused', {}))
                results['health_report'][network_name] = measurement.get('network_health', 1.0)

        # Build final measured state
        results['measured_state'] = self._build_measured_state(results)

        return results

    def _build_measured_state(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build a measured state dictionary from network measurements."""
        fused = results.get('fused_state', {})

        measured = {
            'h': fused.get('level', 0),
            'v': fused.get('velocity', 0),
            'T_sun': fused.get('temp_sun', 25),
            'T_shade': fused.get('temp_shade', 25),
            'joint_gap': fused.get('joint_gap', 20),
            'vib_amp': fused.get('vibration', 0),
            'bearing_stress': fused.get('bearing_stress', 30),
            'Q_in': fused.get('flow_in', 80),
            'Q_out': fused.get('flow_out', 80),
            'time': self.simulation_time
        }

        # Calculate Froude number
        if measured['h'] > 0.1:
            measured['fr'] = measured['v'] / np.sqrt(9.81 * measured['h'])
        else:
            measured['fr'] = 0.0

        return measured

    def inject_fault(self, network_name: str, sensor_name: str,
                     degradation_mode: SensorDegradationMode, factor: float = 0.1):
        """Inject a fault into a specific sensor."""
        if network_name in self.networks:
            network = self.networks[network_name]
            if sensor_name in network.sensors:
                sensor = network.sensors[sensor_name]
                sensor.degradation_mode = degradation_mode
                sensor.degradation_factor = factor
                sensor.is_healthy = False

    def clear_faults(self):
        """Clear all injected faults."""
        for network in self.networks.values():
            for sensor in network.sensors.values():
                sensor.degradation_mode = SensorDegradationMode.NONE
                sensor.degradation_factor = 0.0
                sensor.is_healthy = True
                sensor.health_score = 1.0

    def get_full_status(self) -> Dict[str, Any]:
        """Get complete status of the sensor simulation engine."""
        return {
            'simulation_time': self.simulation_time,
            'measurement_count': self.measurement_count,
            'environment': self.environment.copy(),
            'networks': {
                name: network.get_sensor_status()
                for name, network in self.networks.items()
            },
            'overall_health': np.mean([
                network._calculate_network_health()
                for network in self.networks.values()
            ])
        }

    def reset(self):
        """Reset the entire sensor simulation engine."""
        self.simulation_time = 0.0
        self.measurement_count = 0
        for network in self.networks.values():
            for sensor in network.sensors.values():
                sensor.reset()
