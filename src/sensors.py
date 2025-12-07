"""
Sensor Models for TAOS High-Fidelity Simulation

This module implements realistic sensor models with:
- Measurement noise (Gaussian, bias, drift)
- Time delays and sampling rates
- Sensor faults (stuck, drift, noise increase)
- Redundancy and voting logic
- Kalman filtering for state estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class SensorFaultType(Enum):
    """Types of sensor faults."""
    NONE = "none"
    STUCK = "stuck"              # Sensor output frozen
    DRIFT = "drift"              # Gradual drift from true value
    NOISE_INCREASE = "noise_up"  # Increased noise level
    BIAS = "bias"                # Constant offset
    INTERMITTENT = "intermittent"  # Random dropouts
    COMPLETE_FAILURE = "failed"  # No output


@dataclass
class SensorConfig:
    """Configuration for a sensor."""
    name: str
    noise_std: float = 0.0       # Gaussian noise standard deviation
    bias: float = 0.0            # Constant bias
    drift_rate: float = 0.0      # Drift per second
    delay_steps: int = 0         # Measurement delay in steps
    sample_rate: float = 10.0    # Hz
    range_min: float = -np.inf
    range_max: float = np.inf
    fault_type: SensorFaultType = SensorFaultType.NONE
    fault_magnitude: float = 0.0


class Sensor:
    """Individual sensor model with noise, delay, and faults."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self.delay_buffer: deque = deque(maxlen=max(1, config.delay_steps + 1))
        self.drift_accumulated = 0.0
        self.stuck_value = None
        self.last_valid_value = 0.0
        self.sample_counter = 0
        self.is_healthy = True

    def reset(self):
        """Reset sensor state."""
        self.delay_buffer.clear()
        self.drift_accumulated = 0.0
        self.stuck_value = None
        self.last_valid_value = 0.0
        self.sample_counter = 0
        self.is_healthy = True

    def inject_fault(self, fault_type: SensorFaultType, magnitude: float = 0.0):
        """Inject a fault into the sensor."""
        self.config.fault_type = fault_type
        self.config.fault_magnitude = magnitude
        if fault_type == SensorFaultType.STUCK:
            self.stuck_value = self.last_valid_value
        self.is_healthy = fault_type == SensorFaultType.NONE

    def clear_fault(self):
        """Clear any active fault."""
        self.config.fault_type = SensorFaultType.NONE
        self.config.fault_magnitude = 0.0
        self.stuck_value = None
        self.is_healthy = True

    def measure(self, true_value: float, dt: float = 0.1) -> Tuple[float, bool]:
        """
        Take a measurement with noise, delay, and fault effects.

        Args:
            true_value: The actual physical value
            dt: Time step in seconds

        Returns:
            Tuple of (measured_value, is_valid)
        """
        # Handle complete failure
        if self.config.fault_type == SensorFaultType.COMPLETE_FAILURE:
            return np.nan, False

        # Handle intermittent faults
        if self.config.fault_type == SensorFaultType.INTERMITTENT:
            if np.random.random() < self.config.fault_magnitude:
                return np.nan, False

        # Apply measurement noise
        noise = np.random.normal(0, self.config.noise_std)

        # Apply bias
        bias = self.config.bias

        # Apply drift
        self.drift_accumulated += self.config.drift_rate * dt
        if self.config.fault_type == SensorFaultType.DRIFT:
            self.drift_accumulated += self.config.fault_magnitude * dt

        # Apply noise increase fault
        if self.config.fault_type == SensorFaultType.NOISE_INCREASE:
            noise *= (1.0 + self.config.fault_magnitude)

        # Apply bias fault
        if self.config.fault_type == SensorFaultType.BIAS:
            bias += self.config.fault_magnitude

        # Calculate raw measurement
        raw_measurement = true_value + noise + bias + self.drift_accumulated

        # Handle stuck fault
        if self.config.fault_type == SensorFaultType.STUCK:
            if self.stuck_value is None:
                self.stuck_value = raw_measurement
            raw_measurement = self.stuck_value

        # Apply range limits
        raw_measurement = np.clip(raw_measurement, self.config.range_min, self.config.range_max)

        # Apply delay
        self.delay_buffer.append(raw_measurement)
        if len(self.delay_buffer) > self.config.delay_steps:
            delayed_measurement = self.delay_buffer[0]
        else:
            delayed_measurement = raw_measurement

        self.last_valid_value = delayed_measurement
        return delayed_measurement, True


class SensorArray:
    """Array of redundant sensors with voting logic."""

    def __init__(self, name: str, configs: List[SensorConfig]):
        self.name = name
        self.sensors = [Sensor(config) for config in configs]
        self.num_sensors = len(self.sensors)

    def reset(self):
        """Reset all sensors."""
        for sensor in self.sensors:
            sensor.reset()

    def measure(self, true_value: float, dt: float = 0.1) -> Tuple[float, float, int]:
        """
        Take measurements from all sensors and apply voting.

        Returns:
            Tuple of (voted_value, confidence, num_valid)
        """
        measurements = []
        valid_count = 0

        for sensor in self.sensors:
            value, is_valid = sensor.measure(true_value, dt)
            if is_valid and not np.isnan(value):
                measurements.append(value)
                valid_count += 1

        if valid_count == 0:
            return np.nan, 0.0, 0

        # Median voting for robustness
        voted_value = np.median(measurements)

        # Confidence based on agreement
        if valid_count >= 2:
            spread = np.std(measurements)
            confidence = max(0.0, 1.0 - spread / (abs(voted_value) + 1e-6))
        else:
            confidence = 0.5

        return voted_value, confidence, valid_count


class KalmanFilter:
    """Simple Kalman filter for state estimation."""

    def __init__(self, state_dim: int, measurement_dim: int):
        self.n = state_dim
        self.m = measurement_dim

        # State estimate and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1.0

        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(measurement_dim) * 0.1

        # State transition and measurement matrices
        self.F = np.eye(state_dim)
        self.H = np.eye(measurement_dim, state_dim)

    def predict(self, dt: float = 0.1):
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q * dt

    def update(self, z: np.ndarray):
        """Update step with measurement."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x.copy()


class SensorSuite:
    """
    Complete sensor suite for the aqueduct system.
    Includes all sensors needed for autonomous operation.
    """

    def __init__(self, use_redundancy: bool = True, use_kalman: bool = True):
        self.use_redundancy = use_redundancy
        self.use_kalman = use_kalman

        # Define sensor configurations
        if use_redundancy:
            # Triple redundant sensors for critical measurements
            self.water_level_sensors = SensorArray("water_level", [
                SensorConfig("h_1", noise_std=0.02, delay_steps=1, range_min=0, range_max=10),
                SensorConfig("h_2", noise_std=0.02, delay_steps=1, range_min=0, range_max=10),
                SensorConfig("h_3", noise_std=0.03, delay_steps=2, range_min=0, range_max=10),
            ])
            self.velocity_sensors = SensorArray("velocity", [
                SensorConfig("v_1", noise_std=0.05, delay_steps=1, range_min=0, range_max=15),
                SensorConfig("v_2", noise_std=0.05, delay_steps=1, range_min=0, range_max=15),
            ])
            self.temp_sun_sensors = SensorArray("temp_sun", [
                SensorConfig("Ts_1", noise_std=0.5, delay_steps=2, range_min=-50, range_max=80),
                SensorConfig("Ts_2", noise_std=0.5, delay_steps=2, range_min=-50, range_max=80),
            ])
            self.temp_shade_sensors = SensorArray("temp_shade", [
                SensorConfig("Tsh_1", noise_std=0.5, delay_steps=2, range_min=-50, range_max=80),
                SensorConfig("Tsh_2", noise_std=0.5, delay_steps=2, range_min=-50, range_max=80),
            ])
        else:
            # Single sensors (for simplified simulation)
            self.water_level_sensors = SensorArray("water_level", [
                SensorConfig("h", noise_std=0.02, delay_steps=1),
            ])
            self.velocity_sensors = SensorArray("velocity", [
                SensorConfig("v", noise_std=0.05, delay_steps=1),
            ])
            self.temp_sun_sensors = SensorArray("temp_sun", [
                SensorConfig("Ts", noise_std=0.5, delay_steps=2),
            ])
            self.temp_shade_sensors = SensorArray("temp_shade", [
                SensorConfig("Tsh", noise_std=0.5, delay_steps=2),
            ])

        # Non-redundant sensors (typically less critical or unique)
        self.joint_gap_sensor = Sensor(SensorConfig(
            "joint_gap", noise_std=0.1, delay_steps=1, range_min=0, range_max=50
        ))
        self.vibration_sensor = Sensor(SensorConfig(
            "vibration", noise_std=1.0, delay_steps=0, range_min=0, range_max=200
        ))
        self.bearing_stress_sensor = Sensor(SensorConfig(
            "bearing_stress", noise_std=0.5, delay_steps=1, range_min=0, range_max=100
        ))
        self.flow_in_sensor = Sensor(SensorConfig(
            "Q_in", noise_std=1.0, delay_steps=1, range_min=0, range_max=250
        ))
        self.flow_out_sensor = Sensor(SensorConfig(
            "Q_out", noise_std=1.0, delay_steps=1, range_min=0, range_max=250
        ))

        # Bearing lock is a discrete sensor (limit switch)
        self.bearing_lock_sensor = Sensor(SensorConfig(
            "bearing_locked", noise_std=0, delay_steps=0
        ))

        # Kalman filter for state estimation (6 states: h, v, Ts, Tsh, gap, vib)
        if use_kalman:
            self.kalman = KalmanFilter(state_dim=6, measurement_dim=6)
            # Set up process model (simple random walk)
            self.kalman.Q = np.diag([0.001, 0.01, 0.1, 0.1, 0.01, 1.0])
            self.kalman.R = np.diag([0.02, 0.05, 0.5, 0.5, 0.1, 1.0])
        else:
            self.kalman = None

        # Diagnostic counters
        self.measurement_count = 0
        self.fault_count = 0

    def reset(self):
        """Reset all sensors."""
        self.water_level_sensors.reset()
        self.velocity_sensors.reset()
        self.temp_sun_sensors.reset()
        self.temp_shade_sensors.reset()
        self.joint_gap_sensor.reset()
        self.vibration_sensor.reset()
        self.bearing_stress_sensor.reset()
        self.flow_in_sensor.reset()
        self.flow_out_sensor.reset()
        self.bearing_lock_sensor.reset()

        if self.kalman:
            self.kalman.x = np.zeros(6)
            self.kalman.P = np.eye(6)

        self.measurement_count = 0
        self.fault_count = 0

    def measure(self, true_state: Dict[str, Any], dt: float = 0.1) -> Dict[str, Any]:
        """
        Take measurements of all state variables.

        Args:
            true_state: Dictionary of true physical state
            dt: Time step

        Returns:
            Dictionary of measured values with confidence
        """
        self.measurement_count += 1
        measurements = {}

        # Water level (redundant)
        h_val, h_conf, h_valid = self.water_level_sensors.measure(true_state['h'], dt)
        measurements['h'] = h_val
        measurements['h_confidence'] = h_conf
        measurements['h_valid_sensors'] = h_valid

        # Velocity (redundant)
        v_val, v_conf, v_valid = self.velocity_sensors.measure(true_state['v'], dt)
        measurements['v'] = v_val
        measurements['v_confidence'] = v_conf

        # Temperatures (redundant)
        Ts_val, Ts_conf, _ = self.temp_sun_sensors.measure(true_state['T_sun'], dt)
        Tsh_val, Tsh_conf, _ = self.temp_shade_sensors.measure(true_state['T_shade'], dt)
        measurements['T_sun'] = Ts_val
        measurements['T_shade'] = Tsh_val
        measurements['T_confidence'] = min(Ts_conf, Tsh_conf)

        # Non-redundant sensors
        measurements['joint_gap'], _ = self.joint_gap_sensor.measure(true_state['joint_gap'], dt)
        measurements['vib_amp'], _ = self.vibration_sensor.measure(true_state['vib_amp'], dt)
        measurements['bearing_stress'], _ = self.bearing_stress_sensor.measure(
            true_state['bearing_stress'], dt
        )
        measurements['Q_in'], _ = self.flow_in_sensor.measure(true_state['Q_in'], dt)
        measurements['Q_out'], _ = self.flow_out_sensor.measure(true_state['Q_out'], dt)

        # Bearing lock (discrete)
        measurements['bearing_locked'] = true_state['bearing_locked']

        # Calculate Froude number from measurements
        if measurements['h'] > 0.1 and not np.isnan(measurements['v']):
            measurements['fr'] = measurements['v'] / np.sqrt(9.81 * measurements['h'])
        else:
            measurements['fr'] = 0.0

        # Apply Kalman filter if enabled
        if self.kalman and not any(np.isnan([h_val, v_val, Ts_val, Tsh_val])):
            z = np.array([
                h_val, v_val, Ts_val, Tsh_val,
                measurements['joint_gap'], measurements['vib_amp']
            ])

            self.kalman.predict(dt)
            self.kalman.update(z)

            estimated = self.kalman.get_state()
            measurements['h_estimated'] = estimated[0]
            measurements['v_estimated'] = estimated[1]
            measurements['T_sun_estimated'] = estimated[2]
            measurements['T_shade_estimated'] = estimated[3]
            measurements['joint_gap_estimated'] = estimated[4]
            measurements['vib_amp_estimated'] = estimated[5]

            # Use estimated values for critical decisions
            measurements['fr_estimated'] = estimated[1] / np.sqrt(9.81 * max(0.1, estimated[0]))

        # Copy through non-measured state
        measurements['T_ambient'] = true_state.get('T_ambient', 25.0)
        measurements['solar_rad'] = true_state.get('solar_rad', 0.0)
        measurements['wind_speed'] = true_state.get('wind_speed', 0.0)
        measurements['time'] = true_state.get('time', 0.0)

        return measurements

    def inject_sensor_fault(self, sensor_name: str, fault_type: SensorFaultType,
                           magnitude: float = 0.0, sensor_index: int = 0):
        """Inject a fault into a specific sensor."""
        sensor_map = {
            'h': (self.water_level_sensors.sensors, sensor_index),
            'v': (self.velocity_sensors.sensors, sensor_index),
            'T_sun': (self.temp_sun_sensors.sensors, sensor_index),
            'T_shade': (self.temp_shade_sensors.sensors, sensor_index),
            'joint_gap': ([self.joint_gap_sensor], 0),
            'vib_amp': ([self.vibration_sensor], 0),
            'bearing_stress': ([self.bearing_stress_sensor], 0),
        }

        if sensor_name in sensor_map:
            sensors, idx = sensor_map[sensor_name]
            if idx < len(sensors):
                sensors[idx].inject_fault(fault_type, magnitude)
                self.fault_count += 1

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all sensors."""
        status = {
            'total_measurements': self.measurement_count,
            'fault_count': self.fault_count,
            'sensors': {}
        }

        # Check each sensor array
        for name, arr in [
            ('water_level', self.water_level_sensors),
            ('velocity', self.velocity_sensors),
            ('temp_sun', self.temp_sun_sensors),
            ('temp_shade', self.temp_shade_sensors),
        ]:
            healthy = sum(1 for s in arr.sensors if s.is_healthy)
            status['sensors'][name] = {
                'total': arr.num_sensors,
                'healthy': healthy,
                'degraded': arr.num_sensors - healthy
            }

        return status
