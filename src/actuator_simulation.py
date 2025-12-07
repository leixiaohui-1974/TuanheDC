"""
High-Fidelity Actuator Simulation Module for TAOS V3.10

This module provides comprehensive actuator simulation with:
- Physics-based actuator models (hydraulic, electric, pneumatic)
- Multi-order dynamics modeling
- Wear and aging effects
- Energy consumption modeling
- Failure mode simulation
- Position feedback and control loops

Author: TAOS Development Team
Version: 3.10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class ActuatorType(Enum):
    """Physical actuator types."""
    HYDRAULIC_CYLINDER = "hydraulic_cylinder"
    HYDRAULIC_MOTOR = "hydraulic_motor"
    ELECTRIC_SERVO = "electric_servo"
    ELECTRIC_STEPPER = "electric_stepper"
    PNEUMATIC_CYLINDER = "pneumatic_cylinder"
    GEAR_MOTOR = "gear_motor"
    LINEAR_ACTUATOR = "linear_actuator"


class ActuatorFailureMode(Enum):
    """Actuator failure modes."""
    NONE = "none"
    STUCK = "stuck"                          # Actuator frozen
    SLOW_RESPONSE = "slow_response"          # Degraded response
    PARTIAL_STROKE = "partial_stroke"        # Limited range
    OSCILLATION = "oscillation"              # Hunting/oscillation
    RUNAWAY = "runaway"                      # Uncontrolled movement
    LEAK = "leak"                            # Hydraulic/pneumatic leak
    MOTOR_BURNOUT = "motor_burnout"          # Motor failure
    GEAR_FAILURE = "gear_failure"            # Mechanical failure
    SENSOR_FEEDBACK_LOSS = "feedback_loss"   # Position sensor failure


class WearModel(Enum):
    """Actuator wear models."""
    NONE = "none"
    LINEAR = "linear"                        # Linear degradation
    BATHTUB = "bathtub"                      # Bathtub curve
    FATIGUE = "fatigue"                      # Fatigue-based
    USAGE_BASED = "usage_based"              # Based on cycle count


@dataclass
class ActuatorDynamicsModel:
    """Complete actuator dynamics model."""
    actuator_type: ActuatorType

    # Physical limits
    position_min: float = 0.0
    position_max: float = 100.0
    velocity_max: float = 10.0               # Units/second
    acceleration_max: float = 50.0           # Units/second²
    force_max: float = 10000.0               # N or Nm

    # Dynamic parameters (2nd order system)
    natural_frequency: float = 10.0          # Hz
    damping_ratio: float = 0.7               # Dimensionless
    time_constant: float = 0.5               # Seconds (for 1st order approximation)

    # Friction model
    static_friction: float = 0.0             # Units (force to overcome)
    coulomb_friction: float = 0.0            # Units (constant kinetic friction)
    viscous_friction: float = 0.0            # Units per velocity

    # Backlash and dead zone
    backlash: float = 0.0                    # Units
    dead_zone: float = 0.0                   # Units

    # Power and efficiency
    rated_power: float = 5000.0              # W
    efficiency: float = 0.85                 # 0-1
    standby_power: float = 50.0              # W

    # Thermal characteristics
    thermal_mass: float = 1000.0             # J/°C
    thermal_resistance: float = 0.1          # °C/W
    max_temperature: float = 80.0            # °C

    # Wear parameters
    wear_model: WearModel = WearModel.NONE
    design_life_cycles: int = 1000000
    current_cycles: int = 0


@dataclass
class PositionController:
    """Internal position controller for actuator."""
    kp: float = 10.0                         # Proportional gain
    ki: float = 1.0                          # Integral gain
    kd: float = 0.5                          # Derivative gain
    integral_limit: float = 100.0            # Anti-windup limit
    output_limit: float = 100.0              # Control output limit


class HighFidelityActuator:
    """
    High-fidelity actuator simulation with physics-based modeling.

    Implements:
    - Second-order dynamics
    - Friction models (Stribeck curve)
    - Backlash and hysteresis
    - Thermal modeling
    - Wear and degradation
    """

    def __init__(self, name: str, dynamics: ActuatorDynamicsModel,
                 controller: Optional[PositionController] = None):
        self.name = name
        self.dynamics = dynamics
        self.controller = controller or PositionController()

        # State variables
        self._position = 0.0
        self._velocity = 0.0
        self._acceleration = 0.0
        self._force = 0.0
        self._temperature = 25.0

        # Command and setpoint
        self._command = 0.0
        self._setpoint = 0.0

        # Controller state
        self._integral = 0.0
        self._last_error = 0.0
        self._last_direction = 0

        # History
        self._history = deque(maxlen=1000)
        self._time = 0.0
        self._total_travel = 0.0
        self._cycle_count = 0

        # Wear state
        self._wear_factor = 1.0              # 1.0 = new, 0.0 = worn out
        self._last_position_for_cycle = 0.0

        # Failure state
        self.failure_mode = ActuatorFailureMode.NONE
        self.failure_severity = 0.0
        self._stuck_position = None

        # Energy tracking
        self._power_consumption = 0.0
        self._energy_consumed = 0.0

        # Health
        self.is_healthy = True
        self.health_score = 1.0

    def reset(self):
        """Reset actuator to initial state."""
        self._position = 0.0
        self._velocity = 0.0
        self._acceleration = 0.0
        self._force = 0.0
        self._temperature = 25.0
        self._command = 0.0
        self._setpoint = 0.0
        self._integral = 0.0
        self._last_error = 0.0
        self._last_direction = 0
        self._history.clear()
        self._time = 0.0
        self._wear_factor = 1.0
        self.failure_mode = ActuatorFailureMode.NONE
        self.failure_severity = 0.0
        self._stuck_position = None
        self._power_consumption = 0.0
        self._energy_consumed = 0.0
        self.is_healthy = True
        self.health_score = 1.0

    def command(self, setpoint: float):
        """Set the position setpoint."""
        self._setpoint = np.clip(setpoint,
                                  self.dynamics.position_min,
                                  self.dynamics.position_max)

    def _calculate_friction(self, velocity: float) -> float:
        """Calculate friction force using Stribeck model."""
        if abs(velocity) < 1e-6:
            # Static friction region
            return 0.0  # Will be handled by static check

        # Stribeck curve: F = Fc + (Fs - Fc) * exp(-|v|/vs) + Fv * v
        # Simplified version:
        stribeck_velocity = 0.1
        fs = self.dynamics.static_friction
        fc = self.dynamics.coulomb_friction
        fv = self.dynamics.viscous_friction

        friction = fc + (fs - fc) * np.exp(-abs(velocity) / stribeck_velocity)
        friction += fv * abs(velocity)

        return friction * np.sign(velocity) * self._wear_factor

    def _apply_backlash(self, command: float) -> float:
        """Apply backlash hysteresis."""
        if self.dynamics.backlash <= 0:
            return command

        direction = np.sign(command - self._position)
        if direction != self._last_direction and direction != 0:
            # Direction change - apply backlash
            error = command - self._position
            if abs(error) < self.dynamics.backlash:
                return self._position  # No movement until backlash is taken up
            else:
                command = self._position + (error - direction * self.dynamics.backlash)

        self._last_direction = direction if direction != 0 else self._last_direction
        return command

    def _apply_dead_zone(self, error: float) -> float:
        """Apply dead zone to control error."""
        if abs(error) < self.dynamics.dead_zone:
            return 0.0
        return error - np.sign(error) * self.dynamics.dead_zone

    def _update_wear(self, dt: float, travel: float):
        """Update wear model based on usage."""
        self._total_travel += abs(travel)

        # Count cycles (one cycle = full stroke there and back)
        stroke = self.dynamics.position_max - self.dynamics.position_min
        if stroke > 0:
            if (self._position - self._last_position_for_cycle) * \
               (self._last_position_for_cycle - self._position) < 0:
                # Direction reversal
                self._cycle_count += 0.5
                self._last_position_for_cycle = self._position

        # Update wear factor
        if self.dynamics.wear_model == WearModel.LINEAR:
            wear_rate = 1.0 / self.dynamics.design_life_cycles
            self._wear_factor = max(0.1, 1.0 - self._cycle_count * wear_rate)

        elif self.dynamics.wear_model == WearModel.BATHTUB:
            # Bathtub curve: high early failure, low mid-life, high late-life
            life_ratio = self._cycle_count / self.dynamics.design_life_cycles
            if life_ratio < 0.1:
                # Infant mortality region
                self._wear_factor = 0.9 + 0.1 * (life_ratio / 0.1)
            elif life_ratio < 0.8:
                # Normal life region
                self._wear_factor = 1.0
            else:
                # Wear-out region
                self._wear_factor = max(0.1, 1.0 - 2 * (life_ratio - 0.8))

        elif self.dynamics.wear_model == WearModel.FATIGUE:
            # S-N curve based fatigue
            if self._cycle_count > 0:
                fatigue_factor = (self._cycle_count / self.dynamics.design_life_cycles) ** 0.5
                self._wear_factor = max(0.1, 1.0 - fatigue_factor)

    def _update_thermal(self, dt: float, power: float):
        """Update thermal model."""
        # Heat generation
        heat_in = power * (1 - self.dynamics.efficiency)

        # Heat dissipation
        heat_out = (self._temperature - 25.0) / self.dynamics.thermal_resistance

        # Temperature change
        dT = (heat_in - heat_out) / self.dynamics.thermal_mass * dt
        self._temperature += dT

        # Thermal protection
        if self._temperature > self.dynamics.max_temperature:
            self._apply_thermal_protection()

    def _apply_thermal_protection(self):
        """Apply thermal protection (derate)."""
        overheat = self._temperature - self.dynamics.max_temperature
        derate_factor = max(0.5, 1.0 - overheat * 0.1)
        self.dynamics.velocity_max *= derate_factor

    def _apply_failure_mode(self, dt: float):
        """Apply active failure mode effects."""
        if self.failure_mode == ActuatorFailureMode.NONE:
            return

        elif self.failure_mode == ActuatorFailureMode.STUCK:
            if self._stuck_position is None:
                self._stuck_position = self._position
            self._position = self._stuck_position
            self._velocity = 0.0
            self._acceleration = 0.0

        elif self.failure_mode == ActuatorFailureMode.SLOW_RESPONSE:
            # Reduce velocity limit
            self.dynamics.velocity_max *= (1.0 - self.failure_severity)

        elif self.failure_mode == ActuatorFailureMode.PARTIAL_STROKE:
            # Limit range
            effective_range = (1.0 - self.failure_severity)
            mid = (self.dynamics.position_max + self.dynamics.position_min) / 2
            half_range = (self.dynamics.position_max - self.dynamics.position_min) / 2 * effective_range
            self._position = np.clip(self._position, mid - half_range, mid + half_range)

        elif self.failure_mode == ActuatorFailureMode.OSCILLATION:
            # Add oscillation
            osc_freq = 5.0  # Hz
            osc_amp = self.failure_severity * 2.0
            self._position += osc_amp * np.sin(2 * np.pi * osc_freq * self._time)

        elif self.failure_mode == ActuatorFailureMode.LEAK:
            # Gradual drift (for hydraulic actuators)
            if self.dynamics.actuator_type in [ActuatorType.HYDRAULIC_CYLINDER,
                                                ActuatorType.HYDRAULIC_MOTOR]:
                leak_rate = self.failure_severity * 0.5  # Units/second
                self._position -= leak_rate * dt

        elif self.failure_mode == ActuatorFailureMode.RUNAWAY:
            # Uncontrolled movement
            runaway_velocity = self.failure_severity * self.dynamics.velocity_max
            self._position += runaway_velocity * dt

    def step(self, dt: float, external_load: float = 0.0) -> Dict[str, float]:
        """
        Update actuator state by one time step.

        Args:
            dt: Time step in seconds
            external_load: External load force/torque

        Returns:
            Dict with current state
        """
        self._time += dt

        # Apply failure mode first
        if self.failure_mode != ActuatorFailureMode.NONE:
            self._apply_failure_mode(dt)
            if self.failure_mode == ActuatorFailureMode.STUCK:
                return self.get_state()

        # Position controller (PID)
        target = self._apply_backlash(self._setpoint)
        error = self._apply_dead_zone(target - self._position)

        # PID control
        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                  -self.controller.integral_limit,
                                  self.controller.integral_limit)

        derivative = (error - self._last_error) / dt if dt > 0 else 0.0
        self._last_error = error

        control_output = (self.controller.kp * error +
                         self.controller.ki * self._integral +
                         self.controller.kd * derivative)

        control_output = np.clip(control_output,
                                  -self.controller.output_limit,
                                  self.controller.output_limit)

        # Convert control output to force/torque command
        self._command = control_output
        commanded_force = control_output * self.dynamics.force_max / 100.0

        # Apply wear effect
        commanded_force *= self._wear_factor

        # Calculate friction
        friction_force = self._calculate_friction(self._velocity)

        # Static friction check
        net_force = commanded_force - friction_force - external_load
        if abs(self._velocity) < 1e-6 and abs(net_force) < self.dynamics.static_friction:
            net_force = 0.0
            self._velocity = 0.0

        # Second-order dynamics
        # m * x'' + c * x' + k * (x - x_eq) = F
        # Simplified to 2nd order transfer function form
        omega_n = 2 * np.pi * self.dynamics.natural_frequency
        zeta = self.dynamics.damping_ratio

        # State-space update
        # x'' = omega_n^2 * (u - x) - 2*zeta*omega_n * x'
        target_position = self._position + commanded_force / (omega_n ** 2 + 1)
        self._acceleration = (omega_n ** 2 * (target_position - self._position) -
                             2 * zeta * omega_n * self._velocity)

        # Apply acceleration limit
        self._acceleration = np.clip(self._acceleration,
                                      -self.dynamics.acceleration_max,
                                      self.dynamics.acceleration_max)

        # Update velocity
        old_velocity = self._velocity
        self._velocity += self._acceleration * dt
        self._velocity = np.clip(self._velocity,
                                  -self.dynamics.velocity_max,
                                  self.dynamics.velocity_max)

        # Update position
        old_position = self._position
        self._position += self._velocity * dt
        self._position = np.clip(self._position,
                                  self.dynamics.position_min,
                                  self.dynamics.position_max)

        # Handle position limits (stop at limits)
        if self._position == self.dynamics.position_min or \
           self._position == self.dynamics.position_max:
            self._velocity = 0.0
            self._acceleration = 0.0

        # Calculate power consumption
        mechanical_power = abs(commanded_force * self._velocity)
        electrical_power = mechanical_power / self.dynamics.efficiency + self.dynamics.standby_power
        self._power_consumption = electrical_power
        self._energy_consumed += electrical_power * dt

        # Update wear
        travel = abs(self._position - old_position)
        self._update_wear(dt, travel)

        # Update thermal
        self._update_thermal(dt, electrical_power)

        # Update health score
        self.health_score = self._calculate_health_score()

        # Store history
        self._history.append({
            'time': self._time,
            'position': self._position,
            'velocity': self._velocity,
            'setpoint': self._setpoint,
            'power': self._power_consumption
        })

        return self.get_state()

    def _calculate_health_score(self) -> float:
        """Calculate actuator health score."""
        score = 1.0

        # Wear factor contribution
        score *= self._wear_factor

        # Temperature contribution
        if self._temperature > self.dynamics.max_temperature * 0.8:
            temp_factor = 1.0 - (self._temperature - self.dynamics.max_temperature * 0.8) / \
                         (self.dynamics.max_temperature * 0.2)
            score *= max(0, temp_factor)

        # Failure mode contribution
        if self.failure_mode != ActuatorFailureMode.NONE:
            score *= (1.0 - self.failure_severity)

        return score

    def get_state(self) -> Dict[str, float]:
        """Get current actuator state."""
        return {
            'position': self._position,
            'velocity': self._velocity,
            'acceleration': self._acceleration,
            'setpoint': self._setpoint,
            'command': self._command,
            'temperature': self._temperature,
            'power_consumption': self._power_consumption,
            'energy_consumed': self._energy_consumed,
            'wear_factor': self._wear_factor,
            'cycle_count': self._cycle_count,
            'health_score': self.health_score,
            'failure_mode': self.failure_mode.value,
            'time': self._time
        }

    def inject_failure(self, mode: ActuatorFailureMode, severity: float = 0.5):
        """Inject a failure mode."""
        self.failure_mode = mode
        self.failure_severity = np.clip(severity, 0.0, 1.0)
        self.is_healthy = mode == ActuatorFailureMode.NONE
        if mode == ActuatorFailureMode.STUCK:
            self._stuck_position = self._position

    def clear_failure(self):
        """Clear active failure."""
        self.failure_mode = ActuatorFailureMode.NONE
        self.failure_severity = 0.0
        self.is_healthy = True
        self._stuck_position = None


class GateActuatorSystem(HighFidelityActuator):
    """
    Specialized gate actuator for aqueduct flow control.

    Models sluice gate mechanics including:
    - Non-linear flow characteristic
    - Hydraulic forces on gate
    - Seal friction
    """

    def __init__(self, name: str, max_flow: float = 200.0,
                 gate_width: float = 5.0, gate_height: float = 5.0):
        dynamics = ActuatorDynamicsModel(
            actuator_type=ActuatorType.HYDRAULIC_CYLINDER,
            position_min=0.0,
            position_max=100.0,      # Percent open
            velocity_max=5.0,        # %/second (slow for stability)
            acceleration_max=10.0,
            force_max=100000.0,      # N
            natural_frequency=2.0,   # Hz (slow response)
            damping_ratio=0.8,
            time_constant=2.0,
            static_friction=5.0,
            coulomb_friction=2.0,
            viscous_friction=0.5,
            backlash=0.5,
            dead_zone=0.2,
            rated_power=15000.0,     # W
            efficiency=0.75,
            wear_model=WearModel.USAGE_BASED,
            design_life_cycles=100000
        )

        controller = PositionController(
            kp=5.0,
            ki=0.5,
            kd=1.0,
            integral_limit=50.0,
            output_limit=100.0
        )

        super().__init__(name, dynamics, controller)

        self.max_flow = max_flow
        self.gate_width = gate_width
        self.gate_height = gate_height
        self._water_level_upstream = 0.0
        self._water_level_downstream = 0.0

    def set_water_levels(self, upstream: float, downstream: float):
        """Set water levels for hydraulic force calculation."""
        self._water_level_upstream = upstream
        self._water_level_downstream = downstream

    def get_flow_rate(self) -> float:
        """
        Calculate flow rate based on gate position and water levels.

        Uses orifice flow equation with contraction coefficient.
        """
        if self._position <= 0:
            return 0.0

        # Gate opening height
        opening = (self._position / 100.0) * self.gate_height

        # Head difference
        head = max(0, self._water_level_upstream - self._water_level_downstream)

        if head <= 0:
            return 0.0

        # Contraction coefficient (typical for sluice gates)
        Cd = 0.61

        # Orifice equation: Q = Cd * A * sqrt(2 * g * h)
        area = opening * self.gate_width
        g = 9.81

        flow = Cd * area * np.sqrt(2 * g * head)

        # Limit to max flow
        return min(flow, self.max_flow)

    def command_flow(self, desired_flow: float):
        """Command a specific flow rate."""
        if desired_flow <= 0:
            self.command(0.0)
            return

        # Inverse calculation of required position
        # This is approximate - actual relationship is non-linear
        normalized_flow = np.clip(desired_flow / self.max_flow, 0, 1)

        # Approximate inverse (accounting for sqrt relationship)
        desired_position = 100.0 * (normalized_flow ** 0.5)

        self.command(desired_position)

    def calculate_hydraulic_load(self) -> float:
        """Calculate hydraulic force on gate."""
        # Hydrostatic pressure on gate
        avg_depth = self._water_level_upstream / 2.0
        rho = 1000.0  # Water density kg/m³
        g = 9.81

        pressure = rho * g * avg_depth
        gate_area = self.gate_width * self.gate_height
        force = pressure * gate_area

        return force

    def step(self, dt: float) -> Dict[str, float]:
        """Step with hydraulic load calculation."""
        load = self.calculate_hydraulic_load()
        state = super().step(dt, external_load=load)
        state['flow_rate'] = self.get_flow_rate()
        state['hydraulic_load'] = load
        return state


class ActuatorSimulationEngine:
    """
    Complete actuator simulation engine for the aqueduct system.

    Manages all actuators with coordinated control and monitoring.
    """

    def __init__(self):
        self.actuators: Dict[str, HighFidelityActuator] = {}
        self.gate_actuators: Dict[str, GateActuatorSystem] = {}

        # Initialize default actuators
        self._initialize_default_actuators()

        # Simulation state
        self.simulation_time = 0.0
        self.total_power_consumption = 0.0
        self.total_energy_consumed = 0.0

        # Safety interlocks
        self.emergency_stop = False
        self.interlock_active = False

    def _initialize_default_actuators(self):
        """Initialize default actuators for the aqueduct."""

        # Inlet gate
        inlet_gate = GateActuatorSystem(
            "inlet_gate",
            max_flow=250.0,
            gate_width=6.0,
            gate_height=8.0
        )
        self.gate_actuators["inlet"] = inlet_gate
        self.actuators["inlet_gate"] = inlet_gate

        # Outlet gate
        outlet_gate = GateActuatorSystem(
            "outlet_gate",
            max_flow=250.0,
            gate_width=6.0,
            gate_height=8.0
        )
        self.gate_actuators["outlet"] = outlet_gate
        self.actuators["outlet_gate"] = outlet_gate

        # Emergency dump valve (fast-acting)
        dump_dynamics = ActuatorDynamicsModel(
            actuator_type=ActuatorType.HYDRAULIC_CYLINDER,
            position_min=0.0,
            position_max=100.0,
            velocity_max=50.0,           # Fast emergency response
            acceleration_max=200.0,
            force_max=50000.0,
            natural_frequency=10.0,
            damping_ratio=0.6,
            time_constant=0.2,
            rated_power=10000.0,
            efficiency=0.8
        )
        dump_valve = HighFidelityActuator("dump_valve", dump_dynamics)
        self.actuators["dump_valve"] = dump_valve

        # Bypass valve
        bypass_dynamics = ActuatorDynamicsModel(
            actuator_type=ActuatorType.ELECTRIC_SERVO,
            position_min=0.0,
            position_max=100.0,
            velocity_max=20.0,
            acceleration_max=100.0,
            force_max=20000.0,
            natural_frequency=5.0,
            damping_ratio=0.7,
            time_constant=0.5,
            rated_power=5000.0,
            efficiency=0.9
        )
        bypass_valve = HighFidelityActuator("bypass_valve", bypass_dynamics)
        self.actuators["bypass_valve"] = bypass_valve

    def command_flows(self, Q_in: float, Q_out: float,
                      emergency_dump: bool = False):
        """
        Command inlet and outlet flow rates.

        Args:
            Q_in: Desired inlet flow (m³/s)
            Q_out: Desired outlet flow (m³/s)
            emergency_dump: Activate emergency dump valve
        """
        if self.emergency_stop:
            # Emergency stop - close all
            self.gate_actuators["inlet"].command(0.0)
            self.gate_actuators["outlet"].command(100.0)
            self.actuators["dump_valve"].command(100.0)
            return

        # Command gates
        self.gate_actuators["inlet"].command_flow(Q_in)
        self.gate_actuators["outlet"].command_flow(Q_out)

        # Emergency dump valve
        if emergency_dump:
            self.actuators["dump_valve"].command(100.0)
        else:
            self.actuators["dump_valve"].command(0.0)

    def set_water_levels(self, h_upstream: float, h_downstream: float):
        """Set water levels for all gate actuators."""
        for gate in self.gate_actuators.values():
            gate.set_water_levels(h_upstream, h_downstream)

    def step(self, dt: float) -> Dict[str, Any]:
        """
        Update all actuators.

        Returns:
            Dict with actuator states and flow rates
        """
        self.simulation_time += dt

        results = {
            'timestamp': self.simulation_time,
            'actuators': {},
            'flows': {},
            'power': {}
        }

        self.total_power_consumption = 0.0

        # Update all actuators
        for name, actuator in self.actuators.items():
            if isinstance(actuator, GateActuatorSystem):
                state = actuator.step(dt)
            else:
                state = actuator.step(dt)

            results['actuators'][name] = state
            self.total_power_consumption += state.get('power_consumption', 0)

        # Calculate actual flows
        inlet_flow = self.gate_actuators["inlet"].get_flow_rate()
        outlet_flow = self.gate_actuators["outlet"].get_flow_rate()

        # Add dump valve flow if open
        dump_position = self.actuators["dump_valve"]._position
        dump_flow = dump_position / 100.0 * 200.0  # Max 200 m³/s

        results['flows'] = {
            'Q_in_actual': inlet_flow,
            'Q_out_actual': outlet_flow + dump_flow,
            'inlet_flow': inlet_flow,
            'outlet_flow': outlet_flow,
            'dump_flow': dump_flow
        }

        results['power'] = {
            'total_power_kW': self.total_power_consumption / 1000.0,
            'total_energy_kWh': self.total_energy_consumed / 3600000.0
        }

        self.total_energy_consumed += self.total_power_consumption * dt

        return results

    def emergency_shutdown(self):
        """Activate emergency shutdown."""
        self.emergency_stop = True
        # Close inlet, open outlet and dump
        self.gate_actuators["inlet"].command(0.0)
        self.gate_actuators["outlet"].command(100.0)
        self.actuators["dump_valve"].command(100.0)

    def reset_emergency(self):
        """Reset emergency shutdown."""
        self.emergency_stop = False

    def inject_failure(self, actuator_name: str, mode: ActuatorFailureMode,
                       severity: float = 0.5):
        """Inject failure into an actuator."""
        if actuator_name in self.actuators:
            self.actuators[actuator_name].inject_failure(mode, severity)

    def clear_all_failures(self):
        """Clear all actuator failures."""
        for actuator in self.actuators.values():
            actuator.clear_failure()

    def get_full_status(self) -> Dict[str, Any]:
        """Get complete actuator system status."""
        return {
            'simulation_time': self.simulation_time,
            'emergency_stop': self.emergency_stop,
            'total_power_kW': self.total_power_consumption / 1000.0,
            'total_energy_kWh': self.total_energy_consumed / 3600000.0,
            'actuators': {
                name: actuator.get_state()
                for name, actuator in self.actuators.items()
            },
            'health': {
                name: {
                    'is_healthy': actuator.is_healthy,
                    'health_score': actuator.health_score,
                    'failure_mode': actuator.failure_mode.value
                }
                for name, actuator in self.actuators.items()
            }
        }

    def reset(self):
        """Reset all actuators."""
        self.simulation_time = 0.0
        self.total_power_consumption = 0.0
        self.total_energy_consumed = 0.0
        self.emergency_stop = False
        self.interlock_active = False
        for actuator in self.actuators.values():
            actuator.reset()
