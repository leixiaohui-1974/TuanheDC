"""
Actuator Models for TAOS High-Fidelity Simulation

This module implements realistic actuator models with:
- Response dynamics (first/second order)
- Rate limits and saturation
- Backlash and dead zones
- Actuator faults (stuck, slow, partial failure)
- Power consumption modeling
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class ActuatorFaultType(Enum):
    """Types of actuator faults."""
    NONE = "none"
    STUCK = "stuck"              # Actuator frozen at position
    SLOW = "slow"                # Reduced response speed
    PARTIAL = "partial"          # Reduced range of motion
    OSCILLATION = "oscillation"  # Unwanted oscillation
    COMPLETE_FAILURE = "failed"  # No response


@dataclass
class ActuatorConfig:
    """Configuration for an actuator."""
    name: str
    min_value: float = 0.0
    max_value: float = 200.0
    rate_limit: float = 50.0     # Max change per second
    time_constant: float = 0.5   # Response time constant (seconds)
    dead_zone: float = 0.0       # Dead zone around setpoint
    backlash: float = 0.0        # Backlash hysteresis
    power_consumption: float = 0.0  # kW at full operation
    fault_type: ActuatorFaultType = ActuatorFaultType.NONE
    fault_magnitude: float = 0.0


class Actuator:
    """
    Single actuator model with dynamics and faults.

    Implements first-order dynamics with rate limiting:
        τ * dx/dt + x = u (with rate limit)
    """

    def __init__(self, config: ActuatorConfig):
        self.config = config
        self.position = 0.0           # Current actual position
        self.command = 0.0            # Current command
        self.velocity = 0.0           # Current rate of change
        self.stuck_position = None    # Position when stuck
        self.last_direction = 0       # For backlash tracking
        self.is_healthy = True
        self.power_draw = 0.0         # Current power consumption

    def reset(self):
        """Reset actuator state."""
        self.position = 0.0
        self.command = 0.0
        self.velocity = 0.0
        self.stuck_position = None
        self.last_direction = 0
        self.is_healthy = True
        self.power_draw = 0.0

    def inject_fault(self, fault_type: ActuatorFaultType, magnitude: float = 0.0):
        """Inject a fault."""
        self.config.fault_type = fault_type
        self.config.fault_magnitude = magnitude
        if fault_type == ActuatorFaultType.STUCK:
            self.stuck_position = self.position
        self.is_healthy = fault_type == ActuatorFaultType.NONE

    def clear_fault(self):
        """Clear active fault."""
        self.config.fault_type = ActuatorFaultType.NONE
        self.config.fault_magnitude = 0.0
        self.stuck_position = None
        self.is_healthy = True

    def command_to(self, setpoint: float):
        """Set the command setpoint."""
        # Apply saturation
        self.command = np.clip(setpoint, self.config.min_value, self.config.max_value)

    def step(self, dt: float) -> float:
        """
        Update actuator position based on dynamics.

        Args:
            dt: Time step in seconds

        Returns:
            Current actuator position
        """
        # Handle complete failure
        if self.config.fault_type == ActuatorFaultType.COMPLETE_FAILURE:
            self.power_draw = 0.0
            return self.position

        # Handle stuck fault
        if self.config.fault_type == ActuatorFaultType.STUCK:
            self.power_draw = 0.0
            if self.stuck_position is not None:
                self.position = self.stuck_position
            return self.position

        # Calculate target with partial failure
        target = self.command
        if self.config.fault_type == ActuatorFaultType.PARTIAL:
            # Reduce effective range
            range_factor = 1.0 - self.config.fault_magnitude
            center = (self.config.max_value + self.config.min_value) / 2
            target = center + (target - center) * range_factor

        # Apply dead zone
        error = target - self.position
        if abs(error) < self.config.dead_zone:
            error = 0.0

        # Apply backlash
        if self.config.backlash > 0:
            current_direction = np.sign(error)
            if current_direction != self.last_direction and current_direction != 0:
                # Direction change - apply backlash delay
                if abs(error) < self.config.backlash:
                    error = 0.0
                else:
                    error -= np.sign(error) * self.config.backlash
            self.last_direction = current_direction if current_direction != 0 else self.last_direction

        # Calculate desired velocity based on first-order dynamics
        time_constant = self.config.time_constant
        if self.config.fault_type == ActuatorFaultType.SLOW:
            time_constant *= (1.0 + self.config.fault_magnitude)

        desired_velocity = error / max(time_constant, 0.01)

        # Apply rate limit
        rate_limit = self.config.rate_limit
        if self.config.fault_type == ActuatorFaultType.SLOW:
            rate_limit /= (1.0 + self.config.fault_magnitude)

        self.velocity = np.clip(desired_velocity, -rate_limit, rate_limit)

        # Apply oscillation fault
        if self.config.fault_type == ActuatorFaultType.OSCILLATION:
            oscillation = self.config.fault_magnitude * np.sin(10.0 * np.pi * self.position)
            self.velocity += oscillation

        # Update position
        self.position += self.velocity * dt

        # Apply saturation
        self.position = np.clip(self.position, self.config.min_value, self.config.max_value)

        # Calculate power consumption
        self.power_draw = self.config.power_consumption * abs(self.velocity) / self.config.rate_limit

        return self.position

    def get_status(self) -> Dict[str, Any]:
        """Get actuator status."""
        return {
            'name': self.config.name,
            'position': self.position,
            'command': self.command,
            'velocity': self.velocity,
            'is_healthy': self.is_healthy,
            'fault': self.config.fault_type.value,
            'power_draw': self.power_draw
        }


class GateActuator(Actuator):
    """
    Specialized gate actuator for flow control.
    Maps gate position (0-100%) to flow rate.
    """

    def __init__(self, name: str, max_flow: float = 200.0,
                 rate_limit: float = 20.0, time_constant: float = 2.0):
        config = ActuatorConfig(
            name=name,
            min_value=0.0,
            max_value=100.0,  # 0-100% open
            rate_limit=rate_limit,  # % per second
            time_constant=time_constant,
            dead_zone=0.5,
            backlash=1.0,
            power_consumption=5.0  # kW
        )
        super().__init__(config)
        self.max_flow = max_flow

    def get_flow_rate(self) -> float:
        """Get current flow rate based on gate position."""
        # Non-linear gate characteristic (typical for sluice gates)
        # Q = Qmax * (position/100)^0.5 for free flow
        normalized_position = self.position / 100.0
        return self.max_flow * np.sqrt(max(0, normalized_position))

    def command_flow(self, desired_flow: float):
        """Command a specific flow rate (inverse of get_flow_rate)."""
        normalized_flow = np.clip(desired_flow / self.max_flow, 0, 1)
        # Inverse: position = 100 * (Q/Qmax)^2
        desired_position = 100.0 * (normalized_flow ** 2)
        self.command_to(desired_position)


class ValveActuator(Actuator):
    """
    Fast-acting valve for emergency response.
    """

    def __init__(self, name: str, max_flow: float = 200.0):
        config = ActuatorConfig(
            name=name,
            min_value=0.0,
            max_value=100.0,
            rate_limit=100.0,  # Fast response
            time_constant=0.2,  # Quick
            dead_zone=0.0,
            backlash=0.0,
            power_consumption=2.0
        )
        super().__init__(config)
        self.max_flow = max_flow

    def get_flow_rate(self) -> float:
        """Linear flow characteristic for valves."""
        return self.max_flow * (self.position / 100.0)


class ActuatorSuite:
    """
    Complete actuator suite for the aqueduct system.
    """

    def __init__(self):
        # Main flow control gates
        self.inlet_gate = GateActuator(
            "inlet_gate",
            max_flow=200.0,
            rate_limit=10.0,    # Slow movement for stability
            time_constant=3.0
        )

        self.outlet_gate = GateActuator(
            "outlet_gate",
            max_flow=200.0,
            rate_limit=15.0,
            time_constant=2.5
        )

        # Emergency dump valve
        self.dump_valve = ValveActuator("dump_valve", max_flow=200.0)

        # Bypass valve for fine control
        self.bypass_valve = ValveActuator("bypass_valve", max_flow=50.0)

        # All actuators
        self.all_actuators = [
            self.inlet_gate,
            self.outlet_gate,
            self.dump_valve,
            self.bypass_valve
        ]

        # Power monitoring
        self.total_power = 0.0

    def reset(self):
        """Reset all actuators."""
        for actuator in self.all_actuators:
            actuator.reset()
        self.total_power = 0.0

    def command(self, Q_in: float, Q_out: float, emergency_dump: bool = False):
        """
        Command flow rates.

        Args:
            Q_in: Desired inlet flow (m³/s)
            Q_out: Desired outlet flow (m³/s)
            emergency_dump: If True, open dump valve
        """
        self.inlet_gate.command_flow(Q_in)
        self.outlet_gate.command_flow(Q_out)

        if emergency_dump:
            self.dump_valve.command_to(100.0)  # Fully open
        else:
            self.dump_valve.command_to(0.0)    # Closed

    def step(self, dt: float) -> Dict[str, float]:
        """
        Update all actuators.

        Returns:
            Actual flow rates achieved
        """
        self.total_power = 0.0

        for actuator in self.all_actuators:
            actuator.step(dt)
            self.total_power += actuator.power_draw

        # Calculate actual flows
        Q_in_actual = self.inlet_gate.get_flow_rate()
        Q_out_actual = self.outlet_gate.get_flow_rate() + self.dump_valve.get_flow_rate()

        return {
            'Q_in_actual': Q_in_actual,
            'Q_out_actual': Q_out_actual,
            'inlet_position': self.inlet_gate.position,
            'outlet_position': self.outlet_gate.position,
            'dump_position': self.dump_valve.position,
            'total_power_kW': self.total_power
        }

    def inject_fault(self, actuator_name: str, fault_type: ActuatorFaultType,
                    magnitude: float = 0.0):
        """Inject fault into specific actuator."""
        actuator_map = {
            'inlet': self.inlet_gate,
            'outlet': self.outlet_gate,
            'dump': self.dump_valve,
            'bypass': self.bypass_valve
        }

        if actuator_name in actuator_map:
            actuator_map[actuator_name].inject_fault(fault_type, magnitude)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all actuators."""
        return {
            'inlet_gate': self.inlet_gate.get_status(),
            'outlet_gate': self.outlet_gate.get_status(),
            'dump_valve': self.dump_valve.get_status(),
            'bypass_valve': self.bypass_valve.get_status(),
            'total_power_kW': self.total_power,
            'all_healthy': all(a.is_healthy for a in self.all_actuators)
        }


class ActuatorController:
    """
    Low-level actuator controller with safety interlocks.
    """

    def __init__(self, actuators: ActuatorSuite):
        self.actuators = actuators
        self.emergency_mode = False
        self.manual_override = False

        # Safety limits
        self.max_rate_of_change = 50.0  # m³/s per second
        self.min_outlet_flow = 10.0     # Minimum to prevent backflow

        # Command history for rate limiting
        self.last_Q_in_cmd = 80.0
        self.last_Q_out_cmd = 80.0

    def execute_command(self, Q_in: float, Q_out: float, dt: float,
                       emergency: bool = False) -> Dict[str, float]:
        """
        Execute flow commands with safety checks.

        Args:
            Q_in: Desired inlet flow
            Q_out: Desired outlet flow
            dt: Time step
            emergency: Emergency mode flag

        Returns:
            Actual achieved flows
        """
        # Rate limiting
        max_change = self.max_rate_of_change * dt

        Q_in_limited = np.clip(
            Q_in,
            self.last_Q_in_cmd - max_change,
            self.last_Q_in_cmd + max_change
        )

        Q_out_limited = np.clip(
            Q_out,
            self.last_Q_out_cmd - max_change,
            self.last_Q_out_cmd + max_change
        )

        # Safety: minimum outlet flow
        Q_out_limited = max(Q_out_limited, self.min_outlet_flow)

        # Emergency override
        if emergency or self.emergency_mode:
            self.actuators.command(0.0, 200.0, emergency_dump=True)
            self.emergency_mode = True
        else:
            self.actuators.command(Q_in_limited, Q_out_limited, emergency_dump=False)

        # Update actuator positions
        result = self.actuators.step(dt)

        # Store for next iteration
        self.last_Q_in_cmd = Q_in_limited
        self.last_Q_out_cmd = Q_out_limited

        return result

    def reset_emergency(self):
        """Clear emergency mode."""
        self.emergency_mode = False
        self.actuators.dump_valve.command_to(0.0)

    def get_actuator_delays(self) -> Dict[str, float]:
        """Get effective delays for each actuator."""
        return {
            'inlet_delay_s': self.actuators.inlet_gate.config.time_constant,
            'outlet_delay_s': self.actuators.outlet_gate.config.time_constant,
            'dump_delay_s': self.actuators.dump_valve.config.time_constant
        }
