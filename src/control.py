import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Any


class ControlMode(Enum):
    """Operating modes for the autonomous controller."""
    AUTO = "AUTO"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


class RiskLevel(Enum):
    """Risk severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PerceptionSystem:
    """
    Analyzes system state to detect active scenarios and risks.
    Implements multi-physics risk detection for hydraulic, thermal,
    structural, and seismic conditions.
    """

    # Configurable thresholds
    FROUDE_CRITICAL = 0.9
    FROUDE_WARNING = 0.7
    THERMAL_DELTA_CRITICAL = 10.0
    THERMAL_DELTA_WARNING = 6.0
    BEARING_STRESS_CRITICAL = 25.0
    BEARING_STRESS_WARNING = 20.0
    JOINT_GAP_MAX_WARNING = 25.0
    JOINT_GAP_MAX_CRITICAL = 30.0
    JOINT_GAP_MIN_WARNING = 10.0
    JOINT_GAP_MIN_CRITICAL = 5.0
    VIBRATION_CRITICAL = 50.0
    VIBRATION_WARNING = 30.0
    WATER_LEVEL_MIN = 1.0
    WATER_LEVEL_MAX = 7.5

    def analyze(self, state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Analyzes the state to detect active scenarios and risks.

        Args:
            state: Current system state dictionary

        Returns:
            Tuple of (detected_scenarios, risk_messages)
        """
        detected_scenarios = []
        risks = []

        # S1.1 Hydraulic Jump Risk
        fr = state['fr']
        if fr > self.FROUDE_CRITICAL:
            detected_scenarios.append("S1.1")
            risks.append(f"CRITICAL: Flow instability (Fr={fr:.2f})")
        elif fr > self.FROUDE_WARNING:
            risks.append(f"WARNING: Flow velocity high (Fr={fr:.2f})")

        # S3.1 Thermal Bending
        delta_T = state['T_sun'] - state['T_shade']
        if delta_T > self.THERMAL_DELTA_CRITICAL:
            detected_scenarios.append("S3.1")
            risks.append(f"CRITICAL: Thermal bending risk (dT={delta_T:.1f}C)")
        elif delta_T > self.THERMAL_DELTA_WARNING:
            risks.append(f"WARNING: Thermal gradient increasing (dT={delta_T:.1f}C)")

        # S3.3 Bearing Lock / Stress
        if state['bearing_locked']:
            detected_scenarios.append("S3.3")
            risks.append("CRITICAL: Bearing LOCKED")

        if state['bearing_stress'] > self.BEARING_STRESS_CRITICAL:
            risks.append(f"CRITICAL: Bearing Stress High ({state['bearing_stress']:.1f} MPa)")
        elif state['bearing_stress'] > self.BEARING_STRESS_WARNING:
            risks.append(f"WARNING: Bearing Stress Elevated ({state['bearing_stress']:.1f} MPa)")

        # S4.1 Joint Tearing (Cold - gap expanding)
        if state['joint_gap'] > self.JOINT_GAP_MAX_CRITICAL:
            detected_scenarios.append("S4.1")
            risks.append(f"CRITICAL: Joint gap critical ({state['joint_gap']:.1f} mm)")
        elif state['joint_gap'] > self.JOINT_GAP_MAX_WARNING:
            detected_scenarios.append("S4.1")
            risks.append(f"WARNING: Joint gap expanding ({state['joint_gap']:.1f} mm)")

        # S4.2 Joint Compression (Hot - gap closing)
        if state['joint_gap'] < self.JOINT_GAP_MIN_CRITICAL:
            risks.append(f"CRITICAL: Joint gap closing ({state['joint_gap']:.1f} mm)")
        elif state['joint_gap'] < self.JOINT_GAP_MIN_WARNING:
            risks.append(f"WARNING: Joint gap narrowing ({state['joint_gap']:.1f} mm)")

        # S5.1 Earthquake
        if state['vib_amp'] > self.VIBRATION_CRITICAL:
            detected_scenarios.append("S5.1")
            risks.append("CRITICAL: SEISMIC ACTIVITY DETECTED")
        elif state['vib_amp'] > self.VIBRATION_WARNING:
            risks.append(f"WARNING: Vibration elevated ({state['vib_amp']:.1f} mm)")

        # Water level warnings
        h = state['h']
        if h < self.WATER_LEVEL_MIN:
            risks.append(f"WARNING: Water level critically low ({h:.2f} m)")
        elif h > self.WATER_LEVEL_MAX:
            risks.append(f"WARNING: Water level critically high ({h:.2f} m)")

        return detected_scenarios, risks

    def get_risk_level(self, risks: List[str]) -> RiskLevel:
        """Determine overall risk level from risk messages."""
        for risk in risks:
            if "CRITICAL" in risk:
                return RiskLevel.CRITICAL
        for risk in risks:
            if "WARNING" in risk:
                return RiskLevel.WARNING
        return RiskLevel.INFO

class AutonomousController:
    """
    Autonomous controller for the aqueduct system.
    Implements PID-based level control with scenario-specific overrides.
    """

    # Control parameters
    DEFAULT_TARGET_H = 4.0
    KP_LEVEL = 10.0  # Proportional gain for level control
    KI_LEVEL = 0.1   # Integral gain for level control
    Q_OUT_MAX = 200.0
    Q_OUT_MIN = 0.0
    Q_IN_MAX = 200.0
    Q_IN_MIN = 0.0
    COOLING_RAMP_RATE = 5.0  # m³/s per step
    COOLING_FLOW_TARGET = 120.0

    def __init__(self):
        self.mode = ControlMode.AUTO
        self.target_h = self.DEFAULT_TARGET_H
        self.perception = PerceptionSystem()
        self.integral_error = 0.0
        self.manual_Q_in = None
        self.manual_Q_out = None

    def set_mode(self, mode: ControlMode):
        """Switch control mode."""
        self.mode = mode
        if mode == ControlMode.AUTO:
            self.manual_Q_in = None
            self.manual_Q_out = None

    def set_manual_control(self, Q_in: float = None, Q_out: float = None):
        """Set manual control values (only effective in MANUAL mode)."""
        if Q_in is not None:
            self.manual_Q_in = max(self.Q_IN_MIN, min(self.Q_IN_MAX, Q_in))
        if Q_out is not None:
            self.manual_Q_out = max(self.Q_OUT_MIN, min(self.Q_OUT_MAX, Q_out))

    def _clamp_flow(self, Q: float, min_val: float, max_val: float) -> float:
        """Clamp flow value to valid range."""
        return max(min_val, min(max_val, Q))

    def _compute_pid_output(self, error: float, dt: float = 1.0) -> float:
        """Compute PID controller output for level control."""
        # Update integral (with anti-windup)
        self.integral_error += error * dt
        self.integral_error = max(-50.0, min(50.0, self.integral_error))

        # P + I control
        output = self.KP_LEVEL * error + self.KI_LEVEL * self.integral_error
        return output

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main control loop. Returns actions dict.

        Args:
            state: Current system state

        Returns:
            Dictionary containing control actions and status
        """
        actions = {}
        scenarios, risks = self.perception.analyze(state)
        risk_level = self.perception.get_risk_level(risks)

        # Store common info
        actions['risks'] = risks
        actions['active_scenarios'] = scenarios
        actions['risk_level'] = risk_level.value
        actions['mode'] = self.mode.value if isinstance(self.mode, ControlMode) else self.mode

        # Handle MANUAL mode
        if self.mode == ControlMode.MANUAL:
            actions['Q_in'] = self.manual_Q_in if self.manual_Q_in is not None else state['Q_in']
            actions['Q_out'] = self.manual_Q_out if self.manual_Q_out is not None else state['Q_out']
            actions['status'] = "MANUAL CONTROL"
            return actions

        # --- EMERGENCY Scenarios (Highest Priority) ---

        # S5.1 + S3.3: Earthquake + Locked Bearing -> EMERGENCY RELEASE
        if "S5.1" in scenarios and "S3.3" in scenarios:
            self.mode = ControlMode.EMERGENCY
            actions['status'] = "EMERGENCY: DUMP WATER"
            actions['Q_out'] = self.Q_OUT_MAX
            actions['Q_in'] = self.Q_IN_MIN
            actions['mode'] = ControlMode.EMERGENCY.value
            return actions

        # S5.1 alone: Earthquake - reduce flow, prepare for emergency
        if "S5.1" in scenarios:
            actions['status'] = "EMERGENCY: SEISMIC RESPONSE"
            actions['Q_out'] = state['Q_in'] * 1.2  # Slightly increase outflow
            actions['Q_in'] = state['Q_in'] * 0.8   # Reduce inflow
            actions['Q_out'] = self._clamp_flow(actions['Q_out'], self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = self._clamp_flow(actions['Q_in'], self.Q_IN_MIN, self.Q_IN_MAX)
            return actions

        # --- Critical Scenarios ---

        # S1.1: High Froude Number -> Increase Level (Submerge hydraulic jump)
        if "S1.1" in scenarios:
            actions['status'] = "STABILIZING FLOW (S1.1)"
            self.target_h = 7.0  # Target higher water level
            error_h = self.target_h - state['h']
            # Aggressive close of downstream gate
            target_Q_out = state['Q_in'] - (error_h * self.KP_LEVEL * 2.0)
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # S3.1: Thermal Bending -> Water Cooling (Increase Flow)
        if "S3.1" in scenarios:
            actions['status'] = "COOLING MODE (S3.1)"
            current_Q = state['Q_in']
            if current_Q < self.COOLING_FLOW_TARGET:
                actions['Q_in'] = current_Q + self.COOLING_RAMP_RATE
                actions['Q_out'] = actions['Q_in']  # Pass through
            else:
                actions['Q_in'] = self.COOLING_FLOW_TARGET
                actions['Q_out'] = self.COOLING_FLOW_TARGET
            return actions

        # S3.3: Bearing Lock (without earthquake) -> Controlled release
        if "S3.3" in scenarios:
            actions['status'] = "BEARING LOCK RESPONSE (S3.3)"
            # Reduce water level gradually to decrease load
            self.target_h = 3.0
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h)
            target_Q_out = state['Q_in'] - adjustment
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # S4.1: Joint Gap Expanding (Cold) -> Increase water for thermal mass
        if "S4.1" in scenarios:
            actions['status'] = "JOINT PROTECTION MODE (S4.1)"
            # Increase water level slightly for thermal buffering
            self.target_h = 5.0
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h)
            target_Q_out = state['Q_in'] - adjustment
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # --- Normal Operation (No critical scenarios) ---
        self.target_h = self.DEFAULT_TARGET_H
        self.mode = ControlMode.AUTO

        # PID Level Control
        error_h = self.target_h - state['h']
        adjustment = self._compute_pid_output(error_h)
        nominal_flow = state['Q_in']
        target_Q_out = nominal_flow - adjustment

        actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
        actions['Q_in'] = state['Q_in']
        actions['status'] = "NORMAL"

        return actions

    def reset(self):
        """Reset controller to initial state."""
        self.mode = ControlMode.AUTO
        self.target_h = self.DEFAULT_TARGET_H
        self.integral_error = 0.0
        self.manual_Q_in = None
        self.manual_Q_out = None
