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

    Covers all 14+ scenarios:
    - S1.1: Hydraulic Jump, S1.2: Surge Wave
    - S2.1: Vortex-Induced Vibration
    - S3.1: Thermal Bending, S3.2: Rapid Cooling, S3.3: Bearing Lock
    - S4.1: Joint Expansion, S4.2: Joint Compression
    - S5.1: Seismic Event, S5.2: Aftershock Sequence
    - S6.1: Sensor Degradation, S6.2: Actuator Fault
    - MULTI_PHYSICS: Combined scenarios
    """

    # Configurable thresholds - Hydraulic
    FROUDE_CRITICAL = 0.9
    FROUDE_WARNING = 0.7
    SURGE_FLOW_THRESHOLD = 130.0  # S1.2 surge detection

    # Thresholds - Thermal
    THERMAL_DELTA_CRITICAL = 10.0
    THERMAL_DELTA_WARNING = 6.0
    COOLING_RATE_CRITICAL = 3.0  # °C per minute for S3.2
    COOLING_RATE_WARNING = 1.5

    # Thresholds - Structural
    BEARING_STRESS_CRITICAL = 40.0  # Calibrated for normal ~31 MPa
    BEARING_STRESS_WARNING = 35.0
    JOINT_GAP_MAX_WARNING = 25.0
    JOINT_GAP_MAX_CRITICAL = 30.0
    JOINT_GAP_MIN_WARNING = 10.0
    JOINT_GAP_MIN_CRITICAL = 5.0

    # Thresholds - Vibration/Seismic
    VIBRATION_CRITICAL = 50.0  # S5.1 main shock
    VIBRATION_WARNING = 30.0
    VIBRATION_AFTERSHOCK = 20.0  # S5.2 aftershock
    WIND_VIV_CRITICAL = 12.0  # S2.1 vortex-induced vibration
    WIND_VIV_WARNING = 8.0

    # Thresholds - Water Level
    WATER_LEVEL_MIN = 1.0
    WATER_LEVEL_MAX = 7.5

    # State tracking for rate-based detection
    def __init__(self):
        self.prev_T_ambient = None
        self.prev_time = None
        self.prev_vib_history = []
        self.sensor_confidence_history = []
        self.actuator_response_history = []

    def analyze(self, state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Analyzes the state to detect all 14+ scenarios and risks.

        Args:
            state: Current system state dictionary

        Returns:
            Tuple of (detected_scenarios, risk_messages)
        """
        detected_scenarios = []
        risks = []
        current_time = state.get('time', 0.0)

        # ========== S1.1 Hydraulic Jump ==========
        fr = state.get('fr', 0.0)
        if fr > self.FROUDE_CRITICAL:
            detected_scenarios.append("S1.1")
            risks.append(f"CRITICAL: Flow instability (Fr={fr:.2f})")
        elif fr > self.FROUDE_WARNING:
            risks.append(f"WARNING: Flow velocity high (Fr={fr:.2f})")

        # ========== S1.2 Surge Wave ==========
        Q_in = state.get('Q_in', 80.0)
        if Q_in > self.SURGE_FLOW_THRESHOLD:
            detected_scenarios.append("S1.2")
            risks.append(f"WARNING: Surge flow detected (Q={Q_in:.1f} m³/s)")

        # ========== S2.1 Vortex-Induced Vibration ==========
        wind_speed = state.get('wind_speed', 0.0)
        vib_amp = state.get('vib_amp', 0.0)
        # VIV occurs at critical wind speeds with structural vibration
        if wind_speed > self.WIND_VIV_CRITICAL and vib_amp > 10.0:
            detected_scenarios.append("S2.1")
            risks.append(f"CRITICAL: Vortex vibration (wind={wind_speed:.1f}m/s, vib={vib_amp:.1f}mm)")
        elif wind_speed > self.WIND_VIV_WARNING:
            risks.append(f"WARNING: High wind condition ({wind_speed:.1f} m/s)")

        # ========== S3.1 Thermal Bending ==========
        T_sun = state.get('T_sun', 20.0)
        T_shade = state.get('T_shade', 20.0)
        delta_T = T_sun - T_shade
        if delta_T > self.THERMAL_DELTA_CRITICAL:
            detected_scenarios.append("S3.1")
            risks.append(f"CRITICAL: Thermal bending risk (ΔT={delta_T:.1f}°C)")
        elif delta_T > self.THERMAL_DELTA_WARNING:
            risks.append(f"WARNING: Thermal gradient increasing (ΔT={delta_T:.1f}°C)")

        # ========== S3.2 Rapid Cooling ==========
        T_ambient = state.get('T_ambient', 25.0)
        if self.prev_T_ambient is not None and self.prev_time is not None:
            dt_minutes = (current_time - self.prev_time) / 60.0
            if dt_minutes > 0:
                cooling_rate = (self.prev_T_ambient - T_ambient) / dt_minutes
                if cooling_rate > self.COOLING_RATE_CRITICAL:
                    detected_scenarios.append("S3.2")
                    risks.append(f"CRITICAL: Rapid cooling ({cooling_rate:.1f}°C/min)")
                elif cooling_rate > self.COOLING_RATE_WARNING:
                    risks.append(f"WARNING: Temperature dropping ({cooling_rate:.1f}°C/min)")
        self.prev_T_ambient = T_ambient
        self.prev_time = current_time

        # ========== S3.3 Bearing Lock ==========
        bearing_locked = state.get('bearing_locked', False)
        bearing_stress = state.get('bearing_stress', 0.0)
        if bearing_locked:
            detected_scenarios.append("S3.3")
            risks.append("CRITICAL: Bearing LOCKED - thermal expansion blocked")

        if bearing_stress > self.BEARING_STRESS_CRITICAL:
            risks.append(f"CRITICAL: Bearing stress critical ({bearing_stress:.1f} MPa)")
        elif bearing_stress > self.BEARING_STRESS_WARNING:
            risks.append(f"WARNING: Bearing stress elevated ({bearing_stress:.1f} MPa)")

        # ========== S4.1 Joint Expansion (Cold) ==========
        joint_gap = state.get('joint_gap', 20.0)
        if joint_gap > self.JOINT_GAP_MAX_CRITICAL:
            detected_scenarios.append("S4.1")
            risks.append(f"CRITICAL: Joint gap critical ({joint_gap:.1f} mm)")
        elif joint_gap > self.JOINT_GAP_MAX_WARNING:
            detected_scenarios.append("S4.1")
            risks.append(f"WARNING: Joint gap expanding ({joint_gap:.1f} mm)")

        # ========== S4.2 Joint Compression (Hot) ==========
        if joint_gap < self.JOINT_GAP_MIN_CRITICAL:
            detected_scenarios.append("S4.2")
            risks.append(f"CRITICAL: Joint compression ({joint_gap:.1f} mm)")
        elif joint_gap < self.JOINT_GAP_MIN_WARNING:
            detected_scenarios.append("S4.2")
            risks.append(f"WARNING: Joint gap narrowing ({joint_gap:.1f} mm)")

        # ========== S5.1 Seismic Event (Main Shock) ==========
        if vib_amp > self.VIBRATION_CRITICAL:
            detected_scenarios.append("S5.1")
            risks.append(f"CRITICAL: SEISMIC ACTIVITY ({vib_amp:.1f} mm)")

        # ========== S5.2 Aftershock Sequence ==========
        # Track vibration history for aftershock detection
        self.prev_vib_history.append(vib_amp)
        if len(self.prev_vib_history) > 20:
            self.prev_vib_history.pop(0)

        # Aftershock: multiple moderate vibrations after a main shock
        if len(self.prev_vib_history) >= 5:
            recent_high = sum(1 for v in self.prev_vib_history[-5:] if v > self.VIBRATION_AFTERSHOCK)
            peak_in_history = max(self.prev_vib_history) > self.VIBRATION_CRITICAL
            if recent_high >= 2 and peak_in_history and vib_amp < self.VIBRATION_CRITICAL:
                detected_scenarios.append("S5.2")
                risks.append("WARNING: Aftershock sequence detected")

        # ========== S6.1 Sensor Degradation ==========
        h_confidence = state.get('h_confidence', 1.0)
        v_confidence = state.get('v_confidence', 1.0)
        T_confidence = state.get('T_confidence', 1.0)
        min_confidence = min(h_confidence, v_confidence, T_confidence)

        self.sensor_confidence_history.append(min_confidence)
        if len(self.sensor_confidence_history) > 10:
            self.sensor_confidence_history.pop(0)

        if len(self.sensor_confidence_history) >= 5:
            avg_confidence = sum(self.sensor_confidence_history[-5:]) / 5
            if avg_confidence < 0.6:
                detected_scenarios.append("S6.1")
                risks.append(f"WARNING: Sensor degradation (conf={avg_confidence:.2f})")
            elif avg_confidence < 0.8:
                risks.append(f"INFO: Sensor confidence low ({avg_confidence:.2f})")

        # ========== S6.2 Actuator Fault ==========
        # Detect actuator fault by comparing commanded vs actual flow
        Q_in_cmd = state.get('Q_in_cmd', Q_in)
        Q_out_cmd = state.get('Q_out_cmd', state.get('Q_out', 80.0))
        Q_out_actual = state.get('Q_out', 80.0)

        in_error = abs(Q_in - Q_in_cmd) / max(Q_in_cmd, 1.0)
        out_error = abs(Q_out_actual - Q_out_cmd) / max(Q_out_cmd, 1.0)

        if in_error > 0.2 or out_error > 0.2:
            detected_scenarios.append("S6.2")
            risks.append(f"WARNING: Actuator response anomaly (err={max(in_error,out_error):.1%})")

        # ========== MULTI_PHYSICS Detection ==========
        # Multiple physics coupling detected
        multi_count = 0
        if "S1.1" in detected_scenarios or "S1.2" in detected_scenarios:
            multi_count += 1  # Hydraulic
        if "S2.1" in detected_scenarios:
            multi_count += 1  # Wind
        if any(s in detected_scenarios for s in ["S3.1", "S3.2", "S3.3"]):
            multi_count += 1  # Thermal
        if any(s in detected_scenarios for s in ["S4.1", "S4.2"]):
            multi_count += 1  # Structural
        if any(s in detected_scenarios for s in ["S5.1", "S5.2"]):
            multi_count += 1  # Seismic

        if multi_count >= 2:
            detected_scenarios.append("MULTI_PHYSICS")
            risks.append(f"CRITICAL: Multi-physics coupling ({multi_count} domains active)")

        # ========== Water Level Warnings ==========
        h = state.get('h', 4.0)
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

        # S4.2: Joint Compression (Hot) -> Reduce water temperature effect
        if "S4.2" in scenarios:
            actions['status'] = "JOINT COMPRESSION MODE (S4.2)"
            # Increase flow to cool structure
            current_Q = state['Q_in']
            actions['Q_in'] = min(current_Q + 10.0, 150.0)
            actions['Q_out'] = actions['Q_in']
            return actions

        # S1.2: Surge Wave -> Attenuate flow
        if "S1.2" in scenarios:
            actions['status'] = "SURGE ATTENUATION (S1.2)"
            # Gradually increase outflow to handle surge
            actions['Q_out'] = min(state['Q_in'] * 1.1, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # S2.1: Vortex-Induced Vibration -> Increase damping (water mass)
        if "S2.1" in scenarios:
            actions['status'] = "VIV DAMPING MODE (S2.1)"
            # Increase water level for mass damping
            self.target_h = 6.0
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h)
            target_Q_out = state['Q_in'] - adjustment
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # S3.2: Rapid Cooling -> Maintain thermal inertia
        if "S3.2" in scenarios:
            actions['status'] = "RAPID COOLING RESPONSE (S3.2)"
            # Maintain higher water level for thermal buffer
            self.target_h = 5.5
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h)
            target_Q_out = state['Q_in'] - adjustment
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in']
            return actions

        # S5.2: Aftershock -> Cautious operation
        if "S5.2" in scenarios:
            actions['status'] = "AFTERSHOCK MONITORING (S5.2)"
            # Reduce water level but don't dump
            self.target_h = 3.5
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h)
            target_Q_out = state['Q_in'] - adjustment
            actions['Q_out'] = self._clamp_flow(target_Q_out, self.Q_OUT_MIN, self.Q_OUT_MAX)
            actions['Q_in'] = state['Q_in'] * 0.9
            return actions

        # S6.1: Sensor Degradation -> Conservative operation
        if "S6.1" in scenarios:
            actions['status'] = "SENSOR DEGRADATION MODE (S6.1)"
            # Use estimated values, maintain safe level
            self.target_h = 4.0
            actions['Q_out'] = state['Q_in'] * 0.95  # Slightly conservative
            actions['Q_in'] = state['Q_in']
            return actions

        # S6.2: Actuator Fault -> Reduce control authority
        if "S6.2" in scenarios:
            actions['status'] = "ACTUATOR FAULT MODE (S6.2)"
            # Reduce control changes to avoid making things worse
            actions['Q_out'] = state['Q_out']  # Hold current
            actions['Q_in'] = state['Q_in']
            return actions

        # MULTI_PHYSICS: Complex coupling -> Priority-based response
        if "MULTI_PHYSICS" in scenarios:
            actions['status'] = "MULTI-PHYSICS RESPONSE"
            # Default to conservative safe operation
            self.target_h = 4.0
            error_h = self.target_h - state['h']
            adjustment = self._compute_pid_output(error_h) * 0.5  # Reduced gain
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
