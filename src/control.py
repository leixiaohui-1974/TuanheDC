import numpy as np

class PerceptionSystem:
    def analyze(self, state):
        """
        Analyzes the state to detect active scenarios and risks.
        """
        detected_scenarios = []
        risks = []

        # S1.1 Hydraulic Jump Risk
        fr = state['fr']
        if fr > 0.9:
            detected_scenarios.append("S1.1")
            risks.append(f"CRITICAL: Flow instability (Fr={fr:.2f})")
        elif fr > 0.7:
            risks.append(f"WARNING: Flow velocity high (Fr={fr:.2f})")

        # S3.1 Thermal Bending
        delta_T = state['T_sun'] - state['T_shade']
        if delta_T > 10.0:
            detected_scenarios.append("S3.1")
            risks.append(f"CRITICAL: Thermal bending risk (dT={delta_T:.1f}C)")

        # S3.3 Bearing Lock / Stress
        if state['bearing_locked']:
            detected_scenarios.append("S3.3")
            risks.append("CRITICAL: Bearing LOCKED")

        if state['bearing_stress'] > 25.0: # Arbitrary high stress threshold
            risks.append(f"CRITICAL: Bearing Stress High ({state['bearing_stress']:.1f} MPa)")

        # S4.1 Joint Tearing (Cold)
        # Gap > 30mm or so
        if state['joint_gap'] > 25.0:
             detected_scenarios.append("S4.1")
             risks.append(f"WARNING: Joint gap expanding ({state['joint_gap']:.1f} mm)")

        # S5.1 Earthquake
        if state['vib_amp'] > 50.0: # mm
             detected_scenarios.append("S5.1")
             risks.append("CRITICAL: SEISMIC ACTIVITY DETECTED")

        return detected_scenarios, risks

class AutonomousController:
    def __init__(self):
        self.mode = "AUTO" # AUTO, MANUAL, EMERGENCY
        self.target_h = 4.0
        self.perception = PerceptionSystem()

    def decide(self, state):
        """
        Main control loop. Returns actions dict.
        """
        actions = {}

        scenarios, risks = self.perception.analyze(state)

        # Default Actions (PID for Level Control)
        # Goal: Keep h constant at target (unless overridden)
        # Simple P-controller
        error_h = self.target_h - state['h']
        kp = 10.0
        # To raise level, we need net inflow. Q_out = Q_in - adjustment
        # If error > 0 (too low), we want Q_in > Q_out.
        # Assume we control Q_out (downstream gate) mainly for level,
        # and Q_in is disturbance or requested from upstream.
        # Let's say we request Q_out.
        nominal_flow = state['Q_in']
        target_Q_out = nominal_flow - (error_h * kp)

        # Constraints
        if target_Q_out < 0: target_Q_out = 0
        if target_Q_out > 200: target_Q_out = 200

        actions['Q_out'] = target_Q_out
        actions['Q_in'] = state['Q_in'] # Default: don't change upstream
        actions['status'] = "NORMAL"
        actions['risks'] = risks
        actions['active_scenarios'] = scenarios

        # --- Override Logic based on Scenarios ---

        # S5.1 + S3.3: Earthquake + Locked Bearing -> EMERGENCY RELEASE
        if "S5.1" in scenarios and "S3.3" in scenarios:
            actions['status'] = "EMERGENCY: DUMP WATER"
            actions['Q_out'] = 200.0 # Max open
            actions['Q_in'] = 0.0 # Close upstream
            return actions

        # S1.1: High Froude Number -> Increase Level (Submerge hydraulic jump)
        if "S1.1" in scenarios:
            actions['status'] = "STABILIZING FLOW (S1.1)"
            # Target higher water level to reduce velocity (v = Q/A)
            # If we are already at max Q, we must raise h.
            # Override target_h temporarily
            self.target_h = 7.0
            # Recalculate Q_out for new target
            error_h = self.target_h - state['h']
            target_Q_out = state['Q_in'] - (error_h * kp * 2.0) # Aggressive close
            actions['Q_out'] = target_Q_out

        # S3.1: Thermal Bending -> Water Cooling (Increase Flow)
        elif "S3.1" in scenarios:
            actions['status'] = "COOLING MODE (S3.1)"
            # We need high velocity to cool. v = Q / (W*h)
            # Increase Q_in and Q_out together (Pass-through)
            current_Q = state['Q_in']
            if current_Q < 120.0:
                actions['Q_in'] = current_Q + 5.0 # Ramp up
                actions['Q_out'] = actions['Q_in'] # Pass through

        else:
            # Reset target if no special scenario
            self.target_h = 4.0

        return actions
