import numpy as np
import math

class AqueductSimulation:
    def __init__(self):
        # --- System Constants ---
        self.L_span = 40.0  # Span length (m)
        self.Width = 10.0   # Width (m)
        self.H_max = 8.0    # Max depth (m)
        self.mass_concrete = 1600.0 * 1000 # kg (approx per span)
        self.g = 9.81

        # Thermal props
        self.alpha_c = 1e-5 # Thermal expansion coeff concrete
        # Reduced coefficients to simulate thermal inertia (time constants in minutes/hours)
        self.k_air = 0.0005   # Heat transfer coeff air
        self.k_water = 0.002  # Heat transfer coeff water
        self.k_sun = 0.001    # Solar heating gain factor

        # --- State Variables ---
        # Water
        self.h = 4.0        # Water level (m)
        self.v = 2.0        # Flow velocity (m/s)
        self.T_water = 15.0 # Water temp (C)
        self.Q_in = 80.0    # Inflow (m3/s)
        self.Q_out = 80.0   # Outflow (m3/s) - controlled by gates

        # Concrete
        self.T_sun = 20.0   # Sun-side temp (C)
        self.T_shade = 20.0 # Shade-side temp (C)
        self.joint_gap = 20.0 # mm (Design gap)

        # Vibration/Structure
        self.vib_amp = 0.0  # Vibration amplitude (mm)
        self.bearing_stress = 10.0 # MPa

        # --- Environment Inputs (Disturbances) ---
        self.T_ambient = 25.0
        self.solar_rad = 0.0 # 0 to 1 scale
        self.wind_speed = 0.0 # m/s
        self.ground_accel = 0.0 # m/s2 (Earthquake)

        # Scenario Flags
        self.ice_plug = False
        self.bearing_locked = False

        self.time = 0.0

    def get_state(self):
        return {
            'time': self.time,
            'h': self.h,
            'v': self.v,
            'T_water': self.T_water,
            'T_sun': self.T_sun,
            'T_shade': self.T_shade,
            'T_ambient': self.T_ambient,
            'joint_gap': self.joint_gap,
            'fr': self.calculate_froude(),
            'vib_amp': self.vib_amp,
            'bearing_stress': self.bearing_stress,
            'wind_speed': self.wind_speed,
            'solar_rad': self.solar_rad,
            'Q_in': self.Q_in,
            'Q_out': self.Q_out,
            'bearing_locked': self.bearing_locked
        }

    def calculate_froude(self):
        if self.h <= 0.1: return 0
        return self.v / np.sqrt(self.g * self.h)

    def step(self, dt, control_actions):
        """
        Evolve the system state by dt.
        control_actions: dict with 'Q_in', 'Q_out', 'T_water_target'(conceptually), etc.
        """
        # 1. Apply Controls
        target_Q_in = control_actions.get('Q_in', self.Q_in)
        target_Q_out = control_actions.get('Q_out', self.Q_out)

        # Actuator dynamics
        self.Q_in += (target_Q_in - self.Q_in) * 0.1
        self.Q_out += (target_Q_out - self.Q_out) * 0.1

        # 2. Water Dynamics
        area = self.Width * self.L_span
        dh_dt = (self.Q_in - self.Q_out) / area
        self.h += dh_dt * dt

        if self.h < 0.1: self.h = 0.1
        if self.h > self.H_max: self.h = self.H_max

        # Update Velocity
        avg_Q = (self.Q_in + self.Q_out) / 2.0
        self.v = avg_Q / (self.Width * self.h)

        # 3. Thermal Dynamics
        cooling_factor = self.k_water * (self.h / self.H_max) * (1 + 0.5 * self.v)

        dTs_dt = self.k_sun * self.solar_rad \
                 + self.k_air * (self.T_ambient - self.T_sun) \
                 - cooling_factor * (self.T_sun - self.T_water)

        dTsh_dt = self.k_air * (self.T_ambient - self.T_shade) \
                  - cooling_factor * (self.T_shade - self.T_water)

        self.T_sun += dTs_dt * dt
        self.T_shade += dTsh_dt * dt

        if 'T_water_input' in control_actions:
             self.T_water = control_actions['T_water_input']

        # 4. Structural Dynamics
        avg_concrete_temp = (self.T_sun + self.T_shade) / 2.0
        delta_T = avg_concrete_temp - 20.0
        expansion = self.alpha_c * (self.L_span * 1000) * delta_T # mm
        self.joint_gap = 20.0 - expansion

        breathing = (self.h - 4.0) * 0.5 # mm
        self.joint_gap += breathing

        weight_stress = (self.mass_concrete + self.h * self.Width * self.L_span * 1000) * 9.81 / 1e6

        thermal_stress = 0.0
        if self.bearing_locked:
            E_concrete = 30e3 # MPa
            # Stress depends on how much it WANTS to expand but CAN'T.
            # Simplified: Stress proportional to delta_T
            thermal_stress = E_concrete * self.alpha_c * abs(delta_T) * 0.5 # Factor 0.5 for partial constraint

        self.bearing_stress = weight_stress + thermal_stress

        # Vibration
        target_amp = 0.0
        if self.wind_speed > 10.0 and self.wind_speed < 15.0:
            target_amp = 20.0

        if self.ground_accel > 0:
            target_amp += self.ground_accel * 100.0

        if self.calculate_froude() > 0.9:
            target_amp += 10.0 * (self.calculate_froude() - 0.9)

        self.vib_amp += (target_amp - self.vib_amp) * 0.2

        self.time += dt

    def inject_scenario(self, scenario_id):
        """
        Sets parameters to mimic a scenario.
        """
        if scenario_id == "NORMAL":
            self.T_ambient = 25.0
            self.solar_rad = 0.5
            self.wind_speed = 2.0
            self.bearing_locked = False
            self.ground_accel = 0.0
            self.Q_in = 80.0

        elif scenario_id == "S1.1": # Hydraulic Jump / Surge
            # Needs Fr > 0.9. Fr = v / sqrt(gh).
            # If h = 2.0, sqrt(gh) ~ 4.4. Need v > 4.0.
            # v = Q / (W*h) = Q / 20. Need Q > 80.
            self.Q_in = 140.0
            self.h = 2.0
            # Force update v immediately for initial check
            self.v = self.Q_in / (self.Width * self.h)

        elif scenario_id == "S3.1": # Thermal Bending (Summer Noon)
            self.T_ambient = 35.0
            self.solar_rad = 1.0 # Max sun
            self.wind_speed = 0.5
            # Force temps for immediate effect
            self.T_sun = 45.0
            self.T_shade = 28.0

        elif scenario_id == "S3.3": # Bearing Lock + Cold
            self.T_ambient = -10.0
            self.solar_rad = 0.0
            self.bearing_locked = True

        elif scenario_id == "S5.1": # Earthquake
            self.ground_accel = 0.5 # g

        elif scenario_id == "S4.1": # Joint Failure potential (Cold)
            self.T_ambient = -15.0
            self.solar_rad = 0.0
            self.T_sun = -10.0
            self.T_shade = -10.0

    def reset(self):
        """Reset simulation to initial safe state."""
        # Water
        self.h = 4.0
        self.v = 2.0
        self.T_water = 15.0
        self.Q_in = 80.0
        self.Q_out = 80.0

        # Concrete
        self.T_sun = 20.0
        self.T_shade = 20.0
        self.joint_gap = 20.0

        # Vibration/Structure
        self.vib_amp = 0.0
        self.bearing_stress = 10.0

        # Environment
        self.T_ambient = 25.0
        self.solar_rad = 0.0
        self.wind_speed = 0.0
        self.ground_accel = 0.0

        # Flags
        self.ice_plug = False
        self.bearing_locked = False

        self.time = 0.0

    def get_froude_number(self):
        """Alias for calculate_froude for clearer API."""
        return self.calculate_froude()

    def is_safe_state(self):
        """Check if the system is in a safe operational state."""
        fr = self.calculate_froude()
        delta_T = abs(self.T_sun - self.T_shade)

        return (
            fr < 0.9 and
            delta_T < 10.0 and
            self.joint_gap > 5.0 and
            self.joint_gap < 35.0 and
            self.bearing_stress < 25.0 and
            self.vib_amp < 50.0 and
            not self.bearing_locked
        )
