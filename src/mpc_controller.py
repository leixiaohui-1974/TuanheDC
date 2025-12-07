"""
Adaptive Model Predictive Controller (MPC) for TAOS

This module implements:
- Linear MPC with quadratic cost
- Adaptive gain scheduling based on scenarios
- Constraint handling (flow limits, rate limits)
- Prediction horizon optimization
- Fallback to PID when MPC fails
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


@dataclass
class MPCConfig:
    """MPC controller configuration."""
    prediction_horizon: int = 10     # Np: prediction steps
    control_horizon: int = 5         # Nc: control steps
    dt: float = 0.5                  # Sample time (seconds)

    # State weights (Q matrix diagonal)
    w_h: float = 10.0               # Water level weight
    w_v: float = 1.0                # Velocity weight
    w_fr: float = 5.0               # Froude number weight
    w_T_delta: float = 2.0          # Thermal differential weight

    # Control weights (R matrix diagonal)
    w_Q_in: float = 0.1             # Inlet flow weight
    w_Q_out: float = 0.1            # Outlet flow weight
    w_dQ_in: float = 1.0            # Inlet rate change weight
    w_dQ_out: float = 1.0           # Outlet rate change weight

    # Constraints
    Q_min: float = 0.0
    Q_max: float = 200.0
    dQ_max: float = 20.0            # Max rate of change per step

    # Targets
    h_target: float = 4.0
    v_target: float = 2.0
    fr_target: float = 0.32


class AdaptiveMPC:
    """
    Adaptive Model Predictive Controller with scenario-based gain scheduling.
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        self.config = config or MPCConfig()
        self.last_u = np.array([80.0, 80.0])  # [Q_in, Q_out]

        # System model parameters (linearized around operating point)
        self.A = None  # State transition
        self.B = None  # Control input
        self.C = None  # Output

        # Adaptive parameters - Full scenario coverage
        self.scenario_gains = {
            # Normal operation
            'NORMAL': {'w_h': 10.0, 'w_fr': 5.0, 'w_T': 2.0, 'target_h': 4.0},

            # S1.x Hydraulic scenarios
            'S1.1': {'w_h': 5.0, 'w_fr': 50.0, 'w_T': 1.0, 'target_h': 7.0},    # Prioritize Fr, raise level
            'S1.2': {'w_h': 8.0, 'w_fr': 30.0, 'w_T': 1.0, 'target_h': 5.0},    # Surge attenuation

            # S2.x Wind scenarios
            'S2.1': {'w_h': 15.0, 'w_fr': 5.0, 'w_T': 2.0, 'target_h': 6.0},    # VIV damping, higher water

            # S3.x Thermal scenarios
            'S3.1': {'w_h': 5.0, 'w_fr': 5.0, 'w_T': 50.0, 'target_h': 4.0},    # Prioritize thermal
            'S3.2': {'w_h': 12.0, 'w_fr': 5.0, 'w_T': 30.0, 'target_h': 5.5},   # Rapid cooling buffer
            'S3.3': {'w_h': 20.0, 'w_fr': 5.0, 'w_T': 2.0, 'target_h': 3.0},    # Bearing lock, reduce load

            # S4.x Joint scenarios
            'S4.1': {'w_h': 15.0, 'w_fr': 3.0, 'w_T': 10.0, 'target_h': 5.0},   # Cold joint protection
            'S4.2': {'w_h': 12.0, 'w_fr': 5.0, 'w_T': 15.0, 'target_h': 4.0},   # Hot joint cooling

            # S5.x Seismic scenarios
            'S5.1': {'w_h': 30.0, 'w_fr': 10.0, 'w_T': 1.0, 'target_h': 2.5},   # Emergency priority, low level
            'S5.2': {'w_h': 25.0, 'w_fr': 8.0, 'w_T': 2.0, 'target_h': 3.5},    # Aftershock caution

            # S6.x Fault scenarios
            'S6.1': {'w_h': 8.0, 'w_fr': 4.0, 'w_T': 2.0, 'target_h': 4.0},     # Sensor fault, conservative
            'S6.2': {'w_h': 5.0, 'w_fr': 3.0, 'w_T': 1.0, 'target_h': 4.0},     # Actuator fault, minimal control

            # Combined scenarios
            'MULTI_PHYSICS': {'w_h': 15.0, 'w_fr': 10.0, 'w_T': 10.0, 'target_h': 4.0},
            'COMBINED_THERMAL_SEISMIC': {'w_h': 35.0, 'w_fr': 15.0, 'w_T': 5.0, 'target_h': 2.0},
        }

        # PID fallback controller
        self.pid_kp = 10.0
        self.pid_ki = 0.5
        self.pid_kd = 2.0
        self.pid_integral = 0.0
        self.pid_last_error = 0.0

        # Performance tracking
        self.solve_count = 0
        self.fallback_count = 0

        # Initialize model
        self._build_model()

    def _build_model(self):
        """Build linearized system model."""
        # Simplified model:
        # State: [h, v, T_delta]
        # Input: [Q_in, Q_out]
        # Dynamics: dh/dt = (Q_in - Q_out) / A
        #           dv/dt = (Q - Q_prev) / (2*A*h) (simplified)
        #           dT/dt = -k * v * T (cooling effect)

        dt = self.config.dt
        A_cross = 400.0  # m² (Width * Length)

        # State transition matrix (discrete)
        self.A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, -0.01, 0.98]
        ])

        # Control input matrix
        self.B = np.array([
            [dt / A_cross, -dt / A_cross],
            [0.001, -0.001],
            [0.0, 0.0]
        ])

        # Output matrix (we observe all states)
        self.C = np.eye(3)

    def _update_gains(self, scenarios: List[str]):
        """Update MPC gains based on active scenarios with full coverage."""
        if not scenarios:
            gains = self.scenario_gains['NORMAL']
        else:
            # Priority order: Emergency > Seismic > Thermal/Structural > Hydraulic > Faults
            priority = [
                'COMBINED_THERMAL_SEISMIC',  # Highest priority
                'S5.1',   # Main seismic
                'S5.2',   # Aftershock
                'S3.3',   # Bearing lock
                'S3.1',   # Thermal bending
                'S3.2',   # Rapid cooling
                'S1.1',   # Hydraulic jump
                'S1.2',   # Surge wave
                'S2.1',   # VIV
                'S4.1',   # Joint expansion
                'S4.2',   # Joint compression
                'S6.1',   # Sensor fault
                'S6.2',   # Actuator fault
                'MULTI_PHYSICS',  # General multi-physics
            ]
            gains = None
            for s in priority:
                if s in scenarios:
                    gains = self.scenario_gains.get(s)
                    if gains:
                        break
            if gains is None:
                gains = self.scenario_gains['NORMAL']

        self.config.w_h = gains['w_h']
        self.config.w_fr = gains['w_fr']
        self.config.w_T_delta = gains['w_T']
        self.config.h_target = gains.get('target_h', 4.0)

    def _build_qp_matrices(self, x0: np.ndarray, x_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build QP matrices for the MPC problem.

        minimize: (1/2) u' H u + f' u
        subject to: A_ineq u <= b_ineq

        Returns:
            H, f, A_ineq, b_ineq
        """
        Np = self.config.prediction_horizon
        Nc = self.config.control_horizon
        nx = 3  # State dimension
        nu = 2  # Control dimension

        # Build prediction matrices
        # x(k+i) = A^i x(k) + sum(A^(i-j-1) B u(k+j))
        Psi = np.zeros((Np * nx, nx))  # Free response
        Theta = np.zeros((Np * nx, Nc * nu))  # Forced response

        A_power = np.eye(nx)
        for i in range(Np):
            Psi[i*nx:(i+1)*nx, :] = A_power @ self.A
            A_power = A_power @ self.A

            for j in range(min(i + 1, Nc)):
                power = i - j
                A_pow = np.eye(nx)
                for _ in range(power):
                    A_pow = A_pow @ self.A
                Theta[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_pow @ self.B

        # Weight matrices
        Q = np.diag([self.config.w_h, self.config.w_fr, self.config.w_T_delta])
        Q_bar = np.kron(np.eye(Np), Q)

        R = np.diag([self.config.w_Q_in, self.config.w_Q_out])
        R_bar = np.kron(np.eye(Nc), R)

        # Rate weight
        dR = np.diag([self.config.w_dQ_in, self.config.w_dQ_out])
        dR_bar = np.kron(np.eye(Nc), dR)

        # Difference matrix for rate penalty
        D = np.zeros((Nc * nu, Nc * nu))
        for i in range(Nc):
            D[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = np.eye(nu)
            if i > 0:
                D[i*nu:(i+1)*nu, (i-1)*nu:i*nu] = -np.eye(nu)

        # Reference trajectory
        X_ref = np.tile(x_ref, Np)

        # QP matrices
        H = Theta.T @ Q_bar @ Theta + R_bar + D.T @ dR_bar @ D
        H = (H + H.T) / 2  # Ensure symmetry

        # Free response
        x_free = Psi @ x0

        f = Theta.T @ Q_bar @ (x_free - X_ref)

        # Add term for rate from previous control
        u_prev = np.zeros(Nc * nu)
        u_prev[:nu] = self.last_u
        f += D.T @ dR_bar @ (-D @ np.zeros(Nc * nu) + np.tile(-self.last_u, Nc))

        # Constraints: Q_min <= u <= Q_max
        # -dQ_max <= du <= dQ_max
        n_u = Nc * nu

        # Box constraints
        A_ineq = np.vstack([np.eye(n_u), -np.eye(n_u), D, -D])

        b_ineq = np.hstack([
            np.ones(n_u) * self.config.Q_max,
            -np.ones(n_u) * self.config.Q_min,
            np.ones(n_u) * self.config.dQ_max + np.tile(self.last_u, Nc),
            np.ones(n_u) * self.config.dQ_max - np.tile(self.last_u, Nc)
        ])

        return H, f, A_ineq, b_ineq

    def _solve_qp(self, H: np.ndarray, f: np.ndarray,
                  A_ineq: np.ndarray, b_ineq: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve the QP problem using a simple active-set method.
        For production, use cvxpy or qpsolvers.
        """
        n = len(f)
        max_iter = 100

        # Start with unconstrained solution
        try:
            u = np.linalg.solve(H, -f)
        except np.linalg.LinAlgError:
            return None

        # Project onto constraints iteratively
        for _ in range(max_iter):
            violations = A_ineq @ u - b_ineq
            if np.all(violations <= 1e-6):
                break

            # Find most violated constraint
            worst_idx = np.argmax(violations)
            if violations[worst_idx] <= 1e-6:
                break

            # Project onto constraint
            a = A_ineq[worst_idx, :]
            b = b_ineq[worst_idx]

            # u_new = u - (a'u - b) / (a'a) * a
            u = u - (np.dot(a, u) - b) / (np.dot(a, a) + 1e-10) * a

        # Final constraint enforcement
        u = np.clip(u, self.config.Q_min, self.config.Q_max)

        return u

    def _pid_fallback(self, state: Dict[str, Any]) -> np.ndarray:
        """PID controller as fallback when MPC fails."""
        self.fallback_count += 1

        h = state.get('h', 4.0)
        target_h = self.config.h_target

        error = target_h - h

        # PID calculation
        self.pid_integral += error * self.config.dt
        self.pid_integral = np.clip(self.pid_integral, -10.0, 10.0)

        derivative = (error - self.pid_last_error) / self.config.dt
        self.pid_last_error = error

        output = self.pid_kp * error + self.pid_ki * self.pid_integral + self.pid_kd * derivative

        Q_in = state.get('Q_in', 80.0)
        Q_out = Q_in - output

        Q_out = np.clip(Q_out, self.config.Q_min, self.config.Q_max)

        return np.array([Q_in, Q_out])

    def compute(self, state: Dict[str, Any], scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Compute optimal control action.

        Args:
            state: Current system state
            scenarios: Active scenarios for gain adaptation

        Returns:
            Control actions and diagnostics
        """
        self.solve_count += 1

        # Update gains based on scenarios
        if scenarios:
            self._update_gains(scenarios)

        # Extract state vector [h, v, T_delta]
        h = state.get('h', 4.0)
        v = state.get('v', 2.0)
        T_sun = state.get('T_sun', 20.0)
        T_shade = state.get('T_shade', 20.0)
        T_delta = T_sun - T_shade

        x0 = np.array([h, v, T_delta])

        # Reference state
        x_ref = np.array([self.config.h_target, self.config.v_target, 0.0])

        # Build and solve QP
        try:
            H, f, A_ineq, b_ineq = self._build_qp_matrices(x0, x_ref)
            u_optimal = self._solve_qp(H, f, A_ineq, b_ineq)

            if u_optimal is None:
                u = self._pid_fallback(state)
                method = 'PID_FALLBACK'
            else:
                # Extract first control action
                u = u_optimal[:2]
                method = 'MPC'

        except Exception:
            u = self._pid_fallback(state)
            method = 'PID_FALLBACK'

        # Apply rate limiting
        du = u - self.last_u
        du = np.clip(du, -self.config.dQ_max, self.config.dQ_max)
        u = self.last_u + du

        # Update last control
        self.last_u = u.copy()

        return {
            'Q_in': float(u[0]),
            'Q_out': float(u[1]),
            'method': method,
            'solve_count': self.solve_count,
            'fallback_count': self.fallback_count,
            'gains': {
                'w_h': self.config.w_h,
                'w_fr': self.config.w_fr,
                'w_T': self.config.w_T_delta
            }
        }

    def reset(self):
        """Reset controller state."""
        self.last_u = np.array([80.0, 80.0])
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.solve_count = 0
        self.fallback_count = 0


class HybridController:
    """
    Hybrid controller combining MPC with scenario-specific overrides.
    Provides seamless switching between control modes.
    """

    def __init__(self):
        self.mpc = AdaptiveMPC()
        self.mode = 'AUTO'
        self.emergency_active = False

        # Emergency thresholds
        self.emergency_vibration = 50.0
        self.emergency_fr = 1.5

        # Performance metrics
        self.control_history = []
        self.max_history = 1000

    def decide(self, state: Dict[str, Any], scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Main control decision function.

        Args:
            state: Current measured state
            scenarios: Detected scenarios

        Returns:
            Control actions with status
        """
        scenarios = scenarios or []

        # Emergency check
        if self._check_emergency(state, scenarios):
            return self._emergency_response(state)

        # Compute MPC control
        mpc_result = self.mpc.compute(state, scenarios)

        # Apply scenario-specific modifications
        if 'S1.1' in scenarios:
            # Hydraulic jump: be more aggressive with level control
            mpc_result = self._modify_for_hydraulic(mpc_result, state)

        elif 'S3.1' in scenarios:
            # Thermal: increase flow for cooling
            mpc_result = self._modify_for_thermal(mpc_result, state)

        elif 'S4.1' in scenarios:
            # Joint gap: careful level control
            mpc_result = self._modify_for_joint(mpc_result, state)

        # Build response
        actions = {
            'Q_in': mpc_result['Q_in'],
            'Q_out': mpc_result['Q_out'],
            'status': f"MPC_AUTO ({mpc_result['method']})",
            'mode': self.mode,
            'active_scenarios': scenarios,
            'mpc_diagnostics': mpc_result
        }

        # Record history
        self._record_history(actions)

        return actions

    def _check_emergency(self, state: Dict[str, Any], scenarios: List[str]) -> bool:
        """Check if emergency response is needed."""
        # S5.1 + S3.3 combination
        if 'S5.1' in scenarios and 'S3.3' in scenarios:
            return True

        # Extreme vibration
        if state.get('vib_amp', 0) > self.emergency_vibration * 1.5:
            return True

        # Extreme Froude number
        if state.get('fr', 0) > self.emergency_fr:
            return True

        return False

    def _emergency_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emergency response."""
        self.emergency_active = True

        return {
            'Q_in': 0.0,
            'Q_out': 200.0,
            'emergency_dump': True,
            'status': 'EMERGENCY: DUMP ACTIVE',
            'mode': 'EMERGENCY',
            'active_scenarios': ['EMERGENCY'],
            'mpc_diagnostics': None
        }

    def _modify_for_hydraulic(self, result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Modify control for hydraulic jump scenario."""
        # Reduce outflow more aggressively to raise level
        h = state.get('h', 4.0)
        if h < 6.0:
            result['Q_out'] = max(0, result['Q_out'] - 20.0)
        return result

    def _modify_for_thermal(self, result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Modify control for thermal bending scenario."""
        # Increase flow for cooling
        Q_in = result['Q_in']
        if Q_in < 120.0:
            result['Q_in'] = min(150.0, Q_in + 10.0)
            result['Q_out'] = result['Q_in']  # Pass-through
        return result

    def _modify_for_joint(self, result: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Modify control for joint gap scenario."""
        # Slightly higher water level for thermal mass
        h = state.get('h', 4.0)
        if h < 5.0:
            result['Q_out'] = max(0, result['Q_out'] - 5.0)
        return result

    def _record_history(self, actions: Dict[str, Any]):
        """Record control action for analysis."""
        self.control_history.append({
            'Q_in': actions['Q_in'],
            'Q_out': actions['Q_out'],
            'status': actions['status']
        })

        if len(self.control_history) > self.max_history:
            self.control_history.pop(0)

    def reset(self):
        """Reset controller."""
        self.mpc.reset()
        self.mode = 'AUTO'
        self.emergency_active = False
        self.control_history = []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get controller performance statistics."""
        if not self.control_history:
            return {}

        Q_in_values = [h['Q_in'] for h in self.control_history]
        Q_out_values = [h['Q_out'] for h in self.control_history]

        return {
            'total_decisions': len(self.control_history),
            'mpc_solves': self.mpc.solve_count,
            'pid_fallbacks': self.mpc.fallback_count,
            'fallback_rate': self.mpc.fallback_count / max(1, self.mpc.solve_count),
            'Q_in_stats': {
                'mean': np.mean(Q_in_values),
                'std': np.std(Q_in_values),
                'min': np.min(Q_in_values),
                'max': np.max(Q_in_values)
            },
            'Q_out_stats': {
                'mean': np.mean(Q_out_values),
                'std': np.std(Q_out_values),
                'min': np.min(Q_out_values),
                'max': np.max(Q_out_values)
            }
        }
