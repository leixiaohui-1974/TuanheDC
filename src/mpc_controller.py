"""
TAOS V3.10 Model Predictive Control (MPC) Module
模型预测控制模块

Features:
- Linear and nonlinear MPC implementation
- Multi-variable control with constraints
- Receding horizon optimization
- Disturbance rejection
- Reference tracking
- Economic MPC for cost optimization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import logging
import threading
import time
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are

logger = logging.getLogger(__name__)


class MPCType(Enum):
    """MPC controller types"""
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    ECONOMIC = "economic"
    ROBUST = "robust"


class SolverType(Enum):
    """Optimization solver types"""
    QP = "qp"
    SLSQP = "slsqp"
    IPOPT = "ipopt"
    SQPMETHOD = "sqp"


@dataclass
class MPCConfig:
    """MPC controller configuration"""
    prediction_horizon: int = 20
    control_horizon: int = 10
    sample_time: float = 1.0
    n_states: int = 4
    n_inputs: int = 2
    n_outputs: int = 2
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    du_min: Optional[np.ndarray] = None
    du_max: Optional[np.ndarray] = None
    y_min: Optional[np.ndarray] = None
    y_max: Optional[np.ndarray] = None
    solver: SolverType = SolverType.SLSQP
    max_iterations: int = 100
    tolerance: float = 1e-6
    use_terminal_constraint: bool = False
    use_terminal_cost: bool = True

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(self.n_outputs)
        if self.R is None:
            self.R = 0.1 * np.eye(self.n_inputs)
        if self.S is None:
            self.S = 0.01 * np.eye(self.n_inputs)


@dataclass
class MPCResult:
    """MPC optimization result"""
    optimal_control: np.ndarray
    predicted_states: np.ndarray
    predicted_outputs: np.ndarray
    cost: float
    success: bool
    iterations: int
    solve_time: float
    active_constraints: List[str] = field(default_factory=list)


class SystemModel(ABC):
    """Abstract base class for system models"""

    @abstractmethod
    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def output(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_linearization(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class LinearStateSpaceModel(SystemModel):
    """Discrete-time linear state-space model"""

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, D: Optional[np.ndarray] = None):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D) if D is not None else np.zeros((C.shape[0], B.shape[1]))
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u

    def output(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        if u is None:
            u = np.zeros(self.n_inputs)
        return self.C @ x + self.D @ u

    def get_linearization(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.A, self.B


class NonlinearModel(SystemModel):
    """Nonlinear system model with user-defined dynamics"""

    def __init__(self, f: Callable, h: Callable,
                 n_states: int, n_inputs: int, n_outputs: int,
                 sample_time: float = 1.0):
        self.f = f
        self.h = h
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.sample_time = sample_time

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x, u)

    def output(self, x: np.ndarray) -> np.ndarray:
        return self.h(x)

    def get_linearization(self, x: np.ndarray, u: np.ndarray,
                          eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((self.n_states, self.n_states))
        B = np.zeros((self.n_states, self.n_inputs))
        f0 = self.f(x, u)
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            A[:, i] = (self.f(x_plus, u) - f0) / eps
        for i in range(self.n_inputs):
            u_plus = u.copy()
            u_plus[i] += eps
            B[:, i] = (self.f(x, u_plus) - f0) / eps
        return A, B


class AqueductModel(SystemModel):
    """Water aqueduct system model for TAOS"""

    def __init__(self, channel_params: Dict[str, float] = None):
        self.params = channel_params or {
            'length': 1000.0,
            'width': 5.0,
            'slope': 0.0001,
            'manning_n': 0.015,
            'gravity': 9.81,
            'gate_coeff': 0.6,
        }
        self.n_states = 4
        self.n_inputs = 2
        self.n_outputs = 2
        self.sample_time = 60.0

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        h1, h2, q1, q2 = x
        g1, g2 = np.clip(u, 0, 1)
        p = self.params
        dt = self.sample_time
        g_coeff = p['gate_coeff']
        gravity = p['gravity']
        q_gate1 = g_coeff * g1 * p['width'] * np.sqrt(2 * gravity * max(h1, 0.01)) if h1 > 0 else 0
        q_gate2 = g_coeff * g2 * p['width'] * np.sqrt(2 * gravity * max(h2, 0.01)) if h2 > 0 else 0
        area = p['width'] * p['length'] / 2
        dh1_dt = (q1 - q_gate1) / area
        dh2_dt = (q_gate1 - q_gate2) / area
        friction = p['manning_n'] ** 2 * gravity * abs(q1) * q1 / (max(h1, 0.01) ** (10/3))
        dq1_dt = gravity * p['width'] * h1 * p['slope'] - friction
        friction2 = p['manning_n'] ** 2 * gravity * abs(q2) * q2 / (max(h2, 0.01) ** (10/3))
        dq2_dt = gravity * p['width'] * h2 * p['slope'] - friction2
        return np.array([
            max(0, h1 + dh1_dt * dt),
            max(0, h2 + dh2_dt * dt),
            max(0, q1 + dq1_dt * dt),
            max(0, q2 + dq2_dt * dt)
        ])

    def output(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1]])

    def get_linearization(self, x: np.ndarray, u: np.ndarray,
                          eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((self.n_states, self.n_states))
        B = np.zeros((self.n_states, self.n_inputs))
        f0 = self.predict(x, u)
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            A[:, i] = (self.predict(x_plus, u) - f0) / eps
        for i in range(self.n_inputs):
            u_plus = u.copy()
            u_plus[i] += eps
            B[:, i] = (self.predict(x, u_plus) - f0) / eps
        return A, B


class MPCController:
    """Model Predictive Controller"""

    def __init__(self, model: SystemModel, config: MPCConfig):
        self.model = model
        self.config = config
        self.N_p = config.prediction_horizon
        self.N_c = config.control_horizon
        self.n_x = model.n_states
        self.n_u = model.n_inputs
        self.n_y = model.n_outputs
        self.u_prev = np.zeros(self.n_u)
        self.reference = np.zeros((self.N_p, self.n_y))
        self.P_terminal = None
        self._solve_count = 0
        self._total_solve_time = 0.0

    def set_reference(self, reference: np.ndarray):
        if reference.ndim == 1:
            self.reference = np.tile(reference, (self.N_p, 1))
        else:
            self.reference = reference[:self.N_p]

    def compute_terminal_cost(self, A: np.ndarray, B: np.ndarray):
        try:
            Q = self.config.Q
            R = self.config.R
            if Q.shape[0] != A.shape[0]:
                C = getattr(self.model, 'C', np.eye(A.shape[0])[:self.n_y])
                Q_state = C.T @ Q @ C
            else:
                Q_state = Q
            self.P_terminal = solve_discrete_are(A, B, Q_state, R)
        except Exception as e:
            logger.warning(f"Failed to compute terminal cost: {e}")
            self.P_terminal = np.eye(self.n_x) * 10

    def _build_prediction_matrices(self, x0: np.ndarray, u0: np.ndarray) -> Tuple:
        A, B = self.model.get_linearization(x0, u0)
        C = getattr(self.model, 'C', np.eye(self.n_x)[:self.n_y])
        Psi = np.zeros((self.N_p * self.n_y, self.n_x))
        Theta = np.zeros((self.N_p * self.n_y, self.N_c * self.n_u))
        A_power = np.eye(self.n_x)
        for i in range(self.N_p):
            A_power = A_power @ A
            Psi[i*self.n_y:(i+1)*self.n_y, :] = C @ A_power
            for j in range(min(i+1, self.N_c)):
                A_power_j = np.linalg.matrix_power(A, i-j)
                Theta[i*self.n_y:(i+1)*self.n_y, j*self.n_u:(j+1)*self.n_u] = C @ A_power_j @ B
        return Psi, Theta, A, B

    def _objective_function(self, U_flat: np.ndarray, x0: np.ndarray,
                            Psi: np.ndarray, Theta: np.ndarray) -> float:
        Q = self.config.Q
        R = self.config.R
        S = self.config.S
        U = U_flat.reshape(self.N_c, self.n_u)
        Y_ref = self.reference.flatten()
        Y_pred = Psi @ x0 + Theta @ U_flat
        e = Y_pred - Y_ref
        cost_tracking = e.T @ np.kron(np.eye(self.N_p), Q) @ e
        cost_control = U_flat.T @ np.kron(np.eye(self.N_c), R) @ U_flat
        dU = np.zeros_like(U)
        dU[0] = U[0] - self.u_prev
        dU[1:] = np.diff(U, axis=0)
        dU_flat = dU.flatten()
        cost_rate = dU_flat.T @ np.kron(np.eye(self.N_c), S) @ dU_flat
        cost_terminal = 0.0
        if self.config.use_terminal_cost and self.P_terminal is not None:
            x_terminal = x0.copy()
            A, B = self.model.get_linearization(x0, U[0])
            for i in range(self.N_p):
                u_i = U[min(i, self.N_c-1)]
                x_terminal = A @ x_terminal + B @ u_i
            cost_terminal = x_terminal.T @ self.P_terminal @ x_terminal
        return float(cost_tracking + cost_control + cost_rate + cost_terminal)

    def _get_constraints(self, x0: np.ndarray, Psi: np.ndarray, Theta: np.ndarray) -> List[Dict]:
        constraints = []
        cfg = self.config
        if cfg.u_min is not None or cfg.u_max is not None:
            def input_bounds(U_flat):
                U = U_flat.reshape(self.N_c, self.n_u)
                violations = []
                for i in range(self.N_c):
                    if cfg.u_min is not None:
                        violations.extend(U[i] - cfg.u_min)
                    if cfg.u_max is not None:
                        violations.extend(cfg.u_max - U[i])
                return np.array(violations)
            constraints.append({'type': 'ineq', 'fun': input_bounds})
        if cfg.du_min is not None or cfg.du_max is not None:
            def rate_bounds(U_flat):
                U = U_flat.reshape(self.N_c, self.n_u)
                violations = []
                dU0 = U[0] - self.u_prev
                if cfg.du_min is not None:
                    violations.extend(dU0 - cfg.du_min)
                if cfg.du_max is not None:
                    violations.extend(cfg.du_max - dU0)
                for i in range(1, self.N_c):
                    dU = U[i] - U[i-1]
                    if cfg.du_min is not None:
                        violations.extend(dU - cfg.du_min)
                    if cfg.du_max is not None:
                        violations.extend(cfg.du_max - dU)
                return np.array(violations)
            constraints.append({'type': 'ineq', 'fun': rate_bounds})
        if cfg.y_min is not None or cfg.y_max is not None:
            def output_bounds(U_flat):
                Y_pred = Psi @ x0 + Theta @ U_flat
                Y = Y_pred.reshape(self.N_p, self.n_y)
                violations = []
                for i in range(self.N_p):
                    if cfg.y_min is not None:
                        violations.extend(Y[i] - cfg.y_min)
                    if cfg.y_max is not None:
                        violations.extend(cfg.y_max - Y[i])
                return np.array(violations)
            constraints.append({'type': 'ineq', 'fun': output_bounds})
        return constraints

    def solve(self, x0: np.ndarray, disturbance: Optional[np.ndarray] = None) -> MPCResult:
        start_time = time.time()
        if disturbance is not None:
            x0 = x0 + disturbance
        Psi, Theta, A, B = self._build_prediction_matrices(x0, self.u_prev)
        if self.config.use_terminal_cost and self.P_terminal is None:
            self.compute_terminal_cost(A, B)
        U0 = np.tile(self.u_prev, self.N_c)
        bounds = None
        if self.config.u_min is not None or self.config.u_max is not None:
            lb = np.tile(self.config.u_min if self.config.u_min is not None
                         else -np.inf * np.ones(self.n_u), self.N_c)
            ub = np.tile(self.config.u_max if self.config.u_max is not None
                         else np.inf * np.ones(self.n_u), self.N_c)
            bounds = list(zip(lb, ub))
        constraints = self._get_constraints(x0, Psi, Theta)
        result = minimize(
            fun=lambda U: self._objective_function(U, x0, Psi, Theta),
            x0=U0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        solve_time = time.time() - start_time
        self._solve_count += 1
        self._total_solve_time += solve_time
        U_optimal = result.x.reshape(self.N_c, self.n_u)
        Y_pred = (Psi @ x0 + Theta @ result.x).reshape(self.N_p, self.n_y)
        X_pred = np.zeros((self.N_p + 1, self.n_x))
        X_pred[0] = x0
        for i in range(self.N_p):
            u_i = U_optimal[min(i, self.N_c-1)]
            X_pred[i+1] = self.model.predict(X_pred[i], u_i)
        return MPCResult(
            optimal_control=U_optimal,
            predicted_states=X_pred,
            predicted_outputs=Y_pred,
            cost=result.fun,
            success=result.success,
            iterations=result.nit,
            solve_time=solve_time
        )

    def step(self, x0: np.ndarray, disturbance: Optional[np.ndarray] = None) -> np.ndarray:
        result = self.solve(x0, disturbance)
        if result.success:
            u_optimal = result.optimal_control[0]
            self.u_prev = u_optimal
            return u_optimal
        else:
            logger.warning("MPC solve failed, using previous control")
            return self.u_prev

    def get_statistics(self) -> Dict[str, Any]:
        avg_time = self._total_solve_time / max(1, self._solve_count)
        return {
            'solve_count': self._solve_count,
            'total_solve_time': self._total_solve_time,
            'average_solve_time': avg_time,
            'prediction_horizon': self.N_p,
            'control_horizon': self.N_c
        }

    def reset(self):
        self.u_prev = np.zeros(self.n_u)
        self._solve_count = 0
        self._total_solve_time = 0.0


class EconomicMPC(MPCController):
    """Economic MPC for cost optimization"""

    def __init__(self, model: SystemModel, config: MPCConfig,
                 economic_cost: Callable[[np.ndarray, np.ndarray], float]):
        super().__init__(model, config)
        self.economic_cost = economic_cost
        self.stage_cost_weight = 1.0

    def _objective_function(self, U_flat: np.ndarray, x0: np.ndarray,
                            Psi: np.ndarray, Theta: np.ndarray) -> float:
        tracking_cost = super()._objective_function(U_flat, x0, Psi, Theta)
        U = U_flat.reshape(self.N_c, self.n_u)
        X = np.zeros((self.N_p + 1, self.n_x))
        X[0] = x0
        economic_total = 0.0
        for i in range(self.N_p):
            u_i = U[min(i, self.N_c-1)]
            X[i+1] = self.model.predict(X[i], u_i)
            economic_total += self.economic_cost(X[i+1], u_i)
        return tracking_cost + self.stage_cost_weight * economic_total


class DisturbanceObserver:
    """Disturbance observer for MPC"""

    def __init__(self, model: SystemModel, observer_gain: float = 0.5):
        self.model = model
        self.gain = observer_gain
        self.disturbance_estimate = np.zeros(model.n_states)
        self.x_predicted = None

    def update(self, x_measured: np.ndarray, u_applied: np.ndarray):
        if self.x_predicted is not None:
            error = x_measured - self.x_predicted
            self.disturbance_estimate = (1 - self.gain) * self.disturbance_estimate + self.gain * error
        self.x_predicted = self.model.predict(x_measured, u_applied)

    def get_estimate(self) -> np.ndarray:
        return self.disturbance_estimate

    def reset(self):
        self.disturbance_estimate = np.zeros(self.model.n_states)
        self.x_predicted = None


class MPCManager:
    """MPC Manager for multiple controllers"""

    def __init__(self):
        self.controllers: Dict[str, MPCController] = {}
        self.observers: Dict[str, DisturbanceObserver] = {}
        self._lock = threading.Lock()

    def add_controller(self, name: str, controller: MPCController,
                       observer: Optional[DisturbanceObserver] = None):
        with self._lock:
            self.controllers[name] = controller
            if observer:
                self.observers[name] = observer

    def remove_controller(self, name: str):
        with self._lock:
            if name in self.controllers:
                del self.controllers[name]
            if name in self.observers:
                del self.observers[name]

    def step(self, name: str, state: np.ndarray,
             measurement: Optional[np.ndarray] = None,
             u_prev: Optional[np.ndarray] = None) -> np.ndarray:
        with self._lock:
            if name not in self.controllers:
                raise ValueError(f"Controller '{name}' not found")
            controller = self.controllers[name]
            observer = self.observers.get(name)
            disturbance = None
            if observer and measurement is not None and u_prev is not None:
                observer.update(measurement, u_prev)
                disturbance = observer.get_estimate()
            return controller.step(state, disturbance)

    def set_reference(self, name: str, reference: np.ndarray):
        with self._lock:
            if name in self.controllers:
                self.controllers[name].set_reference(reference)

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: ctrl.get_statistics() for name, ctrl in self.controllers.items()}


def create_aqueduct_mpc(sample_time: float = 60.0) -> Tuple[MPCController, AqueductModel]:
    """Create MPC controller for aqueduct system"""
    model = AqueductModel()
    model.sample_time = sample_time
    config = MPCConfig(
        prediction_horizon=20,
        control_horizon=10,
        sample_time=sample_time,
        n_states=4,
        n_inputs=2,
        n_outputs=2,
        Q=np.diag([10.0, 10.0]),
        R=np.diag([1.0, 1.0]),
        S=np.diag([0.1, 0.1]),
        u_min=np.array([0.0, 0.0]),
        u_max=np.array([1.0, 1.0]),
        du_min=np.array([-0.1, -0.1]),
        du_max=np.array([0.1, 0.1]),
        y_min=np.array([0.5, 0.5]),
        y_max=np.array([5.0, 5.0])
    )
    controller = MPCController(model, config)
    return controller, model


_mpc_manager: Optional[MPCManager] = None


def get_mpc_manager() -> MPCManager:
    global _mpc_manager
    if _mpc_manager is None:
        _mpc_manager = MPCManager()
    return _mpc_manager


# ============================================================
# Backward Compatibility Classes for server.py
# ============================================================

@dataclass
class MPCResultCompat:
    """MPC result compatible with old interface"""
    Q_cmd: float = 0.0
    theta_cmd: float = 0.0
    u: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    method: str = "MPC"

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for backward compatibility"""
        return getattr(self, key, default)


class AdaptiveMPC:
    """
    Adaptive MPC Controller - Backward compatible wrapper
    自适应MPC控制器 - 向后兼容包装器
    """

    def __init__(self):
        self.controller, self.model = create_aqueduct_mpc()
        self.scenario_gains: Dict[str, float] = {}
        self.last_u = np.zeros(2)
        self.solve_count = 0
        self.fallback_count = 0
        self.config = self._create_config()

    def _create_config(self):
        """Create config object with expected attributes"""
        class Config:
            prediction_horizon = 20
            control_horizon = 10
            dt = 60.0
            Q_min = 0.0
            Q_max = 100.0
            dQ_max = 10.0
            w_h = 10.0
            w_fr = 1.0
            w_T_delta = 0.5
            h_target = 2.5
        return Config()

    def _update_gains(self, scenarios: List[str]):
        """Update scenario-specific gains"""
        for scenario in scenarios:
            if scenario not in self.scenario_gains:
                self.scenario_gains[scenario] = 1.0

    def compute(self, state: Dict[str, float], scenarios: List[str] = None) -> MPCResultCompat:
        """Compute MPC control action"""
        self.solve_count += 1

        # Convert state dict to numpy array
        x = np.array([
            state.get('h', 2.5),
            state.get('Q', 50.0),
            state.get('T_structure', 20.0),
            state.get('Fr', 0.3)
        ])

        # Set reference
        y_ref = np.array([self.config.h_target, 0.35])
        self.controller.set_reference(y_ref)

        try:
            result = self.controller.compute(x)
            if result.success:
                self.last_u = result.u_optimal[0] if len(result.u_optimal) > 0 else np.zeros(2)
                return MPCResultCompat(
                    Q_cmd=float(self.last_u[0] * 100),
                    theta_cmd=float(self.last_u[1]),
                    u=self.last_u
                )
        except Exception:
            self.fallback_count += 1

        return MPCResultCompat(Q_cmd=50.0, theta_cmd=0.5, u=np.array([0.5, 0.5]))


class HybridController:
    """
    Hybrid Controller - Combines MPC with PID fallback
    混合控制器 - 结合MPC与PID后备
    """

    def __init__(self):
        self.mpc = AdaptiveMPC()
        self.mode = 'mpc'
        self.pid_gains = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.05}
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, state: Dict[str, float], setpoint: float = 2.5) -> Dict[str, float]:
        """Compute hybrid control action"""
        if self.mode == 'mpc':
            result = self.mpc.compute(state)
            return {
                'Q_cmd': result.Q_cmd,
                'theta_cmd': result.theta_cmd,
                'mode': 'mpc'
            }
        else:
            # PID fallback
            error = setpoint - state.get('h', 2.5)
            self.integral += error
            derivative = error - self.last_error
            self.last_error = error

            output = (self.pid_gains['Kp'] * error +
                     self.pid_gains['Ki'] * self.integral +
                     self.pid_gains['Kd'] * derivative)

            return {
                'Q_cmd': max(0, min(100, 50 + output * 10)),
                'theta_cmd': 0.5,
                'mode': 'pid'
            }

    def set_mode(self, mode: str):
        """Set controller mode"""
        if mode in ['mpc', 'pid', 'hybrid']:
            self.mode = mode
