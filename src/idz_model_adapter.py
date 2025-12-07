"""
IDZ Model Parameter Adaptation Module for TAOS V3.10

This module provides dynamic parameter updating for IDZ (Integrated Dynamics Zone)
simplified models based on high-fidelity simulation states:
- Online parameter identification
- Recursive least squares estimation
- Neural network-based adaptation
- Model uncertainty quantification
- Multi-fidelity model fusion

IDZ Model: Simplified control-oriented model for real-time applications
High-Fidelity Model: Physics-based detailed simulation model

Author: TAOS Development Team
Version: 3.10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
from scipy import linalg
from scipy.optimize import minimize, least_squares


class AdaptationMethod(Enum):
    """Parameter adaptation methods."""
    RECURSIVE_LEAST_SQUARES = "rls"
    EXTENDED_LEAST_SQUARES = "els"
    GRADIENT_DESCENT = "gradient"
    KALMAN_FILTER = "kalman"
    NEURAL_NETWORK = "neural"
    ENSEMBLE_AVERAGING = "ensemble"
    BAYESIAN = "bayesian"


class ModelFidelity(Enum):
    """Model fidelity levels."""
    HIGH_FIDELITY = "high_fidelity"      # Physics-based detailed model
    MEDIUM_FIDELITY = "medium_fidelity"  # Simplified physics model
    LOW_FIDELITY = "low_fidelity"        # Linear/IDZ model
    SURROGATE = "surrogate"              # Data-driven surrogate


@dataclass
class IDZModelParameters:
    """IDZ model parameters for the aqueduct system."""
    # Hydraulic parameters
    hydraulic_resistance: float = 0.001      # Flow resistance coefficient
    storage_coefficient: float = 360.0       # m³/m (area)
    manning_n: float = 0.013                 # Manning's roughness
    inlet_discharge_coeff: float = 0.95      # Gate discharge coefficient
    outlet_discharge_coeff: float = 0.95

    # Thermal parameters
    thermal_capacity: float = 5000.0         # J/(m³·K)
    heat_transfer_coeff: float = 10.0        # W/(m²·K)
    solar_absorption: float = 0.7            # Solar absorption factor
    thermal_diffusivity: float = 1.0e-6      # m²/s

    # Structural parameters
    thermal_expansion_coeff: float = 1.0e-5  # /°C
    joint_stiffness: float = 1.0e6           # N/m
    damping_ratio: float = 0.05              # Structural damping

    # Coupling parameters
    hydro_thermal_coupling: float = 0.01     # Water cooling effect
    thermal_structural_coupling: float = 1.0  # Thermal stress effect
    hydro_structural_coupling: float = 0.001 # Water load effect

    # Time constants
    hydraulic_time_constant: float = 100.0   # seconds
    thermal_time_constant: float = 3600.0    # seconds
    structural_time_constant: float = 10.0   # seconds

    def to_vector(self) -> np.ndarray:
        """Convert parameters to vector."""
        return np.array([
            self.hydraulic_resistance,
            self.storage_coefficient,
            self.manning_n,
            self.inlet_discharge_coeff,
            self.outlet_discharge_coeff,
            self.thermal_capacity,
            self.heat_transfer_coeff,
            self.solar_absorption,
            self.thermal_diffusivity,
            self.thermal_expansion_coeff,
            self.joint_stiffness,
            self.damping_ratio,
            self.hydro_thermal_coupling,
            self.thermal_structural_coupling,
            self.hydro_structural_coupling,
            self.hydraulic_time_constant,
            self.thermal_time_constant,
            self.structural_time_constant
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'IDZModelParameters':
        """Create parameters from vector."""
        return cls(
            hydraulic_resistance=vec[0],
            storage_coefficient=vec[1],
            manning_n=vec[2],
            inlet_discharge_coeff=vec[3],
            outlet_discharge_coeff=vec[4],
            thermal_capacity=vec[5],
            heat_transfer_coeff=vec[6],
            solar_absorption=vec[7],
            thermal_diffusivity=vec[8],
            thermal_expansion_coeff=vec[9],
            joint_stiffness=vec[10],
            damping_ratio=vec[11],
            hydro_thermal_coupling=vec[12],
            thermal_structural_coupling=vec[13],
            hydro_structural_coupling=vec[14],
            hydraulic_time_constant=vec[15],
            thermal_time_constant=vec[16],
            structural_time_constant=vec[17]
        )


@dataclass
class ParameterBounds:
    """Parameter bounds for constrained adaptation."""
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray


@dataclass
class AdaptationConfig:
    """Configuration for parameter adaptation."""
    method: AdaptationMethod = AdaptationMethod.RECURSIVE_LEAST_SQUARES
    learning_rate: float = 0.01
    forgetting_factor: float = 0.99          # For RLS
    regularization: float = 0.001            # L2 regularization
    adaptation_window: int = 100             # Data points to use
    min_samples: int = 10                    # Minimum samples before adaptation
    max_parameter_change: float = 0.1        # Max relative change per step
    uncertainty_threshold: float = 0.5       # Trigger adaptation threshold


class RecursiveLeastSquares:
    """
    Recursive Least Squares estimator for online parameter identification.
    """

    def __init__(self, num_params: int, forgetting_factor: float = 0.99,
                 initial_covariance: float = 1000.0):
        self.n = num_params
        self.lambda_ = forgetting_factor

        # Parameter estimate
        self.theta = np.zeros(num_params)

        # Covariance matrix (inverse of information matrix)
        self.P = np.eye(num_params) * initial_covariance

        # History
        self.estimation_count = 0

    def initialize(self, theta0: np.ndarray, P0: Optional[np.ndarray] = None):
        """Initialize estimator."""
        self.theta = theta0.copy()
        if P0 is not None:
            self.P = P0.copy()

    def update(self, phi: np.ndarray, y: float) -> Tuple[np.ndarray, float]:
        """
        Update parameter estimate with new data.

        Args:
            phi: Regressor vector (n,)
            y: Observed output

        Returns:
            Updated parameters and prediction error
        """
        # Prediction
        y_pred = np.dot(phi, self.theta)
        error = y - y_pred

        # Kalman gain
        Pphi = self.P @ phi
        denominator = self.lambda_ + np.dot(phi, Pphi)
        K = Pphi / denominator

        # Update parameters
        self.theta = self.theta + K * error

        # Update covariance (with forgetting factor)
        self.P = (self.P - np.outer(K, Pphi)) / self.lambda_

        # Ensure positive definiteness
        self.P = (self.P + self.P.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-10:
            self.P += np.eye(self.n) * (1e-10 - min_eig)

        self.estimation_count += 1

        return self.theta.copy(), error

    def get_uncertainty(self) -> np.ndarray:
        """Get parameter uncertainty (standard deviations)."""
        return np.sqrt(np.diag(self.P))


class IDZModel:
    """
    IDZ (Integrated Dynamics Zone) simplified model.

    Provides a computationally efficient model for real-time control
    that captures the essential dynamics of the aqueduct system.
    """

    def __init__(self, params: Optional[IDZModelParameters] = None):
        self.params = params or IDZModelParameters()

        # State: [h, v, T_avg, T_diff, joint_gap, vib]
        self.state_dim = 6
        self.state = np.array([4.0, 2.0, 20.0, 0.0, 20.0, 0.0])

        # State names
        self.state_names = ['h', 'v', 'T_avg', 'T_diff', 'joint_gap', 'vib_amp']

    def reset(self, initial_state: Optional[Dict[str, float]] = None):
        """Reset model to initial state."""
        if initial_state:
            self.state = np.array([
                initial_state.get('h', 4.0),
                initial_state.get('v', 2.0),
                initial_state.get('T_avg', 20.0),
                initial_state.get('T_diff', 0.0),
                initial_state.get('joint_gap', 20.0),
                initial_state.get('vib_amp', 0.0)
            ])
        else:
            self.state = np.array([4.0, 2.0, 20.0, 0.0, 20.0, 0.0])

    def step(self, u: Dict[str, float], env: Dict[str, float],
             dt: float = 0.1) -> Dict[str, float]:
        """
        Advance model by one time step.

        Args:
            u: Control inputs (Q_in, Q_out)
            env: Environmental inputs (T_ambient, solar_rad, wind_speed)
            dt: Time step

        Returns:
            New state dictionary
        """
        h, v, T_avg, T_diff, joint_gap, vib = self.state
        p = self.params

        Q_in = u.get('Q_in', 80.0)
        Q_out = u.get('Q_out', 80.0)
        T_amb = env.get('T_ambient', 25.0)
        solar = env.get('solar_rad', 0.0)
        wind = env.get('wind_speed', 0.0)

        # Hydraulic dynamics
        # dh/dt = (Q_in - Q_out) / A
        dh_dt = (Q_in - Q_out) / p.storage_coefficient

        # Velocity from continuity
        width = 9.0  # Aqueduct width
        if h > 0.1:
            v_new = (Q_in + Q_out) / 2.0 / (width * h)
        else:
            v_new = 0.0

        # Thermal dynamics
        # Average temperature
        cooling = p.hydro_thermal_coupling * h * v  # Water cooling effect
        heating = p.solar_absorption * solar
        exchange = p.heat_transfer_coeff * (T_amb - T_avg) / p.thermal_capacity

        dT_avg_dt = heating + exchange - cooling

        # Temperature differential (sun-shade)
        dT_diff_dt = (solar * 0.5 - T_diff / p.thermal_time_constant)

        # Structural dynamics
        # Joint gap depends on temperature
        delta_T = T_avg - 20.0  # Reference temperature
        expansion = p.thermal_expansion_coeff * 40000.0 * delta_T  # 40m span in mm
        target_gap = 20.0 - expansion

        # Joint gap dynamics
        djoint_dt = (target_gap - joint_gap) / p.structural_time_constant

        # Vibration dynamics
        target_vib = 0.0
        # Wind-induced vibration
        if 10.0 < wind < 15.0:
            target_vib += 20.0

        # Froude number effect
        fr = v / np.sqrt(9.81 * max(h, 0.1))
        if fr > 0.9:
            target_vib += 10.0 * (fr - 0.9)

        dvib_dt = (target_vib - vib) / p.structural_time_constant

        # Euler integration
        h_new = h + dh_dt * dt
        T_avg_new = T_avg + dT_avg_dt * dt
        T_diff_new = T_diff + dT_diff_dt * dt
        joint_new = joint_gap + djoint_dt * dt
        vib_new = vib + dvib_dt * dt

        # Constraints
        h_new = np.clip(h_new, 0.1, 8.0)
        v_new = np.clip(v_new, 0.0, 15.0)
        T_avg_new = np.clip(T_avg_new, -50.0, 80.0)
        T_diff_new = np.clip(T_diff_new, 0.0, 30.0)
        joint_new = np.clip(joint_new, 0.0, 50.0)
        vib_new = np.clip(vib_new, 0.0, 200.0)

        self.state = np.array([h_new, v_new, T_avg_new, T_diff_new, joint_new, vib_new])

        return {
            'h': h_new,
            'v': v_new,
            'T_avg': T_avg_new,
            'T_diff': T_diff_new,
            'T_sun': T_avg_new + T_diff_new / 2,
            'T_shade': T_avg_new - T_diff_new / 2,
            'joint_gap': joint_new,
            'vib_amp': vib_new,
            'fr': v_new / np.sqrt(9.81 * max(h_new, 0.1))
        }

    def get_regressor(self, state: np.ndarray, u: Dict[str, float],
                      env: Dict[str, float]) -> np.ndarray:
        """
        Get regressor vector for parameter identification.

        Returns features that relate parameters to state derivatives.
        """
        h, v, T_avg, T_diff, joint_gap, vib = state
        Q_in = u.get('Q_in', 80.0)
        Q_out = u.get('Q_out', 80.0)
        T_amb = env.get('T_ambient', 25.0)
        solar = env.get('solar_rad', 0.0)
        wind = env.get('wind_speed', 0.0)

        # Build regressor vector
        phi = np.array([
            (Q_in - Q_out),           # Storage coefficient
            h * v,                     # Hydro-thermal coupling
            solar,                     # Solar absorption
            T_amb - T_avg,            # Heat transfer
            T_avg - 20.0,             # Thermal expansion
            wind ** 2,                 # Wind effect
            v ** 2 / max(h, 0.1),     # Froude effect
            joint_gap - 20.0,          # Joint restoring
            vib                        # Vibration damping
        ])

        return phi


class IDZModelAdapter:
    """
    Dynamic parameter adapter for IDZ model.

    Continuously updates IDZ model parameters based on
    high-fidelity simulation data.
    """

    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()

        # IDZ model
        self.idz_model = IDZModel()

        # Parameter estimator
        num_params = len(self.idz_model.params.to_vector())
        self.estimator = RecursiveLeastSquares(
            num_params,
            forgetting_factor=self.config.forgetting_factor
        )

        # Initialize with nominal parameters
        self.estimator.initialize(self.idz_model.params.to_vector())

        # Parameter bounds
        nominal = self.idz_model.params.to_vector()
        self.bounds = ParameterBounds(
            lower=nominal * 0.1,
            upper=nominal * 10.0,
            nominal=nominal
        )

        # Data buffers
        self.hifi_buffer: deque = deque(maxlen=self.config.adaptation_window)
        self.idz_buffer: deque = deque(maxlen=self.config.adaptation_window)
        self.error_buffer: deque = deque(maxlen=1000)

        # Adaptation history
        self.adaptation_history: deque = deque(maxlen=10000)
        self.parameter_history: deque = deque(maxlen=1000)

        # Metrics
        self.total_adaptations = 0
        self.start_time = time.time()

    def update_from_hifi(self, hifi_state: Dict[str, float],
                         control: Dict[str, float],
                         environment: Dict[str, float],
                         dt: float = 0.1) -> Dict[str, Any]:
        """
        Update IDZ model parameters based on high-fidelity state.

        Args:
            hifi_state: State from high-fidelity simulation
            control: Control inputs
            environment: Environmental conditions
            dt: Time step

        Returns:
            Adaptation results
        """
        # Convert high-fidelity state to IDZ state
        hifi_idz_state = np.array([
            hifi_state.get('h', 4.0),
            hifi_state.get('v', 2.0),
            (hifi_state.get('T_sun', 20.0) + hifi_state.get('T_shade', 20.0)) / 2,
            abs(hifi_state.get('T_sun', 20.0) - hifi_state.get('T_shade', 20.0)),
            hifi_state.get('joint_gap', 20.0),
            hifi_state.get('vib_amp', 0.0)
        ])

        # Get IDZ prediction
        idz_pred = self.idz_model.step(control, environment, dt)
        idz_state = self.idz_model.state.copy()

        # Calculate prediction error
        error = hifi_idz_state - idz_state
        error_norm = np.linalg.norm(error)

        # Store in buffers
        self.hifi_buffer.append({
            'state': hifi_idz_state.copy(),
            'control': control.copy(),
            'environment': environment.copy(),
            'timestamp': time.time()
        })

        self.idz_buffer.append({
            'state': idz_state.copy(),
            'prediction': idz_pred.copy()
        })

        self.error_buffer.append({
            'error': error.copy(),
            'error_norm': error_norm,
            'timestamp': time.time()
        })

        # Perform adaptation if enough data
        result = {
            'adapted': False,
            'error_norm': error_norm,
            'prediction': idz_pred
        }

        if len(self.hifi_buffer) >= self.config.min_samples:
            if error_norm > self.config.uncertainty_threshold or \
               self.total_adaptations % 100 == 0:
                result = self._adapt_parameters()
                result['error_norm'] = error_norm
                result['prediction'] = idz_pred

        # Sync IDZ state with high-fidelity
        self.idz_model.state = hifi_idz_state.copy()

        return result

    def _adapt_parameters(self) -> Dict[str, Any]:
        """Perform parameter adaptation."""
        method = self.config.method

        if method == AdaptationMethod.RECURSIVE_LEAST_SQUARES:
            result = self._adapt_rls()
        elif method == AdaptationMethod.GRADIENT_DESCENT:
            result = self._adapt_gradient()
        elif method == AdaptationMethod.BAYESIAN:
            result = self._adapt_bayesian()
        else:
            result = self._adapt_rls()

        self.total_adaptations += 1

        # Store parameter history
        self.parameter_history.append({
            'timestamp': time.time(),
            'parameters': self.idz_model.params.to_vector().copy(),
            'method': method.value
        })

        return result

    def _adapt_rls(self) -> Dict[str, Any]:
        """Adapt using Recursive Least Squares."""
        if len(self.hifi_buffer) < 2:
            return {'adapted': False, 'reason': 'insufficient_data'}

        # Get recent data
        recent_hifi = list(self.hifi_buffer)[-10:]

        total_error = 0.0
        for i in range(1, len(recent_hifi)):
            prev = recent_hifi[i - 1]
            curr = recent_hifi[i]

            # Build regressor
            phi = self.idz_model.get_regressor(
                prev['state'],
                prev['control'],
                prev['environment']
            )

            # Output is the state change
            dy = curr['state'] - prev['state']

            # Update for each state variable
            for j, y in enumerate(dy):
                if len(phi) > 0:
                    # Simplified: use first few parameters
                    theta, error = self.estimator.update(phi[:self.estimator.n], y)
                    total_error += error ** 2

        # Update IDZ model parameters (with bounds)
        new_params = np.clip(
            self.estimator.theta,
            self.bounds.lower,
            self.bounds.upper
        )

        # Limit parameter change rate
        old_params = self.idz_model.params.to_vector()
        max_change = self.config.max_parameter_change * np.abs(old_params)
        param_change = np.clip(new_params - old_params, -max_change, max_change)
        new_params = old_params + param_change

        # Update model
        self.idz_model.params = IDZModelParameters.from_vector(new_params)

        self.adaptation_history.append({
            'timestamp': time.time(),
            'method': 'rls',
            'error': np.sqrt(total_error),
            'param_change_norm': np.linalg.norm(param_change)
        })

        return {
            'adapted': True,
            'method': 'rls',
            'rmse': np.sqrt(total_error / max(1, len(recent_hifi) - 1)),
            'param_change': param_change.tolist()
        }

    def _adapt_gradient(self) -> Dict[str, Any]:
        """Adapt using gradient descent."""
        if len(self.hifi_buffer) < self.config.min_samples:
            return {'adapted': False, 'reason': 'insufficient_data'}

        # Cost function: prediction error
        def cost(params):
            temp_model = IDZModel(IDZModelParameters.from_vector(params))
            total_error = 0.0

            for i, data in enumerate(list(self.hifi_buffer)[-20:]):
                if i > 0:
                    prev = list(self.hifi_buffer)[-20:][i - 1]
                    temp_model.state = prev['state'].copy()
                    pred = temp_model.step(data['control'], data['environment'], 0.1)

                    error = data['state'] - temp_model.state
                    total_error += np.sum(error ** 2)

            # Regularization
            total_error += self.config.regularization * np.sum(
                (params - self.bounds.nominal) ** 2
            )

            return total_error

        # Optimize
        x0 = self.idz_model.params.to_vector()

        result = minimize(
            cost, x0,
            method='L-BFGS-B',
            bounds=list(zip(self.bounds.lower, self.bounds.upper)),
            options={'maxiter': 50}
        )

        if result.success:
            # Limit change rate
            old_params = self.idz_model.params.to_vector()
            max_change = self.config.max_parameter_change * np.abs(old_params)
            param_change = np.clip(result.x - old_params, -max_change, max_change)
            new_params = old_params + param_change

            self.idz_model.params = IDZModelParameters.from_vector(new_params)

            return {
                'adapted': True,
                'method': 'gradient',
                'final_cost': result.fun,
                'iterations': result.nit
            }

        return {'adapted': False, 'reason': 'optimization_failed'}

    def _adapt_bayesian(self) -> Dict[str, Any]:
        """Adapt using Bayesian inference (simplified)."""
        if len(self.hifi_buffer) < self.config.min_samples:
            return {'adapted': False, 'reason': 'insufficient_data'}

        # Use RLS covariance as approximate posterior
        theta = self.estimator.theta
        P = self.estimator.P

        # Sample from posterior
        num_samples = 20
        samples = np.random.multivariate_normal(theta, P, num_samples)

        # Evaluate each sample
        best_params = theta.copy()
        best_error = np.inf

        for sample in samples:
            sample = np.clip(sample, self.bounds.lower, self.bounds.upper)
            temp_model = IDZModel(IDZModelParameters.from_vector(sample))

            total_error = 0.0
            for data in list(self.hifi_buffer)[-10:]:
                temp_model.step(data['control'], data['environment'], 0.1)
                error = data['state'] - temp_model.state
                total_error += np.sum(error ** 2)

            if total_error < best_error:
                best_error = total_error
                best_params = sample.copy()

        # Update model
        old_params = self.idz_model.params.to_vector()
        max_change = self.config.max_parameter_change * np.abs(old_params)
        param_change = np.clip(best_params - old_params, -max_change, max_change)
        new_params = old_params + param_change

        self.idz_model.params = IDZModelParameters.from_vector(new_params)

        return {
            'adapted': True,
            'method': 'bayesian',
            'best_error': best_error
        }

    def predict(self, control: Dict[str, float],
                environment: Dict[str, float],
                horizon: int = 10,
                dt: float = 0.1) -> List[Dict[str, float]]:
        """
        Make predictions using adapted IDZ model.

        Args:
            control: Control inputs
            environment: Environmental conditions
            horizon: Prediction horizon (steps)
            dt: Time step

        Returns:
            List of predicted states
        """
        predictions = []

        # Save current state
        saved_state = self.idz_model.state.copy()

        for _ in range(horizon):
            pred = self.idz_model.step(control, environment, dt)
            predictions.append(pred)

        # Restore state
        self.idz_model.state = saved_state

        return predictions

    def get_model_uncertainty(self) -> Dict[str, float]:
        """Get model parameter uncertainty."""
        uncertainty = self.estimator.get_uncertainty()
        params = self.idz_model.params

        return {
            'hydraulic_resistance': uncertainty[0] if len(uncertainty) > 0 else 0,
            'storage_coefficient': uncertainty[1] if len(uncertainty) > 1 else 0,
            'thermal_capacity': uncertainty[5] if len(uncertainty) > 5 else 0,
            'overall_uncertainty': np.mean(uncertainty)
        }

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation performance metrics."""
        if len(self.error_buffer) < 2:
            return {'status': 'insufficient_data'}

        errors = [e['error_norm'] for e in self.error_buffer]

        return {
            'total_adaptations': self.total_adaptations,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'recent_error': errors[-1] if errors else 0,
            'uptime_seconds': time.time() - self.start_time,
            'parameter_uncertainty': self.get_model_uncertainty()
        }

    def get_current_parameters(self) -> Dict[str, float]:
        """Get current IDZ model parameters."""
        p = self.idz_model.params

        return {
            'hydraulic_resistance': p.hydraulic_resistance,
            'storage_coefficient': p.storage_coefficient,
            'manning_n': p.manning_n,
            'inlet_discharge_coeff': p.inlet_discharge_coeff,
            'outlet_discharge_coeff': p.outlet_discharge_coeff,
            'thermal_capacity': p.thermal_capacity,
            'heat_transfer_coeff': p.heat_transfer_coeff,
            'solar_absorption': p.solar_absorption,
            'thermal_expansion_coeff': p.thermal_expansion_coeff,
            'joint_stiffness': p.joint_stiffness,
            'damping_ratio': p.damping_ratio,
            'hydro_thermal_coupling': p.hydro_thermal_coupling,
            'thermal_structural_coupling': p.thermal_structural_coupling,
            'hydraulic_time_constant': p.hydraulic_time_constant,
            'thermal_time_constant': p.thermal_time_constant,
            'structural_time_constant': p.structural_time_constant
        }

    def reset(self):
        """Reset adapter to initial state."""
        self.idz_model = IDZModel()
        nominal = self.idz_model.params.to_vector()
        self.estimator.initialize(nominal)
        self.hifi_buffer.clear()
        self.idz_buffer.clear()
        self.error_buffer.clear()
        self.total_adaptations = 0


class MultiFidelityModelManager:
    """
    Manages multiple fidelity models and their coordination.

    Handles model switching, uncertainty quantification,
    and multi-fidelity fusion.
    """

    def __init__(self):
        self.idz_adapter = IDZModelAdapter()

        # Model weights for fusion
        self.model_weights = {
            ModelFidelity.HIGH_FIDELITY: 0.6,
            ModelFidelity.LOW_FIDELITY: 0.4
        }

        # Model performance tracking
        self.model_errors: Dict[ModelFidelity, deque] = {
            fidelity: deque(maxlen=100)
            for fidelity in ModelFidelity
        }

        # Current active model
        self.active_model = ModelFidelity.LOW_FIDELITY

    def update(self, hifi_state: Dict[str, float],
               control: Dict[str, float],
               environment: Dict[str, float],
               dt: float = 0.1) -> Dict[str, Any]:
        """
        Update multi-fidelity model system.

        Returns:
            Fused state estimate and metrics
        """
        # Update IDZ adapter
        idz_result = self.idz_adapter.update_from_hifi(
            hifi_state, control, environment, dt
        )

        # Track errors
        if 'error_norm' in idz_result:
            self.model_errors[ModelFidelity.LOW_FIDELITY].append(
                idz_result['error_norm']
            )

        # Adaptive weight update based on performance
        self._update_weights()

        # Fuse predictions
        fused_state = self._fuse_states(hifi_state, idz_result.get('prediction', {}))

        return {
            'fused_state': fused_state,
            'idz_state': idz_result.get('prediction', {}),
            'hifi_state': hifi_state,
            'weights': self.model_weights.copy(),
            'adaptation': idz_result
        }

    def _update_weights(self):
        """Update model weights based on performance."""
        idz_errors = list(self.model_errors[ModelFidelity.LOW_FIDELITY])

        if len(idz_errors) >= 10:
            recent_error = np.mean(idz_errors[-10:])

            # Adjust IDZ weight based on error
            if recent_error < 0.1:
                # Good IDZ performance - increase weight
                self.model_weights[ModelFidelity.LOW_FIDELITY] = min(
                    0.5, self.model_weights[ModelFidelity.LOW_FIDELITY] + 0.01
                )
            elif recent_error > 0.5:
                # Poor IDZ performance - decrease weight
                self.model_weights[ModelFidelity.LOW_FIDELITY] = max(
                    0.2, self.model_weights[ModelFidelity.LOW_FIDELITY] - 0.01
                )

            self.model_weights[ModelFidelity.HIGH_FIDELITY] = \
                1.0 - self.model_weights[ModelFidelity.LOW_FIDELITY]

    def _fuse_states(self, hifi_state: Dict[str, float],
                     idz_state: Dict[str, float]) -> Dict[str, float]:
        """Fuse high-fidelity and IDZ states."""
        w_hifi = self.model_weights[ModelFidelity.HIGH_FIDELITY]
        w_idz = self.model_weights[ModelFidelity.LOW_FIDELITY]

        fused = {}

        # Fuse common states
        common_keys = set(hifi_state.keys()) & set(idz_state.keys())
        for key in common_keys:
            if isinstance(hifi_state[key], (int, float)) and \
               isinstance(idz_state[key], (int, float)):
                fused[key] = w_hifi * hifi_state[key] + w_idz * idz_state[key]
            else:
                fused[key] = hifi_state[key]

        # Include high-fidelity only states
        for key in hifi_state:
            if key not in fused:
                fused[key] = hifi_state[key]

        return fused

    def get_status(self) -> Dict[str, Any]:
        """Get multi-fidelity model status."""
        return {
            'active_model': self.active_model.value,
            'weights': self.model_weights,
            'idz_metrics': self.idz_adapter.get_adaptation_metrics(),
            'idz_parameters': self.idz_adapter.get_current_parameters()
        }
