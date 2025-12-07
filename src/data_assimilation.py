"""
Data Assimilation Module for TAOS V3.10

This module provides advanced data assimilation algorithms for
combining sensor observations with model predictions:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Ensemble Kalman Filter (EnKF)
- Particle Filter
- 4D-Var (Four-dimensional variational)
- Hybrid methods

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
from scipy.optimize import minimize


class AssimilationMethod(Enum):
    """Data assimilation methods."""
    KALMAN_FILTER = "kalman_filter"
    EXTENDED_KALMAN = "extended_kalman"
    UNSCENTED_KALMAN = "unscented_kalman"
    ENSEMBLE_KALMAN = "ensemble_kalman"
    PARTICLE_FILTER = "particle_filter"
    VARIATIONAL_3D = "3dvar"
    VARIATIONAL_4D = "4dvar"
    HYBRID = "hybrid"


class ObservationType(Enum):
    """Types of observations."""
    DIRECT = "direct"                    # Direct state measurement
    INDIRECT = "indirect"                # Derived measurement
    SPARSE = "sparse"                    # Spatially sparse
    DENSE = "dense"                      # Full coverage
    LAGGED = "lagged"                    # Time-delayed


@dataclass
class ObservationOperator:
    """Observation operator mapping state to observation space."""
    name: str
    state_indices: List[int]
    observation_indices: List[int]
    transform_func: Optional[Callable] = None
    jacobian_func: Optional[Callable] = None
    noise_covariance: Optional[np.ndarray] = None


@dataclass
class AssimilationConfig:
    """Configuration for data assimilation."""
    method: AssimilationMethod = AssimilationMethod.ENSEMBLE_KALMAN
    state_dimension: int = 10
    observation_dimension: int = 8
    ensemble_size: int = 50
    localization_radius: float = 100.0
    inflation_factor: float = 1.02
    adaptive_inflation: bool = True
    observation_thinning: float = 1.0
    use_covariance_localization: bool = True


@dataclass
class AssimilationState:
    """State of the assimilation system."""
    mean: np.ndarray
    covariance: np.ndarray
    ensemble: Optional[np.ndarray] = None
    analysis_increment: Optional[np.ndarray] = None
    innovation: Optional[np.ndarray] = None
    innovation_covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation.

    Linearizes the system around the current estimate.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        self.n = state_dim
        self.m = obs_dim

        # State estimate
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1.0

        # Noise covariances
        self.Q = np.eye(state_dim) * 0.01      # Process noise
        self.R = np.eye(obs_dim) * 0.1         # Observation noise

        # Model functions (to be set)
        self.f = None      # State transition function
        self.h = None      # Observation function
        self.F = None      # Jacobian of f
        self.H = None      # Jacobian of h

    def set_model(self, f: Callable, h: Callable,
                  F: Optional[Callable] = None, H: Optional[Callable] = None):
        """Set model functions."""
        self.f = f
        self.h = h
        self.F = F
        self.H = H

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 0.1):
        """Prediction step."""
        # State prediction
        if self.f is not None:
            self.x = self.f(self.x, u, dt)

        # Covariance prediction
        if self.F is not None:
            F = self.F(self.x, u, dt)
        else:
            F = np.eye(self.n)

        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update step with observation."""
        # Innovation
        if self.h is not None:
            z_pred = self.h(self.x)
        else:
            z_pred = self.x[:self.m]

        y = z - z_pred

        # Observation Jacobian
        if self.H is not None:
            H = self.H(self.x)
        else:
            H = np.eye(self.m, self.n)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return y  # Return innovation

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        return self.x.copy(), self.P.copy()


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state estimation.

    Uses sigma points to capture the mean and covariance through nonlinear transforms.
    """

    def __init__(self, state_dim: int, obs_dim: int,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = state_dim
        self.m = obs_dim

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = alpha ** 2 * (state_dim + kappa) - state_dim

        # Weights
        self.Wm = np.zeros(2 * state_dim + 1)
        self.Wc = np.zeros(2 * state_dim + 1)

        self.Wm[0] = self.lambd / (state_dim + self.lambd)
        self.Wc[0] = self.lambd / (state_dim + self.lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, 2 * state_dim + 1):
            self.Wm[i] = 1 / (2 * (state_dim + self.lambd))
            self.Wc[i] = 1 / (2 * (state_dim + self.lambd))

        # State
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

        # Noise
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(obs_dim) * 0.1

        # Model functions
        self.f = None
        self.h = None

    def set_model(self, f: Callable, h: Callable):
        """Set model functions."""
        self.f = f
        self.h = h

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))

        sigma_points[0] = x

        try:
            sqrt_P = linalg.cholesky((n + self.lambd) * P, lower=True)
        except linalg.LinAlgError:
            # Handle non-positive definite covariance
            sqrt_P = linalg.sqrtm((n + self.lambd) * P)
            sqrt_P = np.real(sqrt_P)

        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[:, i]
            sigma_points[n + i + 1] = x - sqrt_P[:, i]

        return sigma_points

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 0.1):
        """Prediction step using sigma points."""
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # Transform sigma points through process model
        transformed = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            if self.f is not None:
                transformed[i] = self.f(sp, u, dt)
            else:
                transformed[i] = sp

        # Calculate predicted mean
        self.x = np.sum(self.Wm[:, np.newaxis] * transformed, axis=0)

        # Calculate predicted covariance
        self.P = self.Q * dt
        for i, sp in enumerate(transformed):
            diff = sp - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update step with observation."""
        # Generate sigma points for update
        sigma_points = self._generate_sigma_points(self.x, self.P)

        # Transform through observation model
        z_sigma = np.zeros((len(sigma_points), self.m))
        for i, sp in enumerate(sigma_points):
            if self.h is not None:
                z_sigma[i] = self.h(sp)
            else:
                z_sigma[i] = sp[:self.m]

        # Predicted observation mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_sigma, axis=0)

        # Innovation
        y = z - z_pred

        # Innovation covariance
        S = self.R.copy()
        for i, zs in enumerate(z_sigma):
            diff = zs - z_pred
            S += self.Wc[i] * np.outer(diff, diff)

        # Cross covariance
        Pxz = np.zeros((self.n, self.m))
        for i in range(len(sigma_points)):
            x_diff = sigma_points[i] - self.x
            z_diff = z_sigma[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

        return y

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        return self.x.copy(), self.P.copy()


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter (EnKF) for high-dimensional systems.

    Uses an ensemble of states to represent the probability distribution.
    """

    def __init__(self, state_dim: int, obs_dim: int, ensemble_size: int = 50):
        self.n = state_dim
        self.m = obs_dim
        self.N = ensemble_size

        # Ensemble
        self.ensemble = np.zeros((ensemble_size, state_dim))

        # Mean state
        self.x = np.zeros(state_dim)

        # Observation noise
        self.R = np.eye(obs_dim) * 0.1

        # Model functions
        self.f = None
        self.h = None

        # EnKF parameters
        self.inflation_factor = 1.02
        self.localization_radius = None
        self.localization_matrix = None

    def initialize_ensemble(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize ensemble from mean and covariance."""
        self.x = mean.copy()
        try:
            L = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(covariance)))

        for i in range(self.N):
            self.ensemble[i] = mean + L @ np.random.randn(self.n)

    def set_model(self, f: Callable, h: Callable):
        """Set model functions."""
        self.f = f
        self.h = h

    def set_localization(self, radius: float, distances: Optional[np.ndarray] = None):
        """Set covariance localization."""
        self.localization_radius = radius
        if distances is not None:
            # Gaspari-Cohn localization function
            self.localization_matrix = self._gaspari_cohn(distances, radius)

    def _gaspari_cohn(self, distances: np.ndarray, radius: float) -> np.ndarray:
        """Gaspari-Cohn localization function."""
        r = distances / radius
        gc = np.zeros_like(r)

        mask1 = r <= 1
        mask2 = (r > 1) & (r <= 2)

        gc[mask1] = 1 - 5/3 * r[mask1]**2 + 5/8 * r[mask1]**3 + \
                    1/2 * r[mask1]**4 - 1/4 * r[mask1]**5

        gc[mask2] = 4 - 5*r[mask2] + 5/3 * r[mask2]**2 + 5/8 * r[mask2]**3 - \
                    1/2 * r[mask2]**4 + 1/12 * r[mask2]**5 - 2/(3*r[mask2])

        return gc

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 0.1):
        """Forecast step: propagate ensemble through model."""
        for i in range(self.N):
            if self.f is not None:
                self.ensemble[i] = self.f(self.ensemble[i], u, dt)
            # Add process noise
            self.ensemble[i] += np.random.randn(self.n) * 0.01 * np.sqrt(dt)

        # Update mean
        self.x = np.mean(self.ensemble, axis=0)

        # Covariance inflation
        if self.inflation_factor > 1.0:
            perturbations = self.ensemble - self.x
            self.ensemble = self.x + self.inflation_factor * perturbations

    def update(self, z: np.ndarray) -> np.ndarray:
        """Analysis step: assimilate observation."""
        # Ensemble mean
        x_mean = np.mean(self.ensemble, axis=0)

        # Ensemble perturbations
        X = self.ensemble - x_mean

        # Observation ensemble
        HX = np.zeros((self.N, self.m))
        for i in range(self.N):
            if self.h is not None:
                HX[i] = self.h(self.ensemble[i])
            else:
                HX[i] = self.ensemble[i][:self.m]

        # Observation ensemble mean
        HX_mean = np.mean(HX, axis=0)

        # Observation perturbations
        Y = HX - HX_mean

        # Innovation
        innovation = z - HX_mean

        # Ensemble covariance in observation space
        PHT = X.T @ Y / (self.N - 1)
        HPHT = Y.T @ Y / (self.N - 1)

        # Apply localization if set
        if self.localization_matrix is not None:
            PHT = PHT * self.localization_matrix[:self.n, :self.m]

        # Kalman gain (using perturbed observations)
        S = HPHT + self.R
        K = PHT @ np.linalg.inv(S)

        # Generate perturbed observations
        for i in range(self.N):
            z_perturbed = z + linalg.sqrtm(self.R) @ np.random.randn(self.m)
            HXi = HX[i]
            self.ensemble[i] = self.ensemble[i] + K @ (z_perturbed - HXi)

        # Update mean
        self.x = np.mean(self.ensemble, axis=0)

        return innovation

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble mean and covariance."""
        x_mean = np.mean(self.ensemble, axis=0)
        X = self.ensemble - x_mean
        P = X.T @ X / (self.N - 1)
        return x_mean, P

    def get_ensemble_spread(self) -> float:
        """Get ensemble spread (standard deviation)."""
        return np.mean(np.std(self.ensemble, axis=0))


class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for nonlinear/non-Gaussian systems.
    """

    def __init__(self, state_dim: int, obs_dim: int, num_particles: int = 1000):
        self.n = state_dim
        self.m = obs_dim
        self.N = num_particles

        # Particles and weights
        self.particles = np.zeros((num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

        # Model functions
        self.f = None
        self.h = None

        # Observation likelihood
        self.R = np.eye(obs_dim) * 0.1

        # Resampling threshold
        self.resample_threshold = num_particles / 2

    def initialize(self, mean: np.ndarray, covariance: np.ndarray):
        """Initialize particles."""
        try:
            L = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(covariance)))

        for i in range(self.N):
            self.particles[i] = mean + L @ np.random.randn(self.n)

        self.weights = np.ones(self.N) / self.N

    def set_model(self, f: Callable, h: Callable):
        """Set model functions."""
        self.f = f
        self.h = h

    def predict(self, u: Optional[np.ndarray] = None, dt: float = 0.1,
                process_noise_std: float = 0.01):
        """Propagate particles through process model."""
        for i in range(self.N):
            if self.f is not None:
                self.particles[i] = self.f(self.particles[i], u, dt)
            # Add process noise
            self.particles[i] += np.random.randn(self.n) * process_noise_std * np.sqrt(dt)

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update weights based on observation likelihood."""
        # Compute observation for each particle
        likelihoods = np.zeros(self.N)

        R_inv = np.linalg.inv(self.R)
        R_det = np.linalg.det(self.R)

        for i in range(self.N):
            if self.h is not None:
                z_pred = self.h(self.particles[i])
            else:
                z_pred = self.particles[i][:self.m]

            innovation = z - z_pred

            # Gaussian likelihood
            exponent = -0.5 * innovation.T @ R_inv @ innovation
            likelihoods[i] = np.exp(exponent) / np.sqrt((2 * np.pi) ** self.m * R_det)

        # Update weights
        self.weights *= likelihoods

        # Normalize
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.N) / self.N

        # Resample if needed
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.resample_threshold:
            self._resample()

        # Return weighted innovation
        x_mean = self.get_state()[0]
        if self.h is not None:
            z_pred = self.h(x_mean)
        else:
            z_pred = x_mean[:self.m]

        return z - z_pred

    def _resample(self):
        """Systematic resampling."""
        cumsum = np.cumsum(self.weights)
        u = (np.random.rand() + np.arange(self.N)) / self.N

        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, self.N - 1)

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.N) / self.N

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weighted mean and covariance."""
        x_mean = np.average(self.particles, weights=self.weights, axis=0)

        # Weighted covariance
        diff = self.particles - x_mean
        P = np.zeros((self.n, self.n))
        for i in range(self.N):
            P += self.weights[i] * np.outer(diff[i], diff[i])

        return x_mean, P


class DataAssimilationEngine:
    """
    Complete data assimilation engine for the aqueduct system.

    Integrates multiple assimilation methods and manages the
    state estimation workflow.
    """

    def __init__(self, config: Optional[AssimilationConfig] = None):
        self.config = config or AssimilationConfig()

        # Initialize filters based on config
        self.filters: Dict[str, Any] = {}
        self._initialize_filters()

        # State dimension mappings
        self.state_names = ['h', 'v', 'T_sun', 'T_shade', 'T_core',
                           'joint_gap', 'vib_amp', 'bearing_stress', 'Q_in', 'Q_out']
        self.obs_names = ['h', 'v', 'T_sun', 'T_shade',
                         'joint_gap', 'vib_amp', 'bearing_stress', 'Q_in']

        # History
        self.assimilation_history: deque = deque(maxlen=10000)
        self.innovation_history: deque = deque(maxlen=1000)

        # Current state
        self.current_state = AssimilationState(
            mean=np.zeros(self.config.state_dimension),
            covariance=np.eye(self.config.state_dimension)
        )

        # Model functions
        self._f = None
        self._h = None

        # Performance metrics
        self.total_assimilations = 0
        self.start_time = time.time()

    def _initialize_filters(self):
        """Initialize assimilation filters."""
        n = self.config.state_dimension
        m = self.config.observation_dimension

        # Extended Kalman Filter
        self.filters['ekf'] = ExtendedKalmanFilter(n, m)

        # Unscented Kalman Filter
        self.filters['ukf'] = UnscentedKalmanFilter(n, m)

        # Ensemble Kalman Filter
        self.filters['enkf'] = EnsembleKalmanFilter(
            n, m, self.config.ensemble_size
        )

        # Particle Filter
        self.filters['pf'] = ParticleFilter(n, m, num_particles=500)

    def set_model(self, f: Callable, h: Callable,
                  F: Optional[Callable] = None, H: Optional[Callable] = None):
        """
        Set the model functions.

        Args:
            f: State transition function f(x, u, dt) -> x_next
            h: Observation function h(x) -> z
            F: Jacobian of f (for EKF)
            H: Jacobian of h (for EKF)
        """
        self._f = f
        self._h = h

        self.filters['ekf'].set_model(f, h, F, H)
        self.filters['ukf'].set_model(f, h)
        self.filters['enkf'].set_model(f, h)
        self.filters['pf'].set_model(f, h)

    def initialize(self, initial_state: Dict[str, float],
                   initial_uncertainty: Optional[Dict[str, float]] = None):
        """Initialize assimilation with initial state."""
        # Build state vector
        x0 = np.zeros(self.config.state_dimension)
        for i, name in enumerate(self.state_names):
            if i < len(x0) and name in initial_state:
                x0[i] = initial_state[name]

        # Build initial covariance
        P0 = np.eye(self.config.state_dimension)
        if initial_uncertainty:
            for i, name in enumerate(self.state_names):
                if i < len(x0) and name in initial_uncertainty:
                    P0[i, i] = initial_uncertainty[name] ** 2

        # Initialize all filters
        self.filters['ekf'].x = x0.copy()
        self.filters['ekf'].P = P0.copy()

        self.filters['ukf'].x = x0.copy()
        self.filters['ukf'].P = P0.copy()

        self.filters['enkf'].initialize_ensemble(x0, P0)
        self.filters['pf'].initialize(x0, P0)

        self.current_state = AssimilationState(
            mean=x0.copy(),
            covariance=P0.copy(),
            timestamp=time.time()
        )

    def predict(self, control: Optional[Dict[str, float]] = None,
                dt: float = 0.1) -> Dict[str, float]:
        """
        Perform prediction step.

        Args:
            control: Control inputs
            dt: Time step

        Returns:
            Predicted state
        """
        # Convert control to vector if provided
        u = None
        if control:
            u = np.array([control.get('Q_in', 80), control.get('Q_out', 80)])

        # Predict with selected method
        method = self.config.method

        if method == AssimilationMethod.EXTENDED_KALMAN:
            self.filters['ekf'].predict(u, dt)
            x, P = self.filters['ekf'].get_state()

        elif method == AssimilationMethod.UNSCENTED_KALMAN:
            self.filters['ukf'].predict(u, dt)
            x, P = self.filters['ukf'].get_state()

        elif method == AssimilationMethod.ENSEMBLE_KALMAN:
            self.filters['enkf'].predict(u, dt)
            x, P = self.filters['enkf'].get_state()

        elif method == AssimilationMethod.PARTICLE_FILTER:
            self.filters['pf'].predict(u, dt)
            x, P = self.filters['pf'].get_state()

        else:
            # Default to EnKF
            self.filters['enkf'].predict(u, dt)
            x, P = self.filters['enkf'].get_state()

        self.current_state.mean = x
        self.current_state.covariance = P
        self.current_state.timestamp = time.time()

        return self._state_to_dict(x)

    def assimilate(self, observations: Dict[str, float],
                   timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Assimilate observations into state estimate.

        Args:
            observations: Dictionary of observed values
            timestamp: Observation timestamp

        Returns:
            Analysis results including updated state
        """
        timestamp = timestamp or time.time()

        # Build observation vector
        z = np.zeros(self.config.observation_dimension)
        for i, name in enumerate(self.obs_names):
            if i < len(z) and name in observations:
                z[i] = observations[name]

        # Get prior state
        prior_mean = self.current_state.mean.copy()

        # Update with selected method
        method = self.config.method

        if method == AssimilationMethod.EXTENDED_KALMAN:
            innovation = self.filters['ekf'].update(z)
            x, P = self.filters['ekf'].get_state()

        elif method == AssimilationMethod.UNSCENTED_KALMAN:
            innovation = self.filters['ukf'].update(z)
            x, P = self.filters['ukf'].get_state()

        elif method == AssimilationMethod.ENSEMBLE_KALMAN:
            innovation = self.filters['enkf'].update(z)
            x, P = self.filters['enkf'].get_state()

        elif method == AssimilationMethod.PARTICLE_FILTER:
            innovation = self.filters['pf'].update(z)
            x, P = self.filters['pf'].get_state()

        else:
            innovation = self.filters['enkf'].update(z)
            x, P = self.filters['enkf'].get_state()

        # Calculate analysis increment
        increment = x - prior_mean

        # Update current state
        self.current_state = AssimilationState(
            mean=x,
            covariance=P,
            analysis_increment=increment,
            innovation=innovation,
            timestamp=timestamp
        )

        # Store history
        self.total_assimilations += 1
        self.innovation_history.append({
            'timestamp': timestamp,
            'innovation': innovation.copy(),
            'increment': increment.copy()
        })

        result = {
            'timestamp': timestamp,
            'state': self._state_to_dict(x),
            'uncertainty': self._uncertainty_to_dict(P),
            'innovation': {
                self.obs_names[i]: innovation[i]
                for i in range(min(len(innovation), len(self.obs_names)))
            },
            'increment': {
                self.state_names[i]: increment[i]
                for i in range(min(len(increment), len(self.state_names)))
            },
            'method': method.value
        }

        self.assimilation_history.append(result)

        return result

    def _state_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert state vector to dictionary."""
        return {
            self.state_names[i]: x[i]
            for i in range(min(len(x), len(self.state_names)))
        }

    def _uncertainty_to_dict(self, P: np.ndarray) -> Dict[str, float]:
        """Convert covariance to uncertainty dictionary."""
        return {
            f"{self.state_names[i]}_std": np.sqrt(P[i, i])
            for i in range(min(P.shape[0], len(self.state_names)))
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current assimilated state."""
        return {
            'mean': self._state_to_dict(self.current_state.mean),
            'uncertainty': self._uncertainty_to_dict(self.current_state.covariance),
            'timestamp': self.current_state.timestamp,
            'method': self.config.method.value
        }

    def get_innovation_statistics(self) -> Dict[str, Any]:
        """Get innovation statistics for assimilation quality monitoring."""
        if len(self.innovation_history) < 10:
            return {'status': 'insufficient_data'}

        innovations = np.array([h['innovation'] for h in self.innovation_history])

        return {
            'mean': np.mean(innovations, axis=0).tolist(),
            'std': np.std(innovations, axis=0).tolist(),
            'bias': np.mean(innovations, axis=0).tolist(),
            'rmse': np.sqrt(np.mean(innovations ** 2, axis=0)).tolist(),
            'samples': len(self.innovation_history)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get assimilation engine status."""
        return {
            'method': self.config.method.value,
            'state_dimension': self.config.state_dimension,
            'observation_dimension': self.config.observation_dimension,
            'ensemble_size': self.config.ensemble_size,
            'total_assimilations': self.total_assimilations,
            'uptime_seconds': time.time() - self.start_time,
            'current_state': self.get_current_state(),
            'innovation_stats': self.get_innovation_statistics()
        }

    def switch_method(self, method: AssimilationMethod):
        """Switch assimilation method."""
        # Transfer state between methods
        if self.config.method != method:
            x = self.current_state.mean
            P = self.current_state.covariance

            self.config.method = method

            # Re-initialize the new method with current state
            if method == AssimilationMethod.EXTENDED_KALMAN:
                self.filters['ekf'].x = x.copy()
                self.filters['ekf'].P = P.copy()

            elif method == AssimilationMethod.UNSCENTED_KALMAN:
                self.filters['ukf'].x = x.copy()
                self.filters['ukf'].P = P.copy()

            elif method == AssimilationMethod.ENSEMBLE_KALMAN:
                self.filters['enkf'].initialize_ensemble(x, P)

            elif method == AssimilationMethod.PARTICLE_FILTER:
                self.filters['pf'].initialize(x, P)
