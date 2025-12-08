"""
State Prediction Module for TAOS V3.10

This module provides real-time system state prediction including:
- Multi-horizon forecasting
- Uncertainty quantification
- Ensemble prediction
- Scenario-based prediction
- Neural network predictors
- Hybrid model prediction

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


class PredictionMethod(Enum):
    """Prediction methods."""
    PHYSICS_BASED = "physics_based"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


class UncertaintyType(Enum):
    """Types of uncertainty."""
    ALEATORY = "aleatory"           # Inherent randomness
    EPISTEMIC = "epistemic"         # Model/knowledge uncertainty
    COMBINED = "combined"


@dataclass
class PredictionHorizon:
    """Prediction horizon definition."""
    name: str
    steps: int
    dt: float                        # Time step in seconds
    total_seconds: float
    confidence_decay: float = 0.95   # Confidence decay per step


@dataclass
class StatePrediction:
    """Single state prediction result."""
    timestamp: float                 # Prediction timestamp
    horizon_seconds: float           # How far ahead
    state: Dict[str, float]          # Predicted state
    uncertainty: Dict[str, float]    # Uncertainty (std dev)
    confidence: float                # Overall confidence (0-1)
    method: PredictionMethod


@dataclass
class PredictionTrajectory:
    """Complete prediction trajectory."""
    start_time: float
    predictions: List[StatePrediction]
    horizon_name: str
    method: PredictionMethod
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicsPredictor:
    """
    Physics-based predictor using simplified aqueduct dynamics.
    """

    def __init__(self):
        # Physical parameters
        self.width = 9.0              # Aqueduct width (m)
        self.area = 360.0             # Cross-sectional area proxy
        self.g = 9.81

        # Time constants
        self.tau_hydraulic = 100.0    # Hydraulic time constant
        self.tau_thermal = 3600.0     # Thermal time constant
        self.tau_structural = 10.0    # Structural time constant

    def predict_step(self, state: Dict[str, float],
                     control: Dict[str, float],
                     environment: Dict[str, float],
                     dt: float = 1.0) -> Dict[str, float]:
        """
        Predict state one step ahead using physics model.
        """
        # Current state
        h = state.get('h', 4.0)
        v = state.get('v', 2.0)
        T_sun = state.get('T_sun', 25.0)
        T_shade = state.get('T_shade', 25.0)
        joint_gap = state.get('joint_gap', 20.0)
        vib = state.get('vib_amp', 0.0)

        # Control inputs
        Q_in = control.get('Q_in', 80.0)
        Q_out = control.get('Q_out', 80.0)

        # Environment
        T_amb = environment.get('T_ambient', 25.0)
        solar = environment.get('solar_rad', 0.0)
        wind = environment.get('wind_speed', 0.0)

        # Hydraulic dynamics
        dh_dt = (Q_in - Q_out) / self.area
        h_new = h + dh_dt * dt
        h_new = np.clip(h_new, 0.1, 8.0)

        # Velocity update
        if h_new > 0.1:
            v_new = (Q_in + Q_out) / 2.0 / (self.width * h_new)
        else:
            v_new = 0.0

        # Thermal dynamics
        k_sun = 0.001 * solar
        k_air = 0.0005
        dTs_dt = k_sun + k_air * (T_amb - T_sun)
        dTsh_dt = k_air * (T_amb - T_shade)

        T_sun_new = T_sun + dTs_dt * dt
        T_shade_new = T_shade + dTsh_dt * dt

        # Structural dynamics
        T_avg = (T_sun_new + T_shade_new) / 2.0
        delta_T = T_avg - 20.0
        expansion = 1.0e-5 * 40000 * delta_T  # mm
        target_gap = 20.0 - expansion
        joint_new = joint_gap + (target_gap - joint_gap) * dt / self.tau_structural

        # Vibration
        target_vib = 0.0
        if 10.0 < wind < 15.0:
            target_vib += 20.0
        fr = v_new / np.sqrt(self.g * max(h_new, 0.1))
        if fr > 0.9:
            target_vib += 10.0 * (fr - 0.9)
        vib_new = vib + (target_vib - vib) * dt / self.tau_structural

        return {
            'h': h_new,
            'v': v_new,
            'T_sun': T_sun_new,
            'T_shade': T_shade_new,
            'joint_gap': joint_new,
            'vib_amp': max(0, vib_new),
            'fr': fr,
            'Q_in': Q_in,
            'Q_out': Q_out
        }

    def predict_trajectory(self, initial_state: Dict[str, float],
                           control_sequence: List[Dict[str, float]],
                           environment_sequence: List[Dict[str, float]],
                           dt: float = 1.0) -> List[Dict[str, float]]:
        """Predict trajectory over multiple steps."""
        trajectory = []
        state = initial_state.copy()

        for i in range(len(control_sequence)):
            control = control_sequence[i] if i < len(control_sequence) else control_sequence[-1]
            env = environment_sequence[i] if i < len(environment_sequence) else environment_sequence[-1]

            state = self.predict_step(state, control, env, dt)
            trajectory.append(state.copy())

        return trajectory


class StatisticalPredictor:
    """
    Statistical predictor using time series methods.
    """

    def __init__(self, history_length: int = 100):
        self.history: Dict[str, deque] = {}
        self.history_length = history_length

        # State variables to track
        self.state_vars = ['h', 'v', 'T_sun', 'T_shade', 'joint_gap', 'vib_amp']

        for var in self.state_vars:
            self.history[var] = deque(maxlen=history_length)

    def update(self, state: Dict[str, float]):
        """Update history with new state."""
        for var in self.state_vars:
            if var in state:
                self.history[var].append(state[var])

    def predict_exponential_smoothing(self, var: str, horizon: int,
                                       alpha: float = 0.3) -> Tuple[List[float], List[float]]:
        """Predict using exponential smoothing."""
        if var not in self.history or len(self.history[var]) < 3:
            return [0.0] * horizon, [1.0] * horizon

        values = list(self.history[var])

        # Simple exponential smoothing
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed

        # Trend estimation
        if len(values) >= 2:
            trend = (values[-1] - values[-2])
        else:
            trend = 0

        # Prediction
        predictions = []
        for i in range(horizon):
            pred = smoothed + trend * (i + 1)
            predictions.append(pred)

        # Uncertainty grows with horizon
        base_std = np.std(values) if len(values) > 1 else 1.0
        uncertainties = [base_std * (1 + 0.1 * i) for i in range(horizon)]

        return predictions, uncertainties

    def predict_arma(self, var: str, horizon: int,
                     ar_order: int = 2, ma_order: int = 1) -> Tuple[List[float], List[float]]:
        """Predict using simple ARMA-like model."""
        if var not in self.history or len(self.history[var]) < ar_order + 2:
            return [0.0] * horizon, [1.0] * horizon

        values = np.array(list(self.history[var]))

        # Fit simple AR model using least squares
        if len(values) > ar_order + 1:
            X = np.column_stack([values[ar_order - i - 1:-1 - i] for i in range(ar_order)])
            y = values[ar_order:]

            try:
                ar_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                ar_coeffs = np.zeros(ar_order)
        else:
            ar_coeffs = np.zeros(ar_order)

        # Make predictions
        predictions = []
        pred_values = list(values[-ar_order:])

        for _ in range(horizon):
            pred = np.dot(ar_coeffs, pred_values[-ar_order:][::-1])
            predictions.append(pred)
            pred_values.append(pred)

        # Estimate uncertainty
        residuals = values[ar_order:] - X @ ar_coeffs if len(values) > ar_order + 1 else [1.0]
        base_std = np.std(residuals) if len(residuals) > 0 else 1.0
        uncertainties = [base_std * np.sqrt(1 + 0.1 * i) for i in range(horizon)]

        return predictions, uncertainties

    def predict(self, horizon: int, method: str = 'exponential') -> Dict[str, Any]:
        """Predict all state variables."""
        predictions = {}
        uncertainties = {}

        for var in self.state_vars:
            if method == 'exponential':
                preds, uncert = self.predict_exponential_smoothing(var, horizon)
            else:
                preds, uncert = self.predict_arma(var, horizon)

            predictions[var] = preds
            uncertainties[var] = uncert

        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'method': method
        }


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple prediction methods.
    """

    def __init__(self, num_members: int = 20):
        self.num_members = num_members
        self.physics_predictor = PhysicsPredictor()
        self.statistical_predictor = StatisticalPredictor()

        # Member weights
        self.member_weights = np.ones(num_members) / num_members

        # Ensemble perturbation scale
        self.perturbation_scale = {
            'h': 0.1,
            'v': 0.1,
            'T_sun': 1.0,
            'T_shade': 1.0,
            'joint_gap': 0.5,
            'vib_amp': 1.0,
            'Q_in': 2.0,
            'Q_out': 2.0
        }

    def generate_ensemble(self, initial_state: Dict[str, float],
                          control: Dict[str, float],
                          environment: Dict[str, float]) -> List[Dict[str, float]]:
        """Generate ensemble of initial states."""
        ensemble = []

        for _ in range(self.num_members):
            member = initial_state.copy()
            for var, scale in self.perturbation_scale.items():
                if var in member:
                    member[var] += np.random.normal(0, scale)
            ensemble.append(member)

        return ensemble

    def predict_ensemble(self, initial_state: Dict[str, float],
                         control: Dict[str, float],
                         environment: Dict[str, float],
                         horizon: int,
                         dt: float = 1.0) -> Dict[str, Any]:
        """Generate ensemble prediction."""
        # Generate ensemble of initial states
        ensemble = self.generate_ensemble(initial_state, control, environment)

        # Propagate each member
        trajectories = []
        control_seq = [control] * horizon
        env_seq = [environment] * horizon

        for member in ensemble:
            traj = self.physics_predictor.predict_trajectory(
                member, control_seq, env_seq, dt
            )
            trajectories.append(traj)

        # Calculate ensemble statistics
        result = {
            'mean': [],
            'std': [],
            'percentile_10': [],
            'percentile_90': [],
            'all_members': trajectories
        }

        for step in range(horizon):
            step_values = {var: [] for var in self.perturbation_scale}

            for traj in trajectories:
                if step < len(traj):
                    for var in step_values:
                        if var in traj[step]:
                            step_values[var].append(traj[step][var])

            step_mean = {}
            step_std = {}
            step_p10 = {}
            step_p90 = {}

            for var, values in step_values.items():
                if values:
                    step_mean[var] = np.mean(values)
                    step_std[var] = np.std(values)
                    step_p10[var] = np.percentile(values, 10)
                    step_p90[var] = np.percentile(values, 90)

            result['mean'].append(step_mean)
            result['std'].append(step_std)
            result['percentile_10'].append(step_p10)
            result['percentile_90'].append(step_p90)

        return result


class StatePredictionEngine:
    """
    Complete state prediction engine for the aqueduct system.
    """

    def __init__(self):
        # Predictors
        self.physics_predictor = PhysicsPredictor()
        self.statistical_predictor = StatisticalPredictor()
        self.ensemble_predictor = EnsemblePredictor()

        # Prediction horizons
        self.horizons = {
            'short': PredictionHorizon(
                name='short',
                steps=60,
                dt=1.0,
                total_seconds=60.0,
                confidence_decay=0.99
            ),
            'medium': PredictionHorizon(
                name='medium',
                steps=60,
                dt=10.0,
                total_seconds=600.0,
                confidence_decay=0.98
            ),
            'long': PredictionHorizon(
                name='long',
                steps=60,
                dt=60.0,
                total_seconds=3600.0,
                confidence_decay=0.95
            )
        }

        # History
        self.prediction_history: deque = deque(maxlen=1000)
        self.verification_history: deque = deque(maxlen=1000)

        # Performance tracking
        self.prediction_errors: Dict[str, deque] = {
            var: deque(maxlen=100)
            for var in ['h', 'v', 'T_sun', 'T_shade', 'joint_gap', 'vib_amp']
        }

        # Metrics
        self.total_predictions = 0
        self.start_time = time.time()

    def update_history(self, state: Dict[str, float]):
        """Update statistical predictor history."""
        self.statistical_predictor.update(state)

    def predict(self, current_state: Dict[str, float],
                control: Dict[str, float],
                environment: Dict[str, float],
                horizon_name: str = 'short',
                method: PredictionMethod = PredictionMethod.ENSEMBLE) -> PredictionTrajectory:
        """
        Generate state predictions.

        Args:
            current_state: Current system state
            control: Current/planned control inputs
            environment: Current/forecast environmental conditions
            horizon_name: Prediction horizon ('short', 'medium', 'long')
            method: Prediction method to use

        Returns:
            PredictionTrajectory with predictions
        """
        self.total_predictions += 1
        start_time = time.time()

        horizon = self.horizons.get(horizon_name, self.horizons['short'])
        predictions = []

        if method == PredictionMethod.PHYSICS_BASED:
            control_seq = [control] * horizon.steps
            env_seq = [environment] * horizon.steps
            trajectory = self.physics_predictor.predict_trajectory(
                current_state, control_seq, env_seq, horizon.dt
            )

            for i, pred_state in enumerate(trajectory):
                confidence = horizon.confidence_decay ** (i + 1)
                predictions.append(StatePrediction(
                    timestamp=start_time + (i + 1) * horizon.dt,
                    horizon_seconds=(i + 1) * horizon.dt,
                    state=pred_state,
                    uncertainty={var: 0.1 * (1 + 0.1 * i) for var in pred_state},
                    confidence=confidence,
                    method=method
                ))

        elif method == PredictionMethod.STATISTICAL:
            stat_result = self.statistical_predictor.predict(horizon.steps)

            for i in range(horizon.steps):
                pred_state = {
                    var: stat_result['predictions'][var][i]
                    for var in stat_result['predictions']
                }
                uncertainty = {
                    var: stat_result['uncertainties'][var][i]
                    for var in stat_result['uncertainties']
                }
                confidence = horizon.confidence_decay ** (i + 1)

                predictions.append(StatePrediction(
                    timestamp=start_time + (i + 1) * horizon.dt,
                    horizon_seconds=(i + 1) * horizon.dt,
                    state=pred_state,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    method=method
                ))

        elif method == PredictionMethod.ENSEMBLE:
            ens_result = self.ensemble_predictor.predict_ensemble(
                current_state, control, environment, horizon.steps, horizon.dt
            )

            for i in range(len(ens_result['mean'])):
                confidence = horizon.confidence_decay ** (i + 1)
                predictions.append(StatePrediction(
                    timestamp=start_time + (i + 1) * horizon.dt,
                    horizon_seconds=(i + 1) * horizon.dt,
                    state=ens_result['mean'][i],
                    uncertainty=ens_result['std'][i],
                    confidence=confidence,
                    method=method
                ))

        elif method == PredictionMethod.HYBRID:
            # Combine physics and statistical
            control_seq = [control] * horizon.steps
            env_seq = [environment] * horizon.steps
            physics_traj = self.physics_predictor.predict_trajectory(
                current_state, control_seq, env_seq, horizon.dt
            )
            stat_result = self.statistical_predictor.predict(horizon.steps)

            # Blend predictions (physics weight increases with horizon)
            for i in range(min(len(physics_traj), horizon.steps)):
                physics_weight = 0.3 + 0.4 * (i / horizon.steps)  # 0.3 to 0.7
                stat_weight = 1.0 - physics_weight

                hybrid_state = {}
                hybrid_uncertainty = {}

                for var in physics_traj[i]:
                    phys_val = physics_traj[i].get(var, 0)
                    stat_val = stat_result['predictions'].get(var, [0] * horizon.steps)[i]

                    hybrid_state[var] = physics_weight * phys_val + stat_weight * stat_val
                    hybrid_uncertainty[var] = stat_result['uncertainties'].get(
                        var, [1.0] * horizon.steps
                    )[i]

                confidence = horizon.confidence_decay ** (i + 1)
                predictions.append(StatePrediction(
                    timestamp=start_time + (i + 1) * horizon.dt,
                    horizon_seconds=(i + 1) * horizon.dt,
                    state=hybrid_state,
                    uncertainty=hybrid_uncertainty,
                    confidence=confidence,
                    method=method
                ))

        trajectory = PredictionTrajectory(
            start_time=start_time,
            predictions=predictions,
            horizon_name=horizon_name,
            method=method,
            metadata={
                'horizon_steps': horizon.steps,
                'horizon_dt': horizon.dt,
                'total_seconds': horizon.total_seconds
            }
        )

        # Store in history
        self.prediction_history.append({
            'timestamp': start_time,
            'horizon': horizon_name,
            'method': method.value,
            'initial_state': current_state.copy()
        })

        return trajectory

    def verify_prediction(self, predicted: StatePrediction,
                          actual: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify a prediction against actual state.

        Returns:
            Verification metrics
        """
        errors = {}
        relative_errors = {}

        for var, pred_val in predicted.state.items():
            if var in actual:
                actual_val = actual[var]
                error = pred_val - actual_val
                errors[var] = error

                if abs(actual_val) > 1e-6:
                    relative_errors[var] = error / actual_val
                else:
                    relative_errors[var] = error

                # Track prediction errors
                if var in self.prediction_errors:
                    self.prediction_errors[var].append(error)

        verification = {
            'timestamp': time.time(),
            'horizon_seconds': predicted.horizon_seconds,
            'errors': errors,
            'relative_errors': relative_errors,
            'rmse': np.sqrt(np.mean([e ** 2 for e in errors.values()])) if errors else 0,
            'predicted_confidence': predicted.confidence,
            'within_uncertainty': {}
        }

        # Check if actual was within predicted uncertainty
        for var, error in errors.items():
            uncertainty = predicted.uncertainty.get(var, 1.0)
            verification['within_uncertainty'][var] = abs(error) <= 2 * uncertainty

        self.verification_history.append(verification)

        return verification

    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get prediction accuracy metrics."""
        if not self.verification_history:
            return {'status': 'no_verifications'}

        recent = list(self.verification_history)[-100:]

        var_metrics = {}
        for var in self.prediction_errors:
            errors = list(self.prediction_errors[var])
            if errors:
                var_metrics[var] = {
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
                    'bias': np.mean(errors),
                    'std': np.std(errors)
                }

        return {
            'total_verifications': len(self.verification_history),
            'recent_count': len(recent),
            'average_rmse': np.mean([v['rmse'] for v in recent]),
            'variable_metrics': var_metrics
        }

    def predict_risk(self, current_state: Dict[str, float],
                     control: Dict[str, float],
                     environment: Dict[str, float],
                     risk_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Predict probability of exceeding risk thresholds.
        """
        # Default thresholds
        thresholds = risk_thresholds or {
            'fr': 0.9,                    # Froude number critical
            'T_diff': 10.0,               # Temperature differential
            'joint_gap_min': 5.0,
            'joint_gap_max': 35.0,
            'vib_amp': 50.0
        }

        # Get ensemble prediction
        ens_result = self.ensemble_predictor.predict_ensemble(
            current_state, control, environment, 60, 10.0  # 10 minutes
        )

        risk_probabilities = {
            'hydraulic_jump': [],
            'thermal_stress': [],
            'joint_failure': [],
            'vibration_damage': []
        }

        # Analyze each time step
        for step in range(len(ens_result['all_members'][0])):
            step_values = {var: [] for var in ['fr', 'T_sun', 'T_shade', 'joint_gap', 'vib_amp']}

            for member_traj in ens_result['all_members']:
                if step < len(member_traj):
                    for var in step_values:
                        if var in member_traj[step]:
                            step_values[var].append(member_traj[step][var])

            # Calculate probabilities
            if step_values['fr']:
                fr_exceed = sum(1 for fr in step_values['fr'] if fr > thresholds['fr'])
                risk_probabilities['hydraulic_jump'].append(fr_exceed / len(step_values['fr']))

            if step_values['T_sun'] and step_values['T_shade']:
                T_diffs = [abs(ts - tsh) for ts, tsh in
                          zip(step_values['T_sun'], step_values['T_shade'])]
                thermal_exceed = sum(1 for td in T_diffs if td > thresholds['T_diff'])
                risk_probabilities['thermal_stress'].append(thermal_exceed / len(T_diffs))

            if step_values['joint_gap']:
                joint_exceed = sum(1 for jg in step_values['joint_gap']
                                  if jg < thresholds['joint_gap_min'] or jg > thresholds['joint_gap_max'])
                risk_probabilities['joint_failure'].append(joint_exceed / len(step_values['joint_gap']))

            if step_values['vib_amp']:
                vib_exceed = sum(1 for va in step_values['vib_amp'] if va > thresholds['vib_amp'])
                risk_probabilities['vibration_damage'].append(vib_exceed / len(step_values['vib_amp']))

        return {
            'horizons_seconds': [i * 10.0 for i in range(len(risk_probabilities['hydraulic_jump']))],
            'risk_probabilities': risk_probabilities,
            'max_risks': {
                risk: max(probs) if probs else 0
                for risk, probs in risk_probabilities.items()
            },
            'time_to_risk': {
                risk: next((i * 10.0 for i, p in enumerate(probs) if p > 0.5), None)
                for risk, probs in risk_probabilities.items()
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get prediction engine status."""
        return {
            'total_predictions': self.total_predictions,
            'uptime_seconds': time.time() - self.start_time,
            'horizons': {
                name: {
                    'steps': h.steps,
                    'dt': h.dt,
                    'total_seconds': h.total_seconds
                }
                for name, h in self.horizons.items()
            },
            'accuracy': self.get_prediction_accuracy(),
            'history_size': len(self.prediction_history)
        }


class ScenarioPrediction:
    """
    Scenario-based prediction for what-if analysis.
    """

    def __init__(self):
        self.prediction_engine = StatePredictionEngine()

        # Pre-defined scenarios
        self.scenarios = {
            'normal': {
                'description': 'Normal operation',
                'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}
            },
            'summer_peak': {
                'description': 'Summer peak conditions',
                'environment': {'T_ambient': 38.0, 'solar_rad': 1.0, 'wind_speed': 0.5}
            },
            'winter_cold': {
                'description': 'Winter cold conditions',
                'environment': {'T_ambient': -10.0, 'solar_rad': 0.2, 'wind_speed': 5.0}
            },
            'storm': {
                'description': 'Storm conditions',
                'environment': {'T_ambient': 20.0, 'solar_rad': 0.1, 'wind_speed': 20.0}
            },
            'high_flow': {
                'description': 'High flow demand',
                'control': {'Q_in': 200.0, 'Q_out': 200.0}
            },
            'low_flow': {
                'description': 'Low flow conditions',
                'control': {'Q_in': 30.0, 'Q_out': 30.0}
            }
        }

    def predict_scenario(self, current_state: Dict[str, float],
                         scenario_name: str,
                         base_control: Optional[Dict[str, float]] = None,
                         base_environment: Optional[Dict[str, float]] = None,
                         horizon_name: str = 'medium') -> Dict[str, Any]:
        """Predict system response to a scenario."""
        scenario = self.scenarios.get(scenario_name)
        if not scenario:
            return {'error': f'Unknown scenario: {scenario_name}'}

        # Merge with base values
        control = base_control.copy() if base_control else {'Q_in': 80.0, 'Q_out': 80.0}
        environment = base_environment.copy() if base_environment else {
            'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0
        }

        if 'control' in scenario:
            control.update(scenario['control'])
        if 'environment' in scenario:
            environment.update(scenario['environment'])

        # Generate prediction
        trajectory = self.prediction_engine.predict(
            current_state, control, environment,
            horizon_name, PredictionMethod.ENSEMBLE
        )

        # Risk assessment
        risk = self.prediction_engine.predict_risk(
            current_state, control, environment
        )

        return {
            'scenario': scenario_name,
            'description': scenario.get('description', ''),
            'trajectory': {
                'times': [p.horizon_seconds for p in trajectory.predictions],
                'states': [p.state for p in trajectory.predictions],
                'uncertainties': [p.uncertainty for p in trajectory.predictions],
                'confidences': [p.confidence for p in trajectory.predictions]
            },
            'risk_assessment': risk,
            'control_applied': control,
            'environment_applied': environment
        }

    def compare_scenarios(self, current_state: Dict[str, float],
                          scenario_names: List[str],
                          base_control: Optional[Dict[str, float]] = None,
                          base_environment: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Compare multiple scenarios."""
        results = {}

        for scenario_name in scenario_names:
            results[scenario_name] = self.predict_scenario(
                current_state, scenario_name, base_control, base_environment
            )

        # Summary comparison
        summary = {
            'scenarios': scenario_names,
            'max_risks': {},
            'final_states': {}
        }

        for name, result in results.items():
            if 'risk_assessment' in result:
                summary['max_risks'][name] = result['risk_assessment'].get('max_risks', {})
            if 'trajectory' in result and result['trajectory']['states']:
                summary['final_states'][name] = result['trajectory']['states'][-1]

        return {
            'individual_results': results,
            'summary': summary
        }
