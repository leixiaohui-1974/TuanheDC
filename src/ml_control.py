#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.8 - Advanced AI/ML Control System
团河渡槽自主运行系统 - 高级AI/ML控制模块

Features:
- Deep learning-based prediction
- Reinforcement learning control
- Neural network state estimation
- Anomaly detection with autoencoders
- Multi-model ensemble
- Online learning and adaptation
- Explainable AI for control decisions
"""

import time
import json
import math
import random
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque
from pathlib import Path
import sqlite3


class ModelType(Enum):
    """ML model types"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    MLP = "mlp"
    CNN = "cnn"
    AUTOENCODER = "autoencoder"
    RL_DQN = "rl_dqn"
    RL_PPO = "rl_ppo"
    RL_SAC = "rl_sac"
    ENSEMBLE = "ensemble"


class TrainingStatus(Enum):
    """Model training status"""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    UPDATING = "updating"
    FAILED = "failed"


class PredictionConfidence(Enum):
    """Prediction confidence level"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class ModelConfig:
    """ML model configuration"""
    model_id: str
    model_type: ModelType
    input_features: List[str]
    output_features: List[str]
    hidden_layers: List[int]
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 10
    dropout_rate: float = 0.2
    activation: str = "relu"
    optimizer: str = "adam"
    loss_function: str = "mse"
    regularization: float = 0.01
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type.value,
            'input_features': self.input_features,
            'output_features': self.output_features,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'regularization': self.regularization,
            'metadata': self.metadata
        }


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    mape: float = 0.0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    inference_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2_score': self.r2_score,
            'mape': self.mape,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'inference_time_ms': self.inference_time_ms,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class Prediction:
    """Model prediction result"""
    model_id: str
    timestamp: datetime
    predictions: Dict[str, float]
    confidence: PredictionConfidence
    confidence_scores: Dict[str, float]
    horizon_minutes: int
    explanation: str = ""
    contributing_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'predictions': self.predictions,
            'confidence': self.confidence.value,
            'confidence_scores': self.confidence_scores,
            'horizon_minutes': self.horizon_minutes,
            'explanation': self.explanation,
            'contributing_factors': self.contributing_factors
        }


class NeuralNetworkSimulator:
    """
    Simulated neural network for demonstration
    (In production, would use PyTorch/TensorFlow)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.weights: Dict[str, List[float]] = {}
        self.biases: Dict[str, float] = {}
        self.training_history: List[Dict[str, float]] = []
        self.status = TrainingStatus.NOT_TRAINED
        self.epochs_trained = 0

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights (Xavier initialization)"""
        layer_sizes = [len(self.config.input_features)] + \
                      self.config.hidden_layers + \
                      [len(self.config.output_features)]

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = math.sqrt(2.0 / (fan_in + fan_out))

            self.weights[f'layer_{i}'] = [
                random.gauss(0, scale) for _ in range(fan_in * fan_out)
            ]
            self.biases[f'layer_{i}'] = random.gauss(0, 0.1)

    def _activate(self, x: float) -> float:
        """Apply activation function"""
        if self.config.activation == 'relu':
            return max(0, x)
        elif self.config.activation == 'tanh':
            return math.tanh(x)
        elif self.config.activation == 'sigmoid':
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        else:
            return x

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through network"""
        current = inputs

        layer_sizes = [len(self.config.input_features)] + \
                      self.config.hidden_layers + \
                      [len(self.config.output_features)]

        for i in range(len(layer_sizes) - 1):
            weights = self.weights[f'layer_{i}']
            bias = self.biases[f'layer_{i}']

            next_size = layer_sizes[i + 1]
            next_layer = []

            for j in range(next_size):
                value = bias
                for k, inp in enumerate(current):
                    w_idx = j * len(current) + k
                    if w_idx < len(weights):
                        value += inp * weights[w_idx]

                # Apply activation (except last layer)
                if i < len(layer_sizes) - 2:
                    value = self._activate(value)

                next_layer.append(value)

            current = next_layer

        return current

    def train_step(self, inputs: List[List[float]], targets: List[List[float]]) -> float:
        """Perform one training step (simplified gradient descent)"""
        total_loss = 0.0
        lr = self.config.learning_rate

        for inp, target in zip(inputs, targets):
            # Forward pass
            output = self.forward(inp)

            # Compute loss
            loss = sum((o - t) ** 2 for o, t in zip(output, target)) / len(output)
            total_loss += loss

            # Simplified weight update (gradient approximation)
            for key in self.weights:
                for i in range(len(self.weights[key])):
                    # Add small random perturbation to escape local minima
                    gradient_approx = loss * random.gauss(0, 0.1)
                    self.weights[key][i] -= lr * gradient_approx

                self.biases[key] -= lr * loss * 0.1

        avg_loss = total_loss / len(inputs) if inputs else 0
        self.training_history.append({
            'epoch': self.epochs_trained,
            'loss': avg_loss
        })

        return avg_loss


class DeepLearningPredictor:
    """
    Deep learning-based predictor for system state
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "ml"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Models for different prediction tasks
        self.models: Dict[str, NeuralNetworkSimulator] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}

        # Data buffers
        self.input_buffer: deque = deque(maxlen=1000)
        self.sequence_buffer: deque = deque(maxlen=100)

        # Feature normalization
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        # Initialize default models
        self._init_default_models()

    def _init_default_models(self):
        """Initialize default prediction models"""
        # Water level predictor
        self._create_model(ModelConfig(
            model_id='water_level_lstm',
            model_type=ModelType.LSTM,
            input_features=['h', 'Q_in', 'Q_out', 'v', 'fr'],
            output_features=['h_pred'],
            hidden_layers=[64, 32],
            sequence_length=20,
            learning_rate=0.001
        ))

        # Flow rate predictor
        self._create_model(ModelConfig(
            model_id='flow_predictor',
            model_type=ModelType.GRU,
            input_features=['Q_in', 'Q_out', 'h', 'gate_position'],
            output_features=['Q_in_pred', 'Q_out_pred'],
            hidden_layers=[32, 16],
            sequence_length=10,
            learning_rate=0.001
        ))

        # Temperature predictor
        self._create_model(ModelConfig(
            model_id='thermal_predictor',
            model_type=ModelType.MLP,
            input_features=['T_sun', 'T_shade', 'hour', 'solar_radiation'],
            output_features=['T_sun_pred', 'T_shade_pred', 'thermal_bending_pred'],
            hidden_layers=[32, 16, 8],
            learning_rate=0.001
        ))

        # Scenario classifier
        self._create_model(ModelConfig(
            model_id='scenario_classifier',
            model_type=ModelType.MLP,
            input_features=['h', 'v', 'fr', 'T_sun', 'T_shade', 'vib_amp',
                          'joint_gap', 'bearing_stress', 'ground_accel'],
            output_features=['scenario_prob_hydraulic', 'scenario_prob_thermal',
                          'scenario_prob_structural', 'scenario_prob_seismic'],
            hidden_layers=[64, 32, 16],
            activation='sigmoid',
            learning_rate=0.001
        ))

        # Anomaly detector (autoencoder)
        self._create_model(ModelConfig(
            model_id='anomaly_detector',
            model_type=ModelType.AUTOENCODER,
            input_features=['h', 'v', 'Q_in', 'Q_out', 'T_sun', 'T_shade',
                          'vib_amp', 'joint_gap', 'bearing_stress'],
            output_features=['h', 'v', 'Q_in', 'Q_out', 'T_sun', 'T_shade',
                           'vib_amp', 'joint_gap', 'bearing_stress'],
            hidden_layers=[16, 8, 4, 8, 16],  # Bottleneck architecture
            learning_rate=0.001
        ))

    def _create_model(self, config: ModelConfig):
        """Create and register a model"""
        self.model_configs[config.model_id] = config
        self.models[config.model_id] = NeuralNetworkSimulator(config)
        self.model_metrics[config.model_id] = ModelMetrics(model_id=config.model_id)

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using stored statistics"""
        normalized = {}
        for key, value in features.items():
            mean = self.feature_means.get(key, 0)
            std = self.feature_stds.get(key, 1)
            normalized[key] = (value - mean) / std if std > 0 else 0
        return normalized

    def _update_statistics(self, features: Dict[str, float]):
        """Update running statistics for normalization"""
        alpha = 0.01  # Exponential smoothing factor

        for key, value in features.items():
            if key not in self.feature_means:
                self.feature_means[key] = value
                self.feature_stds[key] = 1.0
            else:
                old_mean = self.feature_means[key]
                self.feature_means[key] = alpha * value + (1 - alpha) * old_mean
                variance = alpha * (value - old_mean) ** 2 + (1 - alpha) * self.feature_stds[key] ** 2
                self.feature_stds[key] = math.sqrt(max(variance, 0.01))

    def add_observation(self, state: Dict[str, Any]):
        """Add observation to training buffer"""
        self._update_statistics(state)
        self.input_buffer.append({
            'timestamp': datetime.now(),
            'state': state
        })

        # Build sequence if enough data
        if len(self.input_buffer) >= 20:
            sequence = [obs['state'] for obs in list(self.input_buffer)[-20:]]
            self.sequence_buffer.append(sequence)

    def predict(self, model_id: str, current_state: Dict[str, Any],
                horizon_minutes: int = 30) -> Optional[Prediction]:
        """Make prediction using specified model"""
        if model_id not in self.models:
            return None

        model = self.models[model_id]
        config = self.model_configs[model_id]

        # Prepare input features
        normalized_state = self._normalize_features(current_state)
        inputs = [normalized_state.get(f, 0) for f in config.input_features]

        # Forward pass
        start_time = time.time()
        outputs = model.forward(inputs)
        inference_time = (time.time() - start_time) * 1000

        # Build predictions dict
        predictions = {}
        for i, feature in enumerate(config.output_features):
            if i < len(outputs):
                # Denormalize output
                mean = self.feature_means.get(feature.replace('_pred', ''), 0)
                std = self.feature_stds.get(feature.replace('_pred', ''), 1)
                predictions[feature] = outputs[i] * std + mean

        # Calculate confidence
        confidence_scores = {}
        for feature in config.output_features:
            # Confidence based on model performance
            base_confidence = 0.8 if model.status == TrainingStatus.TRAINED else 0.5
            confidence_scores[feature] = base_confidence

        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0

        if avg_confidence >= 0.8:
            confidence = PredictionConfidence.HIGH
        elif avg_confidence >= 0.6:
            confidence = PredictionConfidence.MEDIUM
        elif avg_confidence >= 0.4:
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.UNCERTAIN

        # Feature importance (simplified)
        contributing_factors = {}
        for i, feature in enumerate(config.input_features):
            if i < len(inputs):
                contributing_factors[feature] = abs(inputs[i]) / (sum(abs(x) for x in inputs) + 1e-6)

        # Update metrics
        self.model_metrics[model_id].inference_time_ms = inference_time

        return Prediction(
            model_id=model_id,
            timestamp=datetime.now(),
            predictions=predictions,
            confidence=confidence,
            confidence_scores=confidence_scores,
            horizon_minutes=horizon_minutes,
            explanation=self._generate_explanation(model_id, predictions, contributing_factors),
            contributing_factors=contributing_factors
        )

    def _generate_explanation(self, model_id: str, predictions: Dict[str, float],
                             factors: Dict[str, float]) -> str:
        """Generate human-readable explanation"""
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]

        explanation = f"Prediction by {model_id}: "
        for feature, value in predictions.items():
            explanation += f"{feature}={value:.2f}, "

        explanation += "\nTop contributing factors: "
        for feature, importance in top_factors:
            explanation += f"{feature} ({importance*100:.1f}%), "

        return explanation.rstrip(", ")

    def train_model(self, model_id: str, epochs: int = 100) -> Dict[str, Any]:
        """Train a specific model"""
        if model_id not in self.models:
            return {'error': 'Model not found'}

        model = self.models[model_id]
        config = self.model_configs[model_id]
        model.status = TrainingStatus.TRAINING

        # Prepare training data from buffer
        if len(self.input_buffer) < config.batch_size * 2:
            model.status = TrainingStatus.NOT_TRAINED
            return {'error': 'Insufficient training data'}

        # Create training batches
        training_losses = []

        for epoch in range(epochs):
            # Sample batch
            batch_indices = random.sample(range(len(self.input_buffer) - 1),
                                         min(config.batch_size, len(self.input_buffer) - 1))

            inputs = []
            targets = []

            for idx in batch_indices:
                obs = self.input_buffer[idx]
                next_obs = self.input_buffer[idx + 1]

                # Normalize and extract features
                inp = [self._normalize_features(obs['state']).get(f, 0)
                      for f in config.input_features]
                tgt = [self._normalize_features(next_obs['state']).get(f.replace('_pred', ''), 0)
                      for f in config.output_features]

                inputs.append(inp)
                targets.append(tgt)

            # Training step
            loss = model.train_step(inputs, targets)
            training_losses.append(loss)
            model.epochs_trained += 1

        model.status = TrainingStatus.TRAINED

        # Update metrics
        metrics = self.model_metrics[model_id]
        metrics.training_loss = training_losses[-1] if training_losses else 0
        metrics.last_updated = datetime.now()

        return {
            'model_id': model_id,
            'epochs': epochs,
            'final_loss': training_losses[-1] if training_losses else 0,
            'training_history': training_losses[-10:],  # Last 10 losses
            'status': model.status.value
        }

    def detect_anomaly(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using autoencoder"""
        prediction = self.predict('anomaly_detector', state)
        if not prediction:
            return {'is_anomaly': False, 'score': 0}

        # Calculate reconstruction error
        total_error = 0
        errors = {}

        for feature in self.model_configs['anomaly_detector'].input_features:
            original = state.get(feature, 0)
            reconstructed = prediction.predictions.get(feature, original)
            error = abs(original - reconstructed) / (abs(original) + 1e-6)
            errors[feature] = error
            total_error += error ** 2

        anomaly_score = math.sqrt(total_error / len(errors)) if errors else 0
        threshold = 0.5

        return {
            'is_anomaly': anomaly_score > threshold,
            'anomaly_score': anomaly_score,
            'threshold': threshold,
            'feature_errors': errors,
            'top_anomalies': sorted(errors.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information"""
        if model_id not in self.models:
            return {'error': 'Model not found'}

        model = self.models[model_id]
        config = self.model_configs[model_id]
        metrics = self.model_metrics[model_id]

        return {
            'config': config.to_dict(),
            'metrics': metrics.to_dict(),
            'status': model.status.value,
            'epochs_trained': model.epochs_trained,
            'training_history_size': len(model.training_history)
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models"""
        return [self.get_model_info(model_id) for model_id in self.models]


class ReinforcementLearningController:
    """
    Reinforcement learning-based controller
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "rl"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # State and action spaces
        self.state_dim = 9  # h, v, Q_in, Q_out, T_sun, T_shade, vib_amp, joint_gap, bearing_stress
        self.action_dim = 2  # Q_in_target, Q_out_target

        # Q-network parameters (simplified tabular Q-learning for demo)
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1  # Exploration rate

        # Experience replay
        self.replay_buffer: deque = deque(maxlen=10000)

        # Training statistics
        self.episodes_trained = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []

        # Policy parameters
        self.action_bounds = {
            'Q_in': (0, 150),
            'Q_out': (0, 200)
        }

    def _discretize_state(self, state: Dict[str, Any]) -> str:
        """Discretize continuous state for tabular Q-learning"""
        # Simple discretization
        h_bin = int(state.get('h', 4.5) / 0.5)
        Q_in_bin = int(state.get('Q_in', 85) / 20)
        Q_out_bin = int(state.get('Q_out', 85) / 20)
        risk_bin = 0 if state.get('risk_level', 'LOW') == 'LOW' else 1

        return f"{h_bin}_{Q_in_bin}_{Q_out_bin}_{risk_bin}"

    def _discretize_action(self, Q_in: float, Q_out: float) -> str:
        """Discretize continuous action"""
        Q_in_bin = int(Q_in / 20)
        Q_out_bin = int(Q_out / 20)
        return f"{Q_in_bin}_{Q_out_bin}"

    def _continuous_action(self, action_key: str) -> Tuple[float, float]:
        """Convert discrete action to continuous"""
        parts = action_key.split('_')
        Q_in = int(parts[0]) * 20 + 10
        Q_out = int(parts[1]) * 20 + 10
        return Q_in, Q_out

    def get_action(self, state: Dict[str, Any], explore: bool = True) -> Dict[str, float]:
        """Select action using epsilon-greedy policy"""
        state_key = self._discretize_state(state)

        # Initialize Q-values if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            for q_in_bin in range(8):
                for q_out_bin in range(10):
                    action_key = f"{q_in_bin}_{q_out_bin}"
                    self.q_table[state_key][action_key] = 0.0

        # Epsilon-greedy action selection
        if explore and random.random() < self.epsilon:
            # Explore: random action
            Q_in = random.uniform(*self.action_bounds['Q_in'])
            Q_out = random.uniform(*self.action_bounds['Q_out'])
        else:
            # Exploit: best action
            best_action = max(self.q_table[state_key].items(),
                            key=lambda x: x[1])
            Q_in, Q_out = self._continuous_action(best_action[0])

        return {
            'Q_in': Q_in,
            'Q_out': Q_out,
            'exploration': explore and random.random() < self.epsilon
        }

    def _calculate_reward(self, state: Dict[str, Any], action: Dict[str, float],
                         next_state: Dict[str, Any]) -> float:
        """Calculate reward for state transition"""
        reward = 0.0

        # Target water level
        h_target = 4.5
        h_error = abs(next_state.get('h', 4.5) - h_target)
        reward -= h_error * 10  # Penalize deviation

        # Flow balance
        Q_diff = abs(action.get('Q_in', 85) - action.get('Q_out', 85))
        reward -= Q_diff * 0.1

        # Safety bonus
        risk_level = next_state.get('risk_level', 'LOW')
        if risk_level == 'LOW':
            reward += 5
        elif risk_level == 'MEDIUM':
            reward += 0
        else:
            reward -= 20

        # Froude number constraint
        fr = next_state.get('fr', 0.3)
        if fr > 0.9:
            reward -= 30  # Critical Froude

        # Energy efficiency (minimize large flow changes)
        Q_in_change = abs(action.get('Q_in', 85) - state.get('Q_in', 85))
        reward -= Q_in_change * 0.05

        return reward

    def store_experience(self, state: Dict[str, Any], action: Dict[str, float],
                        reward: float, next_state: Dict[str, Any], done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        self.total_reward += reward

    def train_step(self, batch_size: int = 32) -> float:
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.replay_buffer), batch_size)

        total_loss = 0.0

        for exp in batch:
            state_key = self._discretize_state(exp['state'])
            action_key = self._discretize_action(exp['action']['Q_in'], exp['action']['Q_out'])
            next_state_key = self._discretize_state(exp['next_state'])

            # Initialize if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            if action_key not in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0

            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {}

            # Q-learning update
            current_q = self.q_table[state_key][action_key]

            if exp['done']:
                target_q = exp['reward']
            else:
                max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
                target_q = exp['reward'] + self.discount_factor * max_next_q

            # Update Q-value
            td_error = target_q - current_q
            self.q_table[state_key][action_key] += self.learning_rate * td_error

            total_loss += td_error ** 2

        return total_loss / batch_size

    def update(self, state: Dict[str, Any], action: Dict[str, float],
               next_state: Dict[str, Any], done: bool = False) -> Dict[str, Any]:
        """Full update cycle: calculate reward, store experience, train"""
        reward = self._calculate_reward(state, action, next_state)
        self.store_experience(state, action, reward, next_state, done)

        loss = self.train_step()

        return {
            'reward': reward,
            'loss': loss,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer)
        }

    def get_policy_info(self) -> Dict[str, Any]:
        """Get policy information"""
        return {
            'episodes_trained': self.episodes_trained,
            'total_reward': self.total_reward,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'replay_buffer_size': len(self.replay_buffer),
            'recent_rewards': self.episode_rewards[-10:] if self.episode_rewards else []
        }


class MLControlManager:
    """
    Main ML control management system
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "ml"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.predictor = DeepLearningPredictor(str(self.data_dir / "predictor"))
        self.rl_controller = ReinforcementLearningController(str(self.data_dir / "rl"))

        # State
        self.running = False
        self.ml_enabled = True
        self.rl_enabled = False  # Start with RL disabled until trained

        # Performance tracking
        self.prediction_history: deque = deque(maxlen=1000)
        self.control_history: deque = deque(maxlen=1000)

        # Threading
        self.lock = threading.Lock()
        self.training_thread = None

    def start(self):
        """Start ML control manager"""
        self.running = True
        print("[MLControl] Started")

    def stop(self):
        """Stop ML control manager"""
        self.running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state update - predictions and control"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'anomaly': {},
            'rl_action': {},
            'recommendations': []
        }

        # Add observation to predictor
        self.predictor.add_observation(state)

        # Generate predictions
        if self.ml_enabled:
            # Water level prediction
            pred = self.predictor.predict('water_level_lstm', state, horizon_minutes=30)
            if pred:
                results['predictions']['water_level'] = pred.to_dict()

            # Scenario classification
            pred = self.predictor.predict('scenario_classifier', state)
            if pred:
                results['predictions']['scenarios'] = pred.to_dict()

            # Anomaly detection
            anomaly = self.predictor.detect_anomaly(state)
            results['anomaly'] = anomaly

            if anomaly.get('is_anomaly'):
                results['recommendations'].append({
                    'type': 'ANOMALY_ALERT',
                    'message': f"Anomaly detected (score: {anomaly['anomaly_score']:.2f})",
                    'severity': 'HIGH'
                })

        # RL control action
        if self.rl_enabled:
            action = self.rl_controller.get_action(state, explore=False)
            results['rl_action'] = action

            # Generate recommendation based on RL
            current_Q_in = state.get('Q_in', 85)
            suggested_Q_in = action['Q_in']

            if abs(suggested_Q_in - current_Q_in) > 10:
                results['recommendations'].append({
                    'type': 'FLOW_ADJUSTMENT',
                    'message': f"RL suggests adjusting Q_in from {current_Q_in:.1f} to {suggested_Q_in:.1f}",
                    'severity': 'MEDIUM'
                })

        # Store history
        with self.lock:
            self.prediction_history.append(results)

        return results

    def update_rl(self, state: Dict[str, Any], action: Dict[str, float],
                  next_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update RL controller with transition"""
        if not self.rl_enabled:
            return {'message': 'RL not enabled'}

        return self.rl_controller.update(state, action, next_state)

    def train_models(self, epochs: int = 100) -> Dict[str, Any]:
        """Train all prediction models"""
        results = {}

        for model_id in self.predictor.models:
            result = self.predictor.train_model(model_id, epochs)
            results[model_id] = result

        return results

    def enable_rl_control(self, enable: bool = True):
        """Enable or disable RL control"""
        self.rl_enabled = enable

    def get_prediction(self, model_id: str, state: Dict[str, Any],
                      horizon: int = 30) -> Optional[Dict[str, Any]]:
        """Get prediction from specific model"""
        pred = self.predictor.predict(model_id, state, horizon)
        return pred.to_dict() if pred else None

    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of all models"""
        return self.predictor.list_models()

    def get_status(self) -> Dict[str, Any]:
        """Get ML control status"""
        return {
            'running': self.running,
            'ml_enabled': self.ml_enabled,
            'rl_enabled': self.rl_enabled,
            'models': len(self.predictor.models),
            'observations': len(self.predictor.input_buffer),
            'predictions_made': len(self.prediction_history),
            'rl_info': self.rl_controller.get_policy_info(),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_ml_manager = None


def get_ml_manager() -> MLControlManager:
    """Get global ML control manager"""
    global _ml_manager
    if _ml_manager is None:
        _ml_manager = MLControlManager()
    return _ml_manager


if __name__ == "__main__":
    # Test ML control
    print("=== ML Control Test ===")

    manager = MLControlManager()
    manager.start()

    # Simulate some observations
    print("\n1. Adding observations...")
    for i in range(50):
        state = {
            'h': 4.5 + 0.1 * math.sin(i * 0.1),
            'v': 2.0 + 0.05 * math.cos(i * 0.1),
            'Q_in': 85 + random.uniform(-5, 5),
            'Q_out': 85 + random.uniform(-5, 5),
            'T_sun': 28 + random.uniform(-2, 2),
            'T_shade': 22 + random.uniform(-1, 1),
            'vib_amp': 2.0 + random.uniform(-0.5, 0.5),
            'joint_gap': 20 + random.uniform(-1, 1),
            'bearing_stress': 25 + random.uniform(-3, 3),
            'fr': 0.32,
            'risk_level': 'LOW',
            'gate_position': 0.5,
            'hour': i % 24,
            'solar_radiation': 500 + random.uniform(-100, 100)
        }
        manager.predictor.add_observation(state)

    # Train models
    print("\n2. Training models...")
    train_results = manager.train_models(epochs=20)
    for model_id, result in train_results.items():
        print(f"   {model_id}: loss={result.get('final_loss', 'N/A'):.4f}")

    # Make predictions
    print("\n3. Making predictions...")
    current_state = {
        'h': 4.6,
        'v': 2.1,
        'Q_in': 88,
        'Q_out': 82,
        'T_sun': 30,
        'T_shade': 23,
        'vib_amp': 2.5,
        'joint_gap': 21,
        'bearing_stress': 26,
        'fr': 0.33,
        'risk_level': 'LOW',
        'gate_position': 0.55,
        'hour': 14,
        'solar_radiation': 600
    }

    results = manager.process_state(current_state)
    print(f"   Anomaly detected: {results['anomaly'].get('is_anomaly', False)}")
    print(f"   Anomaly score: {results['anomaly'].get('anomaly_score', 0):.3f}")

    # Scenario classification
    scenario_pred = manager.get_prediction('scenario_classifier', current_state)
    if scenario_pred:
        print(f"   Scenario probabilities: {scenario_pred['predictions']}")

    # Test RL
    print("\n4. Testing RL Controller:")
    manager.enable_rl_control(True)

    for i in range(10):
        action = manager.rl_controller.get_action(current_state, explore=True)
        next_state = dict(current_state)
        next_state['h'] += (action['Q_in'] - action['Q_out']) * 0.001
        next_state['Q_in'] = action['Q_in']
        next_state['Q_out'] = action['Q_out']

        update_result = manager.update_rl(current_state, action, next_state)
        current_state = next_state

    print(f"   RL buffer size: {manager.rl_controller.get_policy_info()['replay_buffer_size']}")
    print(f"   Total reward: {manager.rl_controller.get_policy_info()['total_reward']:.2f}")

    # Status
    print("\n5. System Status:")
    status = manager.get_status()
    print(f"   Models: {status['models']}")
    print(f"   Observations: {status['observations']}")
    print(f"   ML enabled: {status['ml_enabled']}")
    print(f"   RL enabled: {status['rl_enabled']}")

    manager.stop()
    print("\nML Control test completed!")
