"""
TAOS Intelligence Enhancement Module
智能化提升模块 - 机器学习预测、异常检测、自适应优化

Features:
- Time series prediction (ARIMA-like, Exponential Smoothing)
- Anomaly detection (Statistical, Isolation-based)
- Pattern recognition
- Adaptive parameter optimization
- Scenario transition prediction
- Reinforcement learning-based control optimization
"""

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics as stats


class PredictionMethod(Enum):
    """预测方法"""
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    LINEAR_REGRESSION = "linear_regression"
    ARIMA_SIMPLE = "arima_simple"


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"               # 突增
    DROP = "drop"                 # 骤降
    DRIFT = "drift"               # 漂移
    OSCILLATION = "oscillation"   # 振荡
    FLATLINE = "flatline"         # 僵值
    OUT_OF_RANGE = "out_of_range" # 超范围


@dataclass
class Prediction:
    """预测结果"""
    timestamp: datetime
    variable: str
    value: float
    confidence: float           # 置信度 0-1
    lower_bound: float          # 置信区间下限
    upper_bound: float          # 置信区间上限
    method: PredictionMethod
    horizon_minutes: int        # 预测时域


@dataclass
class Anomaly:
    """异常检测结果"""
    timestamp: datetime
    variable: str
    value: float
    expected_value: float
    deviation: float            # 偏差量
    anomaly_type: AnomalyType
    severity: float             # 严重程度 0-1
    confidence: float           # 检测置信度


@dataclass
class Pattern:
    """模式识别结果"""
    pattern_id: str
    description: str
    variables: List[str]
    frequency: float            # 出现频率
    typical_duration: float     # 典型持续时间
    associated_scenarios: List[str]
    trigger_conditions: Dict[str, Tuple[float, float]]  # 触发条件范围


@dataclass
class OptimizationResult:
    """优化结果"""
    timestamp: datetime
    parameter: str
    old_value: float
    new_value: float
    improvement: float
    method: str


class TimeSeriesPredictor:
    """时间序列预测器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, deque] = {}
        self.models: Dict[str, Dict[str, Any]] = {}

    def update(self, variable: str, value: float, timestamp: datetime = None):
        """更新历史数据"""
        if variable not in self.history:
            self.history[variable] = deque(maxlen=self.window_size)

        self.history[variable].append({
            'value': value,
            'timestamp': timestamp or datetime.now()
        })

    def predict(self, variable: str, horizon_minutes: int = 5,
                method: PredictionMethod = PredictionMethod.EXPONENTIAL_SMOOTHING) -> Optional[Prediction]:
        """预测未来值"""
        if variable not in self.history or len(self.history[variable]) < 3:
            return None

        values = [h['value'] for h in self.history[variable]]

        if method == PredictionMethod.EXPONENTIAL_SMOOTHING:
            predicted, confidence = self._exponential_smoothing(values, horizon_minutes)
        elif method == PredictionMethod.MOVING_AVERAGE:
            predicted, confidence = self._moving_average(values, horizon_minutes)
        elif method == PredictionMethod.LINEAR_REGRESSION:
            predicted, confidence = self._linear_regression(values, horizon_minutes)
        else:
            predicted, confidence = self._simple_arima(values, horizon_minutes)

        # 计算置信区间
        std = stats.stdev(values) if len(values) > 1 else 0.1
        margin = 1.96 * std * (1 + 0.1 * horizon_minutes)  # 扩展随预测时域增加

        return Prediction(
            timestamp=datetime.now() + timedelta(minutes=horizon_minutes),
            variable=variable,
            value=predicted,
            confidence=confidence,
            lower_bound=predicted - margin,
            upper_bound=predicted + margin,
            method=method,
            horizon_minutes=horizon_minutes
        )

    def _exponential_smoothing(self, values: List[float],
                                horizon: int) -> Tuple[float, float]:
        """指数平滑预测"""
        alpha = 0.3  # 平滑系数
        beta = 0.1   # 趋势系数

        # 双指数平滑
        level = values[0]
        trend = 0

        for v in values[1:]:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # 预测
        predicted = level + trend * horizon

        # 置信度随预测时域降低
        confidence = max(0.5, 1.0 - 0.05 * horizon)

        return predicted, confidence

    def _moving_average(self, values: List[float],
                        horizon: int) -> Tuple[float, float]:
        """移动平均预测"""
        window = min(20, len(values))
        recent = values[-window:]
        predicted = sum(recent) / len(recent)

        # 简单趋势
        if len(values) >= 10:
            early = sum(values[-20:-10]) / 10
            late = sum(values[-10:]) / 10
            trend = (late - early) / 10
            predicted += trend * horizon

        confidence = max(0.4, 0.9 - 0.03 * horizon)
        return predicted, confidence

    def _linear_regression(self, values: List[float],
                           horizon: int) -> Tuple[float, float]:
        """线性回归预测"""
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return y_mean, 0.6

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # 预测
        predicted = slope * (n - 1 + horizon * 2) + intercept  # 假设2个采样点/分钟

        # R² 作为置信度
        ss_tot = sum((v - y_mean) ** 2 for v in values)
        ss_res = sum((v - (slope * i + intercept)) ** 2 for i, v in enumerate(values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.5

        confidence = max(0.3, r_squared * (1 - 0.02 * horizon))
        return predicted, confidence

    def _simple_arima(self, values: List[float],
                      horizon: int) -> Tuple[float, float]:
        """简化ARIMA预测 (AR(1)模型)"""
        if len(values) < 5:
            return values[-1], 0.5

        # 差分
        diff = [values[i] - values[i-1] for i in range(1, len(values))]

        # AR(1)系数
        if len(diff) > 1:
            autocorr = sum(diff[i] * diff[i-1] for i in range(1, len(diff)))
            variance = sum(d ** 2 for d in diff[:-1])
            phi = autocorr / variance if variance > 0 else 0
            phi = max(-0.95, min(0.95, phi))  # 限制系数范围
        else:
            phi = 0

        # 预测差分
        last_diff = diff[-1]
        predicted_diff = phi * last_diff

        # 还原
        predicted = values[-1] + predicted_diff * horizon

        confidence = max(0.4, 0.85 - 0.04 * horizon)
        return predicted, confidence

    def get_trend(self, variable: str, window: int = 20) -> Dict[str, Any]:
        """获取趋势信息"""
        if variable not in self.history or len(self.history[variable]) < window:
            return {'trend': 'unknown', 'slope': 0, 'stability': 0}

        values = [h['value'] for h in self.history[variable]][-window:]

        # 计算斜率
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0

        # 稳定性
        std = stats.stdev(values) if n > 1 else 0
        mean = abs(y_mean) if y_mean != 0 else 1
        cv = std / mean  # 变异系数

        if abs(slope) < 0.01 * mean:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        return {
            'trend': trend,
            'slope': slope,
            'stability': 1 - min(1, cv),
            'mean': y_mean,
            'std': std
        }


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.history: Dict[str, deque] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}

        # 默认阈值
        self.default_thresholds = {
            'h': {'min': 1.5, 'max': 7.0, 'rate': 0.5},
            'fr': {'min': 0.0, 'max': 1.0, 'rate': 0.1},
            'T_sun': {'min': -20, 'max': 60, 'rate': 5},
            'T_shade': {'min': -20, 'max': 50, 'rate': 5},
            'vib_amp': {'min': 0, 'max': 100, 'rate': 20},
            'joint_gap': {'min': 5, 'max': 40, 'rate': 2},
            'Q_in': {'min': 0, 'max': 150, 'rate': 20},
            'Q_out': {'min': 0, 'max': 150, 'rate': 20}
        }

    def update(self, variable: str, value: float, timestamp: datetime = None):
        """更新历史数据"""
        if variable not in self.history:
            self.history[variable] = deque(maxlen=500)

        self.history[variable].append({
            'value': value,
            'timestamp': timestamp or datetime.now()
        })

        # 更新基线
        self._update_baseline(variable)

    def _update_baseline(self, variable: str):
        """更新基线统计"""
        if variable not in self.history or len(self.history[variable]) < 10:
            return

        values = [h['value'] for h in self.history[variable]]
        self.baselines[variable] = {
            'mean': stats.mean(values),
            'std': stats.stdev(values) if len(values) > 1 else 1.0,
            'median': stats.median(values),
            'min': min(values),
            'max': max(values)
        }

    def detect(self, variable: str, value: float,
               timestamp: datetime = None) -> Optional[Anomaly]:
        """检测异常"""
        anomalies = []

        # 1. 超范围检测
        anomaly = self._check_range(variable, value, timestamp)
        if anomaly:
            anomalies.append(anomaly)

        # 2. 统计异常检测
        anomaly = self._check_statistical(variable, value, timestamp)
        if anomaly:
            anomalies.append(anomaly)

        # 3. 变化率异常检测
        anomaly = self._check_rate(variable, value, timestamp)
        if anomaly:
            anomalies.append(anomaly)

        # 4. 僵值检测
        anomaly = self._check_flatline(variable, timestamp)
        if anomaly:
            anomalies.append(anomaly)

        # 返回最严重的异常
        if anomalies:
            return max(anomalies, key=lambda a: a.severity)
        return None

    def _check_range(self, variable: str, value: float,
                     timestamp: datetime = None) -> Optional[Anomaly]:
        """超范围检测"""
        thresholds = self.thresholds.get(variable, self.default_thresholds.get(variable))
        if not thresholds:
            return None

        min_val = thresholds.get('min', float('-inf'))
        max_val = thresholds.get('max', float('inf'))

        if value < min_val:
            deviation = min_val - value
            severity = min(1.0, deviation / max(abs(min_val), 1) * self.sensitivity)
            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=value,
                expected_value=min_val,
                deviation=deviation,
                anomaly_type=AnomalyType.OUT_OF_RANGE,
                severity=severity,
                confidence=0.95
            )
        elif value > max_val:
            deviation = value - max_val
            severity = min(1.0, deviation / max(abs(max_val), 1) * self.sensitivity)
            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=value,
                expected_value=max_val,
                deviation=deviation,
                anomaly_type=AnomalyType.OUT_OF_RANGE,
                severity=severity,
                confidence=0.95
            )
        return None

    def _check_statistical(self, variable: str, value: float,
                           timestamp: datetime = None) -> Optional[Anomaly]:
        """统计异常检测 (Z-score)"""
        if variable not in self.baselines:
            return None

        baseline = self.baselines[variable]
        mean = baseline['mean']
        std = baseline['std']

        if std == 0:
            return None

        z_score = abs(value - mean) / std

        # 阈值根据灵敏度调整
        threshold = 3.0 / self.sensitivity

        if z_score > threshold:
            anomaly_type = AnomalyType.SPIKE if value > mean else AnomalyType.DROP
            severity = min(1.0, (z_score - threshold) / threshold)

            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=value,
                expected_value=mean,
                deviation=value - mean,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=1 - math.exp(-z_score)
            )
        return None

    def _check_rate(self, variable: str, value: float,
                    timestamp: datetime = None) -> Optional[Anomaly]:
        """变化率异常检测"""
        if variable not in self.history or len(self.history[variable]) < 2:
            return None

        prev = self.history[variable][-1]['value']
        rate = abs(value - prev)

        thresholds = self.thresholds.get(variable, self.default_thresholds.get(variable))
        if not thresholds:
            return None

        rate_threshold = thresholds.get('rate', float('inf'))

        if rate > rate_threshold:
            anomaly_type = AnomalyType.SPIKE if value > prev else AnomalyType.DROP
            severity = min(1.0, (rate - rate_threshold) / rate_threshold * self.sensitivity)

            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=value,
                expected_value=prev,
                deviation=value - prev,
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=0.85
            )
        return None

    def _check_flatline(self, variable: str,
                        timestamp: datetime = None) -> Optional[Anomaly]:
        """僵值检测"""
        if variable not in self.history or len(self.history[variable]) < 30:
            return None

        recent = [h['value'] for h in list(self.history[variable])[-30:]]

        # 检查是否所有值相同
        unique_values = set(recent)
        if len(unique_values) == 1:
            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=recent[-1],
                expected_value=recent[-1],
                deviation=0,
                anomaly_type=AnomalyType.FLATLINE,
                severity=0.6 * self.sensitivity,
                confidence=0.9
            )

        # 检查变化是否太小
        std = stats.stdev(recent) if len(recent) > 1 else 0
        mean = abs(stats.mean(recent)) if stats.mean(recent) != 0 else 1
        cv = std / mean

        if cv < 0.001:  # 变异系数极小
            return Anomaly(
                timestamp=timestamp or datetime.now(),
                variable=variable,
                value=recent[-1],
                expected_value=stats.mean(recent),
                deviation=0,
                anomaly_type=AnomalyType.FLATLINE,
                severity=0.4 * self.sensitivity,
                confidence=0.7
            )
        return None

    def set_thresholds(self, variable: str, min_val: float = None,
                       max_val: float = None, rate: float = None):
        """设置自定义阈值"""
        if variable not in self.thresholds:
            self.thresholds[variable] = {}

        if min_val is not None:
            self.thresholds[variable]['min'] = min_val
        if max_val is not None:
            self.thresholds[variable]['max'] = max_val
        if rate is not None:
            self.thresholds[variable]['rate'] = rate

    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取异常统计摘要"""
        # 这个方法需要与持久化模块配合使用
        return {
            'variables_monitored': list(self.history.keys()),
            'baselines': self.baselines,
            'sensitivity': self.sensitivity
        }


class PatternRecognizer:
    """模式识别器"""

    def __init__(self):
        self.patterns: List[Pattern] = []
        self.sequence_history: deque = deque(maxlen=1000)
        self._init_predefined_patterns()

    def _init_predefined_patterns(self):
        """初始化预定义模式"""
        self.patterns = [
            Pattern(
                pattern_id="P1_HIGH_FLOW_CYCLE",
                description="高流量周期性波动",
                variables=['Q_in', 'h'],
                frequency=0.0,
                typical_duration=3600,
                associated_scenarios=['S1.1', 'S1.2'],
                trigger_conditions={
                    'Q_in': (100, 150),
                    'h': (5.0, 6.5)
                }
            ),
            Pattern(
                pattern_id="P2_THERMAL_STRESS",
                description="热应力累积模式",
                variables=['T_sun', 'T_shade', 'joint_gap'],
                frequency=0.0,
                typical_duration=7200,
                associated_scenarios=['S3.1', 'S3.3', 'S4.1'],
                trigger_conditions={
                    'T_delta': (10, 25),
                    'joint_gap': (25, 40)
                }
            ),
            Pattern(
                pattern_id="P3_SEISMIC_AFTEREFFECT",
                description="地震后效应模式",
                variables=['vib_amp', 'ground_accel', 'joint_gap'],
                frequency=0.0,
                typical_duration=1800,
                associated_scenarios=['S5.1', 'S5.2'],
                trigger_conditions={
                    'vib_amp': (30, 100),
                    'ground_accel': (0.1, 1.0)
                }
            ),
            Pattern(
                pattern_id="P4_NORMAL_OPERATION",
                description="正常运行模式",
                variables=['h', 'fr', 'Q_in', 'Q_out'],
                frequency=0.0,
                typical_duration=float('inf'),
                associated_scenarios=['NORMAL'],
                trigger_conditions={
                    'h': (3.5, 4.5),
                    'fr': (0, 0.6),
                    'Q_in': (70, 90),
                    'Q_out': (70, 90)
                }
            ),
            Pattern(
                pattern_id="P5_HYDRAULIC_INSTABILITY",
                description="水力不稳定模式",
                variables=['fr', 'v', 'h'],
                frequency=0.0,
                typical_duration=600,
                associated_scenarios=['S2.1', 'S2.2'],
                trigger_conditions={
                    'fr': (0.7, 1.0),
                    'h': (2.0, 3.0)
                }
            )
        ]

    def update(self, state: Dict[str, Any]):
        """更新状态序列"""
        self.sequence_history.append({
            'timestamp': datetime.now(),
            'state': state.copy()
        })

    def recognize(self, state: Dict[str, Any]) -> List[Pattern]:
        """识别当前匹配的模式"""
        matched = []

        for pattern in self.patterns:
            if self._check_pattern_match(state, pattern):
                matched.append(pattern)

        return matched

    def _check_pattern_match(self, state: Dict[str, Any],
                              pattern: Pattern) -> bool:
        """检查模式是否匹配"""
        for var, (low, high) in pattern.trigger_conditions.items():
            # 处理特殊变量
            if var == 'T_delta':
                value = abs(state.get('T_sun', 25) - state.get('T_shade', 25))
            else:
                value = state.get(var)

            if value is None:
                continue

            if not (low <= value <= high):
                return False

        return True

    def get_pattern_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """获取各模式的匹配概率"""
        probabilities = {}

        for pattern in self.patterns:
            prob = self._calculate_pattern_probability(state, pattern)
            probabilities[pattern.pattern_id] = prob

        return probabilities

    def _calculate_pattern_probability(self, state: Dict[str, Any],
                                        pattern: Pattern) -> float:
        """计算模式匹配概率"""
        match_scores = []

        for var, (low, high) in pattern.trigger_conditions.items():
            if var == 'T_delta':
                value = abs(state.get('T_sun', 25) - state.get('T_shade', 25))
            else:
                value = state.get(var)

            if value is None:
                match_scores.append(0.5)
                continue

            mid = (low + high) / 2
            range_width = (high - low) / 2

            if low <= value <= high:
                # 在范围内，计算距离中心的距离
                distance = abs(value - mid) / range_width
                score = 1.0 - 0.5 * distance
            else:
                # 在范围外，快速衰减
                if value < low:
                    distance = (low - value) / max(range_width, 1)
                else:
                    distance = (value - high) / max(range_width, 1)
                score = max(0, 0.5 * math.exp(-distance))

            match_scores.append(score)

        return sum(match_scores) / len(match_scores) if match_scores else 0.0


class AdaptiveOptimizer:
    """自适应参数优化器"""

    def __init__(self):
        self.parameter_history: Dict[str, List[Tuple[float, float]]] = {}  # param -> [(value, performance)]
        self.current_params: Dict[str, float] = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.2

        # 参数范围
        self.param_bounds = {
            'w_h': (0.5, 5.0),
            'w_fr': (0.5, 5.0),
            'w_T': (0.1, 2.0),
            'target_h': (3.5, 4.5),
            'Q_base': (70, 90),
            'kp': (0.1, 2.0),
            'ki': (0.01, 0.5),
            'kd': (0.0, 1.0)
        }

    def suggest_parameter(self, param_name: str,
                          current_performance: float) -> OptimizationResult:
        """建议参数调整"""
        current_value = self.current_params.get(param_name,
            (self.param_bounds.get(param_name, (0, 1))[0] +
             self.param_bounds.get(param_name, (0, 1))[1]) / 2
        )

        # 记录历史
        if param_name not in self.parameter_history:
            self.parameter_history[param_name] = []
        self.parameter_history[param_name].append((current_value, current_performance))

        # 决定探索还是利用
        if random.random() < self.exploration_rate:
            # 探索：随机扰动
            bounds = self.param_bounds.get(param_name, (0, 1))
            noise = (bounds[1] - bounds[0]) * 0.1 * (random.random() - 0.5)
            new_value = current_value + noise
        else:
            # 利用：基于历史梯度
            new_value = self._gradient_step(param_name, current_value)

        # 限制在范围内
        bounds = self.param_bounds.get(param_name, (float('-inf'), float('inf')))
        new_value = max(bounds[0], min(bounds[1], new_value))

        self.current_params[param_name] = new_value

        return OptimizationResult(
            timestamp=datetime.now(),
            parameter=param_name,
            old_value=current_value,
            new_value=new_value,
            improvement=0.0,  # 需要后续评估
            method='adaptive_gradient'
        )

    def _gradient_step(self, param_name: str, current_value: float) -> float:
        """梯度步进"""
        history = self.parameter_history.get(param_name, [])
        if len(history) < 3:
            return current_value

        # 估计梯度
        recent = history[-10:]
        values = [v for v, _ in recent]
        perfs = [p for _, p in recent]

        if len(set(values)) < 2:
            return current_value

        # 简单线性回归估计梯度
        v_mean = sum(values) / len(values)
        p_mean = sum(perfs) / len(perfs)

        numerator = sum((v - v_mean) * (p - p_mean) for v, p in zip(values, perfs))
        denominator = sum((v - v_mean) ** 2 for v in values)

        if denominator == 0:
            return current_value

        gradient = numerator / denominator

        # 梯度上升（假设性能越高越好）
        return current_value + self.learning_rate * gradient

    def evaluate_improvement(self, param_name: str,
                             new_performance: float) -> float:
        """评估改进效果"""
        history = self.parameter_history.get(param_name, [])
        if len(history) < 2:
            return 0.0

        old_performance = history[-2][1] if len(history) >= 2 else history[-1][1]
        improvement = new_performance - old_performance

        return improvement


class ReinforcementLearner:
    """强化学习控制器"""

    def __init__(self, state_dims: int = 8, action_dims: int = 2):
        self.state_dims = state_dims
        self.action_dims = action_dims

        # Q表 (离散化)
        self.q_table: Dict[Tuple, Dict[Tuple, float]] = {}

        # 参数
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

        # 状态离散化参数
        self.state_bins = {
            'h': [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
            'fr': [0.3, 0.5, 0.7, 0.85],
            'T_delta': [5, 10, 15, 20],
            'vib_amp': [10, 30, 50, 80]
        }

        # 动作空间
        self.actions = [
            (70, 70), (75, 75), (80, 80), (85, 85), (90, 90),
            (80, 70), (80, 90), (85, 80), (75, 80)
        ]

        # 经验回放
        self.experience_buffer: deque = deque(maxlen=10000)

    def discretize_state(self, state: Dict[str, Any]) -> Tuple:
        """将连续状态离散化"""
        h = state.get('h', 4.0)
        fr = state.get('fr', 0.5)
        T_delta = abs(state.get('T_sun', 25) - state.get('T_shade', 25))
        vib = state.get('vib_amp', 0)

        h_idx = self._bin_index(h, self.state_bins['h'])
        fr_idx = self._bin_index(fr, self.state_bins['fr'])
        T_idx = self._bin_index(T_delta, self.state_bins['T_delta'])
        vib_idx = self._bin_index(vib, self.state_bins['vib_amp'])

        return (h_idx, fr_idx, T_idx, vib_idx)

    def _bin_index(self, value: float, bins: List[float]) -> int:
        """获取离散化索引"""
        for i, threshold in enumerate(bins):
            if value < threshold:
                return i
        return len(bins)

    def select_action(self, state: Dict[str, Any]) -> Tuple[float, float]:
        """选择动作 (epsilon-greedy)"""
        discrete_state = self.discretize_state(state)

        if random.random() < self.epsilon:
            # 探索
            return random.choice(self.actions)
        else:
            # 利用
            if discrete_state not in self.q_table:
                return (80, 80)  # 默认动作

            action_values = self.q_table[discrete_state]
            if not action_values:
                return (80, 80)

            best_action = max(action_values.items(), key=lambda x: x[1])[0]
            return best_action

    def calculate_reward(self, state: Dict[str, Any],
                         next_state: Dict[str, Any]) -> float:
        """计算奖励"""
        reward = 0.0

        # 水位奖励
        h = next_state.get('h', 4.0)
        h_target = 4.0
        h_error = abs(h - h_target)
        reward -= h_error * 10

        # 弗劳德数惩罚
        fr = next_state.get('fr', 0.5)
        if fr > 0.8:
            reward -= (fr - 0.8) * 50
        elif fr > 0.6:
            reward -= (fr - 0.6) * 10

        # 温差惩罚
        T_delta = abs(next_state.get('T_sun', 25) - next_state.get('T_shade', 25))
        if T_delta > 15:
            reward -= (T_delta - 15) * 2

        # 振动惩罚
        vib = next_state.get('vib_amp', 0)
        if vib > 50:
            reward -= (vib - 50) * 0.5

        # 安全奖励
        if 3.5 <= h <= 4.5 and fr < 0.6 and T_delta < 10 and vib < 30:
            reward += 10

        return reward

    def update(self, state: Dict[str, Any], action: Tuple[float, float],
               reward: float, next_state: Dict[str, Any]):
        """更新Q值"""
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)

        # 初始化Q表条目
        if s not in self.q_table:
            self.q_table[s] = {}
        if action not in self.q_table[s]:
            self.q_table[s][action] = 0.0

        # 获取最大未来Q值
        if s_next in self.q_table and self.q_table[s_next]:
            max_q_next = max(self.q_table[s_next].values())
        else:
            max_q_next = 0.0

        # Q-learning更新
        old_q = self.q_table[s][action]
        new_q = old_q + self.learning_rate * (
            reward + self.discount_factor * max_q_next - old_q
        )
        self.q_table[s][action] = new_q

        # 存储经验
        self.experience_buffer.append((s, action, reward, s_next))

    def experience_replay(self, batch_size: int = 32):
        """经验回放训练"""
        if len(self.experience_buffer) < batch_size:
            return

        batch = random.sample(list(self.experience_buffer), batch_size)

        for s, action, reward, s_next in batch:
            if s not in self.q_table:
                self.q_table[s] = {}
            if action not in self.q_table[s]:
                self.q_table[s][action] = 0.0

            if s_next in self.q_table and self.q_table[s_next]:
                max_q_next = max(self.q_table[s_next].values())
            else:
                max_q_next = 0.0

            self.q_table[s][action] += self.learning_rate * (
                reward + self.discount_factor * max_q_next - self.q_table[s][action]
            )

    def get_policy_stats(self) -> Dict[str, Any]:
        """获取策略统计"""
        return {
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_buffer),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }


class IntelligenceManager:
    """智能化管理器 - 整合所有AI组件"""

    def __init__(self):
        self.predictor = TimeSeriesPredictor()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_recognizer = PatternRecognizer()
        self.optimizer = AdaptiveOptimizer()
        self.rl_agent = ReinforcementLearner()

        self.last_state: Dict[str, Any] = {}
        self.last_action: Tuple[float, float] = (80, 80)

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理状态更新，返回智能分析结果"""
        timestamp = datetime.now()
        results = {
            'predictions': {},
            'anomalies': [],
            'patterns': [],
            'rl_action': None
        }

        # 更新各组件
        for var in ['h', 'fr', 'Q_in', 'Q_out', 'T_sun', 'T_shade', 'vib_amp', 'joint_gap']:
            if var in state:
                self.predictor.update(var, state[var], timestamp)
                self.anomaly_detector.update(var, state[var], timestamp)

                # 预测
                pred = self.predictor.predict(var, horizon_minutes=5)
                if pred:
                    results['predictions'][var] = {
                        'value': pred.value,
                        'confidence': pred.confidence,
                        'lower': pred.lower_bound,
                        'upper': pred.upper_bound
                    }

                # 异常检测
                anomaly = self.anomaly_detector.detect(var, state[var], timestamp)
                if anomaly:
                    results['anomalies'].append({
                        'variable': var,
                        'type': anomaly.anomaly_type.value,
                        'severity': anomaly.severity,
                        'value': anomaly.value,
                        'expected': anomaly.expected_value
                    })

        # 模式识别
        self.pattern_recognizer.update(state)
        matched_patterns = self.pattern_recognizer.recognize(state)
        results['patterns'] = [p.pattern_id for p in matched_patterns]

        # 强化学习
        if self.last_state:
            reward = self.rl_agent.calculate_reward(self.last_state, state)
            self.rl_agent.update(self.last_state, self.last_action, reward, state)

        action = self.rl_agent.select_action(state)
        results['rl_action'] = {'Q_in': action[0], 'Q_out': action[1]}
        self.last_action = action
        self.last_state = state.copy()

        return results

    def get_summary(self) -> Dict[str, Any]:
        """获取智能系统摘要"""
        return {
            'predictor': {
                'variables_tracked': list(self.predictor.history.keys()),
                'window_size': self.predictor.window_size
            },
            'anomaly_detector': self.anomaly_detector.get_anomaly_summary(),
            'pattern_recognizer': {
                'defined_patterns': len(self.pattern_recognizer.patterns),
                'sequence_length': len(self.pattern_recognizer.sequence_history)
            },
            'rl_agent': self.rl_agent.get_policy_stats()
        }


# 全局实例
_intelligence_instance: Optional[IntelligenceManager] = None


def get_intelligence() -> IntelligenceManager:
    """获取全局智能管理器实例"""
    global _intelligence_instance
    if _intelligence_instance is None:
        _intelligence_instance = IntelligenceManager()
    return _intelligence_instance
