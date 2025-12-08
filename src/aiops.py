"""
TAOS V3.10 AIOps Module
智能运维模块

Features:
- Anomaly detection
- Intelligent diagnostics
- Predictive maintenance
- Root cause analysis
- Auto-remediation
- Capacity planning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import threading
import time
from collections import deque, defaultdict
import json
import statistics

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Anomaly types"""
    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    OSCILLATION = "oscillation"
    FLATLINE = "flatline"
    DRIFT = "drift"
    CONTEXTUAL = "contextual"


class SeverityLevel(Enum):
    """Severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MaintenanceType(Enum):
    """Maintenance types"""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"


class HealthStatus(Enum):
    """Equipment health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    entity_id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    detected_at: datetime
    value: float
    expected_value: float
    deviation_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'anomaly_id': self.anomaly_id,
            'entity_id': self.entity_id,
            'metric_name': self.metric_name,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation_score': self.deviation_score,
            'context': self.context,
            'resolved': self.resolved
        }


@dataclass
class DiagnosticResult:
    """Diagnostic analysis result"""
    diagnosis_id: str
    entity_id: str
    timestamp: datetime
    health_status: HealthStatus
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float
    root_causes: List[str] = field(default_factory=list)
    related_anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'diagnosis_id': self.diagnosis_id,
            'entity_id': self.entity_id,
            'timestamp': self.timestamp.isoformat(),
            'health_status': self.health_status.value,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'root_causes': self.root_causes,
            'related_anomalies': self.related_anomalies
        }


@dataclass
class MaintenancePrediction:
    """Maintenance prediction"""
    prediction_id: str
    entity_id: str
    component: str
    maintenance_type: MaintenanceType
    predicted_failure_date: Optional[datetime]
    confidence: float
    remaining_useful_life: Optional[float]  # hours
    risk_score: float
    recommended_actions: List[str]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction_id,
            'entity_id': self.entity_id,
            'component': self.component,
            'maintenance_type': self.maintenance_type.value,
            'predicted_failure_date': self.predicted_failure_date.isoformat() if self.predicted_failure_date else None,
            'confidence': self.confidence,
            'remaining_useful_life': self.remaining_useful_life,
            'risk_score': self.risk_score,
            'recommended_actions': self.recommended_actions
        }


@dataclass
class RemediationAction:
    """Remediation action"""
    action_id: str
    anomaly_id: str
    action_type: str
    parameters: Dict[str, Any]
    auto_execute: bool = False
    executed: bool = False
    executed_at: Optional[datetime] = None
    result: Optional[str] = None
    success: bool = False


class AnomalyDetector:
    """
    Anomaly detection engine
    异常检测引擎
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_buffers: Dict[str, deque] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def add_data_point(self, entity_id: str, metric: str, value: float,
                       timestamp: datetime = None):
        """Add data point for analysis"""
        key = f"{entity_id}:{metric}"
        timestamp = timestamp or datetime.now()

        with self._lock:
            if key not in self.data_buffers:
                self.data_buffers[key] = deque(maxlen=self.window_size)
            self.data_buffers[key].append((timestamp, value))
            self._update_baseline(key)

    def _update_baseline(self, key: str):
        """Update baseline statistics"""
        buffer = self.data_buffers[key]
        if len(buffer) < 10:
            return

        values = [v for _, v in buffer]
        self.baselines[key] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values)
        }

    def set_thresholds(self, entity_id: str, metric: str,
                       sigma_factor: float = 3.0,
                       absolute_min: float = None,
                       absolute_max: float = None):
        """Set detection thresholds"""
        key = f"{entity_id}:{metric}"
        self.thresholds[key] = {
            'sigma_factor': sigma_factor,
            'absolute_min': absolute_min,
            'absolute_max': absolute_max
        }

    def detect(self, entity_id: str, metric: str, value: float) -> Optional[Anomaly]:
        """Detect anomaly in new data point"""
        key = f"{entity_id}:{metric}"

        with self._lock:
            if key not in self.baselines:
                return None

            baseline = self.baselines[key]
            thresholds = self.thresholds.get(key, {'sigma_factor': 3.0})

            mean = baseline['mean']
            std = baseline['std'] or 0.1
            sigma_factor = thresholds['sigma_factor']

            # Calculate deviation score
            deviation = abs(value - mean) / std if std > 0 else 0
            deviation_score = deviation / sigma_factor

            # Check for anomaly
            anomaly_type = None
            severity = None

            # Statistical anomaly
            if deviation > sigma_factor:
                if value > mean:
                    anomaly_type = AnomalyType.SPIKE
                else:
                    anomaly_type = AnomalyType.DROP

                if deviation > sigma_factor * 2:
                    severity = SeverityLevel.CRITICAL
                elif deviation > sigma_factor * 1.5:
                    severity = SeverityLevel.HIGH
                else:
                    severity = SeverityLevel.MEDIUM

            # Absolute threshold check
            abs_min = thresholds.get('absolute_min')
            abs_max = thresholds.get('absolute_max')

            if abs_min is not None and value < abs_min:
                anomaly_type = AnomalyType.DROP
                severity = SeverityLevel.HIGH
            if abs_max is not None and value > abs_max:
                anomaly_type = AnomalyType.SPIKE
                severity = SeverityLevel.HIGH

            # Check for trend/drift
            if len(self.data_buffers[key]) >= 10:
                recent = [v for _, v in list(self.data_buffers[key])[-10:]]
                trend = self._detect_trend(recent)
                if trend and not anomaly_type:
                    anomaly_type = AnomalyType.TREND if trend > 0 else AnomalyType.DRIFT
                    severity = SeverityLevel.LOW

            # Check for oscillation
            if len(self.data_buffers[key]) >= 20:
                oscillation = self._detect_oscillation(key)
                if oscillation and not anomaly_type:
                    anomaly_type = AnomalyType.OSCILLATION
                    severity = SeverityLevel.MEDIUM

            if anomaly_type:
                import uuid
                return Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    entity_id=entity_id,
                    metric_name=metric,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    detected_at=datetime.now(),
                    value=value,
                    expected_value=mean,
                    deviation_score=deviation_score,
                    context={
                        'baseline_mean': mean,
                        'baseline_std': std,
                        'threshold_sigma': sigma_factor
                    }
                )

            return None

    def _detect_trend(self, values: List[float]) -> Optional[float]:
        """Detect monotonic trend"""
        if len(values) < 5:
            return None

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        slope = numerator / denominator

        # Check if trend is significant
        if abs(slope) > 0.1 * statistics.stdev(values) if len(values) > 1 else 0:
            return slope
        return None

    def _detect_oscillation(self, key: str) -> bool:
        """Detect oscillating pattern"""
        values = [v for _, v in list(self.data_buffers[key])[-20:]]
        if len(values) < 10:
            return False

        # Count sign changes
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1)
                          if diffs[i] * diffs[i+1] < 0)

        return sign_changes > len(diffs) * 0.6


class IntelligentDiagnostics:
    """
    Intelligent diagnostics engine
    智能诊断引擎
    """

    def __init__(self):
        self.diagnostic_rules: List[Dict[str, Any]] = []
        self.symptom_patterns: Dict[str, List[Dict]] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_diagnostic_rule(self, rule: Dict[str, Any]):
        """Add diagnostic rule"""
        self.diagnostic_rules.append(rule)

    def add_symptom_pattern(self, pattern_id: str, symptoms: List[Dict],
                            diagnosis: str, recommendations: List[str]):
        """Add symptom pattern for diagnosis"""
        self.symptom_patterns[pattern_id] = {
            'symptoms': symptoms,
            'diagnosis': diagnosis,
            'recommendations': recommendations
        }

    def diagnose(self, entity_id: str,
                 metrics: Dict[str, float],
                 anomalies: List[Anomaly]) -> DiagnosticResult:
        """Perform diagnosis on entity"""
        import uuid

        issues = []
        recommendations = []
        root_causes = []
        confidence = 0.0

        # Analyze anomalies
        for anomaly in anomalies:
            issue = {
                'type': anomaly.anomaly_type.value,
                'metric': anomaly.metric_name,
                'severity': anomaly.severity.value,
                'description': f"{anomaly.metric_name} showing {anomaly.anomaly_type.value} pattern"
            }
            issues.append(issue)

        # Apply diagnostic rules
        for rule in self.diagnostic_rules:
            if self._check_rule_conditions(rule, metrics, anomalies):
                issues.append({
                    'type': 'rule_match',
                    'rule_id': rule.get('rule_id'),
                    'description': rule.get('description', '')
                })
                recommendations.extend(rule.get('recommendations', []))
                if 'root_cause' in rule:
                    root_causes.append(rule['root_cause'])

        # Match symptom patterns
        for pattern_id, pattern in self.symptom_patterns.items():
            match_score = self._match_symptoms(pattern['symptoms'], metrics, anomalies)
            if match_score > 0.7:
                root_causes.append(pattern['diagnosis'])
                recommendations.extend(pattern['recommendations'])
                confidence = max(confidence, match_score)

        # Determine health status
        health_status = self._determine_health_status(issues, anomalies)

        if not confidence:
            confidence = 0.5 if issues else 0.9

        return DiagnosticResult(
            diagnosis_id=str(uuid.uuid4()),
            entity_id=entity_id,
            timestamp=datetime.now(),
            health_status=health_status,
            issues=issues,
            recommendations=list(set(recommendations)),
            confidence=confidence,
            root_causes=list(set(root_causes)),
            related_anomalies=[a.anomaly_id for a in anomalies]
        )

    def _check_rule_conditions(self, rule: Dict, metrics: Dict[str, float],
                                anomalies: List[Anomaly]) -> bool:
        """Check if rule conditions are met"""
        conditions = rule.get('conditions', [])

        for cond in conditions:
            cond_type = cond.get('type')

            if cond_type == 'metric_threshold':
                metric = cond.get('metric')
                threshold = cond.get('threshold')
                operator = cond.get('operator', '>')

                if metric not in metrics:
                    return False

                value = metrics[metric]
                if operator == '>' and not value > threshold:
                    return False
                elif operator == '<' and not value < threshold:
                    return False
                elif operator == '>=' and not value >= threshold:
                    return False
                elif operator == '<=' and not value <= threshold:
                    return False

            elif cond_type == 'anomaly_present':
                anomaly_type = AnomalyType(cond.get('anomaly_type'))
                metric = cond.get('metric')

                found = any(
                    a.anomaly_type == anomaly_type and a.metric_name == metric
                    for a in anomalies
                )
                if not found:
                    return False

        return True

    def _match_symptoms(self, symptoms: List[Dict],
                         metrics: Dict[str, float],
                         anomalies: List[Anomaly]) -> float:
        """Match symptoms against patterns"""
        if not symptoms:
            return 0.0

        matched = 0
        for symptom in symptoms:
            symptom_type = symptom.get('type')

            if symptom_type == 'anomaly':
                for anomaly in anomalies:
                    if (anomaly.metric_name == symptom.get('metric') and
                        anomaly.anomaly_type.value == symptom.get('anomaly_type')):
                        matched += 1
                        break

            elif symptom_type == 'metric_range':
                metric = symptom.get('metric')
                if metric in metrics:
                    value = metrics[metric]
                    min_val = symptom.get('min', float('-inf'))
                    max_val = symptom.get('max', float('inf'))
                    if min_val <= value <= max_val:
                        matched += 1

        return matched / len(symptoms)

    def _determine_health_status(self, issues: List[Dict],
                                   anomalies: List[Anomaly]) -> HealthStatus:
        """Determine overall health status"""
        if not issues and not anomalies:
            return HealthStatus.HEALTHY

        # Check for critical anomalies
        critical_count = sum(
            1 for a in anomalies if a.severity == SeverityLevel.CRITICAL
        )
        high_count = sum(
            1 for a in anomalies if a.severity == SeverityLevel.HIGH
        )

        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif high_count > 1:
            return HealthStatus.WARNING
        elif high_count > 0 or len(issues) > 2:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.DEGRADED


class PredictiveMaintenance:
    """
    Predictive maintenance engine
    预测性维护引擎
    """

    def __init__(self):
        self.equipment_data: Dict[str, Dict] = {}
        self.failure_patterns: Dict[str, Dict] = {}
        self.degradation_models: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_equipment(self, entity_id: str, components: List[str],
                           expected_lifetime: float):
        """Register equipment for monitoring"""
        with self._lock:
            self.equipment_data[entity_id] = {
                'components': components,
                'expected_lifetime': expected_lifetime,
                'operating_hours': 0,
                'maintenance_history': [],
                'health_indicators': {}
            }

    def update_operating_hours(self, entity_id: str, hours: float):
        """Update operating hours"""
        with self._lock:
            if entity_id in self.equipment_data:
                self.equipment_data[entity_id]['operating_hours'] = hours

    def update_health_indicator(self, entity_id: str, component: str,
                                 indicator: str, value: float):
        """Update health indicator"""
        with self._lock:
            if entity_id in self.equipment_data:
                key = f"{component}:{indicator}"
                data = self.equipment_data[entity_id]
                if 'health_indicators' not in data:
                    data['health_indicators'] = {}
                if key not in data['health_indicators']:
                    data['health_indicators'][key] = []
                data['health_indicators'][key].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': value
                })

    def add_failure_pattern(self, pattern_id: str, component: str,
                            indicators: Dict[str, Dict]):
        """Add failure pattern for prediction"""
        self.failure_patterns[pattern_id] = {
            'component': component,
            'indicators': indicators
        }

    def predict_maintenance(self, entity_id: str) -> List[MaintenancePrediction]:
        """Predict maintenance needs"""
        import uuid

        with self._lock:
            if entity_id not in self.equipment_data:
                return []

            data = self.equipment_data[entity_id]
            predictions = []

            for component in data['components']:
                # Calculate remaining useful life
                rul, confidence = self._estimate_rul(entity_id, component)

                # Calculate risk score
                risk_score = self._calculate_risk_score(entity_id, component)

                # Determine maintenance type and actions
                maintenance_type, actions = self._recommend_maintenance(
                    component, rul, risk_score
                )

                # Predict failure date
                failure_date = None
                if rul is not None and rul > 0:
                    failure_date = datetime.now() + timedelta(hours=rul)

                prediction = MaintenancePrediction(
                    prediction_id=str(uuid.uuid4()),
                    entity_id=entity_id,
                    component=component,
                    maintenance_type=maintenance_type,
                    predicted_failure_date=failure_date,
                    confidence=confidence,
                    remaining_useful_life=rul,
                    risk_score=risk_score,
                    recommended_actions=actions
                )
                predictions.append(prediction)

            return predictions

    def _estimate_rul(self, entity_id: str, component: str) -> Tuple[Optional[float], float]:
        """Estimate remaining useful life"""
        data = self.equipment_data[entity_id]
        operating_hours = data['operating_hours']
        expected_lifetime = data['expected_lifetime']

        # Basic linear degradation model
        base_rul = max(0, expected_lifetime - operating_hours)
        confidence = 0.6

        # Adjust based on health indicators
        health_indicators = data.get('health_indicators', {})
        degradation_factor = 1.0

        for key, values in health_indicators.items():
            if key.startswith(component):
                if values:
                    recent_values = [v['value'] for v in values[-10:]]
                    trend = self._calculate_trend(recent_values)
                    if trend > 0:  # Degrading
                        degradation_factor *= (1 + trend * 0.1)
                    confidence = min(0.9, confidence + 0.1)

        adjusted_rul = base_rul / degradation_factor
        return adjusted_rul, confidence

    def _calculate_risk_score(self, entity_id: str, component: str) -> float:
        """Calculate maintenance risk score"""
        data = self.equipment_data[entity_id]
        operating_hours = data['operating_hours']
        expected_lifetime = data['expected_lifetime']

        # Base risk from age
        age_ratio = operating_hours / expected_lifetime if expected_lifetime > 0 else 1.0
        base_risk = min(1.0, age_ratio)

        # Adjust for health indicators
        health_risk = 0.0
        health_indicators = data.get('health_indicators', {})

        for key, values in health_indicators.items():
            if key.startswith(component) and values:
                recent = values[-1]['value']
                # Assume normalized 0-1 scale where higher is worse
                health_risk = max(health_risk, recent)

        return 0.6 * base_risk + 0.4 * health_risk

    def _recommend_maintenance(self, component: str, rul: Optional[float],
                                risk_score: float) -> Tuple[MaintenanceType, List[str]]:
        """Recommend maintenance type and actions"""
        actions = []

        if risk_score > 0.8:
            maintenance_type = MaintenanceType.CORRECTIVE
            actions.append(f"Immediate inspection of {component} required")
            actions.append(f"Prepare replacement parts for {component}")
        elif risk_score > 0.6 or (rul and rul < 100):
            maintenance_type = MaintenanceType.PREDICTIVE
            actions.append(f"Schedule maintenance for {component} within 1 week")
            actions.append(f"Order spare parts for {component}")
        elif risk_score > 0.4:
            maintenance_type = MaintenanceType.CONDITION_BASED
            actions.append(f"Increase monitoring frequency for {component}")
            actions.append(f"Perform visual inspection of {component}")
        else:
            maintenance_type = MaintenanceType.PREVENTIVE
            actions.append(f"Continue routine maintenance for {component}")

        return maintenance_type, actions

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator


class AutoRemediation:
    """
    Auto-remediation engine
    自动修复引擎
    """

    def __init__(self):
        self.remediation_rules: Dict[str, Dict] = {}
        self.action_handlers: Dict[str, Callable] = {}
        self.remediation_history: List[RemediationAction] = []
        self._lock = threading.Lock()

    def add_remediation_rule(self, rule_id: str,
                              anomaly_type: AnomalyType,
                              metric: str,
                              action_type: str,
                              parameters: Dict[str, Any],
                              auto_execute: bool = False):
        """Add remediation rule"""
        self.remediation_rules[rule_id] = {
            'anomaly_type': anomaly_type,
            'metric': metric,
            'action_type': action_type,
            'parameters': parameters,
            'auto_execute': auto_execute
        }

    def register_action_handler(self, action_type: str,
                                  handler: Callable[[Dict[str, Any]], bool]):
        """Register action handler"""
        self.action_handlers[action_type] = handler

    def get_remediation_actions(self, anomaly: Anomaly) -> List[RemediationAction]:
        """Get remediation actions for anomaly"""
        import uuid

        actions = []
        for rule_id, rule in self.remediation_rules.items():
            if (rule['anomaly_type'] == anomaly.anomaly_type and
                rule['metric'] == anomaly.metric_name):

                action = RemediationAction(
                    action_id=str(uuid.uuid4()),
                    anomaly_id=anomaly.anomaly_id,
                    action_type=rule['action_type'],
                    parameters=rule['parameters'].copy(),
                    auto_execute=rule['auto_execute']
                )
                actions.append(action)

        return actions

    def execute_action(self, action: RemediationAction) -> bool:
        """Execute remediation action"""
        with self._lock:
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                logger.warning(f"No handler for action type: {action.action_type}")
                action.result = "No handler available"
                action.success = False
                return False

            try:
                success = handler(action.parameters)
                action.executed = True
                action.executed_at = datetime.now()
                action.success = success
                action.result = "Executed successfully" if success else "Execution failed"
                self.remediation_history.append(action)
                return success
            except Exception as e:
                action.executed = True
                action.executed_at = datetime.now()
                action.success = False
                action.result = str(e)
                self.remediation_history.append(action)
                return False


class AIOpsManager:
    """
    AIOps Manager
    智能运维管理器
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.diagnostics = IntelligentDiagnostics()
        self.maintenance = PredictiveMaintenance()
        self.remediation = AutoRemediation()
        self.active_anomalies: Dict[str, Anomaly] = {}
        self.diagnostics_cache: Dict[str, DiagnosticResult] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[Anomaly], None]] = []

    def process_metric(self, entity_id: str, metric: str, value: float,
                       timestamp: datetime = None) -> Optional[Anomaly]:
        """Process incoming metric data"""
        timestamp = timestamp or datetime.now()

        # Add to detector buffer
        self.anomaly_detector.add_data_point(entity_id, metric, value, timestamp)

        # Detect anomaly
        anomaly = self.anomaly_detector.detect(entity_id, metric, value)

        if anomaly:
            with self._lock:
                self.active_anomalies[anomaly.anomaly_id] = anomaly

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Auto-remediation
            actions = self.remediation.get_remediation_actions(anomaly)
            for action in actions:
                if action.auto_execute:
                    self.remediation.execute_action(action)

        return anomaly

    def diagnose_entity(self, entity_id: str,
                        metrics: Dict[str, float]) -> DiagnosticResult:
        """Diagnose entity health"""
        # Get related anomalies
        related_anomalies = [
            a for a in self.active_anomalies.values()
            if a.entity_id == entity_id and not a.resolved
        ]

        result = self.diagnostics.diagnose(entity_id, metrics, related_anomalies)

        with self._lock:
            self.diagnostics_cache[entity_id] = result

        return result

    def predict_maintenance(self, entity_id: str) -> List[MaintenancePrediction]:
        """Get maintenance predictions"""
        return self.maintenance.predict_maintenance(entity_id)

    def resolve_anomaly(self, anomaly_id: str):
        """Mark anomaly as resolved"""
        with self._lock:
            if anomaly_id in self.active_anomalies:
                anomaly = self.active_anomalies[anomaly_id]
                anomaly.resolved = True
                anomaly.resolved_at = datetime.now()

    def register_anomaly_callback(self, callback: Callable[[Anomaly], None]):
        """Register anomaly callback"""
        self._callbacks.append(callback)

    def set_detection_threshold(self, entity_id: str, metric: str,
                                 sigma_factor: float = 3.0,
                                 min_value: float = None,
                                 max_value: float = None):
        """Set anomaly detection thresholds"""
        self.anomaly_detector.set_thresholds(
            entity_id, metric, sigma_factor, min_value, max_value
        )

    def register_equipment(self, entity_id: str, components: List[str],
                           expected_lifetime: float):
        """Register equipment for maintenance prediction"""
        self.maintenance.register_equipment(entity_id, components, expected_lifetime)

    def add_diagnostic_rule(self, rule: Dict[str, Any]):
        """Add diagnostic rule"""
        self.diagnostics.add_diagnostic_rule(rule)

    def add_remediation_rule(self, rule_id: str, anomaly_type: AnomalyType,
                              metric: str, action_type: str,
                              parameters: Dict[str, Any],
                              auto_execute: bool = False):
        """Add remediation rule"""
        self.remediation.add_remediation_rule(
            rule_id, anomaly_type, metric, action_type, parameters, auto_execute
        )

    def get_active_anomalies(self, entity_id: str = None) -> List[Anomaly]:
        """Get active anomalies"""
        with self._lock:
            anomalies = list(self.active_anomalies.values())
            if entity_id:
                anomalies = [a for a in anomalies if a.entity_id == entity_id]
            return [a for a in anomalies if not a.resolved]

    def get_statistics(self) -> Dict[str, Any]:
        """Get AIOps statistics"""
        with self._lock:
            active_count = sum(1 for a in self.active_anomalies.values() if not a.resolved)
            by_severity = defaultdict(int)
            by_type = defaultdict(int)

            for anomaly in self.active_anomalies.values():
                if not anomaly.resolved:
                    by_severity[anomaly.severity.name] += 1
                    by_type[anomaly.anomaly_type.name] += 1

            return {
                'active_anomalies': active_count,
                'total_detected': len(self.active_anomalies),
                'by_severity': dict(by_severity),
                'by_type': dict(by_type),
                'monitored_equipment': len(self.maintenance.equipment_data),
                'diagnostic_rules': len(self.diagnostics.diagnostic_rules),
                'remediation_rules': len(self.remediation.remediation_rules)
            }


# Singleton instance
_aiops_manager: Optional[AIOpsManager] = None


def get_aiops_manager() -> AIOpsManager:
    """Get singleton AIOps manager"""
    global _aiops_manager
    if _aiops_manager is None:
        _aiops_manager = AIOpsManager()
    return _aiops_manager
