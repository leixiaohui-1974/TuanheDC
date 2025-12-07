"""
State Evaluation Module for TAOS V3.10

This module provides real-time system state evaluation including:
- Deviation from control targets
- Performance indices calculation
- Multi-objective evaluation
- Risk assessment
- Compliance monitoring
- Operational efficiency metrics

Author: TAOS Development Team
Version: 3.10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class EvaluationCategory(Enum):
    """Categories of state evaluation."""
    HYDRAULIC = "hydraulic"
    THERMAL = "thermal"
    STRUCTURAL = "structural"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    ENVIRONMENTAL = "environmental"


class DeviationSeverity(Enum):
    """Severity levels for deviations."""
    NOMINAL = "nominal"           # Within target band
    MINOR = "minor"               # Slight deviation
    MODERATE = "moderate"         # Notable deviation
    SIGNIFICANT = "significant"   # Large deviation
    CRITICAL = "critical"         # Emergency level


class ComplianceStatus(Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL_VIOLATION = "critical_violation"


@dataclass
class ControlTarget:
    """Definition of a control target."""
    name: str
    category: EvaluationCategory
    target_value: float
    tolerance_band: float           # Acceptable deviation
    warning_threshold: float        # Warning level
    critical_threshold: float       # Critical level
    unit: str = ""
    weight: float = 1.0             # Importance weight
    is_hard_constraint: bool = False


@dataclass
class DeviationResult:
    """Result of deviation analysis."""
    name: str
    target: float
    actual: float
    deviation: float
    relative_deviation: float
    severity: DeviationSeverity
    within_tolerance: bool
    timestamp: float = 0.0


@dataclass
class PerformanceIndex:
    """Performance index definition and value."""
    name: str
    category: EvaluationCategory
    value: float
    target: float
    unit: str = ""
    description: str = ""


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    category: str
    risk_level: float              # 0-1
    probability: float             # 0-1
    consequence: float             # 0-1
    mitigation_available: bool = True
    description: str = ""


class StateEvaluator:
    """
    Real-time state evaluation engine.

    Evaluates system state against control targets and
    calculates performance indices.
    """

    def __init__(self):
        # Control targets
        self.targets: Dict[str, ControlTarget] = {}

        # Evaluation history
        self.evaluation_history: deque = deque(maxlen=10000)
        self.deviation_history: deque = deque(maxlen=1000)

        # Performance tracking
        self.cumulative_deviation = 0.0
        self.evaluation_count = 0
        self.start_time = time.time()

        # Initialize default targets
        self._initialize_default_targets()

    def _initialize_default_targets(self):
        """Initialize default control targets for the aqueduct."""

        # Hydraulic targets
        self.add_target(ControlTarget(
            name="water_level",
            category=EvaluationCategory.HYDRAULIC,
            target_value=4.0,
            tolerance_band=0.5,
            warning_threshold=1.0,
            critical_threshold=2.0,
            unit="m",
            weight=1.5,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="velocity",
            category=EvaluationCategory.HYDRAULIC,
            target_value=2.0,
            tolerance_band=0.5,
            warning_threshold=1.0,
            critical_threshold=2.0,
            unit="m/s",
            weight=1.0
        ))

        self.add_target(ControlTarget(
            name="froude_number",
            category=EvaluationCategory.HYDRAULIC,
            target_value=0.5,
            tolerance_band=0.2,
            warning_threshold=0.3,
            critical_threshold=0.4,
            unit="",
            weight=2.0,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="flow_balance",
            category=EvaluationCategory.HYDRAULIC,
            target_value=0.0,
            tolerance_band=5.0,
            warning_threshold=10.0,
            critical_threshold=20.0,
            unit="m³/s",
            weight=1.2
        ))

        # Thermal targets
        self.add_target(ControlTarget(
            name="temperature_differential",
            category=EvaluationCategory.THERMAL,
            target_value=0.0,
            tolerance_band=5.0,
            warning_threshold=8.0,
            critical_threshold=10.0,
            unit="°C",
            weight=1.5,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="sun_temperature",
            category=EvaluationCategory.THERMAL,
            target_value=25.0,
            tolerance_band=15.0,
            warning_threshold=25.0,
            critical_threshold=35.0,
            unit="°C",
            weight=0.8
        ))

        # Structural targets
        self.add_target(ControlTarget(
            name="joint_gap",
            category=EvaluationCategory.STRUCTURAL,
            target_value=20.0,
            tolerance_band=10.0,
            warning_threshold=12.0,
            critical_threshold=15.0,
            unit="mm",
            weight=1.8,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="vibration_amplitude",
            category=EvaluationCategory.STRUCTURAL,
            target_value=0.0,
            tolerance_band=10.0,
            warning_threshold=30.0,
            critical_threshold=50.0,
            unit="mm",
            weight=2.0,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="bearing_stress",
            category=EvaluationCategory.STRUCTURAL,
            target_value=31.0,
            tolerance_band=5.0,
            warning_threshold=8.0,
            critical_threshold=15.0,
            unit="MPa",
            weight=1.5
        ))

        # Safety targets
        self.add_target(ControlTarget(
            name="water_level_min",
            category=EvaluationCategory.SAFETY,
            target_value=1.0,
            tolerance_band=0.5,
            warning_threshold=0.3,
            critical_threshold=0.1,
            unit="m",
            weight=2.5,
            is_hard_constraint=True
        ))

        self.add_target(ControlTarget(
            name="water_level_max",
            category=EvaluationCategory.SAFETY,
            target_value=7.5,
            tolerance_band=0.5,
            warning_threshold=0.3,
            critical_threshold=0.1,
            unit="m",
            weight=2.5,
            is_hard_constraint=True
        ))

    def add_target(self, target: ControlTarget):
        """Add a control target."""
        self.targets[target.name] = target

    def remove_target(self, name: str):
        """Remove a control target."""
        if name in self.targets:
            del self.targets[name]

    def update_target(self, name: str, **kwargs):
        """Update target parameters."""
        if name in self.targets:
            target = self.targets[name]
            for key, value in kwargs.items():
                if hasattr(target, key):
                    setattr(target, key, value)

    def _map_state_to_targets(self, state: Dict[str, float]) -> Dict[str, float]:
        """Map system state to target values."""
        mapping = {
            'water_level': state.get('h', 4.0),
            'velocity': state.get('v', 2.0),
            'froude_number': state.get('fr', 0.5),
            'flow_balance': state.get('Q_in', 80) - state.get('Q_out', 80),
            'temperature_differential': abs(
                state.get('T_sun', 20) - state.get('T_shade', 20)
            ),
            'sun_temperature': state.get('T_sun', 25),
            'joint_gap': state.get('joint_gap', 20),
            'vibration_amplitude': state.get('vib_amp', 0),
            'bearing_stress': state.get('bearing_stress', 31),
            'water_level_min': state.get('h', 4.0),
            'water_level_max': state.get('h', 4.0)
        }
        return mapping

    def _calculate_severity(self, target: ControlTarget,
                            deviation: float) -> DeviationSeverity:
        """Calculate deviation severity."""
        abs_dev = abs(deviation)

        if abs_dev <= target.tolerance_band:
            return DeviationSeverity.NOMINAL
        elif abs_dev <= target.warning_threshold:
            return DeviationSeverity.MINOR
        elif abs_dev <= target.critical_threshold:
            return DeviationSeverity.MODERATE
        elif abs_dev <= target.critical_threshold * 1.5:
            return DeviationSeverity.SIGNIFICANT
        else:
            return DeviationSeverity.CRITICAL

    def evaluate_deviation(self, state: Dict[str, float],
                           timestamp: Optional[float] = None) -> Dict[str, DeviationResult]:
        """
        Evaluate deviations from control targets.

        Args:
            state: Current system state
            timestamp: Evaluation timestamp

        Returns:
            Dictionary of deviation results for each target
        """
        timestamp = timestamp or time.time()
        mapped_values = self._map_state_to_targets(state)

        results = {}

        for name, target in self.targets.items():
            actual = mapped_values.get(name, target.target_value)

            # Calculate deviation
            deviation = actual - target.target_value

            # Handle special cases (min/max constraints)
            if name == 'water_level_min':
                deviation = target.target_value - actual if actual < target.target_value else 0
            elif name == 'water_level_max':
                deviation = actual - target.target_value if actual > target.target_value else 0

            # Relative deviation
            if target.target_value != 0:
                relative_deviation = deviation / target.target_value
            else:
                relative_deviation = deviation

            # Severity
            severity = self._calculate_severity(target, deviation)

            results[name] = DeviationResult(
                name=name,
                target=target.target_value,
                actual=actual,
                deviation=deviation,
                relative_deviation=relative_deviation,
                severity=severity,
                within_tolerance=abs(deviation) <= target.tolerance_band,
                timestamp=timestamp
            )

        # Store in history
        self.deviation_history.append({
            'timestamp': timestamp,
            'results': results.copy()
        })

        return results

    def calculate_performance_indices(self, state: Dict[str, float],
                                       control: Optional[Dict[str, float]] = None) -> List[PerformanceIndex]:
        """
        Calculate performance indices.

        Args:
            state: Current system state
            control: Control inputs (optional)

        Returns:
            List of performance indices
        """
        indices = []

        # Hydraulic efficiency
        Q_in = state.get('Q_in', 80)
        Q_out = state.get('Q_out', 80)
        Q_design = 350.0

        flow_efficiency = min(Q_in, Q_out) / Q_design if Q_design > 0 else 0
        indices.append(PerformanceIndex(
            name="flow_efficiency",
            category=EvaluationCategory.EFFICIENCY,
            value=flow_efficiency,
            target=0.8,
            unit="%",
            description="Flow rate relative to design capacity"
        ))

        # Froude number stability
        fr = state.get('fr', 0.5)
        fr_stability = 1.0 - min(1.0, fr / 0.9)
        indices.append(PerformanceIndex(
            name="froude_stability",
            category=EvaluationCategory.HYDRAULIC,
            value=fr_stability,
            target=0.9,
            description="Distance from critical Froude number"
        ))

        # Thermal stress index
        T_diff = abs(state.get('T_sun', 20) - state.get('T_shade', 20))
        thermal_stress = min(1.0, T_diff / 10.0)
        indices.append(PerformanceIndex(
            name="thermal_stress_index",
            category=EvaluationCategory.THERMAL,
            value=thermal_stress,
            target=0.3,
            description="Thermal gradient stress level"
        ))

        # Structural health index
        joint_gap = state.get('joint_gap', 20)
        vib = state.get('vib_amp', 0)
        bearing = state.get('bearing_stress', 31)

        joint_health = 1.0 - abs(joint_gap - 20) / 15
        vib_health = 1.0 - min(1.0, vib / 50)
        bearing_health = 1.0 - min(1.0, (bearing - 31) / 20) if bearing > 31 else 1.0

        structural_health = (joint_health + vib_health + bearing_health) / 3
        indices.append(PerformanceIndex(
            name="structural_health_index",
            category=EvaluationCategory.STRUCTURAL,
            value=structural_health,
            target=0.9,
            description="Overall structural health"
        ))

        # Control effort (if control provided)
        if control:
            Q_in_cmd = control.get('Q_in', 80)
            Q_out_cmd = control.get('Q_out', 80)
            control_effort = (abs(Q_in_cmd - 80) + abs(Q_out_cmd - 80)) / 200
            indices.append(PerformanceIndex(
                name="control_effort",
                category=EvaluationCategory.EFFICIENCY,
                value=control_effort,
                target=0.2,
                description="Normalized control effort"
            ))

        # Overall performance index
        deviations = self.evaluate_deviation(state)
        weighted_deviation_sum = 0.0
        total_weight = 0.0

        for name, result in deviations.items():
            target = self.targets[name]
            normalized_dev = abs(result.deviation) / max(target.critical_threshold, 0.001)
            weighted_deviation_sum += normalized_dev * target.weight
            total_weight += target.weight

        if total_weight > 0:
            overall_performance = 1.0 - min(1.0, weighted_deviation_sum / total_weight)
        else:
            overall_performance = 1.0

        indices.append(PerformanceIndex(
            name="overall_performance",
            category=EvaluationCategory.EFFICIENCY,
            value=overall_performance,
            target=0.85,
            description="Weighted overall system performance"
        ))

        return indices

    def assess_risk(self, state: Dict[str, float]) -> List[RiskAssessment]:
        """
        Assess system risks based on current state.

        Returns:
            List of risk assessments
        """
        risks = []

        # Hydraulic jump risk
        fr = state.get('fr', 0.5)
        hj_prob = min(1.0, max(0, (fr - 0.7) / 0.3))
        hj_consequence = 0.8  # High consequence
        risks.append(RiskAssessment(
            category="hydraulic_jump",
            risk_level=hj_prob * hj_consequence,
            probability=hj_prob,
            consequence=hj_consequence,
            description="Risk of hydraulic jump formation"
        ))

        # Thermal cracking risk
        T_diff = abs(state.get('T_sun', 20) - state.get('T_shade', 20))
        tc_prob = min(1.0, max(0, (T_diff - 5) / 10))
        tc_consequence = 0.7
        risks.append(RiskAssessment(
            category="thermal_cracking",
            risk_level=tc_prob * tc_consequence,
            probability=tc_prob,
            consequence=tc_consequence,
            description="Risk of thermal stress cracking"
        ))

        # Joint failure risk
        joint_gap = state.get('joint_gap', 20)
        jf_prob = 0.0
        if joint_gap < 5 or joint_gap > 35:
            jf_prob = 1.0
        elif joint_gap < 10 or joint_gap > 30:
            jf_prob = 0.5
        elif joint_gap < 15 or joint_gap > 25:
            jf_prob = 0.2

        jf_consequence = 0.9
        risks.append(RiskAssessment(
            category="joint_failure",
            risk_level=jf_prob * jf_consequence,
            probability=jf_prob,
            consequence=jf_consequence,
            description="Risk of expansion joint failure"
        ))

        # Vibration damage risk
        vib = state.get('vib_amp', 0)
        vd_prob = min(1.0, max(0, (vib - 20) / 40))
        vd_consequence = 0.6
        risks.append(RiskAssessment(
            category="vibration_damage",
            risk_level=vd_prob * vd_consequence,
            probability=vd_prob,
            consequence=vd_consequence,
            description="Risk of vibration-induced damage"
        ))

        # Bearing overload risk
        bearing = state.get('bearing_stress', 31)
        bo_prob = min(1.0, max(0, (bearing - 35) / 15))
        bo_consequence = 0.85
        risks.append(RiskAssessment(
            category="bearing_overload",
            risk_level=bo_prob * bo_consequence,
            probability=bo_prob,
            consequence=bo_consequence,
            description="Risk of bearing support overload"
        ))

        # Water level exceedance risk
        h = state.get('h', 4.0)
        wl_prob = 0.0
        if h < 1.0 or h > 7.5:
            wl_prob = 1.0
        elif h < 1.5 or h > 7.0:
            wl_prob = 0.5
        elif h < 2.0 or h > 6.5:
            wl_prob = 0.2

        wl_consequence = 0.95
        risks.append(RiskAssessment(
            category="water_level_exceedance",
            risk_level=wl_prob * wl_consequence,
            probability=wl_prob,
            consequence=wl_consequence,
            description="Risk of water level constraint violation"
        ))

        return sorted(risks, key=lambda r: r.risk_level, reverse=True)

    def check_compliance(self, state: Dict[str, float]) -> Dict[str, ComplianceStatus]:
        """
        Check compliance with operational constraints.

        Returns:
            Dictionary of compliance status for each constraint
        """
        compliance = {}

        deviations = self.evaluate_deviation(state)

        for name, result in deviations.items():
            target = self.targets[name]

            if result.within_tolerance:
                compliance[name] = ComplianceStatus.COMPLIANT
            elif result.severity in [DeviationSeverity.MINOR, DeviationSeverity.MODERATE]:
                compliance[name] = ComplianceStatus.WARNING
            elif result.severity == DeviationSeverity.SIGNIFICANT:
                compliance[name] = ComplianceStatus.VIOLATION
            else:
                compliance[name] = ComplianceStatus.CRITICAL_VIOLATION

            # Hard constraints are always violations if exceeded
            if target.is_hard_constraint and not result.within_tolerance:
                if result.severity in [DeviationSeverity.SIGNIFICANT, DeviationSeverity.CRITICAL]:
                    compliance[name] = ComplianceStatus.CRITICAL_VIOLATION

        return compliance

    def evaluate(self, state: Dict[str, float],
                 control: Optional[Dict[str, float]] = None,
                 timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive state evaluation.

        Args:
            state: Current system state
            control: Current control inputs
            timestamp: Evaluation timestamp

        Returns:
            Complete evaluation results
        """
        timestamp = timestamp or time.time()
        self.evaluation_count += 1

        # Evaluate all aspects
        deviations = self.evaluate_deviation(state, timestamp)
        performance = self.calculate_performance_indices(state, control)
        risks = self.assess_risk(state)
        compliance = self.check_compliance(state)

        # Calculate summary metrics
        deviation_summary = {
            'nominal': sum(1 for d in deviations.values() if d.severity == DeviationSeverity.NOMINAL),
            'minor': sum(1 for d in deviations.values() if d.severity == DeviationSeverity.MINOR),
            'moderate': sum(1 for d in deviations.values() if d.severity == DeviationSeverity.MODERATE),
            'significant': sum(1 for d in deviations.values() if d.severity == DeviationSeverity.SIGNIFICANT),
            'critical': sum(1 for d in deviations.values() if d.severity == DeviationSeverity.CRITICAL)
        }

        compliance_summary = {
            'compliant': sum(1 for c in compliance.values() if c == ComplianceStatus.COMPLIANT),
            'warning': sum(1 for c in compliance.values() if c == ComplianceStatus.WARNING),
            'violation': sum(1 for c in compliance.values() if c == ComplianceStatus.VIOLATION),
            'critical': sum(1 for c in compliance.values() if c == ComplianceStatus.CRITICAL_VIOLATION)
        }

        # Overall system score
        perf_scores = [p.value for p in performance if p.name == 'overall_performance']
        overall_score = perf_scores[0] if perf_scores else 0.5

        max_risk = max(r.risk_level for r in risks) if risks else 0

        result = {
            'timestamp': timestamp,
            'deviations': {
                name: {
                    'target': d.target,
                    'actual': d.actual,
                    'deviation': d.deviation,
                    'severity': d.severity.value,
                    'within_tolerance': d.within_tolerance
                }
                for name, d in deviations.items()
            },
            'deviation_summary': deviation_summary,
            'performance_indices': [
                {
                    'name': p.name,
                    'category': p.category.value,
                    'value': p.value,
                    'target': p.target,
                    'unit': p.unit
                }
                for p in performance
            ],
            'risk_assessment': [
                {
                    'category': r.category,
                    'risk_level': r.risk_level,
                    'probability': r.probability,
                    'consequence': r.consequence,
                    'description': r.description
                }
                for r in risks
            ],
            'compliance': {name: status.value for name, status in compliance.items()},
            'compliance_summary': compliance_summary,
            'overall_score': overall_score,
            'max_risk_level': max_risk,
            'evaluation_number': self.evaluation_count
        }

        # Store in history
        self.evaluation_history.append(result)

        return result

    def get_evaluation_trend(self, lookback_seconds: float = 3600) -> Dict[str, Any]:
        """Get evaluation trends over time."""
        cutoff = time.time() - lookback_seconds
        recent = [e for e in self.evaluation_history if e['timestamp'] > cutoff]

        if not recent:
            return {'status': 'no_data'}

        scores = [e['overall_score'] for e in recent]
        risks = [e['max_risk_level'] for e in recent]

        return {
            'period_seconds': lookback_seconds,
            'samples': len(recent),
            'score_trend': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'current': scores[-1] if scores else 0
            },
            'risk_trend': {
                'mean': np.mean(risks),
                'std': np.std(risks),
                'max': np.max(risks),
                'current': risks[-1] if risks else 0
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get evaluator status."""
        return {
            'targets_count': len(self.targets),
            'evaluation_count': self.evaluation_count,
            'uptime_seconds': time.time() - self.start_time,
            'history_size': len(self.evaluation_history),
            'trend': self.get_evaluation_trend(600)  # Last 10 minutes
        }


class MultiObjectiveEvaluator:
    """
    Multi-objective evaluation for conflicting objectives.
    """

    def __init__(self):
        self.objectives: Dict[str, Dict[str, Any]] = {}
        self.pareto_history: deque = deque(maxlen=1000)

        # Initialize default objectives
        self._initialize_default_objectives()

    def _initialize_default_objectives(self):
        """Initialize default multi-objective criteria."""
        self.objectives = {
            'safety': {
                'direction': 'maximize',
                'weight': 0.4,
                'components': ['water_level', 'froude_number', 'vibration']
            },
            'efficiency': {
                'direction': 'maximize',
                'weight': 0.3,
                'components': ['flow_rate', 'control_effort']
            },
            'structural_integrity': {
                'direction': 'maximize',
                'weight': 0.2,
                'components': ['joint_gap', 'bearing_stress', 'thermal_stress']
            },
            'operational_cost': {
                'direction': 'minimize',
                'weight': 0.1,
                'components': ['energy_consumption', 'maintenance_indicator']
            }
        }

    def evaluate_objectives(self, state: Dict[str, float],
                            control: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Evaluate multi-objective scores."""
        scores = {}

        # Safety objective
        fr = state.get('fr', 0.5)
        h = state.get('h', 4.0)
        vib = state.get('vib_amp', 0)

        safety = (
            (1.0 - min(1.0, fr / 0.9)) * 0.4 +
            (1.0 - abs(h - 4.0) / 4.0) * 0.3 +
            (1.0 - min(1.0, vib / 50)) * 0.3
        )
        scores['safety'] = safety

        # Efficiency objective
        Q_in = state.get('Q_in', 80)
        Q_out = state.get('Q_out', 80)
        efficiency = min(Q_in, Q_out) / 350.0  # Normalized to design flow
        scores['efficiency'] = efficiency

        # Structural integrity
        joint = state.get('joint_gap', 20)
        bearing = state.get('bearing_stress', 31)
        T_diff = abs(state.get('T_sun', 20) - state.get('T_shade', 20))

        structural = (
            (1.0 - abs(joint - 20) / 15) * 0.4 +
            (1.0 - max(0, bearing - 31) / 20) * 0.3 +
            (1.0 - T_diff / 15) * 0.3
        )
        scores['structural_integrity'] = max(0, structural)

        # Cost objective (lower is better, so we return 1 - cost)
        if control:
            effort = (abs(control.get('Q_in', 80) - 80) +
                     abs(control.get('Q_out', 80) - 80)) / 200
        else:
            effort = 0
        scores['operational_cost'] = 1.0 - effort

        return scores

    def calculate_weighted_score(self, objective_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        total = 0.0
        for name, score in objective_scores.items():
            if name in self.objectives:
                weight = self.objectives[name]['weight']
                total += weight * score
        return total

    def is_pareto_dominant(self, scores1: Dict[str, float],
                           scores2: Dict[str, float]) -> bool:
        """Check if scores1 Pareto-dominates scores2."""
        dominated_in_all = True
        better_in_one = False

        for name in self.objectives:
            if name not in scores1 or name not in scores2:
                continue

            direction = self.objectives[name]['direction']

            if direction == 'maximize':
                if scores1[name] < scores2[name]:
                    dominated_in_all = False
                if scores1[name] > scores2[name]:
                    better_in_one = True
            else:
                if scores1[name] > scores2[name]:
                    dominated_in_all = False
                if scores1[name] < scores2[name]:
                    better_in_one = True

        return dominated_in_all and better_in_one

    def evaluate(self, state: Dict[str, float],
                 control: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Complete multi-objective evaluation."""
        objective_scores = self.evaluate_objectives(state, control)
        weighted_score = self.calculate_weighted_score(objective_scores)

        # Store for Pareto analysis
        self.pareto_history.append({
            'timestamp': time.time(),
            'scores': objective_scores.copy(),
            'weighted': weighted_score
        })

        return {
            'objective_scores': objective_scores,
            'weighted_score': weighted_score,
            'objectives': {
                name: {
                    'score': objective_scores.get(name, 0),
                    'weight': obj['weight'],
                    'direction': obj['direction']
                }
                for name, obj in self.objectives.items()
            }
        }
