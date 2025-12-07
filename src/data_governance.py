"""
Data Governance Module for TAOS V3.10

This module provides comprehensive data governance capabilities:
- Data quality monitoring and validation
- Data lineage tracking
- Metadata management
- Data access control and audit
- Data lifecycle management
- Anomaly detection and data cleansing
- Compliance reporting

Author: TAOS Development Team
Version: 3.10
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import hashlib
import json
import time
import uuid


class DataQualityDimension(Enum):
    """Data quality dimensions following ISO 8000."""
    ACCURACY = "accuracy"               # Correctness of data
    COMPLETENESS = "completeness"       # Presence of required data
    CONSISTENCY = "consistency"         # Conformance to rules
    TIMELINESS = "timeliness"           # Data currency
    VALIDITY = "validity"               # Conformance to format
    UNIQUENESS = "uniqueness"           # No duplicates
    INTEGRITY = "integrity"             # Referential integrity


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CRITICAL = "critical"


class DataLifecycleStage(Enum):
    """Data lifecycle stages."""
    CREATION = "creation"
    PROCESSING = "processing"
    STORAGE = "storage"
    USAGE = "usage"
    ARCHIVAL = "archival"
    DELETION = "deletion"


class ValidationRuleType(Enum):
    """Types of validation rules."""
    RANGE_CHECK = "range_check"
    NULL_CHECK = "null_check"
    TYPE_CHECK = "type_check"
    FORMAT_CHECK = "format_check"
    CONSISTENCY_CHECK = "consistency_check"
    TEMPORAL_CHECK = "temporal_check"
    CROSS_FIELD_CHECK = "cross_field_check"
    STATISTICAL_CHECK = "statistical_check"


@dataclass
class ValidationRule:
    """Data validation rule definition."""
    rule_id: str
    name: str
    rule_type: ValidationRuleType
    field_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"              # error, warning, info
    is_active: bool = True
    description: str = ""

    def validate(self, value: Any, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """Validate a value against this rule."""
        context = context or {}

        if self.rule_type == ValidationRuleType.RANGE_CHECK:
            min_val = self.parameters.get('min', -np.inf)
            max_val = self.parameters.get('max', np.inf)
            if value is None:
                return False, f"{self.field_name} is null"
            if not (min_val <= value <= max_val):
                return False, f"{self.field_name}={value} outside range [{min_val}, {max_val}]"

        elif self.rule_type == ValidationRuleType.NULL_CHECK:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return False, f"{self.field_name} is null/NaN"

        elif self.rule_type == ValidationRuleType.TYPE_CHECK:
            expected_type = self.parameters.get('type', 'float')
            type_map = {'float': (int, float), 'int': int, 'str': str, 'bool': bool}
            if expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    return False, f"{self.field_name} has invalid type"

        elif self.rule_type == ValidationRuleType.TEMPORAL_CHECK:
            max_age_seconds = self.parameters.get('max_age_seconds', 60)
            timestamp = context.get('timestamp', time.time())
            if time.time() - timestamp > max_age_seconds:
                return False, f"{self.field_name} data is stale"

        elif self.rule_type == ValidationRuleType.STATISTICAL_CHECK:
            # Check against expected distribution
            mean = self.parameters.get('mean', 0)
            std = self.parameters.get('std', 1)
            n_sigma = self.parameters.get('n_sigma', 3)
            if value is not None:
                z_score = abs(value - mean) / std if std > 0 else 0
                if z_score > n_sigma:
                    return False, f"{self.field_name}={value} is {z_score:.1f} sigma from mean"

        return True, ""


@dataclass
class DataQualityMetrics:
    """Data quality metrics for a data source."""
    source_name: str
    timestamp: float = field(default_factory=time.time)

    # Quality scores (0-1)
    accuracy_score: float = 1.0
    completeness_score: float = 1.0
    consistency_score: float = 1.0
    timeliness_score: float = 1.0
    validity_score: float = 1.0
    uniqueness_score: float = 1.0
    integrity_score: float = 1.0

    # Counts
    total_records: int = 0
    valid_records: int = 0
    error_count: int = 0
    warning_count: int = 0

    # Details
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall data quality score."""
        scores = [
            self.accuracy_score,
            self.completeness_score,
            self.consistency_score,
            self.timeliness_score,
            self.validity_score,
            self.uniqueness_score,
            self.integrity_score
        ]
        return np.mean(scores)


@dataclass
class DataLineageRecord:
    """Record of data lineage/provenance."""
    record_id: str
    timestamp: float
    source_system: str
    source_id: str
    operation: str                       # create, transform, merge, etc.
    input_records: List[str] = field(default_factory=list)
    output_records: List[str] = field(default_factory=list)
    transformation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "system"
    checksum: str = ""


@dataclass
class DataAccessAudit:
    """Data access audit record."""
    audit_id: str
    timestamp: float
    user_id: str
    action: str                          # read, write, delete, export
    data_source: str
    record_ids: List[str] = field(default_factory=list)
    success: bool = True
    details: str = ""
    ip_address: str = ""
    classification: DataClassification = DataClassification.INTERNAL


class DataQualityValidator:
    """
    Data quality validation engine.

    Validates data against defined rules and calculates quality metrics.
    """

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self._validation_history: deque = deque(maxlen=10000)
        self._field_statistics: Dict[str, Dict[str, float]] = {}

        # Initialize default rules for aqueduct data
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default validation rules for aqueduct data."""

        # Water level rules
        self.add_rule(ValidationRule(
            rule_id="h_range",
            name="Water Level Range Check",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="h",
            parameters={'min': 0.0, 'max': 10.0},
            severity="error",
            description="Water level must be between 0 and 10 meters"
        ))

        self.add_rule(ValidationRule(
            rule_id="h_null",
            name="Water Level Null Check",
            rule_type=ValidationRuleType.NULL_CHECK,
            field_name="h",
            severity="error"
        ))

        # Velocity rules
        self.add_rule(ValidationRule(
            rule_id="v_range",
            name="Velocity Range Check",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="v",
            parameters={'min': 0.0, 'max': 20.0},
            severity="error"
        ))

        # Temperature rules
        self.add_rule(ValidationRule(
            rule_id="T_sun_range",
            name="Sun-side Temperature Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="T_sun",
            parameters={'min': -50.0, 'max': 80.0},
            severity="error"
        ))

        self.add_rule(ValidationRule(
            rule_id="T_shade_range",
            name="Shade-side Temperature Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="T_shade",
            parameters={'min': -50.0, 'max': 80.0},
            severity="error"
        ))

        # Consistency check - temperature differential
        self.add_rule(ValidationRule(
            rule_id="T_diff_check",
            name="Temperature Differential Check",
            rule_type=ValidationRuleType.CROSS_FIELD_CHECK,
            field_name="T_delta",
            parameters={'max_diff': 30.0},
            severity="warning",
            description="Temperature differential should not exceed 30°C"
        ))

        # Froude number check
        self.add_rule(ValidationRule(
            rule_id="fr_range",
            name="Froude Number Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="fr",
            parameters={'min': 0.0, 'max': 2.0},
            severity="warning"
        ))

        # Flow rate rules
        self.add_rule(ValidationRule(
            rule_id="Q_in_range",
            name="Inlet Flow Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="Q_in",
            parameters={'min': 0.0, 'max': 500.0},
            severity="error"
        ))

        self.add_rule(ValidationRule(
            rule_id="Q_out_range",
            name="Outlet Flow Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="Q_out",
            parameters={'min': 0.0, 'max': 500.0},
            severity="error"
        ))

        # Structural rules
        self.add_rule(ValidationRule(
            rule_id="joint_gap_range",
            name="Joint Gap Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="joint_gap",
            parameters={'min': 0.0, 'max': 50.0},
            severity="error"
        ))

        self.add_rule(ValidationRule(
            rule_id="bearing_stress_range",
            name="Bearing Stress Range",
            rule_type=ValidationRuleType.RANGE_CHECK,
            field_name="bearing_stress",
            parameters={'min': 0.0, 'max': 100.0},
            severity="error"
        ))

        # Timeliness rule
        self.add_rule(ValidationRule(
            rule_id="data_freshness",
            name="Data Freshness Check",
            rule_type=ValidationRuleType.TEMPORAL_CHECK,
            field_name="*",
            parameters={'max_age_seconds': 30},
            severity="warning"
        ))

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """Remove a validation rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def update_statistics(self, field_name: str, value: float):
        """Update running statistics for a field."""
        if field_name not in self._field_statistics:
            self._field_statistics[field_name] = {
                'count': 0,
                'mean': 0,
                'M2': 0,
                'min': np.inf,
                'max': -np.inf
            }

        stats = self._field_statistics[field_name]
        stats['count'] += 1
        delta = value - stats['mean']
        stats['mean'] += delta / stats['count']
        delta2 = value - stats['mean']
        stats['M2'] += delta * delta2
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)

    def get_statistics(self, field_name: str) -> Dict[str, float]:
        """Get statistics for a field."""
        if field_name not in self._field_statistics:
            return {}

        stats = self._field_statistics[field_name]
        std = np.sqrt(stats['M2'] / stats['count']) if stats['count'] > 0 else 0

        return {
            'count': stats['count'],
            'mean': stats['mean'],
            'std': std,
            'min': stats['min'],
            'max': stats['max']
        }

    def validate(self, data: Dict[str, Any],
                 timestamp: Optional[float] = None) -> DataQualityMetrics:
        """
        Validate data against all active rules.

        Args:
            data: Dictionary of field values
            timestamp: Data timestamp for timeliness checks

        Returns:
            DataQualityMetrics with validation results
        """
        timestamp = timestamp or time.time()
        context = {'timestamp': timestamp, 'data': data}

        metrics = DataQualityMetrics(
            source_name="sensor_data",
            timestamp=timestamp,
            total_records=1
        )

        error_count = 0
        warning_count = 0
        errors = []
        warnings = []

        # Field validation counts
        fields_total = len(data)
        fields_valid = 0
        fields_present = 0

        for field_name, value in data.items():
            # Update statistics
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.update_statistics(field_name, value)
                fields_present += 1

            field_valid = True

            # Apply all relevant rules
            for rule in self.rules.values():
                if not rule.is_active:
                    continue

                if rule.field_name != "*" and rule.field_name != field_name:
                    continue

                # Handle cross-field checks
                if rule.rule_type == ValidationRuleType.CROSS_FIELD_CHECK:
                    if rule.field_name == "T_delta":
                        T_sun = data.get('T_sun', 0)
                        T_shade = data.get('T_shade', 0)
                        value = abs(T_sun - T_shade)

                is_valid, message = rule.validate(value, context)

                if not is_valid:
                    field_valid = False
                    violation = {
                        'rule_id': rule.rule_id,
                        'field': field_name,
                        'value': value,
                        'message': message,
                        'severity': rule.severity
                    }

                    if rule.severity == "error":
                        error_count += 1
                        errors.append(violation)
                    else:
                        warning_count += 1
                        warnings.append(violation)

            if field_valid:
                fields_valid += 1

        # Calculate quality scores
        if fields_total > 0:
            metrics.accuracy_score = fields_valid / fields_total
            metrics.completeness_score = fields_present / fields_total
            metrics.validity_score = 1.0 - (error_count / max(1, fields_total))

        if error_count == 0:
            metrics.valid_records = 1
            metrics.consistency_score = 1.0
        else:
            metrics.valid_records = 0
            metrics.consistency_score = max(0, 1.0 - error_count * 0.1)

        # Timeliness score based on data age
        data_age = time.time() - timestamp
        metrics.timeliness_score = max(0, 1.0 - data_age / 60.0)

        metrics.error_count = error_count
        metrics.warning_count = warning_count
        metrics.errors = errors
        metrics.warnings = warnings

        # Store in history
        self._validation_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })

        return metrics

    def get_validation_report(self, lookback_seconds: float = 3600) -> Dict[str, Any]:
        """Generate validation report for recent data."""
        cutoff = time.time() - lookback_seconds

        recent = [h for h in self._validation_history if h['timestamp'] > cutoff]

        if not recent:
            return {'status': 'no_data', 'records': 0}

        metrics_list = [h['metrics'] for h in recent]

        return {
            'status': 'ok',
            'period_seconds': lookback_seconds,
            'records': len(recent),
            'average_scores': {
                'overall': np.mean([m.overall_score for m in metrics_list]),
                'accuracy': np.mean([m.accuracy_score for m in metrics_list]),
                'completeness': np.mean([m.completeness_score for m in metrics_list]),
                'consistency': np.mean([m.consistency_score for m in metrics_list]),
                'timeliness': np.mean([m.timeliness_score for m in metrics_list]),
                'validity': np.mean([m.validity_score for m in metrics_list])
            },
            'error_rate': sum(m.error_count for m in metrics_list) / len(recent),
            'warning_rate': sum(m.warning_count for m in metrics_list) / len(recent),
            'field_statistics': {
                name: self.get_statistics(name)
                for name in self._field_statistics.keys()
            }
        }


class DataLineageTracker:
    """
    Track data lineage and provenance.

    Maintains a complete history of data transformations.
    """

    def __init__(self):
        self.lineage_records: Dict[str, DataLineageRecord] = {}
        self._record_index: Dict[str, List[str]] = {}  # source_id -> lineage_records

    def record_creation(self, source_system: str, source_id: str,
                        metadata: Optional[Dict] = None) -> str:
        """Record data creation."""
        record_id = str(uuid.uuid4())
        record = DataLineageRecord(
            record_id=record_id,
            timestamp=time.time(),
            source_system=source_system,
            source_id=source_id,
            operation="create",
            metadata=metadata or {}
        )

        self.lineage_records[record_id] = record
        self._index_record(source_id, record_id)

        return record_id

    def record_transformation(self, input_ids: List[str], output_id: str,
                              transformation: str, metadata: Optional[Dict] = None) -> str:
        """Record data transformation."""
        record_id = str(uuid.uuid4())
        record = DataLineageRecord(
            record_id=record_id,
            timestamp=time.time(),
            source_system="transformation",
            source_id=output_id,
            operation="transform",
            input_records=input_ids,
            output_records=[output_id],
            transformation=transformation,
            metadata=metadata or {}
        )

        self.lineage_records[record_id] = record
        self._index_record(output_id, record_id)

        return record_id

    def record_merge(self, input_ids: List[str], output_id: str,
                     metadata: Optional[Dict] = None) -> str:
        """Record data merge operation."""
        record_id = str(uuid.uuid4())
        record = DataLineageRecord(
            record_id=record_id,
            timestamp=time.time(),
            source_system="merge",
            source_id=output_id,
            operation="merge",
            input_records=input_ids,
            output_records=[output_id],
            metadata=metadata or {}
        )

        self.lineage_records[record_id] = record
        self._index_record(output_id, record_id)

        return record_id

    def _index_record(self, source_id: str, record_id: str):
        """Index a record by source ID."""
        if source_id not in self._record_index:
            self._record_index[source_id] = []
        self._record_index[source_id].append(record_id)

    def get_lineage(self, data_id: str) -> List[DataLineageRecord]:
        """Get complete lineage for a data item."""
        if data_id not in self._record_index:
            return []

        result = []
        visited = set()
        queue = [data_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id in self._record_index:
                for record_id in self._record_index[current_id]:
                    record = self.lineage_records.get(record_id)
                    if record:
                        result.append(record)
                        queue.extend(record.input_records)

        return sorted(result, key=lambda r: r.timestamp)

    def get_impact_analysis(self, source_id: str) -> List[str]:
        """Find all data items affected by a source."""
        affected = set()

        for record in self.lineage_records.values():
            if source_id in record.input_records:
                affected.update(record.output_records)

        # Recursively find downstream impacts
        queue = list(affected)
        while queue:
            current = queue.pop(0)
            for record in self.lineage_records.values():
                if current in record.input_records:
                    for output in record.output_records:
                        if output not in affected:
                            affected.add(output)
                            queue.append(output)

        return list(affected)


class DataAccessController:
    """
    Data access control and audit management.

    Implements role-based access control and comprehensive auditing.
    """

    def __init__(self):
        self.audit_log: deque = deque(maxlen=100000)
        self.access_policies: Dict[str, Dict[str, Set[str]]] = {}  # role -> resource -> actions
        self.user_roles: Dict[str, Set[str]] = {}

        # Initialize default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default access policies."""
        # Admin role - full access
        self.access_policies["admin"] = {
            "*": {"read", "write", "delete", "export", "admin"}
        }

        # Operator role - operational access
        self.access_policies["operator"] = {
            "sensor_data": {"read"},
            "actuator_data": {"read", "write"},
            "control_commands": {"read", "write"},
            "alerts": {"read", "write"}
        }

        # Analyst role - read and export
        self.access_policies["analyst"] = {
            "sensor_data": {"read", "export"},
            "historical_data": {"read", "export"},
            "reports": {"read", "export"}
        }

        # Viewer role - read only
        self.access_policies["viewer"] = {
            "sensor_data": {"read"},
            "status": {"read"}
        }

    def assign_role(self, user_id: str, role: str):
        """Assign a role to a user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)

    def revoke_role(self, user_id: str, role: str):
        """Revoke a role from a user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)

    def check_access(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has access to perform action on resource."""
        if user_id not in self.user_roles:
            return False

        for role in self.user_roles[user_id]:
            if role not in self.access_policies:
                continue

            policy = self.access_policies[role]

            # Check wildcard policy
            if "*" in policy and action in policy["*"]:
                return True

            # Check specific resource policy
            if resource in policy and action in policy[resource]:
                return True

        return False

    def audit_access(self, user_id: str, action: str, resource: str,
                     record_ids: Optional[List[str]] = None,
                     success: bool = True, details: str = "",
                     classification: DataClassification = DataClassification.INTERNAL) -> str:
        """Record an access audit entry."""
        audit_id = str(uuid.uuid4())

        audit = DataAccessAudit(
            audit_id=audit_id,
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            data_source=resource,
            record_ids=record_ids or [],
            success=success,
            details=details,
            classification=classification
        )

        self.audit_log.append(audit)
        return audit_id

    def get_audit_trail(self, user_id: Optional[str] = None,
                        resource: Optional[str] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit trail with filters."""
        result = []

        for audit in reversed(self.audit_log):
            if len(result) >= limit:
                break

            if user_id and audit.user_id != user_id:
                continue

            if resource and audit.data_source != resource:
                continue

            if start_time and audit.timestamp < start_time:
                continue

            if end_time and audit.timestamp > end_time:
                continue

            result.append({
                'audit_id': audit.audit_id,
                'timestamp': audit.timestamp,
                'user_id': audit.user_id,
                'action': audit.action,
                'resource': audit.data_source,
                'success': audit.success,
                'details': audit.details,
                'classification': audit.classification.value
            })

        return result


class DataLifecycleManager:
    """
    Manage data lifecycle including retention and archival.
    """

    def __init__(self):
        self.retention_policies: Dict[str, Dict[str, Any]] = {}
        self.archival_queue: deque = deque(maxlen=10000)
        self.deletion_queue: deque = deque(maxlen=10000)

        # Initialize default retention policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default retention policies."""
        # Real-time data - short retention
        self.retention_policies["realtime"] = {
            'retention_days': 7,
            'aggregation_after_days': 1,
            'aggregation_interval': '1min',
            'archive_after_days': 30
        }

        # Operational data - medium retention
        self.retention_policies["operational"] = {
            'retention_days': 90,
            'aggregation_after_days': 7,
            'aggregation_interval': '5min',
            'archive_after_days': 365
        }

        # Historical data - long retention
        self.retention_policies["historical"] = {
            'retention_days': 365 * 5,  # 5 years
            'aggregation_after_days': 30,
            'aggregation_interval': '1hour',
            'archive_after_days': 365 * 10
        }

        # Audit data - compliance retention
        self.retention_policies["audit"] = {
            'retention_days': 365 * 7,  # 7 years for compliance
            'aggregation_after_days': None,  # No aggregation
            'archive_after_days': 365 * 7
        }

    def apply_retention_policy(self, data_type: str,
                               records: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Apply retention policy to records."""
        if data_type not in self.retention_policies:
            return {'active': records, 'archive': [], 'delete': []}

        policy = self.retention_policies[data_type]
        current_time = time.time()

        active = []
        to_archive = []
        to_delete = []

        for record in records:
            record_time = record.get('timestamp', 0)
            age_days = (current_time - record_time) / 86400

            if age_days > policy['retention_days']:
                to_delete.append(record)
            elif age_days > policy.get('archive_after_days', float('inf')):
                to_archive.append(record)
            else:
                active.append(record)

        return {
            'active': active,
            'archive': to_archive,
            'delete': to_delete
        }

    def get_lifecycle_status(self, data_type: str) -> Dict[str, Any]:
        """Get lifecycle status for a data type."""
        if data_type not in self.retention_policies:
            return {'status': 'no_policy'}

        policy = self.retention_policies[data_type]

        return {
            'data_type': data_type,
            'retention_days': policy['retention_days'],
            'archive_after_days': policy.get('archive_after_days'),
            'aggregation_interval': policy.get('aggregation_interval'),
            'policy': policy
        }


class DataGovernanceEngine:
    """
    Complete data governance engine for TAOS.

    Integrates all governance capabilities.
    """

    def __init__(self):
        self.quality_validator = DataQualityValidator()
        self.lineage_tracker = DataLineageTracker()
        self.access_controller = DataAccessController()
        self.lifecycle_manager = DataLifecycleManager()

        # Governance metrics
        self.metrics_history: deque = deque(maxlen=10000)
        self._start_time = time.time()

    def process_data(self, data: Dict[str, Any], source: str,
                     user_id: str = "system") -> Dict[str, Any]:
        """
        Process data through governance pipeline.

        Args:
            data: Data to process
            source: Data source identifier
            user_id: User performing the operation

        Returns:
            Governance results
        """
        timestamp = data.get('timestamp', time.time())

        # 1. Validate data quality
        quality_metrics = self.quality_validator.validate(data, timestamp)

        # 2. Record lineage
        lineage_id = self.lineage_tracker.record_creation(
            source_system=source,
            source_id=str(uuid.uuid4()),
            metadata={'original_data': data}
        )

        # 3. Audit access
        self.access_controller.audit_access(
            user_id=user_id,
            action="write",
            resource=source,
            success=quality_metrics.error_count == 0,
            details=f"Quality score: {quality_metrics.overall_score:.2f}"
        )

        # 4. Store metrics
        result = {
            'timestamp': timestamp,
            'source': source,
            'quality': {
                'overall_score': quality_metrics.overall_score,
                'is_valid': quality_metrics.error_count == 0,
                'errors': quality_metrics.errors,
                'warnings': quality_metrics.warnings
            },
            'lineage_id': lineage_id,
            'processed': True
        }

        self.metrics_history.append(result)

        return result

    def get_governance_dashboard(self) -> Dict[str, Any]:
        """Get governance dashboard metrics."""
        uptime = time.time() - self._start_time

        # Calculate recent quality metrics
        recent_metrics = list(self.metrics_history)[-1000:]

        if recent_metrics:
            avg_quality = np.mean([m['quality']['overall_score'] for m in recent_metrics])
            valid_rate = np.mean([1 if m['quality']['is_valid'] else 0 for m in recent_metrics])
        else:
            avg_quality = 1.0
            valid_rate = 1.0

        return {
            'uptime_seconds': uptime,
            'total_records_processed': len(self.metrics_history),
            'quality_metrics': {
                'average_quality_score': avg_quality,
                'validation_pass_rate': valid_rate,
                'detailed_report': self.quality_validator.get_validation_report()
            },
            'lineage': {
                'total_records': len(self.lineage_tracker.lineage_records)
            },
            'access_control': {
                'total_audits': len(self.access_controller.audit_log),
                'recent_audits': self.access_controller.get_audit_trail(limit=10)
            },
            'lifecycle': {
                'policies': list(self.lifecycle_manager.retention_policies.keys())
            }
        }

    def export_compliance_report(self, start_time: Optional[float] = None,
                                  end_time: Optional[float] = None) -> Dict[str, Any]:
        """Export compliance report for regulatory purposes."""
        start_time = start_time or (time.time() - 86400 * 30)  # Last 30 days
        end_time = end_time or time.time()

        return {
            'report_type': 'data_governance_compliance',
            'generated_at': time.time(),
            'period': {
                'start': start_time,
                'end': end_time
            },
            'data_quality': self.quality_validator.get_validation_report(end_time - start_time),
            'access_audit': self.access_controller.get_audit_trail(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            ),
            'data_retention': {
                policy_name: self.lifecycle_manager.get_lifecycle_status(policy_name)
                for policy_name in self.lifecycle_manager.retention_policies.keys()
            }
        }
