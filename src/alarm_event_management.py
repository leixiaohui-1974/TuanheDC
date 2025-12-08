"""
TAOS V3.10 Alarm and Event Management Module
告警与事件管理模块

Features:
- Multi-level alarm configuration (warning, alert, critical, emergency)
- Alarm rules engine with condition evaluation
- Event correlation and root cause analysis
- Alarm acknowledgment and escalation
- Notification dispatch (email, SMS, push)
- Alarm statistics and reporting
"""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import json
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class AlarmSeverity(Enum):
    """Alarm severity levels"""
    INFO = 1
    WARNING = 2
    ALERT = 3
    CRITICAL = 4
    EMERGENCY = 5

    @property
    def display_name(self) -> str:
        names = {1: "信息", 2: "警告", 3: "告警", 4: "严重", 5: "紧急"}
        return names.get(self.value, "未知")


class AlarmState(Enum):
    """Alarm state"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    SHELVED = "shelved"
    SUPPRESSED = "suppressed"


class AlarmCategory(Enum):
    """Alarm categories"""
    PROCESS = "process"
    EQUIPMENT = "equipment"
    SAFETY = "safety"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    ENVIRONMENTAL = "environmental"


class ComparisonOperator(Enum):
    """Comparison operators for alarm conditions"""
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    IN_RANGE = "in_range"
    OUT_OF_RANGE = "out_of_range"
    CONTAINS = "contains"
    RATE_OF_CHANGE = "rate_of_change"
    DEVIATION = "deviation"


class NotificationType(Enum):
    """Notification types"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    VOICE = "voice"


@dataclass
class AlarmCondition:
    """Alarm trigger condition"""
    tag_id: str
    operator: ComparisonOperator
    threshold: Any
    threshold_high: Optional[Any] = None  # For range operators
    deadband: float = 0.0
    delay_seconds: float = 0.0  # Time delay before triggering
    # For rate of change
    time_window: float = 60.0  # seconds
    # For deviation
    reference_value: Optional[float] = None

    def evaluate(self, current_value: Any, previous_values: List[tuple] = None) -> bool:
        """Evaluate if condition is met"""
        if current_value is None:
            return False

        try:
            if self.operator == ComparisonOperator.EQUAL:
                return current_value == self.threshold
            elif self.operator == ComparisonOperator.NOT_EQUAL:
                return current_value != self.threshold
            elif self.operator == ComparisonOperator.GREATER:
                return float(current_value) > float(self.threshold) + self.deadband
            elif self.operator == ComparisonOperator.GREATER_EQUAL:
                return float(current_value) >= float(self.threshold)
            elif self.operator == ComparisonOperator.LESS:
                return float(current_value) < float(self.threshold) - self.deadband
            elif self.operator == ComparisonOperator.LESS_EQUAL:
                return float(current_value) <= float(self.threshold)
            elif self.operator == ComparisonOperator.IN_RANGE:
                return self.threshold <= float(current_value) <= self.threshold_high
            elif self.operator == ComparisonOperator.OUT_OF_RANGE:
                return float(current_value) < self.threshold or float(current_value) > self.threshold_high
            elif self.operator == ComparisonOperator.CONTAINS:
                return str(self.threshold) in str(current_value)
            elif self.operator == ComparisonOperator.RATE_OF_CHANGE:
                return self._evaluate_rate_of_change(current_value, previous_values)
            elif self.operator == ComparisonOperator.DEVIATION:
                if self.reference_value is not None:
                    deviation = abs(float(current_value) - self.reference_value)
                    return deviation > float(self.threshold)
                return False
        except (ValueError, TypeError):
            return False

        return False

    def _evaluate_rate_of_change(self, current_value: float,
                                   previous_values: List[tuple]) -> bool:
        """Evaluate rate of change condition"""
        if not previous_values:
            return False

        # Find value from time_window ago
        now = time.time()
        target_time = now - self.time_window

        for timestamp, value in reversed(previous_values):
            if timestamp <= target_time:
                rate = abs(float(current_value) - float(value)) / self.time_window
                return rate > float(self.threshold)

        return False


@dataclass
class AlarmRule:
    """Alarm rule definition"""
    rule_id: str
    name: str
    description: str
    severity: AlarmSeverity
    category: AlarmCategory
    conditions: List[AlarmCondition]
    condition_logic: str = "AND"  # AND, OR
    enabled: bool = True
    # Auto-clear settings
    auto_clear: bool = True
    clear_delay_seconds: float = 0.0
    # Notification settings
    notify: bool = True
    notification_groups: List[str] = field(default_factory=list)
    # Escalation settings
    escalation_enabled: bool = False
    escalation_delay_minutes: int = 30
    escalation_severity: Optional[AlarmSeverity] = None
    # Suppression settings
    suppression_window: float = 0.0  # Minimum time between same alarms
    # Priority
    priority: int = 0
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    # Custom actions
    on_activate: Optional[str] = None
    on_clear: Optional[str] = None

    def evaluate(self, values: Dict[str, Any],
                 history: Dict[str, List[tuple]] = None) -> bool:
        """Evaluate all conditions"""
        if not self.enabled:
            return False

        history = history or {}
        results = []

        for condition in self.conditions:
            value = values.get(condition.tag_id)
            prev_values = history.get(condition.tag_id, [])
            results.append(condition.evaluate(value, prev_values))

        if self.condition_logic == "AND":
            return all(results) if results else False
        else:  # OR
            return any(results) if results else False


@dataclass
class AlarmInstance:
    """Active alarm instance"""
    alarm_id: str
    rule_id: str
    name: str
    description: str
    severity: AlarmSeverity
    category: AlarmCategory
    state: AlarmState
    timestamp: datetime
    source_tag: str
    trigger_value: Any
    threshold: Any
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    cleared_at: Optional[datetime] = None
    escalated: bool = False
    escalation_count: int = 0
    notes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'alarm_id': self.alarm_id,
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'severity': self.severity.value,
            'severity_name': self.severity.display_name,
            'category': self.category.value,
            'state': self.state.value,
            'timestamp': self.timestamp.isoformat(),
            'source_tag': self.source_tag,
            'trigger_value': self.trigger_value,
            'threshold': self.threshold,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'cleared_at': self.cleared_at.isoformat() if self.cleared_at else None,
            'escalated': self.escalated,
            'notes': self.notes,
            'metadata': self.metadata
        }


@dataclass
class Event:
    """System event"""
    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    message: str
    severity: AlarmSeverity = AlarmSeverity.INFO
    category: str = "system"
    user: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    related_alarm_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category,
            'user': self.user,
            'details': self.details,
            'related_alarm_id': self.related_alarm_id
        }


@dataclass
class NotificationConfig:
    """Notification configuration"""
    group_id: str
    name: str
    channels: List[NotificationType]
    recipients: List[str]  # Email addresses, phone numbers, etc.
    enabled: bool = True
    min_severity: AlarmSeverity = AlarmSeverity.WARNING
    schedule: Optional[Dict[str, Any]] = None  # Active hours
    throttle_minutes: int = 5  # Minimum time between notifications


class NotificationDispatcher(ABC):
    """Abstract notification dispatcher"""

    @abstractmethod
    def send(self, recipient: str, subject: str, message: str,
             alarm: Optional[AlarmInstance] = None) -> bool:
        """Send notification"""
        pass


class EmailNotifier(NotificationDispatcher):
    """Email notification dispatcher"""

    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 25):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def send(self, recipient: str, subject: str, message: str,
             alarm: Optional[AlarmInstance] = None) -> bool:
        """Send email notification"""
        try:
            # Simulate email sending
            # In production: use smtplib
            logger.info(f"Email sent to {recipient}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class SMSNotifier(NotificationDispatcher):
    """SMS notification dispatcher"""

    def __init__(self, api_endpoint: str = "", api_key: str = ""):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def send(self, recipient: str, subject: str, message: str,
             alarm: Optional[AlarmInstance] = None) -> bool:
        """Send SMS notification"""
        try:
            # Simulate SMS sending
            logger.info(f"SMS sent to {recipient}: {message[:50]}")
            return True
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return False


class WebhookNotifier(NotificationDispatcher):
    """Webhook notification dispatcher"""

    def __init__(self, url: str = "", auth_token: str = ""):
        self.url = url
        self.auth_token = auth_token

    def send(self, recipient: str, subject: str, message: str,
             alarm: Optional[AlarmInstance] = None) -> bool:
        """Send webhook notification"""
        try:
            payload = {
                'recipient': recipient,
                'subject': subject,
                'message': message,
                'alarm': alarm.to_dict() if alarm else None
            }
            # Simulate webhook call
            logger.info(f"Webhook sent to {self.url}: {subject}")
            return True
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


class EventCorrelator:
    """Event correlation engine for root cause analysis"""

    def __init__(self, correlation_window: float = 60.0):
        self.correlation_window = correlation_window
        self.event_buffer: List[Event] = []
        self.correlation_rules: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def add_event(self, event: Event):
        """Add event to correlation buffer"""
        with self._lock:
            self.event_buffer.append(event)
            # Clean old events
            cutoff = datetime.now() - timedelta(seconds=self.correlation_window * 2)
            self.event_buffer = [e for e in self.event_buffer if e.timestamp > cutoff]

    def add_correlation_rule(self, rule_id: str, pattern: Dict):
        """Add correlation rule"""
        self.correlation_rules[rule_id] = pattern

    def find_correlations(self, alarm: AlarmInstance) -> List[Dict]:
        """Find correlated events for an alarm"""
        correlations = []
        window_start = alarm.timestamp - timedelta(seconds=self.correlation_window)
        window_end = alarm.timestamp + timedelta(seconds=5)

        with self._lock:
            related_events = [
                e for e in self.event_buffer
                if window_start <= e.timestamp <= window_end
                   and e.event_id != alarm.alarm_id
            ]

            # Group by source
            by_source = defaultdict(list)
            for event in related_events:
                by_source[event.source].append(event)

            # Build correlation info
            if related_events:
                correlations.append({
                    'type': 'temporal',
                    'count': len(related_events),
                    'events': [e.to_dict() for e in related_events[:10]],
                    'sources': list(by_source.keys())
                })

        return correlations

    def suggest_root_cause(self, alarm: AlarmInstance,
                            correlations: List[Dict]) -> Optional[str]:
        """Suggest possible root cause"""
        if not correlations:
            return None

        # Simple heuristic: look for communication errors before process alarms
        for corr in correlations:
            if corr['type'] == 'temporal':
                for event in corr.get('events', []):
                    if 'communication' in event.get('category', '').lower():
                        return f"可能的根因: 通信故障 ({event.get('source')})"
                    if 'equipment' in event.get('category', '').lower():
                        return f"可能的根因: 设备故障 ({event.get('source')})"

        return None


class AlarmManager:
    """
    Central alarm and event management system
    告警与事件管理中心
    """

    def __init__(self):
        self.rules: Dict[str, AlarmRule] = {}
        self.active_alarms: Dict[str, AlarmInstance] = {}
        self.alarm_history: List[AlarmInstance] = []
        self.events: List[Event] = []
        self.notification_groups: Dict[str, NotificationConfig] = {}
        self.notifiers: Dict[NotificationType, NotificationDispatcher] = {
            NotificationType.EMAIL: EmailNotifier(),
            NotificationType.SMS: SMSNotifier(),
            NotificationType.WEBHOOK: WebhookNotifier()
        }
        self.correlator = EventCorrelator()
        self._value_history: Dict[str, List[tuple]] = {}
        self._last_notification: Dict[str, datetime] = {}
        self._suppression_tracker: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[AlarmInstance], None]] = []
        self._evaluation_thread: Optional[threading.Thread] = None
        self._running = False

    def add_rule(self, rule: AlarmRule) -> bool:
        """Add alarm rule"""
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added alarm rule: {rule.name}")
            return True

    def remove_rule(self, rule_id: str) -> bool:
        """Remove alarm rule"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                return True
            return False

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update alarm rule"""
        with self._lock:
            if rule_id not in self.rules:
                return False

            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            return True

    def add_notification_group(self, config: NotificationConfig) -> bool:
        """Add notification group"""
        self.notification_groups[config.group_id] = config
        return True

    def process_values(self, values: Dict[str, Any]):
        """Process incoming values and check alarm conditions"""
        timestamp = time.time()

        # Update value history
        for tag_id, value in values.items():
            if tag_id not in self._value_history:
                self._value_history[tag_id] = []
            self._value_history[tag_id].append((timestamp, value))
            # Keep last 1000 values
            if len(self._value_history[tag_id]) > 1000:
                self._value_history[tag_id] = self._value_history[tag_id][-1000:]

        # Evaluate all rules
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            try:
                is_triggered = rule.evaluate(values, self._value_history)
                existing_alarm = self._find_active_alarm_by_rule(rule_id)

                if is_triggered and not existing_alarm:
                    # Check suppression
                    if self._is_suppressed(rule_id, rule.suppression_window):
                        continue

                    # Create new alarm
                    alarm = self._create_alarm(rule, values)
                    self._activate_alarm(alarm)

                elif not is_triggered and existing_alarm:
                    # Clear alarm if auto-clear enabled
                    if rule.auto_clear:
                        self.clear_alarm(existing_alarm.alarm_id)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")

    def _find_active_alarm_by_rule(self, rule_id: str) -> Optional[AlarmInstance]:
        """Find active alarm for a rule"""
        for alarm in self.active_alarms.values():
            if alarm.rule_id == rule_id and alarm.state in [AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED]:
                return alarm
        return None

    def _is_suppressed(self, rule_id: str, window: float) -> bool:
        """Check if alarm is in suppression window"""
        if window <= 0:
            return False

        last_time = self._suppression_tracker.get(rule_id)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            return elapsed < window
        return False

    def _create_alarm(self, rule: AlarmRule, values: Dict[str, Any]) -> AlarmInstance:
        """Create alarm instance from rule"""
        # Get trigger info from first condition
        source_tag = rule.conditions[0].tag_id if rule.conditions else ""
        trigger_value = values.get(source_tag)
        threshold = rule.conditions[0].threshold if rule.conditions else None

        return AlarmInstance(
            alarm_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            category=rule.category,
            state=AlarmState.ACTIVE,
            timestamp=datetime.now(),
            source_tag=source_tag,
            trigger_value=trigger_value,
            threshold=threshold
        )

    def _activate_alarm(self, alarm: AlarmInstance):
        """Activate an alarm"""
        with self._lock:
            self.active_alarms[alarm.alarm_id] = alarm
            self._suppression_tracker[alarm.rule_id] = datetime.now()

            # Add event
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type="alarm_activated",
                timestamp=alarm.timestamp,
                source=alarm.source_tag,
                message=f"告警激活: {alarm.name}",
                severity=alarm.severity,
                category=alarm.category.value,
                related_alarm_id=alarm.alarm_id
            )
            self.events.append(event)
            self.correlator.add_event(event)

            # Find correlations
            correlations = self.correlator.find_correlations(alarm)
            if correlations:
                alarm.metadata['correlations'] = correlations
                root_cause = self.correlator.suggest_root_cause(alarm, correlations)
                if root_cause:
                    alarm.metadata['suggested_root_cause'] = root_cause

            # Notify
            rule = self.rules.get(alarm.rule_id)
            if rule and rule.notify:
                self._send_notifications(alarm, rule)

            # Callbacks
            for callback in self._callbacks:
                try:
                    callback(alarm)
                except Exception as e:
                    logger.error(f"Alarm callback error: {e}")

            logger.info(f"Alarm activated: {alarm.name} (ID: {alarm.alarm_id})")

    def acknowledge_alarm(self, alarm_id: str, user: str,
                          notes: str = "") -> bool:
        """Acknowledge an alarm"""
        with self._lock:
            if alarm_id not in self.active_alarms:
                return False

            alarm = self.active_alarms[alarm_id]
            if alarm.state != AlarmState.ACTIVE:
                return False

            alarm.state = AlarmState.ACKNOWLEDGED
            alarm.acknowledged_by = user
            alarm.acknowledged_at = datetime.now()

            if notes:
                alarm.notes.append({
                    'user': user,
                    'timestamp': datetime.now().isoformat(),
                    'text': notes
                })

            # Add event
            self.events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="alarm_acknowledged",
                timestamp=datetime.now(),
                source=alarm.source_tag,
                message=f"告警确认: {alarm.name}",
                severity=AlarmSeverity.INFO,
                user=user,
                related_alarm_id=alarm_id
            ))

            logger.info(f"Alarm acknowledged: {alarm.name} by {user}")
            return True

    def clear_alarm(self, alarm_id: str, user: Optional[str] = None) -> bool:
        """Clear an alarm"""
        with self._lock:
            if alarm_id not in self.active_alarms:
                return False

            alarm = self.active_alarms[alarm_id]
            alarm.state = AlarmState.CLEARED
            alarm.cleared_at = datetime.now()

            # Move to history
            self.alarm_history.append(alarm)
            del self.active_alarms[alarm_id]

            # Add event
            self.events.append(Event(
                event_id=str(uuid.uuid4()),
                event_type="alarm_cleared",
                timestamp=datetime.now(),
                source=alarm.source_tag,
                message=f"告警清除: {alarm.name}",
                severity=AlarmSeverity.INFO,
                user=user or "system",
                related_alarm_id=alarm_id
            ))

            logger.info(f"Alarm cleared: {alarm.name}")
            return True

    def shelve_alarm(self, alarm_id: str, duration_minutes: int,
                     user: str, reason: str = "") -> bool:
        """Shelve an alarm temporarily"""
        with self._lock:
            if alarm_id not in self.active_alarms:
                return False

            alarm = self.active_alarms[alarm_id]
            alarm.state = AlarmState.SHELVED
            alarm.metadata['shelved_by'] = user
            alarm.metadata['shelved_at'] = datetime.now().isoformat()
            alarm.metadata['shelve_duration'] = duration_minutes
            alarm.metadata['shelve_reason'] = reason
            alarm.metadata['shelve_expires'] = (
                    datetime.now() + timedelta(minutes=duration_minutes)
            ).isoformat()

            logger.info(f"Alarm shelved: {alarm.name} for {duration_minutes} minutes")
            return True

    def _send_notifications(self, alarm: AlarmInstance, rule: AlarmRule):
        """Send notifications for alarm"""
        for group_id in rule.notification_groups:
            config = self.notification_groups.get(group_id)
            if not config or not config.enabled:
                continue

            if alarm.severity.value < config.min_severity.value:
                continue

            # Check throttling
            throttle_key = f"{group_id}:{alarm.rule_id}"
            last_notify = self._last_notification.get(throttle_key)
            if last_notify:
                elapsed = (datetime.now() - last_notify).total_seconds() / 60
                if elapsed < config.throttle_minutes:
                    continue

            # Send to each channel
            subject = f"[{alarm.severity.display_name}] {alarm.name}"
            message = self._format_notification_message(alarm)

            for channel in config.channels:
                notifier = self.notifiers.get(channel)
                if notifier:
                    for recipient in config.recipients:
                        try:
                            notifier.send(recipient, subject, message, alarm)
                        except Exception as e:
                            logger.error(f"Notification failed: {e}")

            self._last_notification[throttle_key] = datetime.now()

    def _format_notification_message(self, alarm: AlarmInstance) -> str:
        """Format alarm notification message"""
        return f"""
告警通知
========
名称: {alarm.name}
级别: {alarm.severity.display_name}
时间: {alarm.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
来源: {alarm.source_tag}
触发值: {alarm.trigger_value}
阈值: {alarm.threshold}
描述: {alarm.description}

告警ID: {alarm.alarm_id}
"""

    def get_active_alarms(self, severity: Optional[AlarmSeverity] = None,
                          category: Optional[AlarmCategory] = None) -> List[AlarmInstance]:
        """Get active alarms with optional filtering"""
        alarms = list(self.active_alarms.values())

        if severity:
            alarms = [a for a in alarms if a.severity.value >= severity.value]

        if category:
            alarms = [a for a in alarms if a.category == category]

        return sorted(alarms, key=lambda a: (-a.severity.value, a.timestamp))

    def get_alarm_history(self, start: datetime, end: datetime,
                          limit: int = 1000) -> List[AlarmInstance]:
        """Get alarm history"""
        filtered = [
            a for a in self.alarm_history
            if start <= a.timestamp <= end
        ]
        return sorted(filtered, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_events(self, start: datetime, end: datetime,
                   event_type: Optional[str] = None,
                   limit: int = 1000) -> List[Event]:
        """Get events with filtering"""
        filtered = [
            e for e in self.events
            if start <= e.timestamp <= end
        ]

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_statistics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get alarm statistics"""
        cutoff = datetime.now() - timedelta(hours=period_hours)

        # Active alarms by severity
        active_by_severity = defaultdict(int)
        for alarm in self.active_alarms.values():
            active_by_severity[alarm.severity.display_name] += 1

        # Historical alarms in period
        period_alarms = [a for a in self.alarm_history if a.timestamp >= cutoff]

        # Average response time (time to acknowledge)
        response_times = []
        for alarm in period_alarms:
            if alarm.acknowledged_at and alarm.timestamp:
                delta = (alarm.acknowledged_at - alarm.timestamp).total_seconds()
                response_times.append(delta)

        avg_response = sum(response_times) / len(response_times) if response_times else 0

        # Top alarm sources
        source_counts = defaultdict(int)
        for alarm in period_alarms:
            source_counts[alarm.source_tag] += 1

        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'active_count': len(self.active_alarms),
            'active_by_severity': dict(active_by_severity),
            'period_hours': period_hours,
            'period_total': len(period_alarms),
            'avg_response_time_seconds': round(avg_response, 1),
            'top_sources': dict(top_sources),
            'unacknowledged_count': sum(
                1 for a in self.active_alarms.values()
                if a.state == AlarmState.ACTIVE
            )
        }

    def register_callback(self, callback: Callable[[AlarmInstance], None]):
        """Register alarm callback"""
        self._callbacks.append(callback)

    def log_event(self, event_type: str, source: str, message: str,
                  severity: AlarmSeverity = AlarmSeverity.INFO,
                  category: str = "system",
                  user: Optional[str] = None,
                  details: Dict[str, Any] = None) -> Event:
        """Log a system event"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            message=message,
            severity=severity,
            category=category,
            user=user,
            details=details or {}
        )

        with self._lock:
            self.events.append(event)
            self.correlator.add_event(event)

        return event

    def get_status(self) -> Dict[str, Any]:
        """Get alarm manager status"""
        return {
            'rules_count': len(self.rules),
            'active_alarms': len(self.active_alarms),
            'notification_groups': len(self.notification_groups),
            'events_in_buffer': len(self.events),
            'history_count': len(self.alarm_history)
        }


# Singleton instance
_alarm_manager: Optional[AlarmManager] = None


def get_alarm_manager() -> AlarmManager:
    """Get singleton instance of AlarmManager"""
    global _alarm_manager
    if _alarm_manager is None:
        _alarm_manager = AlarmManager()
    return _alarm_manager


# Helper functions for creating common alarm rules
def create_high_limit_rule(rule_id: str, name: str, tag_id: str,
                           limit: float, severity: AlarmSeverity = AlarmSeverity.ALERT,
                           deadband: float = 0.0) -> AlarmRule:
    """Create a high limit alarm rule"""
    return AlarmRule(
        rule_id=rule_id,
        name=name,
        description=f"{name} 超过上限 {limit}",
        severity=severity,
        category=AlarmCategory.PROCESS,
        conditions=[
            AlarmCondition(
                tag_id=tag_id,
                operator=ComparisonOperator.GREATER,
                threshold=limit,
                deadband=deadband
            )
        ]
    )


def create_low_limit_rule(rule_id: str, name: str, tag_id: str,
                          limit: float, severity: AlarmSeverity = AlarmSeverity.ALERT,
                          deadband: float = 0.0) -> AlarmRule:
    """Create a low limit alarm rule"""
    return AlarmRule(
        rule_id=rule_id,
        name=name,
        description=f"{name} 低于下限 {limit}",
        severity=severity,
        category=AlarmCategory.PROCESS,
        conditions=[
            AlarmCondition(
                tag_id=tag_id,
                operator=ComparisonOperator.LESS,
                threshold=limit,
                deadband=deadband
            )
        ]
    )


def create_range_rule(rule_id: str, name: str, tag_id: str,
                      low_limit: float, high_limit: float,
                      severity: AlarmSeverity = AlarmSeverity.WARNING) -> AlarmRule:
    """Create an out-of-range alarm rule"""
    return AlarmRule(
        rule_id=rule_id,
        name=name,
        description=f"{name} 超出范围 [{low_limit}, {high_limit}]",
        severity=severity,
        category=AlarmCategory.PROCESS,
        conditions=[
            AlarmCondition(
                tag_id=tag_id,
                operator=ComparisonOperator.OUT_OF_RANGE,
                threshold=low_limit,
                threshold_high=high_limit
            )
        ]
    )
