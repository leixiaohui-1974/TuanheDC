#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - Alert Notification System
团河渡槽自主运行系统 - 告警通知模块

Features:
- Multi-channel notification (email, SMS, webhook)
- Alert severity levels and escalation
- Alert acknowledgment and tracking
- Notification templates
- Rate limiting and deduplication
- Alert history and analytics
"""

import json
import sqlite3
import threading
import time
import hashlib
import smtplib
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict
from pathlib import Path


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ALARM = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlertCategory(Enum):
    """Alert categories"""
    HYDRAULIC = "hydraulic"
    STRUCTURAL = "structural"
    THERMAL = "thermal"
    SAFETY = "safety"
    SYSTEM = "system"
    CONTROL = "control"
    SENSOR = "sensor"
    COMMUNICATION = "communication"


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    LOG = "log"
    DATABASE = "database"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notification_count: int = 0
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.name,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'data': self.data,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'notification_count': self.notification_count,
            'escalation_level': self.escalation_level
        }


@dataclass
class NotificationTemplate:
    """Notification template"""
    template_id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    category_filter: List[AlertCategory] = field(default_factory=list)

    def render(self, alert: Alert) -> tuple:
        """Render template with alert data"""
        context = {
            'alert_id': alert.alert_id,
            'severity': alert.severity.name,
            'category': alert.category.value,
            'title': alert.title,
            'message': alert.message,
            'source': alert.source,
            'timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            **alert.data
        }

        subject = self.subject_template
        body = self.body_template

        for key, value in context.items():
            placeholder = f'{{{key}}}'
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        return subject, body


@dataclass
class NotificationConfig:
    """Notification channel configuration"""
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = "taos@example.com"
    email_recipients: List[str] = field(default_factory=list)

    # SMS settings (placeholder - would use service like Twilio)
    sms_api_url: str = ""
    sms_api_key: str = ""
    sms_recipients: List[str] = field(default_factory=list)

    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    rate_limit_window: int = 300  # 5 minutes
    rate_limit_max: int = 10      # max alerts per window
    dedup_window: int = 60        # deduplication window in seconds


class AlertStorage:
    """SQLite-based alert storage"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                data TEXT,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                resolved_at TEXT,
                notification_count INTEGER DEFAULT 0,
                escalation_level INTEGER DEFAULT 0,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                recipient TEXT,
                status TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                response TEXT,
                FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
            )
        """)

        conn.commit()
        conn.close()

    def save_alert(self, alert: Alert):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO alerts
            (alert_id, severity, category, title, message, source, timestamp,
             status, data, acknowledged_by, acknowledged_at, resolved_at,
             notification_count, escalation_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.severity.name,
            alert.category.value,
            alert.title,
            alert.message,
            alert.source,
            alert.timestamp.isoformat(),
            alert.status.value,
            json.dumps(alert.data),
            alert.acknowledged_by,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            alert.resolved_at.isoformat() if alert.resolved_at else None,
            alert.notification_count,
            alert.escalation_level
        ))

        conn.commit()
        conn.close()

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_alert(row)
        return None

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM alerts
            WHERE status IN ('active', 'acknowledged')
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_alert(row) for row in rows]

    def query_alerts(self, start_time: datetime = None, end_time: datetime = None,
                    severity: AlertSeverity = None, category: AlertCategory = None,
                    status: AlertStatus = None, limit: int = 100) -> List[Alert]:
        """Query alerts with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if severity:
            query += " AND severity = ?"
            params.append(severity.name)
        if category:
            query += " AND category = ?"
            params.append(category.value)
        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_alert(row) for row in rows]

    def save_notification(self, alert_id: str, channel: str, recipient: str,
                         status: str, response: str = None):
        """Save notification record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO notifications
            (alert_id, channel, recipient, status, sent_at, response)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (alert_id, channel, recipient, status, datetime.now().isoformat(), response))

        conn.commit()
        conn.close()

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start = datetime.now() - timedelta(hours=hours)

        # Total count
        cursor.execute("""
            SELECT COUNT(*) FROM alerts WHERE timestamp >= ?
        """, (start.isoformat(),))
        total = cursor.fetchone()[0]

        # By severity
        cursor.execute("""
            SELECT severity, COUNT(*) FROM alerts
            WHERE timestamp >= ?
            GROUP BY severity
        """, (start.isoformat(),))
        by_severity = dict(cursor.fetchall())

        # By category
        cursor.execute("""
            SELECT category, COUNT(*) FROM alerts
            WHERE timestamp >= ?
            GROUP BY category
        """, (start.isoformat(),))
        by_category = dict(cursor.fetchall())

        # By status
        cursor.execute("""
            SELECT status, COUNT(*) FROM alerts
            WHERE timestamp >= ?
            GROUP BY status
        """, (start.isoformat(),))
        by_status = dict(cursor.fetchall())

        # Average resolution time (for resolved alerts)
        cursor.execute("""
            SELECT AVG(
                julianday(resolved_at) - julianday(timestamp)
            ) * 24 * 60  -- minutes
            FROM alerts
            WHERE timestamp >= ? AND resolved_at IS NOT NULL
        """, (start.isoformat(),))
        avg_resolution = cursor.fetchone()[0]

        conn.close()

        return {
            'period_hours': hours,
            'total_alerts': total,
            'by_severity': by_severity,
            'by_category': by_category,
            'by_status': by_status,
            'avg_resolution_minutes': round(avg_resolution, 2) if avg_resolution else None
        }

    def _row_to_alert(self, row) -> Alert:
        """Convert database row to Alert object"""
        return Alert(
            alert_id=row[0],
            severity=AlertSeverity[row[1]],
            category=AlertCategory(row[2]),
            title=row[3],
            message=row[4],
            source=row[5],
            timestamp=datetime.fromisoformat(row[6]),
            status=AlertStatus(row[7]),
            data=json.loads(row[8]) if row[8] else {},
            acknowledged_by=row[9],
            acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None,
            resolved_at=datetime.fromisoformat(row[11]) if row[11] else None,
            notification_count=row[12] or 0,
            escalation_level=row[13] or 0
        )


class NotificationDispatcher:
    """Handles sending notifications through various channels"""

    def __init__(self, config: NotificationConfig, storage: AlertStorage):
        self.config = config
        self.storage = storage
        self.rate_limiter: Dict[str, List[float]] = defaultdict(list)
        self.sent_hashes: Dict[str, float] = {}  # For deduplication

    def send(self, alert: Alert, channels: List[NotificationChannel] = None) -> Dict[str, bool]:
        """Send alert notification through specified channels"""
        if channels is None:
            channels = [NotificationChannel.DATABASE, NotificationChannel.LOG]

            # Add channels based on severity
            if alert.severity.value >= AlertSeverity.ALARM.value:
                channels.append(NotificationChannel.WEBHOOK)
            if alert.severity.value >= AlertSeverity.CRITICAL.value:
                channels.append(NotificationChannel.EMAIL)
            if alert.severity == AlertSeverity.EMERGENCY:
                channels.append(NotificationChannel.SMS)

        results = {}

        # Check rate limiting
        if not self._check_rate_limit(alert):
            return {'rate_limited': True}

        # Check deduplication
        if self._is_duplicate(alert):
            return {'duplicate': True}

        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    success = self._send_email(alert)
                elif channel == NotificationChannel.SMS:
                    success = self._send_sms(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    success = self._send_webhook(alert)
                elif channel == NotificationChannel.LOG:
                    success = self._send_log(alert)
                elif channel == NotificationChannel.DATABASE:
                    success = self._send_database(alert)
                else:
                    success = False

                results[channel.value] = success

                # Record notification
                self.storage.save_notification(
                    alert.alert_id, channel.value, "", "success" if success else "failed"
                )

            except Exception as e:
                results[channel.value] = False
                self.storage.save_notification(
                    alert.alert_id, channel.value, "", "error", str(e)
                )

        return results

    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if rate limit allows sending"""
        now = time.time()
        key = f"{alert.category.value}_{alert.severity.name}"

        # Clean old entries
        self.rate_limiter[key] = [
            t for t in self.rate_limiter[key]
            if now - t < self.config.rate_limit_window
        ]

        # Check limit
        if len(self.rate_limiter[key]) >= self.config.rate_limit_max:
            return False

        # Add current
        self.rate_limiter[key].append(now)
        return True

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is duplicate within dedup window"""
        now = time.time()

        # Clean old hashes
        self.sent_hashes = {
            h: t for h, t in self.sent_hashes.items()
            if now - t < self.config.dedup_window
        }

        # Generate hash
        hash_str = f"{alert.category.value}:{alert.title}:{alert.source}"
        alert_hash = hashlib.md5(hash_str.encode()).hexdigest()

        if alert_hash in self.sent_hashes:
            return True

        self.sent_hashes[alert_hash] = now
        return False

    def _send_email(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.config.email_recipients:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[TAOS {alert.severity.name}] {alert.title}"

            body = f"""
TAOS Alert Notification
=======================

Severity: {alert.severity.name}
Category: {alert.category.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

Title: {alert.title}

Message:
{alert.message}

Data:
{json.dumps(alert.data, indent=2, ensure_ascii=False)}

---
Alert ID: {alert.alert_id}
This is an automated message from TAOS - Tuanhe Aqueduct Autonomous Operation System
            """

            msg.attach(MIMEText(body, 'plain'))

            # Note: In production, this would actually send email
            # For now, just log it
            print(f"[EMAIL] Would send to {self.config.email_recipients}: {alert.title}")
            return True

        except Exception as e:
            print(f"Email send error: {e}")
            return False

    def _send_sms(self, alert: Alert) -> bool:
        """Send SMS notification"""
        if not self.config.sms_recipients or not self.config.sms_api_url:
            return False

        try:
            message = f"[TAOS {alert.severity.name}] {alert.title}: {alert.message[:100]}"

            # Note: In production, this would call SMS API
            print(f"[SMS] Would send to {self.config.sms_recipients}: {message}")
            return True

        except Exception as e:
            print(f"SMS send error: {e}")
            return False

    def _send_webhook(self, alert: Alert) -> bool:
        """Send webhook notification"""
        if not self.config.webhook_urls:
            return False

        success = True
        payload = json.dumps(alert.to_dict()).encode('utf-8')

        for url in self.config.webhook_urls:
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        **self.config.webhook_headers
                    },
                    method='POST'
                )
                # Note: In production, this would actually send
                print(f"[WEBHOOK] Would POST to {url}: {alert.title}")

            except Exception as e:
                print(f"Webhook error for {url}: {e}")
                success = False

        return success

    def _send_log(self, alert: Alert) -> bool:
        """Log alert"""
        log_msg = f"[ALERT:{alert.severity.name}] [{alert.category.value}] {alert.title} - {alert.message}"
        print(log_msg)
        return True

    def _send_database(self, alert: Alert) -> bool:
        """Save alert to database"""
        self.storage.save_alert(alert)
        return True


class AlertManager:
    """
    Main alert manager for TAOS
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "data" / "alerts.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.storage = AlertStorage(db_path)
        self.config = NotificationConfig()
        self.dispatcher = NotificationDispatcher(self.config, self.storage)

        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()

        # Alert rules
        self.rules = self._default_rules()

        # Load active alerts
        self._load_active_alerts()

    def _default_rules(self) -> List[Dict[str, Any]]:
        """Default alert rules"""
        return [
            # Hydraulic alerts
            {'condition': lambda s: s.get('h', 4) > 6.5, 'severity': AlertSeverity.CRITICAL,
             'category': AlertCategory.HYDRAULIC, 'title': '高水位警报',
             'message': '水位超过6.5m，达到{h:.2f}m'},
            {'condition': lambda s: s.get('h', 4) > 6.0, 'severity': AlertSeverity.ALARM,
             'category': AlertCategory.HYDRAULIC, 'title': '水位偏高',
             'message': '水位超过6.0m，当前{h:.2f}m'},
            {'condition': lambda s: s.get('h', 4) < 2.5, 'severity': AlertSeverity.ALARM,
             'category': AlertCategory.HYDRAULIC, 'title': '低水位警报',
             'message': '水位低于2.5m，当前{h:.2f}m'},
            {'condition': lambda s: s.get('fr', 0.3) > 0.8, 'severity': AlertSeverity.CRITICAL,
             'category': AlertCategory.HYDRAULIC, 'title': '水跃风险',
             'message': 'Fr数超过0.8，达到{fr:.2f}，存在水跃风险'},

            # Structural alerts
            {'condition': lambda s: s.get('vib_amp', 0) > 15, 'severity': AlertSeverity.EMERGENCY,
             'category': AlertCategory.STRUCTURAL, 'title': '剧烈振动',
             'message': '振动幅度超过15mm，达到{vib_amp:.1f}mm'},
            {'condition': lambda s: s.get('vib_amp', 0) > 10, 'severity': AlertSeverity.CRITICAL,
             'category': AlertCategory.STRUCTURAL, 'title': '振动超限',
             'message': '振动幅度超过10mm，当前{vib_amp:.1f}mm'},
            {'condition': lambda s: s.get('joint_gap', 20) < 5, 'severity': AlertSeverity.ALARM,
             'category': AlertCategory.STRUCTURAL, 'title': '伸缩缝挤压',
             'message': '伸缩缝间隙过小({joint_gap:.1f}mm)，可能受压'},
            {'condition': lambda s: s.get('joint_gap', 20) > 35, 'severity': AlertSeverity.ALARM,
             'category': AlertCategory.STRUCTURAL, 'title': '伸缩缝拉开',
             'message': '伸缩缝间隙过大({joint_gap:.1f}mm)，可能撕裂'},

            # Thermal alerts
            {'condition': lambda s: abs(s.get('T_sun', 25) - s.get('T_shade', 25)) > 25,
             'severity': AlertSeverity.CRITICAL, 'category': AlertCategory.THERMAL,
             'title': '温差过大', 'message': '向阳侧与背阴侧温差超过25°C'},
            {'condition': lambda s: s.get('T_sun', 25) > 50, 'severity': AlertSeverity.ALARM,
             'category': AlertCategory.THERMAL, 'title': '高温警报',
             'message': '向阳侧温度超过50°C，当前{T_sun:.1f}°C'},
        ]

    def _load_active_alerts(self):
        """Load active alerts from database"""
        alerts = self.storage.get_active_alerts()
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert

    def check_state(self, state: Dict[str, Any]) -> List[Alert]:
        """Check state against alert rules and generate alerts"""
        new_alerts = []

        with self.lock:
            for rule in self.rules:
                try:
                    if rule['condition'](state):
                        # Generate alert
                        alert_id = self._generate_alert_id(rule)

                        # Check if already active
                        if alert_id in self.active_alerts:
                            continue

                        message = rule['message'].format(**state)
                        alert = Alert(
                            alert_id=alert_id,
                            severity=rule['severity'],
                            category=rule['category'],
                            title=rule['title'],
                            message=message,
                            source='state_monitor',
                            timestamp=datetime.now(),
                            data={k: v for k, v in state.items()
                                  if isinstance(v, (int, float, str, bool))}
                        )

                        self.active_alerts[alert_id] = alert
                        new_alerts.append(alert)

                        # Dispatch notification
                        self.dispatcher.send(alert)

                        # Call handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(alert)
                            except Exception:
                                pass

                except Exception as e:
                    print(f"Alert rule error: {e}")

            # Check for resolved alerts
            self._check_resolved(state)

        return new_alerts

    def _generate_alert_id(self, rule: Dict) -> str:
        """Generate unique alert ID"""
        key = f"{rule['category'].value}:{rule['title']}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _check_resolved(self, state: Dict[str, Any]):
        """Check if any active alerts are resolved"""
        resolved = []

        for alert_id, alert in list(self.active_alerts.items()):
            # Find corresponding rule
            for rule in self.rules:
                if self._generate_alert_id(rule) == alert_id:
                    try:
                        if not rule['condition'](state):
                            resolved.append(alert_id)
                    except Exception:
                        pass
                    break

        for alert_id in resolved:
            self.resolve_alert(alert_id, "auto-resolved")

    def create_alert(self, severity: AlertSeverity, category: AlertCategory,
                    title: str, message: str, source: str = "manual",
                    data: Dict[str, Any] = None) -> Alert:
        """Create and send a manual alert"""
        alert_id = hashlib.md5(
            f"{category.value}:{title}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            category=category,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now(),
            data=data or {}
        )

        with self.lock:
            self.active_alerts[alert_id] = alert

        self.dispatcher.send(alert)

        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                self.storage.save_alert(alert)
                return True
        return False

    def resolve_alert(self, alert_id: str, reason: str = "resolved") -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                self.storage.save_alert(alert)
                del self.active_alerts[alert_id]
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        with self.lock:
            if alert_id in self.active_alerts:
                return self.active_alerts[alert_id]
        return self.storage.get_alert(alert_id)

    def query_alerts(self, **kwargs) -> List[Alert]:
        """Query alerts from storage"""
        return self.storage.query_alerts(**kwargs)

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics"""
        stats = self.storage.get_statistics(hours)
        stats['active_count'] = len(self.active_alerts)
        return stats

    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)

    def update_config(self, config: Dict[str, Any]):
        """Update notification configuration"""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# Global instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


if __name__ == "__main__":
    # Test alert system
    print("=== Alert System Test ===")

    manager = AlertManager()

    # Test state check
    test_state = {
        'h': 6.8,  # High water level
        'v': 2.5,
        'fr': 0.4,
        'T_sun': 35,
        'T_shade': 25,
        'vib_amp': 2,
        'joint_gap': 20
    }

    print("\n1. Checking state with high water level...")
    alerts = manager.check_state(test_state)
    print(f"   Generated alerts: {len(alerts)}")
    for alert in alerts:
        print(f"   - [{alert.severity.name}] {alert.title}")

    # Test manual alert
    print("\n2. Creating manual alert...")
    manual_alert = manager.create_alert(
        severity=AlertSeverity.WARNING,
        category=AlertCategory.SYSTEM,
        title="Test Alert",
        message="This is a test alert",
        source="test"
    )
    print(f"   Created: {manual_alert.alert_id}")

    # Get active alerts
    print("\n3. Active alerts:")
    active = manager.get_active_alerts()
    for alert in active:
        print(f"   - {alert.alert_id}: {alert.title} ({alert.status.value})")

    # Acknowledge alert
    print("\n4. Acknowledging alert...")
    manager.acknowledge_alert(manual_alert.alert_id, "test_user")

    # Get statistics
    print("\n5. Statistics:")
    stats = manager.get_statistics(24)
    print(f"   Total alerts: {stats['total_alerts']}")
    print(f"   Active: {stats['active_count']}")
    print(f"   By severity: {stats['by_severity']}")

    print("\nAlert system test completed!")
