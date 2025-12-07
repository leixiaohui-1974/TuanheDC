#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.9 - Mobile API Interface
团河渡槽自主运行系统 - 移动端API接口模块

Features:
- Lightweight JSON responses for mobile
- Push notification integration
- Offline data synchronization
- QR code generation
- Location-based services
- Mobile-optimized endpoints
- Rate limiting for mobile clients
- Token-based authentication
"""

import time
import json
import hashlib
import threading
import base64
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque
from pathlib import Path
import sqlite3


class DevicePlatform(Enum):
    """Mobile device platforms"""
    IOS = "ios"
    ANDROID = "android"
    WECHAT_MINI = "wechat_mini"
    ALIPAY_MINI = "alipay_mini"
    WEB_MOBILE = "web_mobile"


class NotificationType(Enum):
    """Notification types"""
    ALERT = "alert"
    WARNING = "warning"
    INFO = "info"
    EMERGENCY = "emergency"
    SYSTEM = "system"
    MAINTENANCE = "maintenance"


class SyncStatus(Enum):
    """Data sync status"""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass
class MobileDevice:
    """Mobile device information"""
    device_id: str
    platform: DevicePlatform
    device_name: str
    user_id: str
    push_token: Optional[str]
    app_version: str
    os_version: str
    registered_at: datetime
    last_active: datetime
    notifications_enabled: bool = True
    location: Optional[Dict[str, float]] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'platform': self.platform.value,
            'device_name': self.device_name,
            'user_id': self.user_id,
            'push_token': self.push_token,
            'app_version': self.app_version,
            'os_version': self.os_version,
            'registered_at': self.registered_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'notifications_enabled': self.notifications_enabled,
            'location': self.location,
            'settings': self.settings
        }


@dataclass
class PushNotification:
    """Push notification message"""
    notification_id: str
    device_ids: List[str]
    notification_type: NotificationType
    title: str
    body: str
    data: Dict[str, Any]
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered: int = 0
    failed: int = 0
    priority: str = "normal"
    badge: int = 0
    sound: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'device_ids': self.device_ids,
            'notification_type': self.notification_type.value,
            'title': self.title,
            'body': self.body,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'delivered': self.delivered,
            'failed': self.failed,
            'priority': self.priority,
            'badge': self.badge,
            'sound': self.sound
        }


@dataclass
class OfflineSyncPackage:
    """Offline data sync package"""
    package_id: str
    device_id: str
    sync_type: str  # 'full' or 'delta'
    timestamp: datetime
    data: Dict[str, Any]
    checksum: str
    status: SyncStatus
    size_bytes: int
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'package_id': self.package_id,
            'device_id': self.device_id,
            'sync_type': self.sync_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'checksum': self.checksum,
            'status': self.status.value,
            'size_bytes': self.size_bytes,
            'conflicts': self.conflicts
        }


class QRCodeGenerator:
    """
    QR code data generator for mobile scanning
    """

    def __init__(self):
        self.qr_codes: Dict[str, Dict[str, Any]] = {}
        self.expiry_seconds = 300  # 5 minutes default

    def generate_login_qr(self, session_id: str) -> Dict[str, Any]:
        """Generate QR code data for mobile login"""
        qr_data = {
            'type': 'login',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=self.expiry_seconds)).isoformat()
        }

        # Create signed token
        token = self._create_token(qr_data)
        qr_data['token'] = token

        self.qr_codes[session_id] = {
            'data': qr_data,
            'scanned': False,
            'scanned_by': None,
            'created_at': datetime.now()
        }

        return {
            'qr_content': json.dumps(qr_data),
            'session_id': session_id,
            'expires_in': self.expiry_seconds
        }

    def generate_device_qr(self, device_id: str, location: str) -> Dict[str, Any]:
        """Generate QR code for device/location identification"""
        qr_data = {
            'type': 'device',
            'device_id': device_id,
            'location': location,
            'timestamp': datetime.now().isoformat()
        }

        token = self._create_token(qr_data)
        qr_data['token'] = token

        return {
            'qr_content': json.dumps(qr_data),
            'device_id': device_id,
            'location': location
        }

    def generate_inspection_qr(self, checkpoint_id: str, checkpoint_name: str) -> Dict[str, Any]:
        """Generate QR code for inspection checkpoints"""
        qr_data = {
            'type': 'inspection',
            'checkpoint_id': checkpoint_id,
            'checkpoint_name': checkpoint_name,
            'timestamp': datetime.now().isoformat()
        }

        token = self._create_token(qr_data)
        qr_data['token'] = token

        return {
            'qr_content': json.dumps(qr_data),
            'checkpoint_id': checkpoint_id,
            'checkpoint_name': checkpoint_name
        }

    def verify_qr_scan(self, session_id: str, scanned_by: str) -> Dict[str, Any]:
        """Verify QR code scan"""
        if session_id not in self.qr_codes:
            return {'valid': False, 'error': 'Invalid QR code'}

        qr_info = self.qr_codes[session_id]

        # Check expiry
        expires_at = datetime.fromisoformat(qr_info['data']['expires_at'])
        if datetime.now() > expires_at:
            return {'valid': False, 'error': 'QR code expired'}

        # Check if already scanned
        if qr_info['scanned']:
            return {'valid': False, 'error': 'QR code already used'}

        # Mark as scanned
        qr_info['scanned'] = True
        qr_info['scanned_by'] = scanned_by

        return {
            'valid': True,
            'session_id': session_id,
            'scanned_by': scanned_by
        }

    def _create_token(self, data: Dict[str, Any]) -> str:
        """Create verification token"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:32]


class MobileDataCompressor:
    """
    Data compression and optimization for mobile
    """

    def __init__(self):
        self.compression_enabled = True
        self.field_mappings: Dict[str, str] = {
            # Map long field names to short codes
            'water_level': 'h',
            'flow_velocity': 'v',
            'flow_rate_in': 'qi',
            'flow_rate_out': 'qo',
            'temperature_sun': 'ts',
            'temperature_shade': 'tsh',
            'vibration_amplitude': 'vib',
            'joint_gap': 'jg',
            'bearing_stress': 'bs',
            'froude_number': 'fr',
            'timestamp': 't',
            'risk_level': 'rl',
            'active_scenarios': 'as'
        }
        self.reverse_mappings = {v: k for k, v in self.field_mappings.items()}

    def compress_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compress state data for mobile transmission"""
        if not self.compression_enabled:
            return state

        compressed = {}
        for key, value in state.items():
            # Use short field names
            short_key = self.field_mappings.get(key, key)

            # Round floats to reduce size
            if isinstance(value, float):
                value = round(value, 2)
            elif isinstance(value, list):
                # Compress arrays
                value = [v[:3] if isinstance(v, str) else v for v in value]

            compressed[short_key] = value

        return compressed

    def decompress_state(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress state data"""
        decompressed = {}
        for key, value in compressed.items():
            long_key = self.reverse_mappings.get(key, key)
            decompressed[long_key] = value
        return decompressed

    def create_summary(self, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create mobile-friendly summary"""
        return {
            'h': round(full_state.get('h', 0), 2),
            'fr': round(full_state.get('fr', 0), 3),
            'qi': round(full_state.get('Q_in', 0), 1),
            'qo': round(full_state.get('Q_out', 0), 1),
            'risk': full_state.get('risk_level', 'UNKNOWN')[:1],  # First letter
            'safe': full_state.get('is_safe', True),
            't': int(time.time())
        }

    def batch_compress(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compress multiple states for batch transfer"""
        if not states:
            return {'count': 0, 'data': []}

        # Extract common keys
        keys = list(states[0].keys())
        short_keys = [self.field_mappings.get(k, k) for k in keys]

        # Create columnar format
        columns = {sk: [] for sk in short_keys}
        for state in states:
            for key, short_key in zip(keys, short_keys):
                value = state.get(key)
                if isinstance(value, float):
                    value = round(value, 2)
                columns[short_key].append(value)

        return {
            'count': len(states),
            'keys': short_keys,
            'data': columns
        }


class PushNotificationService:
    """
    Push notification service for mobile devices
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "mobile"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.notification_queue: deque = deque(maxlen=1000)
        self.notification_history: List[PushNotification] = []
        self.max_history = 500

        # Platform handlers (simulated)
        self.platform_handlers = {
            DevicePlatform.IOS: self._send_apns,
            DevicePlatform.ANDROID: self._send_fcm,
            DevicePlatform.WECHAT_MINI: self._send_wechat,
            DevicePlatform.ALIPAY_MINI: self._send_alipay,
            DevicePlatform.WEB_MOBILE: self._send_web_push
        }

        # Threading
        self.running = False
        self.sender_thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start notification service"""
        self.running = True
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

    def stop(self):
        """Stop notification service"""
        self.running = False
        if self.sender_thread:
            self.sender_thread.join(timeout=5)

    def _sender_loop(self):
        """Background notification sender"""
        while self.running:
            if self.notification_queue:
                notification = self.notification_queue.popleft()
                self._process_notification(notification)
            time.sleep(0.1)

    def send_notification(self, devices: List[MobileDevice], notification_type: NotificationType,
                         title: str, body: str, data: Dict[str, Any] = None,
                         priority: str = "normal") -> str:
        """Queue a notification for sending"""
        notification = PushNotification(
            notification_id=f"notif_{int(time.time()*1000)}",
            device_ids=[d.device_id for d in devices],
            notification_type=notification_type,
            title=title,
            body=body,
            data=data or {},
            created_at=datetime.now(),
            priority=priority,
            badge=1 if notification_type in [NotificationType.ALERT, NotificationType.EMERGENCY] else 0
        )

        self.notification_queue.append((notification, devices))
        return notification.notification_id

    def _process_notification(self, item: Tuple[PushNotification, List[MobileDevice]]):
        """Process and send a notification"""
        notification, devices = item
        notification.sent_at = datetime.now()

        for device in devices:
            if not device.notifications_enabled:
                continue

            handler = self.platform_handlers.get(device.platform)
            if handler:
                success = handler(device, notification)
                if success:
                    notification.delivered += 1
                else:
                    notification.failed += 1

        # Store in history
        with self.lock:
            self.notification_history.append(notification)
            if len(self.notification_history) > self.max_history:
                self.notification_history.pop(0)

    def _send_apns(self, device: MobileDevice, notification: PushNotification) -> bool:
        """Send via Apple Push Notification Service (simulated)"""
        # In production, would use APNs SDK
        return True

    def _send_fcm(self, device: MobileDevice, notification: PushNotification) -> bool:
        """Send via Firebase Cloud Messaging (simulated)"""
        # In production, would use FCM SDK
        return True

    def _send_wechat(self, device: MobileDevice, notification: PushNotification) -> bool:
        """Send via WeChat Mini Program (simulated)"""
        # In production, would use WeChat API
        return True

    def _send_alipay(self, device: MobileDevice, notification: PushNotification) -> bool:
        """Send via Alipay Mini Program (simulated)"""
        # In production, would use Alipay API
        return True

    def _send_web_push(self, device: MobileDevice, notification: PushNotification) -> bool:
        """Send via Web Push (simulated)"""
        # In production, would use Web Push protocol
        return True

    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notification history"""
        with self.lock:
            return [n.to_dict() for n in self.notification_history[-limit:]]


class OfflineSyncManager:
    """
    Offline data synchronization manager
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "mobile" / "sync"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sync_db = self.data_dir / "sync.db"
        self._init_database()

        # Sync state per device
        self.device_sync_states: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _init_database(self):
        """Initialize sync database"""
        conn = sqlite3.connect(str(self.sync_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_packages (
                package_id TEXT PRIMARY KEY,
                device_id TEXT,
                sync_type TEXT,
                timestamp TEXT,
                data TEXT,
                checksum TEXT,
                status TEXT,
                size_bytes INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_conflicts (
                conflict_id TEXT PRIMARY KEY,
                package_id TEXT,
                local_data TEXT,
                server_data TEXT,
                resolved INTEGER DEFAULT 0,
                resolution TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def create_sync_package(self, device_id: str, sync_type: str,
                           data: Dict[str, Any]) -> OfflineSyncPackage:
        """Create a sync package for a device"""
        data_str = json.dumps(data)
        package = OfflineSyncPackage(
            package_id=f"sync_{device_id}_{int(time.time()*1000)}",
            device_id=device_id,
            sync_type=sync_type,
            timestamp=datetime.now(),
            data=data,
            checksum=hashlib.md5(data_str.encode()).hexdigest(),
            status=SyncStatus.PENDING,
            size_bytes=len(data_str.encode())
        )

        # Store in database
        conn = sqlite3.connect(str(self.sync_db))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sync_packages
            (package_id, device_id, sync_type, timestamp, data, checksum, status, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            package.package_id, package.device_id, package.sync_type,
            package.timestamp.isoformat(), data_str, package.checksum,
            package.status.value, package.size_bytes
        ))
        conn.commit()
        conn.close()

        return package

    def process_device_sync(self, device_id: str, client_data: Dict[str, Any],
                           client_timestamp: datetime) -> Dict[str, Any]:
        """Process sync request from mobile device"""
        with self.lock:
            # Get server state
            server_state = self.device_sync_states.get(device_id, {})
            server_timestamp = server_state.get('timestamp', datetime.min)

            # Detect conflicts
            conflicts = []
            if server_timestamp > client_timestamp:
                # Server has newer data
                for key, client_value in client_data.items():
                    server_value = server_state.get('data', {}).get(key)
                    if server_value is not None and server_value != client_value:
                        conflicts.append({
                            'key': key,
                            'client_value': client_value,
                            'server_value': server_value
                        })

            if conflicts:
                return {
                    'status': 'conflict',
                    'conflicts': conflicts,
                    'server_data': server_state.get('data', {}),
                    'server_timestamp': server_timestamp.isoformat()
                }

            # Update server state
            self.device_sync_states[device_id] = {
                'data': client_data,
                'timestamp': datetime.now()
            }

            return {
                'status': 'synced',
                'timestamp': datetime.now().isoformat()
            }

    def get_delta_updates(self, device_id: str, since_timestamp: datetime) -> Dict[str, Any]:
        """Get incremental updates since timestamp"""
        # In production, would query actual data changes
        return {
            'device_id': device_id,
            'since': since_timestamp.isoformat(),
            'updates': [],
            'has_more': False
        }


class MobileDeviceManager:
    """
    Mobile device management system
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "mobile"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.devices: Dict[str, MobileDevice] = {}
        self.qr_generator = QRCodeGenerator()
        self.compressor = MobileDataCompressor()
        self.notification_service = PushNotificationService(str(self.data_dir))
        self.sync_manager = OfflineSyncManager(str(self.data_dir))

        # Rate limiting
        self.rate_limits: Dict[str, deque] = {}
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 60  # requests per window

        # Threading
        self.lock = threading.Lock()

    def start(self):
        """Start mobile manager"""
        self.notification_service.start()

    def stop(self):
        """Stop mobile manager"""
        self.notification_service.stop()

    def register_device(self, device_info: Dict[str, Any]) -> MobileDevice:
        """Register a new mobile device"""
        device = MobileDevice(
            device_id=device_info['device_id'],
            platform=DevicePlatform(device_info.get('platform', 'android')),
            device_name=device_info.get('device_name', 'Unknown Device'),
            user_id=device_info.get('user_id', 'anonymous'),
            push_token=device_info.get('push_token'),
            app_version=device_info.get('app_version', '1.0.0'),
            os_version=device_info.get('os_version', 'unknown'),
            registered_at=datetime.now(),
            last_active=datetime.now(),
            notifications_enabled=device_info.get('notifications_enabled', True),
            location=device_info.get('location'),
            settings=device_info.get('settings', {})
        )

        self.devices[device.device_id] = device
        return device

    def update_device(self, device_id: str, updates: Dict[str, Any]) -> Optional[MobileDevice]:
        """Update device information"""
        device = self.devices.get(device_id)
        if not device:
            return None

        if 'push_token' in updates:
            device.push_token = updates['push_token']
        if 'notifications_enabled' in updates:
            device.notifications_enabled = updates['notifications_enabled']
        if 'location' in updates:
            device.location = updates['location']
        if 'settings' in updates:
            device.settings.update(updates['settings'])

        device.last_active = datetime.now()
        return device

    def check_rate_limit(self, device_id: str) -> bool:
        """Check if device is within rate limit"""
        now = time.time()

        if device_id not in self.rate_limits:
            self.rate_limits[device_id] = deque()

        # Remove old entries
        while self.rate_limits[device_id] and self.rate_limits[device_id][0] < now - self.rate_limit_window:
            self.rate_limits[device_id].popleft()

        # Check limit
        if len(self.rate_limits[device_id]) >= self.rate_limit_max:
            return False

        # Record request
        self.rate_limits[device_id].append(now)
        return True

    def get_mobile_state(self, device_id: str, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get mobile-optimized state"""
        # Update device activity
        if device_id in self.devices:
            self.devices[device_id].last_active = datetime.now()

        # Compress state
        return self.compressor.compress_state(full_state)

    def get_mobile_summary(self, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get ultra-compact state summary"""
        return self.compressor.create_summary(full_state)

    def send_alert_to_all(self, alert_type: NotificationType, title: str,
                         body: str, data: Dict[str, Any] = None):
        """Send alert to all registered devices"""
        devices = list(self.devices.values())
        if devices:
            return self.notification_service.send_notification(
                devices, alert_type, title, body, data, priority="high"
            )
        return None

    def send_alert_to_device(self, device_id: str, alert_type: NotificationType,
                            title: str, body: str, data: Dict[str, Any] = None):
        """Send alert to specific device"""
        device = self.devices.get(device_id)
        if device:
            return self.notification_service.send_notification(
                [device], alert_type, title, body, data
            )
        return None

    def generate_login_qr(self, session_id: str) -> Dict[str, Any]:
        """Generate login QR code"""
        return self.qr_generator.generate_login_qr(session_id)

    def verify_qr_login(self, session_id: str, device_id: str) -> Dict[str, Any]:
        """Verify QR login scan"""
        return self.qr_generator.verify_qr_scan(session_id, device_id)

    def sync_device_data(self, device_id: str, client_data: Dict[str, Any],
                        client_timestamp: str) -> Dict[str, Any]:
        """Synchronize device data"""
        timestamp = datetime.fromisoformat(client_timestamp)
        return self.sync_manager.process_device_sync(device_id, client_data, timestamp)

    def get_inspection_checkpoints(self) -> List[Dict[str, Any]]:
        """Get inspection checkpoint list with QR codes"""
        # Predefined inspection checkpoints for Tuanhe Aqueduct
        checkpoints = [
            {'id': 'CP001', 'name': '上游进水口', 'chainage': 0},
            {'id': 'CP002', 'name': '第一跨中点', 'chainage': 100},
            {'id': 'CP003', 'name': '第一伸缩缝', 'chainage': 200},
            {'id': 'CP004', 'name': '中游监测点', 'chainage': 600},
            {'id': 'CP005', 'name': '第二伸缩缝', 'chainage': 800},
            {'id': 'CP006', 'name': '下游出水口', 'chainage': 1292}
        ]

        for cp in checkpoints:
            qr = self.qr_generator.generate_inspection_qr(cp['id'], cp['name'])
            cp['qr_content'] = qr['qr_content']

        return checkpoints

    def get_status(self) -> Dict[str, Any]:
        """Get mobile system status"""
        return {
            'registered_devices': len(self.devices),
            'devices_by_platform': {
                p.value: len([d for d in self.devices.values() if d.platform == p])
                for p in DevicePlatform
            },
            'active_devices_24h': len([
                d for d in self.devices.values()
                if (datetime.now() - d.last_active).total_seconds() < 86400
            ]),
            'notifications_sent': len(self.notification_service.notification_history),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_mobile_manager = None


def get_mobile_manager() -> MobileDeviceManager:
    """Get global mobile manager"""
    global _mobile_manager
    if _mobile_manager is None:
        _mobile_manager = MobileDeviceManager()
    return _mobile_manager


if __name__ == "__main__":
    # Test mobile API
    print("=== Mobile API Test ===")

    manager = MobileDeviceManager()
    manager.start()

    # Register device
    print("\n1. Registering mobile device...")
    device = manager.register_device({
        'device_id': 'mobile_001',
        'platform': 'android',
        'device_name': 'Samsung Galaxy S21',
        'user_id': 'operator_001',
        'app_version': '1.0.0',
        'os_version': 'Android 12'
    })
    print(f"   Device ID: {device.device_id}")
    print(f"   Platform: {device.platform.value}")

    # Get mobile state
    print("\n2. Getting mobile-optimized state...")
    full_state = {
        'water_level': 4.523,
        'flow_velocity': 2.15,
        'flow_rate_in': 85.7,
        'flow_rate_out': 84.2,
        'temperature_sun': 28.5,
        'temperature_shade': 23.1,
        'vibration_amplitude': 2.3,
        'froude_number': 0.324,
        'risk_level': 'LOW',
        'active_scenarios': ['S1.1', 'S3.1']
    }
    mobile_state = manager.get_mobile_state('mobile_001', full_state)
    print(f"   Compressed keys: {list(mobile_state.keys())}")

    # Get summary
    print("\n3. Getting ultra-compact summary...")
    summary = manager.get_mobile_summary(full_state)
    print(f"   Summary: {summary}")

    # Generate QR code
    print("\n4. Generating login QR code...")
    qr = manager.generate_login_qr('session_12345')
    print(f"   Session ID: {qr['session_id']}")
    print(f"   Expires in: {qr['expires_in']}s")

    # Get inspection checkpoints
    print("\n5. Getting inspection checkpoints...")
    checkpoints = manager.get_inspection_checkpoints()
    for cp in checkpoints[:3]:
        print(f"   {cp['id']}: {cp['name']} at {cp['chainage']}m")

    # Send notification
    print("\n6. Sending test notification...")
    notif_id = manager.send_alert_to_all(
        NotificationType.INFO,
        "系统状态更新",
        "当前水位正常，流量稳定",
        {'water_level': 4.5}
    )
    print(f"   Notification ID: {notif_id}")

    # System status
    print("\n7. Mobile System Status:")
    status = manager.get_status()
    print(f"   Registered devices: {status['registered_devices']}")
    print(f"   Active (24h): {status['active_devices_24h']}")

    manager.stop()
    print("\nMobile API test completed!")
