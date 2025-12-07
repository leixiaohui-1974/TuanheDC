#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.8 - Edge Computing Integration
团河渡槽自主运行系统 - 边缘计算集成模块

Features:
- Edge device management
- Local data preprocessing
- Edge-cloud synchronization
- Offline operation capability
- Data aggregation at edge
- Real-time local inference
- Bandwidth optimization
"""

import time
import json
import hashlib
import threading
import queue
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque
from pathlib import Path
import sqlite3


class EdgeDeviceType(Enum):
    """Edge device types"""
    GATEWAY = "gateway"           # Edge gateway
    SENSOR_HUB = "sensor_hub"     # Sensor aggregation hub
    PLC = "plc"                   # Programmable Logic Controller
    RTU = "rtu"                   # Remote Terminal Unit
    IPC = "ipc"                   # Industrial PC
    SMART_SENSOR = "smart_sensor" # Smart sensor with processing


class EdgeDeviceStatus(Enum):
    """Device status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    SYNCING = "syncing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class SyncMode(Enum):
    """Data synchronization mode"""
    REAL_TIME = "real_time"       # Immediate sync
    BATCH = "batch"               # Periodic batch sync
    ON_CHANGE = "on_change"       # Sync on significant change
    SCHEDULED = "scheduled"       # Scheduled sync
    MANUAL = "manual"             # Manual trigger


class DataPriority(Enum):
    """Data transmission priority"""
    CRITICAL = 1    # Emergency data
    HIGH = 2        # Important alerts
    NORMAL = 3      # Regular readings
    LOW = 4         # Historical data
    BULK = 5        # Bulk transfer


@dataclass
class EdgeDevice:
    """Edge device information"""
    device_id: str
    device_type: EdgeDeviceType
    name: str
    location: str
    status: EdgeDeviceStatus
    ip_address: str
    connected_sensors: List[str]
    firmware_version: str
    registered_at: datetime
    last_seen: datetime
    last_sync: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    storage_usage: float = 0.0
    temperature: float = 25.0
    battery_level: float = 100.0
    signal_strength: float = -50.0
    data_queue_size: int = 0
    sync_mode: SyncMode = SyncMode.BATCH
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'device_type': self.device_type.value,
            'name': self.name,
            'location': self.location,
            'status': self.status.value,
            'ip_address': self.ip_address,
            'connected_sensors': self.connected_sensors,
            'firmware_version': self.firmware_version,
            'registered_at': self.registered_at.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'last_sync': self.last_sync.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'storage_usage': self.storage_usage,
            'temperature': self.temperature,
            'battery_level': self.battery_level,
            'signal_strength': self.signal_strength,
            'data_queue_size': self.data_queue_size,
            'sync_mode': self.sync_mode.value,
            'metadata': self.metadata
        }


@dataclass
class EdgeDataPacket:
    """Data packet from edge device"""
    packet_id: str
    device_id: str
    timestamp: datetime
    priority: DataPriority
    data_type: str
    payload: Dict[str, Any]
    checksum: str
    compressed: bool = False
    encrypted: bool = False
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'packet_id': self.packet_id,
            'device_id': self.device_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'data_type': self.data_type,
            'payload': self.payload,
            'checksum': self.checksum,
            'compressed': self.compressed,
            'encrypted': self.encrypted,
            'retry_count': self.retry_count
        }


@dataclass
class ProcessingRule:
    """Edge data processing rule"""
    rule_id: str
    name: str
    condition: str  # Simple expression
    action: str     # Action to take
    enabled: bool = True
    priority: int = 0
    cooldown_seconds: float = 0
    last_triggered: Optional[datetime] = None


class EdgeDataProcessor:
    """
    Local data processing at edge
    """

    def __init__(self):
        self.rules: Dict[str, ProcessingRule] = {}
        self.aggregation_buffers: Dict[str, deque] = {}
        self.aggregation_window = 60  # seconds
        self.max_buffer_size = 1000
        self.filters: Dict[str, Dict[str, Any]] = {}

        # Statistical accumulators
        self.stats: Dict[str, Dict[str, float]] = {}

        # Initialize default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default processing rules"""
        self.rules = {
            'high_water_alert': ProcessingRule(
                rule_id='high_water_alert',
                name='High Water Level Alert',
                condition='water_level > 6.5',
                action='alert:HIGH_WATER',
                cooldown_seconds=60
            ),
            'low_water_alert': ProcessingRule(
                rule_id='low_water_alert',
                name='Low Water Level Alert',
                condition='water_level < 2.0',
                action='alert:LOW_WATER',
                cooldown_seconds=60
            ),
            'high_vibration': ProcessingRule(
                rule_id='high_vibration',
                name='High Vibration Alert',
                condition='vibration > 50',
                action='alert:HIGH_VIBRATION',
                cooldown_seconds=30
            ),
            'thermal_warning': ProcessingRule(
                rule_id='thermal_warning',
                name='Thermal Gradient Warning',
                condition='abs(T_sun - T_shade) > 10',
                action='alert:THERMAL_GRADIENT',
                cooldown_seconds=120
            ),
            'seismic_emergency': ProcessingRule(
                rule_id='seismic_emergency',
                name='Seismic Emergency',
                condition='ground_accel > 0.1',
                action='emergency:SEISMIC',
                cooldown_seconds=10,
                priority=1
            )
        }

    def process_reading(self, sensor_id: str, reading: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a sensor reading locally"""
        results = []

        # Apply filters
        filtered_reading = self._apply_filters(sensor_id, reading)
        if filtered_reading is None:
            return results

        # Update statistics
        self._update_statistics(sensor_id, filtered_reading)

        # Check rules
        triggered_rules = self._check_rules(filtered_reading)
        for rule, action in triggered_rules:
            results.append({
                'type': 'rule_triggered',
                'rule_id': rule.rule_id,
                'action': action,
                'reading': filtered_reading
            })

        # Add to aggregation buffer
        self._add_to_buffer(sensor_id, filtered_reading)

        return results

    def _apply_filters(self, sensor_id: str, reading: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply data filters"""
        filter_config = self.filters.get(sensor_id, {})

        # Outlier filter
        if 'outlier_threshold' in filter_config:
            stats = self.stats.get(sensor_id, {})
            for key, value in reading.items():
                if isinstance(value, (int, float)) and key in stats:
                    mean = stats.get(f'{key}_mean', value)
                    std = stats.get(f'{key}_std', 1)
                    if std > 0 and abs(value - mean) > filter_config['outlier_threshold'] * std:
                        return None  # Filter out outlier

        # Rate limit filter
        if 'min_interval' in filter_config:
            last_time = filter_config.get('_last_reading_time', 0)
            if time.time() - last_time < filter_config['min_interval']:
                return None
            filter_config['_last_reading_time'] = time.time()

        return reading

    def _update_statistics(self, sensor_id: str, reading: Dict[str, Any]):
        """Update running statistics"""
        if sensor_id not in self.stats:
            self.stats[sensor_id] = {}

        stats = self.stats[sensor_id]
        alpha = 0.1  # Exponential smoothing factor

        for key, value in reading.items():
            if isinstance(value, (int, float)):
                mean_key = f'{key}_mean'
                var_key = f'{key}_var'
                std_key = f'{key}_std'

                if mean_key not in stats:
                    stats[mean_key] = value
                    stats[var_key] = 0
                else:
                    old_mean = stats[mean_key]
                    stats[mean_key] = alpha * value + (1 - alpha) * old_mean
                    stats[var_key] = alpha * (value - old_mean) ** 2 + (1 - alpha) * stats[var_key]

                stats[std_key] = stats[var_key] ** 0.5

    def _check_rules(self, reading: Dict[str, Any]) -> List[Tuple[ProcessingRule, str]]:
        """Check all rules against reading"""
        triggered = []

        for rule in sorted(self.rules.values(), key=lambda r: r.priority):
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, reading):
                rule.last_triggered = datetime.now()
                triggered.append((rule, rule.action))

        return triggered

    def _evaluate_condition(self, condition: str, reading: Dict[str, Any]) -> bool:
        """Evaluate a simple condition expression"""
        try:
            # Create local variables from reading
            local_vars = dict(reading)
            local_vars['abs'] = abs
            local_vars['min'] = min
            local_vars['max'] = max

            return eval(condition, {"__builtins__": {}}, local_vars)
        except:
            return False

    def _add_to_buffer(self, sensor_id: str, reading: Dict[str, Any]):
        """Add reading to aggregation buffer"""
        if sensor_id not in self.aggregation_buffers:
            self.aggregation_buffers[sensor_id] = deque(maxlen=self.max_buffer_size)

        reading['_timestamp'] = time.time()
        self.aggregation_buffers[sensor_id].append(reading)

    def get_aggregated_data(self, sensor_id: str, window_seconds: float = None) -> Dict[str, Any]:
        """Get aggregated data for a sensor"""
        window = window_seconds or self.aggregation_window
        buffer = self.aggregation_buffers.get(sensor_id, deque())

        cutoff_time = time.time() - window
        recent_readings = [r for r in buffer if r.get('_timestamp', 0) > cutoff_time]

        if not recent_readings:
            return {}

        # Aggregate numeric values
        aggregated = {'count': len(recent_readings)}

        keys = set()
        for r in recent_readings:
            keys.update(k for k, v in r.items()
                       if isinstance(v, (int, float)) and not k.startswith('_'))

        for key in keys:
            values = [r[key] for r in recent_readings if key in r]
            if values:
                aggregated[f'{key}_min'] = min(values)
                aggregated[f'{key}_max'] = max(values)
                aggregated[f'{key}_mean'] = sum(values) / len(values)
                aggregated[f'{key}_last'] = values[-1]

        return aggregated

    def add_rule(self, rule: ProcessingRule):
        """Add a processing rule"""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """Remove a processing rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def set_filter(self, sensor_id: str, filter_config: Dict[str, Any]):
        """Set filter for a sensor"""
        self.filters[sensor_id] = filter_config


class EdgeSyncManager:
    """
    Edge-cloud data synchronization
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "edge"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Sync queues by priority
        self.send_queues: Dict[DataPriority, queue.PriorityQueue] = {
            p: queue.PriorityQueue() for p in DataPriority
        }
        self.pending_acks: Dict[str, EdgeDataPacket] = {}

        # Sync state
        self.sync_status = {
            'last_sync': None,
            'pending_packets': 0,
            'sync_errors': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }

        # Connection state
        self.connected = False
        self.connection_quality = 1.0

        # Offline storage
        self.offline_db = self.data_dir / "offline_queue.db"
        self._init_offline_storage()

        # Threading
        self.running = False
        self.sync_thread = None
        self.lock = threading.Lock()

        # Callbacks
        self.on_sync_complete: Optional[Callable] = None
        self.on_sync_error: Optional[Callable] = None

    def _init_offline_storage(self):
        """Initialize offline storage database"""
        conn = sqlite3.connect(str(self.offline_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offline_packets (
                packet_id TEXT PRIMARY KEY,
                device_id TEXT,
                timestamp TEXT,
                priority INTEGER,
                data_type TEXT,
                payload TEXT,
                checksum TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def start(self):
        """Start sync manager"""
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

    def stop(self):
        """Stop sync manager"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)

    def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            if self.connected:
                self._process_send_queues()
            time.sleep(0.1)

    def _process_send_queues(self):
        """Process send queues by priority"""
        for priority in DataPriority:
            q = self.send_queues[priority]
            while not q.empty():
                try:
                    _, packet = q.get_nowait()
                    if self._send_packet(packet):
                        self.pending_acks[packet.packet_id] = packet
                    else:
                        # Failed to send, store offline
                        self._store_offline(packet)
                        break  # Stop processing on failure
                except queue.Empty:
                    break

    def _send_packet(self, packet: EdgeDataPacket) -> bool:
        """Send a packet to cloud (simulated)"""
        # In production, would actually send via MQTT/HTTP/etc
        try:
            data_size = len(json.dumps(packet.payload))
            self.sync_status['bytes_sent'] += data_size
            return True
        except Exception as e:
            self.sync_status['sync_errors'] += 1
            return False

    def _store_offline(self, packet: EdgeDataPacket):
        """Store packet for offline sync"""
        conn = sqlite3.connect(str(self.offline_db))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO offline_packets
            (packet_id, device_id, timestamp, priority, data_type, payload, checksum, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            packet.packet_id,
            packet.device_id,
            packet.timestamp.isoformat(),
            packet.priority.value,
            packet.data_type,
            json.dumps(packet.payload),
            packet.checksum,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def queue_data(self, packet: EdgeDataPacket):
        """Queue data for sync"""
        priority_value = (packet.priority.value, time.time())
        self.send_queues[packet.priority].put((priority_value, packet))

        with self.lock:
            self.sync_status['pending_packets'] += 1

    def set_connection_status(self, connected: bool, quality: float = 1.0):
        """Update connection status"""
        was_connected = self.connected
        self.connected = connected
        self.connection_quality = quality

        # Sync offline data when reconnecting
        if not was_connected and connected:
            self._sync_offline_data()

    def _sync_offline_data(self):
        """Sync offline stored data"""
        conn = sqlite3.connect(str(self.offline_db))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM offline_packets ORDER BY priority, timestamp')

        for row in cursor.fetchall():
            packet = EdgeDataPacket(
                packet_id=row[0],
                device_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                priority=DataPriority(row[3]),
                data_type=row[4],
                payload=json.loads(row[5]),
                checksum=row[6]
            )
            self.queue_data(packet)

        # Clear synced offline data
        cursor.execute('DELETE FROM offline_packets')
        conn.commit()
        conn.close()

    def acknowledge_packet(self, packet_id: str):
        """Acknowledge a synced packet"""
        if packet_id in self.pending_acks:
            del self.pending_acks[packet_id]
            with self.lock:
                self.sync_status['pending_packets'] -= 1
                self.sync_status['last_sync'] = datetime.now().isoformat()

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            **self.sync_status,
            'connected': self.connected,
            'connection_quality': self.connection_quality,
            'pending_acks': len(self.pending_acks),
            'queue_sizes': {
                p.value: self.send_queues[p].qsize()
                for p in DataPriority
            }
        }


class EdgeDeviceManager:
    """
    Edge device management system
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "edge"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Devices
        self.devices: Dict[str, EdgeDevice] = {}
        self.device_processors: Dict[str, EdgeDataProcessor] = {}

        # Sync manager
        self.sync_manager = EdgeSyncManager(str(self.data_dir))

        # Monitoring
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Events
        self.event_handlers: Dict[str, List[Callable]] = {}

        # Initialize demo devices
        self._init_demo_devices()

    def _init_demo_devices(self):
        """Initialize demonstration edge devices"""
        demo_devices = [
            EdgeDevice(
                device_id='edge-gw-001',
                device_type=EdgeDeviceType.GATEWAY,
                name='Main Edge Gateway',
                location='Control Room',
                status=EdgeDeviceStatus.ONLINE,
                ip_address='192.168.10.1',
                connected_sensors=['water_level_1', 'flow_rate_1', 'temp_1'],
                firmware_version='2.1.0',
                registered_at=datetime.now() - timedelta(days=30),
                last_seen=datetime.now(),
                last_sync=datetime.now()
            ),
            EdgeDevice(
                device_id='edge-hub-001',
                device_type=EdgeDeviceType.SENSOR_HUB,
                name='Upstream Sensor Hub',
                location='Upstream Section A',
                status=EdgeDeviceStatus.ONLINE,
                ip_address='192.168.10.10',
                connected_sensors=['water_level_2', 'vibration_1', 'strain_1'],
                firmware_version='1.5.2',
                registered_at=datetime.now() - timedelta(days=25),
                last_seen=datetime.now(),
                last_sync=datetime.now() - timedelta(minutes=5)
            ),
            EdgeDevice(
                device_id='edge-rtu-001',
                device_type=EdgeDeviceType.RTU,
                name='Gate Control RTU',
                location='Gate Section B',
                status=EdgeDeviceStatus.ONLINE,
                ip_address='192.168.10.20',
                connected_sensors=['gate_position_1', 'gate_position_2'],
                firmware_version='3.0.1',
                registered_at=datetime.now() - timedelta(days=60),
                last_seen=datetime.now(),
                last_sync=datetime.now() - timedelta(seconds=30)
            )
        ]

        for device in demo_devices:
            self.devices[device.device_id] = device
            self.device_processors[device.device_id] = EdgeDataProcessor()

    def start(self):
        """Start edge device manager"""
        self.running = True
        self.sync_manager.start()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop edge device manager"""
        self.running = False
        self.sync_manager.stop()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Device monitoring loop"""
        while self.running:
            self._check_device_health()
            time.sleep(10)

    def _check_device_health(self):
        """Check health of all devices"""
        timeout = timedelta(minutes=2)
        now = datetime.now()

        for device in self.devices.values():
            if now - device.last_seen > timeout:
                if device.status != EdgeDeviceStatus.OFFLINE:
                    old_status = device.status
                    device.status = EdgeDeviceStatus.OFFLINE
                    self._emit_event('device_offline', device, old_status)

    def register_device(self, device_info: Dict[str, Any]) -> EdgeDevice:
        """Register a new edge device"""
        device = EdgeDevice(
            device_id=device_info['device_id'],
            device_type=EdgeDeviceType(device_info.get('device_type', 'gateway')),
            name=device_info.get('name', device_info['device_id']),
            location=device_info.get('location', 'Unknown'),
            status=EdgeDeviceStatus.ONLINE,
            ip_address=device_info.get('ip_address', '0.0.0.0'),
            connected_sensors=device_info.get('sensors', []),
            firmware_version=device_info.get('firmware', '1.0.0'),
            registered_at=datetime.now(),
            last_seen=datetime.now(),
            last_sync=datetime.now()
        )

        self.devices[device.device_id] = device
        self.device_processors[device.device_id] = EdgeDataProcessor()

        self._emit_event('device_registered', device)

        return device

    def unregister_device(self, device_id: str) -> bool:
        """Unregister an edge device"""
        if device_id in self.devices:
            device = self.devices[device_id]
            del self.devices[device_id]
            if device_id in self.device_processors:
                del self.device_processors[device_id]

            self._emit_event('device_unregistered', device)
            return True
        return False

    def process_device_data(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data from an edge device"""
        if device_id not in self.devices:
            return {'error': 'Device not found'}

        device = self.devices[device_id]
        processor = self.device_processors.get(device_id)

        # Update device status
        device.last_seen = datetime.now()
        device.status = EdgeDeviceStatus.ONLINE

        # Process data locally
        results = []
        if processor:
            for sensor_id, reading in data.get('readings', {}).items():
                results.extend(processor.process_reading(sensor_id, reading))

        # Check for alerts
        alerts = [r for r in results if r['type'] == 'rule_triggered']

        # Queue for cloud sync
        if data.get('sync', True):
            packet = EdgeDataPacket(
                packet_id=f"{device_id}-{int(time.time()*1000)}",
                device_id=device_id,
                timestamp=datetime.now(),
                priority=DataPriority.HIGH if alerts else DataPriority.NORMAL,
                data_type='sensor_readings',
                payload=data,
                checksum=hashlib.md5(json.dumps(data).encode()).hexdigest()
            )
            self.sync_manager.queue_data(packet)

        return {
            'device_id': device_id,
            'processed': True,
            'results': results,
            'alerts': alerts,
            'queued_for_sync': data.get('sync', True)
        }

    def get_device(self, device_id: str) -> Optional[EdgeDevice]:
        """Get device information"""
        return self.devices.get(device_id)

    def get_all_devices(self) -> List[EdgeDevice]:
        """Get all registered devices"""
        return list(self.devices.values())

    def get_devices_by_type(self, device_type: EdgeDeviceType) -> List[EdgeDevice]:
        """Get devices by type"""
        return [d for d in self.devices.values() if d.device_type == device_type]

    def get_devices_by_status(self, status: EdgeDeviceStatus) -> List[EdgeDevice]:
        """Get devices by status"""
        return [d for d in self.devices.values() if d.status == status]

    def update_device_metrics(self, device_id: str, metrics: Dict[str, Any]):
        """Update device metrics"""
        device = self.devices.get(device_id)
        if device:
            device.last_seen = datetime.now()
            if 'cpu_usage' in metrics:
                device.cpu_usage = metrics['cpu_usage']
            if 'memory_usage' in metrics:
                device.memory_usage = metrics['memory_usage']
            if 'storage_usage' in metrics:
                device.storage_usage = metrics['storage_usage']
            if 'temperature' in metrics:
                device.temperature = metrics['temperature']
            if 'battery_level' in metrics:
                device.battery_level = metrics['battery_level']
            if 'signal_strength' in metrics:
                device.signal_strength = metrics['signal_strength']

    def get_aggregated_data(self, device_id: str) -> Dict[str, Any]:
        """Get aggregated data from device processor"""
        processor = self.device_processors.get(device_id)
        if not processor:
            return {}

        device = self.devices.get(device_id)
        if not device:
            return {}

        aggregated = {}
        for sensor_id in device.connected_sensors:
            aggregated[sensor_id] = processor.get_aggregated_data(sensor_id)

        return aggregated

    def set_device_sync_mode(self, device_id: str, mode: SyncMode):
        """Set device synchronization mode"""
        device = self.devices.get(device_id)
        if device:
            device.sync_mode = mode

    def send_command_to_device(self, device_id: str, command: str,
                               params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send command to edge device"""
        device = self.devices.get(device_id)
        if not device:
            return {'error': 'Device not found'}

        if device.status == EdgeDeviceStatus.OFFLINE:
            return {'error': 'Device offline'}

        # In production, would actually send command to device
        return {
            'device_id': device_id,
            'command': command,
            'params': params,
            'status': 'sent',
            'timestamp': datetime.now().isoformat()
        }

    def _emit_event(self, event_type: str, *args):
        """Emit event to handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(*args)
            except Exception as e:
                print(f"Event handler error: {e}")

    def on_event(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_status(self) -> Dict[str, Any]:
        """Get edge computing system status"""
        online_count = len([d for d in self.devices.values()
                          if d.status == EdgeDeviceStatus.ONLINE])
        offline_count = len([d for d in self.devices.values()
                           if d.status == EdgeDeviceStatus.OFFLINE])

        return {
            'total_devices': len(self.devices),
            'online_devices': online_count,
            'offline_devices': offline_count,
            'devices': {did: d.to_dict() for did, d in self.devices.items()},
            'sync_status': self.sync_manager.get_sync_status(),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_edge_manager = None


def get_edge_manager() -> EdgeDeviceManager:
    """Get global edge device manager"""
    global _edge_manager
    if _edge_manager is None:
        _edge_manager = EdgeDeviceManager()
    return _edge_manager


if __name__ == "__main__":
    # Test edge computing
    print("=== Edge Computing Test ===")

    manager = EdgeDeviceManager()
    manager.start()

    # Get all devices
    print("\n1. Registered Devices:")
    for device in manager.get_all_devices():
        print(f"   - {device.device_id}: {device.name} ({device.device_type.value})")

    # Process some data
    print("\n2. Processing Device Data:")
    result = manager.process_device_data('edge-gw-001', {
        'readings': {
            'water_level_1': {'water_level': 4.5, 'flow_rate': 85.0},
            'temp_1': {'T_sun': 35, 'T_shade': 25}
        }
    })
    print(f"   Processed: {result['processed']}")
    print(f"   Alerts: {len(result['alerts'])}")

    # Test alert trigger
    print("\n3. Testing Alert Rules:")
    result = manager.process_device_data('edge-gw-001', {
        'readings': {
            'water_level_1': {'water_level': 7.0, 'vibration': 60}
        }
    })
    for alert in result['alerts']:
        print(f"   Alert: {alert['rule_id']} -> {alert['action']}")

    # Get aggregated data
    print("\n4. Aggregated Data:")
    aggregated = manager.get_aggregated_data('edge-gw-001')
    for sensor, data in aggregated.items():
        if data:
            print(f"   {sensor}: count={data.get('count', 0)}")

    # System status
    print("\n5. Edge System Status:")
    status = manager.get_status()
    print(f"   Total devices: {status['total_devices']}")
    print(f"   Online: {status['online_devices']}")
    print(f"   Sync connected: {status['sync_status']['connected']}")

    manager.stop()
    print("\nEdge computing test completed!")
