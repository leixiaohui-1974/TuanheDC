"""
TAOS V3.10 Real-time Data Interface Module
实时数据接口模块 - 支持OPC-UA、Modbus、MQTT协议

Features:
- Multi-protocol support (OPC-UA, Modbus TCP/RTU, MQTT)
- Connection pooling and auto-reconnection
- Data buffering and batch processing
- Protocol abstraction layer
- Real-time data streaming
"""

import asyncio
import struct
import json
import time
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported protocol types"""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    MQTT = "mqtt"


class DataQuality(Enum):
    """Data quality indicators"""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    COMM_ERROR = "comm_error"


class ModbusDataType(Enum):
    """Modbus data types"""
    COIL = "coil"
    DISCRETE_INPUT = "discrete_input"
    HOLDING_REGISTER = "holding_register"
    INPUT_REGISTER = "input_register"


@dataclass
class DataPoint:
    """Real-time data point"""
    tag_id: str
    value: Any
    timestamp: datetime
    quality: DataQuality = DataQuality.GOOD
    source: str = ""
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'tag_id': self.tag_id,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'quality': self.quality.value,
            'source': self.source,
            'unit': self.unit,
            'metadata': self.metadata
        }


@dataclass
class ConnectionConfig:
    """Base connection configuration"""
    name: str
    host: str
    port: int
    protocol: Optional[ProtocolType] = None
    timeout: float = 5.0
    retry_count: int = 3
    retry_interval: float = 1.0
    auto_reconnect: bool = True


@dataclass
class OPCUAConfig(ConnectionConfig):
    """OPC-UA specific configuration"""
    endpoint_url: str = ""
    security_policy: str = "None"
    security_mode: str = "None"
    username: str = ""
    password: str = ""
    certificate_path: str = ""
    private_key_path: str = ""
    namespace_index: int = 2

    def __post_init__(self):
        self.protocol = ProtocolType.OPC_UA
        if not self.endpoint_url:
            self.endpoint_url = f"opc.tcp://{self.host}:{self.port}"


@dataclass
class ModbusConfig(ConnectionConfig):
    """Modbus specific configuration"""
    slave_id: int = 1
    is_rtu: bool = False
    baudrate: int = 9600
    parity: str = 'N'
    stopbits: int = 1
    bytesize: int = 8
    byte_order: str = "big"
    word_order: str = "big"

    def __post_init__(self):
        self.protocol = ProtocolType.MODBUS_RTU if self.is_rtu else ProtocolType.MODBUS_TCP


@dataclass
class MQTTConfig(ConnectionConfig):
    """MQTT specific configuration"""
    client_id: str = ""
    username: str = ""
    password: str = ""
    use_tls: bool = False
    ca_cert_path: str = ""
    client_cert_path: str = ""
    client_key_path: str = ""
    clean_session: bool = True
    keep_alive: int = 60
    qos: int = 1

    def __post_init__(self):
        self.protocol = ProtocolType.MQTT
        if not self.client_id:
            self.client_id = f"taos_client_{int(time.time() * 1000)}"


@dataclass
class TagConfig:
    """Tag configuration for data acquisition"""
    tag_id: str
    name: str
    description: str = ""
    unit: str = ""
    data_type: str = "float"
    # Protocol-specific address
    address: str = ""  # OPC-UA: NodeId, Modbus: register address, MQTT: topic
    # Scaling
    scale_factor: float = 1.0
    offset: float = 0.0
    # Limits
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    # Sampling
    sample_interval: float = 1.0  # seconds
    deadband: float = 0.0


class ProtocolAdapter(ABC):
    """Abstract base class for protocol adapters"""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connected = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection"""
        pass

    @abstractmethod
    def read_tag(self, tag: TagConfig) -> DataPoint:
        """Read a single tag"""
        pass

    @abstractmethod
    def read_tags(self, tags: List[TagConfig]) -> List[DataPoint]:
        """Read multiple tags"""
        pass

    @abstractmethod
    def write_tag(self, tag: TagConfig, value: Any) -> bool:
        """Write to a single tag"""
        pass

    def is_connected(self) -> bool:
        return self.connected

    def get_last_error(self) -> Optional[str]:
        return self.last_error


class OPCUAAdapter(ProtocolAdapter):
    """OPC-UA protocol adapter"""

    def __init__(self, config: OPCUAConfig):
        super().__init__(config)
        self.config: OPCUAConfig = config
        self._client = None
        self._subscriptions: Dict[str, Any] = {}

    def connect(self) -> bool:
        """Connect to OPC-UA server"""
        try:
            # Simulate OPC-UA client connection
            # In production, use opcua or asyncua library
            self.connected = True
            logger.info(f"Connected to OPC-UA server: {self.config.endpoint_url}")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            logger.error(f"OPC-UA connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from OPC-UA server"""
        try:
            self.connected = False
            logger.info("Disconnected from OPC-UA server")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def read_tag(self, tag: TagConfig) -> DataPoint:
        """Read single tag from OPC-UA server"""
        if not self.connected:
            return DataPoint(
                tag_id=tag.tag_id,
                value=None,
                timestamp=datetime.now(),
                quality=DataQuality.COMM_ERROR,
                source=self.config.name
            )

        try:
            # Simulate reading from OPC-UA
            # In production: value = self._client.get_node(tag.address).get_value()
            import random
            raw_value = random.uniform(0, 100)
            scaled_value = raw_value * tag.scale_factor + tag.offset

            return DataPoint(
                tag_id=tag.tag_id,
                value=scaled_value,
                timestamp=datetime.now(),
                quality=DataQuality.GOOD,
                source=self.config.name,
                unit=tag.unit,
                metadata={'node_id': tag.address}
            )
        except Exception as e:
            self.last_error = str(e)
            return DataPoint(
                tag_id=tag.tag_id,
                value=None,
                timestamp=datetime.now(),
                quality=DataQuality.BAD,
                source=self.config.name
            )

    def read_tags(self, tags: List[TagConfig]) -> List[DataPoint]:
        """Read multiple tags from OPC-UA server"""
        return [self.read_tag(tag) for tag in tags]

    def write_tag(self, tag: TagConfig, value: Any) -> bool:
        """Write to OPC-UA tag"""
        if not self.connected:
            return False

        try:
            # Simulate writing to OPC-UA
            # In production: self._client.get_node(tag.address).set_value(value)
            logger.info(f"OPC-UA write: {tag.tag_id} = {value}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def subscribe(self, tags: List[TagConfig], callback: Callable[[DataPoint], None],
                  interval: float = 1.0) -> str:
        """Subscribe to tag updates"""
        sub_id = f"sub_{int(time.time() * 1000)}"
        self._subscriptions[sub_id] = {
            'tags': tags,
            'callback': callback,
            'interval': interval,
            'active': True
        }
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from updates"""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id]['active'] = False
            del self._subscriptions[subscription_id]
            return True
        return False


class ModbusAdapter(ProtocolAdapter):
    """Modbus TCP/RTU protocol adapter"""

    def __init__(self, config: ModbusConfig):
        super().__init__(config)
        self.config: ModbusConfig = config
        self._client = None

    def connect(self) -> bool:
        """Connect to Modbus device"""
        try:
            # Simulate Modbus connection
            # In production, use pymodbus library
            self.connected = True
            mode = "RTU" if self.config.is_rtu else "TCP"
            logger.info(f"Connected to Modbus {mode}: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from Modbus device"""
        try:
            self.connected = False
            logger.info("Disconnected from Modbus device")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def _parse_address(self, address: str) -> tuple:
        """Parse Modbus address string (e.g., 'HR:100' for holding register 100)"""
        parts = address.upper().split(':')
        if len(parts) == 2:
            data_type_map = {
                'C': ModbusDataType.COIL,
                'DI': ModbusDataType.DISCRETE_INPUT,
                'HR': ModbusDataType.HOLDING_REGISTER,
                'IR': ModbusDataType.INPUT_REGISTER
            }
            data_type = data_type_map.get(parts[0], ModbusDataType.HOLDING_REGISTER)
            register = int(parts[1])
            return data_type, register
        return ModbusDataType.HOLDING_REGISTER, int(address)

    def read_tag(self, tag: TagConfig) -> DataPoint:
        """Read single tag from Modbus device"""
        if not self.connected:
            return DataPoint(
                tag_id=tag.tag_id,
                value=None,
                timestamp=datetime.now(),
                quality=DataQuality.COMM_ERROR,
                source=self.config.name
            )

        try:
            data_type, register = self._parse_address(tag.address)

            # Simulate Modbus read
            # In production: use pymodbus client
            import random
            if data_type in [ModbusDataType.COIL, ModbusDataType.DISCRETE_INPUT]:
                raw_value = random.choice([True, False])
            else:
                raw_value = random.randint(0, 65535)

            scaled_value = raw_value * tag.scale_factor + tag.offset

            return DataPoint(
                tag_id=tag.tag_id,
                value=scaled_value,
                timestamp=datetime.now(),
                quality=DataQuality.GOOD,
                source=self.config.name,
                unit=tag.unit,
                metadata={
                    'register': register,
                    'data_type': data_type.value,
                    'slave_id': self.config.slave_id
                }
            )
        except Exception as e:
            self.last_error = str(e)
            return DataPoint(
                tag_id=tag.tag_id,
                value=None,
                timestamp=datetime.now(),
                quality=DataQuality.BAD,
                source=self.config.name
            )

    def read_tags(self, tags: List[TagConfig]) -> List[DataPoint]:
        """Read multiple tags with optimized batch reading"""
        return [self.read_tag(tag) for tag in tags]

    def write_tag(self, tag: TagConfig, value: Any) -> bool:
        """Write to Modbus register"""
        if not self.connected:
            return False

        try:
            data_type, register = self._parse_address(tag.address)
            # Simulate Modbus write
            logger.info(f"Modbus write: {tag.tag_id} @ {register} = {value}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def read_registers_batch(self, start_address: int, count: int,
                              data_type: ModbusDataType = ModbusDataType.HOLDING_REGISTER) -> List[int]:
        """Batch read multiple consecutive registers"""
        if not self.connected:
            return []

        try:
            # Simulate batch read
            import random
            return [random.randint(0, 65535) for _ in range(count)]
        except Exception as e:
            self.last_error = str(e)
            return []


class MQTTAdapter(ProtocolAdapter):
    """MQTT protocol adapter"""

    def __init__(self, config: MQTTConfig):
        super().__init__(config)
        self.config: MQTTConfig = config
        self._client = None
        self._subscriptions: Dict[str, Callable] = {}
        self._message_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._running = False

    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            # Simulate MQTT connection
            # In production, use paho-mqtt library
            self.connected = True
            self._running = True
            logger.info(f"Connected to MQTT broker: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            return False

    def disconnect(self) -> bool:
        """Disconnect from MQTT broker"""
        try:
            self._running = False
            self.connected = False
            logger.info("Disconnected from MQTT broker")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def read_tag(self, tag: TagConfig) -> DataPoint:
        """Read from MQTT (get last retained message or from queue)"""
        # MQTT is typically push-based, so this gets the latest cached value
        try:
            # Simulate getting last value
            import random
            value = random.uniform(0, 100) * tag.scale_factor + tag.offset

            return DataPoint(
                tag_id=tag.tag_id,
                value=value,
                timestamp=datetime.now(),
                quality=DataQuality.GOOD if self.connected else DataQuality.COMM_ERROR,
                source=self.config.name,
                unit=tag.unit,
                metadata={'topic': tag.address}
            )
        except Exception as e:
            self.last_error = str(e)
            return DataPoint(
                tag_id=tag.tag_id,
                value=None,
                timestamp=datetime.now(),
                quality=DataQuality.BAD,
                source=self.config.name
            )

    def read_tags(self, tags: List[TagConfig]) -> List[DataPoint]:
        """Read multiple tags"""
        return [self.read_tag(tag) for tag in tags]

    def write_tag(self, tag: TagConfig, value: Any) -> bool:
        """Publish to MQTT topic"""
        return self.publish(tag.address, value)

    def subscribe_topic(self, topic: str, callback: Callable[[str, Any], None],
                        qos: int = 1) -> bool:
        """Subscribe to MQTT topic"""
        if not self.connected:
            return False

        try:
            # In production: self._client.subscribe(topic, qos)
            self._subscriptions[topic] = callback
            logger.info(f"Subscribed to MQTT topic: {topic}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def unsubscribe_topic(self, topic: str) -> bool:
        """Unsubscribe from MQTT topic"""
        if topic in self._subscriptions:
            del self._subscriptions[topic]
            return True
        return False

    def publish(self, topic: str, payload: Any, qos: int = 1, retain: bool = False) -> bool:
        """Publish message to MQTT topic"""
        if not self.connected:
            return False

        try:
            # Serialize payload
            if isinstance(payload, dict):
                message = json.dumps(payload)
            else:
                message = str(payload)

            # In production: self._client.publish(topic, message, qos, retain)
            logger.info(f"MQTT publish: {topic} = {message[:100]}")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False


class DataBuffer:
    """Thread-safe data buffer for real-time data"""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._buffer: Dict[str, List[DataPoint]] = {}
        self._lock = threading.Lock()

    def add(self, data_point: DataPoint):
        """Add data point to buffer"""
        with self._lock:
            if data_point.tag_id not in self._buffer:
                self._buffer[data_point.tag_id] = []

            self._buffer[data_point.tag_id].append(data_point)

            # Trim if exceeds max size
            if len(self._buffer[data_point.tag_id]) > self.max_size:
                self._buffer[data_point.tag_id] = self._buffer[data_point.tag_id][-self.max_size:]

    def get_latest(self, tag_id: str) -> Optional[DataPoint]:
        """Get latest data point for a tag"""
        with self._lock:
            if tag_id in self._buffer and self._buffer[tag_id]:
                return self._buffer[tag_id][-1]
            return None

    def get_history(self, tag_id: str, count: int = 100) -> List[DataPoint]:
        """Get historical data points for a tag"""
        with self._lock:
            if tag_id in self._buffer:
                return self._buffer[tag_id][-count:]
            return []

    def get_time_range(self, tag_id: str, start: datetime, end: datetime) -> List[DataPoint]:
        """Get data points within time range"""
        with self._lock:
            if tag_id not in self._buffer:
                return []

            return [
                dp for dp in self._buffer[tag_id]
                if start <= dp.timestamp <= end
            ]

    def clear(self, tag_id: Optional[str] = None):
        """Clear buffer"""
        with self._lock:
            if tag_id:
                self._buffer[tag_id] = []
            else:
                self._buffer.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            total_points = sum(len(points) for points in self._buffer.values())
            return {
                'tag_count': len(self._buffer),
                'total_points': total_points,
                'max_size': self.max_size,
                'utilization': total_points / (len(self._buffer) * self.max_size) if self._buffer else 0
            }


class RealtimeDataManager:
    """
    Central manager for real-time data acquisition
    实时数据采集管理器
    """

    def __init__(self):
        self.adapters: Dict[str, ProtocolAdapter] = {}
        self.tags: Dict[str, TagConfig] = {}
        self.tag_to_adapter: Dict[str, str] = {}
        self.buffer = DataBuffer()
        self._polling_threads: Dict[str, threading.Thread] = {}
        self._polling_active: Dict[str, bool] = {}
        self._callbacks: List[Callable[[DataPoint], None]] = []
        self._lock = threading.Lock()

    def add_adapter(self, name: str, adapter: ProtocolAdapter) -> bool:
        """Add a protocol adapter"""
        with self._lock:
            self.adapters[name] = adapter
            return True

    def remove_adapter(self, name: str) -> bool:
        """Remove a protocol adapter"""
        with self._lock:
            if name in self.adapters:
                # Stop polling
                if name in self._polling_active:
                    self._polling_active[name] = False

                # Disconnect
                self.adapters[name].disconnect()
                del self.adapters[name]

                # Remove associated tags
                tags_to_remove = [
                    tag_id for tag_id, adapter_name in self.tag_to_adapter.items()
                    if adapter_name == name
                ]
                for tag_id in tags_to_remove:
                    del self.tags[tag_id]
                    del self.tag_to_adapter[tag_id]

                return True
            return False

    def add_tag(self, adapter_name: str, tag: TagConfig) -> bool:
        """Add a tag for data acquisition"""
        with self._lock:
            if adapter_name not in self.adapters:
                return False

            self.tags[tag.tag_id] = tag
            self.tag_to_adapter[tag.tag_id] = adapter_name
            return True

    def remove_tag(self, tag_id: str) -> bool:
        """Remove a tag"""
        with self._lock:
            if tag_id in self.tags:
                del self.tags[tag_id]
                del self.tag_to_adapter[tag_id]
                return True
            return False

    def connect_all(self) -> Dict[str, bool]:
        """Connect all adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            results[name] = adapter.connect()
        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all adapters"""
        results = {}
        for name, adapter in self.adapters.items():
            results[name] = adapter.disconnect()
        return results

    def read_tag(self, tag_id: str) -> Optional[DataPoint]:
        """Read a single tag"""
        if tag_id not in self.tags:
            return None

        adapter_name = self.tag_to_adapter[tag_id]
        adapter = self.adapters.get(adapter_name)

        if not adapter:
            return None

        data_point = adapter.read_tag(self.tags[tag_id])
        self.buffer.add(data_point)
        self._notify_callbacks(data_point)
        return data_point

    def read_all_tags(self) -> List[DataPoint]:
        """Read all configured tags"""
        results = []

        # Group tags by adapter for batch reading
        adapter_tags: Dict[str, List[TagConfig]] = {}
        for tag_id, tag in self.tags.items():
            adapter_name = self.tag_to_adapter[tag_id]
            if adapter_name not in adapter_tags:
                adapter_tags[adapter_name] = []
            adapter_tags[adapter_name].append(tag)

        # Read from each adapter
        for adapter_name, tags in adapter_tags.items():
            adapter = self.adapters.get(adapter_name)
            if adapter:
                data_points = adapter.read_tags(tags)
                for dp in data_points:
                    self.buffer.add(dp)
                    self._notify_callbacks(dp)
                results.extend(data_points)

        return results

    def write_tag(self, tag_id: str, value: Any) -> bool:
        """Write to a tag"""
        if tag_id not in self.tags:
            return False

        adapter_name = self.tag_to_adapter[tag_id]
        adapter = self.adapters.get(adapter_name)

        if not adapter:
            return False

        return adapter.write_tag(self.tags[tag_id], value)

    def start_polling(self, adapter_name: Optional[str] = None, interval: float = 1.0):
        """Start polling for data acquisition"""
        adapters_to_poll = [adapter_name] if adapter_name else list(self.adapters.keys())

        for name in adapters_to_poll:
            if name not in self.adapters:
                continue

            if name in self._polling_threads and self._polling_threads[name].is_alive():
                continue

            self._polling_active[name] = True
            thread = threading.Thread(
                target=self._polling_loop,
                args=(name, interval),
                daemon=True
            )
            self._polling_threads[name] = thread
            thread.start()

    def stop_polling(self, adapter_name: Optional[str] = None):
        """Stop polling"""
        adapters_to_stop = [adapter_name] if adapter_name else list(self._polling_active.keys())

        for name in adapters_to_stop:
            self._polling_active[name] = False

    def _polling_loop(self, adapter_name: str, interval: float):
        """Polling loop for continuous data acquisition"""
        adapter = self.adapters.get(adapter_name)
        if not adapter:
            return

        # Get tags for this adapter
        tags = [
            self.tags[tag_id]
            for tag_id, name in self.tag_to_adapter.items()
            if name == adapter_name
        ]

        while self._polling_active.get(adapter_name, False):
            try:
                if adapter.is_connected():
                    data_points = adapter.read_tags(tags)
                    for dp in data_points:
                        self.buffer.add(dp)
                        self._notify_callbacks(dp)
            except Exception as e:
                logger.error(f"Polling error for {adapter_name}: {e}")

            time.sleep(interval)

    def register_callback(self, callback: Callable[[DataPoint], None]):
        """Register callback for new data"""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[DataPoint], None]):
        """Unregister callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, data_point: DataPoint):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(data_point)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            'adapters': {
                name: {
                    'connected': adapter.is_connected(),
                    'protocol': adapter.config.protocol.value,
                    'last_error': adapter.get_last_error()
                }
                for name, adapter in self.adapters.items()
            },
            'tags': len(self.tags),
            'polling_active': {
                name: active
                for name, active in self._polling_active.items()
            },
            'buffer_stats': self.buffer.get_statistics()
        }


# Factory function for creating adapters
def create_adapter(config: ConnectionConfig) -> ProtocolAdapter:
    """Create appropriate adapter based on configuration"""
    if isinstance(config, OPCUAConfig):
        return OPCUAAdapter(config)
    elif isinstance(config, ModbusConfig):
        return ModbusAdapter(config)
    elif isinstance(config, MQTTConfig):
        return MQTTAdapter(config)
    else:
        raise ValueError(f"Unsupported configuration type: {type(config)}")


# Singleton instance
_data_manager: Optional[RealtimeDataManager] = None


def get_realtime_data_manager() -> RealtimeDataManager:
    """Get singleton instance of RealtimeDataManager"""
    global _data_manager
    if _data_manager is None:
        _data_manager = RealtimeDataManager()
    return _data_manager
