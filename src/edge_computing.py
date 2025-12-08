"""
TAOS V3.10 Edge Computing Module
边缘计算模块

Features:
- Edge node management
- Distributed task execution
- Local data processing
- Edge-cloud synchronization
- Offline operation support
- Resource management
"""

import threading
import time
import uuid
import json
import queue
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Edge node status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    SYNCING = "syncing"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SyncStrategy(Enum):
    """Data synchronization strategy"""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"


@dataclass
class EdgeNode:
    """Edge computing node"""
    node_id: str
    name: str
    location: str
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_concurrent_tasks: int = 10
    
    # Current state
    current_tasks: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'name': self.name,
            'location': self.location,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'resources': self.resources,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'current_tasks': self.current_tasks,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage
        }

    def is_available(self) -> bool:
        """Check if node is available for tasks"""
        if self.status != NodeStatus.ONLINE:
            return False
        if self.current_tasks >= self.max_concurrent_tasks:
            return False
        if self.cpu_usage >= self.max_cpu_percent:
            return False
        if self.memory_usage >= self.max_memory_percent:
            return False
        return True


@dataclass
class EdgeTask:
    """Edge computing task"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timeout_seconds: float = 300.0
    retries: int = 0
    max_retries: int = 3
    required_capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_node': self.assigned_node,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error
        }


@dataclass
class SyncRecord:
    """Data synchronization record"""
    record_id: str
    source_node: str
    target: str  # "cloud" or node_id
    data_type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    synced: bool = False
    sync_attempts: int = 0
    checksum: str = ""

    def compute_checksum(self):
        """Compute data checksum"""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        self.checksum = hashlib.md5(data_str.encode()).hexdigest()


class TaskHandler(ABC):
    """Abstract task handler"""

    @abstractmethod
    def handle(self, task: EdgeTask) -> Any:
        """Execute the task"""
        pass

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Check if handler can process task type"""
        pass


class DataProcessingHandler(TaskHandler):
    """Handler for data processing tasks"""

    def can_handle(self, task_type: str) -> bool:
        return task_type in ["data_aggregate", "data_filter", "data_transform"]

    def handle(self, task: EdgeTask) -> Any:
        payload = task.payload
        task_type = task.task_type
        data = payload.get('data', [])
        
        if task_type == "data_aggregate":
            operation = payload.get('operation', 'mean')
            if operation == 'mean':
                return sum(data) / len(data) if data else 0
            elif operation == 'sum':
                return sum(data)
            elif operation == 'min':
                return min(data) if data else None
            elif operation == 'max':
                return max(data) if data else None
        
        elif task_type == "data_filter":
            threshold = payload.get('threshold', 0)
            operator = payload.get('operator', '>')
            if operator == '>':
                return [x for x in data if x > threshold]
            elif operator == '<':
                return [x for x in data if x < threshold]
            elif operator == '>=':
                return [x for x in data if x >= threshold]
            elif operator == '<=':
                return [x for x in data if x <= threshold]
        
        elif task_type == "data_transform":
            transform = payload.get('transform', 'identity')
            if transform == 'normalize':
                min_val, max_val = min(data), max(data)
                if max_val > min_val:
                    return [(x - min_val) / (max_val - min_val) for x in data]
                return data
            elif transform == 'scale':
                factor = payload.get('factor', 1.0)
                return [x * factor for x in data]
        
        return data


class ControlHandler(TaskHandler):
    """Handler for control tasks"""

    def can_handle(self, task_type: str) -> bool:
        return task_type in ["set_gate", "set_pump", "emergency_stop"]

    def handle(self, task: EdgeTask) -> Any:
        payload = task.payload
        task_type = task.task_type
        
        if task_type == "set_gate":
            gate_id = payload.get('gate_id')
            position = payload.get('position')
            logger.info(f"Setting gate {gate_id} to position {position}")
            return {'gate_id': gate_id, 'new_position': position, 'success': True}
        
        elif task_type == "set_pump":
            pump_id = payload.get('pump_id')
            speed = payload.get('speed')
            logger.info(f"Setting pump {pump_id} to speed {speed}")
            return {'pump_id': pump_id, 'new_speed': speed, 'success': True}
        
        elif task_type == "emergency_stop":
            device_id = payload.get('device_id')
            logger.warning(f"Emergency stop for device {device_id}")
            return {'device_id': device_id, 'stopped': True, 'success': True}
        
        return None


class AlarmHandler(TaskHandler):
    """Handler for alarm processing tasks"""

    def can_handle(self, task_type: str) -> bool:
        return task_type in ["check_alarm", "process_alarm"]

    def handle(self, task: EdgeTask) -> Any:
        payload = task.payload
        task_type = task.task_type
        
        if task_type == "check_alarm":
            value = payload.get('value', 0)
            threshold = payload.get('threshold', 0)
            condition = payload.get('condition', '>')
            
            alarm_triggered = False
            if condition == '>' and value > threshold:
                alarm_triggered = True
            elif condition == '<' and value < threshold:
                alarm_triggered = True
            elif condition == '>=' and value >= threshold:
                alarm_triggered = True
            elif condition == '<=' and value <= threshold:
                alarm_triggered = True
            
            return {
                'alarm_triggered': alarm_triggered,
                'value': value,
                'threshold': threshold,
                'condition': condition
            }
        
        elif task_type == "process_alarm":
            alarm_id = payload.get('alarm_id')
            action = payload.get('action', 'acknowledge')
            
            logger.info(f"Processing alarm {alarm_id} with action {action}")
            return {'alarm_id': alarm_id, 'action': action, 'processed': True}
        
        return None


class EdgeNodeExecutor:
    """Local task executor on edge node"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.handlers: List[TaskHandler] = []
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, EdgeTask] = {}
        self._running = False
        self._worker_threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        
        # Register default handlers
        self.register_handler(DataProcessingHandler())
        self.register_handler(ControlHandler())
        self.register_handler(AlarmHandler())

    def register_handler(self, handler: TaskHandler):
        """Register task handler"""
        self.handlers.append(handler)

    def submit_task(self, task: EdgeTask):
        """Submit task for execution"""
        # Priority queue orders by (priority, timestamp)
        # Negate priority so higher priority comes first
        self.task_queue.put((-task.priority.value, task.created_at.timestamp(), task))

    def start(self, num_workers: int = 4):
        """Start executor with worker threads"""
        self._running = True
        for i in range(num_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self._worker_threads.append(thread)

    def stop(self):
        """Stop executor"""
        self._running = False
        for thread in self._worker_threads:
            thread.join(timeout=5.0)

    def _worker_loop(self):
        """Worker thread loop"""
        while self._running:
            try:
                # Wait for task with timeout
                try:
                    _, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._execute_task(task)
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _execute_task(self, task: EdgeTask):
        """Execute a single task"""
        with self._lock:
            self.running_tasks[task.task_id] = task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
        
        try:
            # Find appropriate handler
            handler = None
            for h in self.handlers:
                if h.can_handle(task.task_type):
                    handler = h
                    break
            
            if handler is None:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute task
            result = handler.handle(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            # Retry if possible
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                self.submit_task(task)
        
        finally:
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]


class EdgeCloudSync:
    """Edge-cloud data synchronization"""

    def __init__(self, node_id: str, strategy: SyncStrategy = SyncStrategy.BATCH):
        self.node_id = node_id
        self.strategy = strategy
        self.sync_queue: List[SyncRecord] = []
        self.batch_size = 100
        self.sync_interval = 60.0  # seconds
        self._lock = threading.Lock()
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        
        # Offline buffer
        self.offline_buffer: List[SyncRecord] = []
        self.is_connected = True

    def add_to_sync(self, data_type: str, data: Any, target: str = "cloud"):
        """Add data to sync queue"""
        record = SyncRecord(
            record_id=str(uuid.uuid4()),
            source_node=self.node_id,
            target=target,
            data_type=data_type,
            data=data
        )
        record.compute_checksum()
        
        with self._lock:
            if self.is_connected:
                self.sync_queue.append(record)
                
                if self.strategy == SyncStrategy.IMMEDIATE:
                    self._sync_now()
                elif self.strategy == SyncStrategy.BATCH and len(self.sync_queue) >= self.batch_size:
                    self._sync_now()
            else:
                self.offline_buffer.append(record)

    def start_sync(self):
        """Start periodic sync"""
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def stop_sync(self):
        """Stop sync"""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)

    def _sync_loop(self):
        """Periodic sync loop"""
        while self._running:
            time.sleep(self.sync_interval)
            if self.sync_queue:
                self._sync_now()

    def _sync_now(self):
        """Execute synchronization"""
        with self._lock:
            if not self.sync_queue:
                return
            
            records_to_sync = self.sync_queue[:self.batch_size]
            
            try:
                # Simulate cloud sync
                # In production: send to cloud API
                success = self._send_to_cloud(records_to_sync)
                
                if success:
                    for record in records_to_sync:
                        record.synced = True
                    self.sync_queue = self.sync_queue[self.batch_size:]
                    logger.debug(f"Synced {len(records_to_sync)} records to cloud")
                else:
                    for record in records_to_sync:
                        record.sync_attempts += 1
                        
            except Exception as e:
                logger.error(f"Sync error: {e}")
                self.is_connected = False

    def _send_to_cloud(self, records: List[SyncRecord]) -> bool:
        """Send records to cloud (simulated)"""
        # In production: HTTP/MQTT/gRPC call to cloud
        return True

    def sync_offline_buffer(self):
        """Sync buffered offline data when connection restored"""
        with self._lock:
            if self.offline_buffer:
                self.sync_queue.extend(self.offline_buffer)
                self.offline_buffer.clear()
                self._sync_now()


class EdgeComputingManager:
    """
    Central edge computing manager
    边缘计算管理器
    """

    def __init__(self):
        self.nodes: Dict[str, EdgeNode] = {}
        self.executors: Dict[str, EdgeNodeExecutor] = {}
        self.sync_handlers: Dict[str, EdgeCloudSync] = {}
        self.tasks: Dict[str, EdgeTask] = {}
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

    def register_node(self, node: EdgeNode) -> str:
        """Register edge node"""
        with self._lock:
            self.nodes[node.node_id] = node
            
            # Create executor for node
            executor = EdgeNodeExecutor(node.node_id)
            executor.start()
            self.executors[node.node_id] = executor
            
            # Create sync handler
            sync = EdgeCloudSync(node.node_id)
            sync.start_sync()
            self.sync_handlers[node.node_id] = sync
            
            node.status = NodeStatus.ONLINE
            logger.info(f"Registered edge node: {node.name}")
            return node.node_id

    def unregister_node(self, node_id: str) -> bool:
        """Unregister edge node"""
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            # Stop executor
            if node_id in self.executors:
                self.executors[node_id].stop()
                del self.executors[node_id]
            
            # Stop sync
            if node_id in self.sync_handlers:
                self.sync_handlers[node_id].stop_sync()
                del self.sync_handlers[node_id]
            
            del self.nodes[node_id]
            return True

    def update_heartbeat(self, node_id: str, metrics: Dict[str, Any] = None):
        """Update node heartbeat and metrics"""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = datetime.now()
                node.status = NodeStatus.ONLINE
                
                if metrics:
                    node.cpu_usage = metrics.get('cpu_usage', node.cpu_usage)
                    node.memory_usage = metrics.get('memory_usage', node.memory_usage)
                    node.current_tasks = metrics.get('current_tasks', node.current_tasks)

    def submit_task(self, task: EdgeTask) -> str:
        """Submit task for execution"""
        with self._lock:
            # Find suitable node
            node = self._select_node(task)
            if not node:
                raise RuntimeError("No available node for task")
            
            task.assigned_node = node.node_id
            self.tasks[task.task_id] = task
            
            # Submit to executor
            executor = self.executors.get(node.node_id)
            if executor:
                executor.submit_task(task)
            
            return task.task_id

    def _select_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Select best node for task"""
        available_nodes = []
        
        for node in self.nodes.values():
            if not node.is_available():
                continue
            
            # Check required capabilities
            if task.required_capabilities:
                if not all(cap in node.capabilities for cap in task.required_capabilities):
                    continue
            
            available_nodes.append(node)
        
        if not available_nodes:
            return None
        
        # Select node with lowest load
        return min(available_nodes, key=lambda n: n.cpu_usage + n.memory_usage)

    def get_task_status(self, task_id: str) -> Optional[EdgeTask]:
        """Get task status"""
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
                    return True
        return False

    def sync_data(self, node_id: str, data_type: str, data: Any):
        """Sync data from node to cloud"""
        with self._lock:
            sync = self.sync_handlers.get(node_id)
            if sync:
                sync.add_to_sync(data_type, data)

    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node status"""
        node = self.nodes.get(node_id)
        return node.to_dict() if node else None

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes status"""
        return [node.to_dict() for node in self.nodes.values()]

    def start_monitoring(self):
        """Start node health monitoring"""
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Monitor node health"""
        timeout_seconds = 60.0
        
        while self._running:
            time.sleep(10.0)
            
            now = datetime.now()
            with self._lock:
                for node in self.nodes.values():
                    if node.status == NodeStatus.ONLINE:
                        elapsed = (now - node.last_heartbeat).total_seconds()
                        if elapsed > timeout_seconds:
                            node.status = NodeStatus.OFFLINE
                            logger.warning(f"Node {node.name} went offline (no heartbeat)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self._lock:
            online_count = sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE)
            task_counts = defaultdict(int)
            for task in self.tasks.values():
                task_counts[task.status.value] += 1
            
            return {
                'total_nodes': len(self.nodes),
                'online_nodes': online_count,
                'total_tasks': len(self.tasks),
                'task_counts': dict(task_counts)
            }


_edge_manager: Optional[EdgeComputingManager] = None


def get_edge_computing_manager() -> EdgeComputingManager:
    """Get singleton edge computing manager"""
    global _edge_manager
    if _edge_manager is None:
        _edge_manager = EdgeComputingManager()
    return _edge_manager


def create_edge_node(name: str, location: str,
                     capabilities: List[str] = None) -> EdgeNode:
    """Create edge node"""
    return EdgeNode(
        node_id=str(uuid.uuid4()),
        name=name,
        location=location,
        capabilities=capabilities or ["data_processing", "control", "alarm"]
    )


def create_edge_task(task_type: str, payload: Dict[str, Any],
                     priority: TaskPriority = TaskPriority.NORMAL) -> EdgeTask:
    """Create edge task"""
    return EdgeTask(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        payload=payload,
        priority=priority
    )


# ============================================================
# Backward Compatibility for server.py
# ============================================================

class EdgeDeviceType(Enum):
    """Edge device types - backward compatible"""
    GATEWAY = "gateway"
    SENSOR_HUB = "sensor_hub"
    PLC = "plc"
    RTU = "rtu"
    IPC = "ipc"
    SMART_SENSOR = "smart_sensor"


class EdgeDeviceStatus(Enum):
    """Edge device status - backward compatible"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class EdgeDeviceManager:
    """
    Edge Device Manager - Backward compatible wrapper
    边缘设备管理器 - 向后兼容包装器
    """

    def __init__(self):
        self._manager = get_edge_computing_manager()
        self._devices: Dict[str, Dict] = {}
        self._running = False

    def start(self):
        """Start edge device manager"""
        self._running = True
        self._manager.start()

    def stop(self):
        """Stop edge device manager"""
        self._running = False
        self._manager.stop()

    def register_device(self, device_id: str, device_type: EdgeDeviceType,
                       name: str, location: str = "") -> bool:
        """Register an edge device"""
        self._devices[device_id] = {
            'id': device_id,
            'type': device_type.value,
            'name': name,
            'location': location,
            'status': EdgeDeviceStatus.ONLINE.value
        }
        return True

    def get_device(self, device_id: str) -> Optional[Dict]:
        """Get device info"""
        return self._devices.get(device_id)

    def get_all_devices(self) -> List[Dict]:
        """Get all registered devices"""
        return list(self._devices.values())

    def process_device_data(self, device_id: str, data: Dict) -> Dict:
        """Process data from device"""
        return {'status': 'processed', 'device_id': device_id, 'data': data}

    def send_command(self, device_id: str, command: Dict) -> bool:
        """Send command to device"""
        return device_id in self._devices

    def get_sync_status(self) -> Dict:
        """Get sync status"""
        return {
            'synced_devices': len(self._devices),
            'pending_sync': 0,
            'last_sync': datetime.now().isoformat()
        }

    def get_status(self) -> Dict:
        """Get manager status"""
        return {
            'running': self._running,
            'total_devices': len(self._devices),
            'online_devices': len([d for d in self._devices.values()
                                   if d.get('status') == 'online'])
        }


def get_edge_manager() -> EdgeDeviceManager:
    """Get edge manager (backward compatible)"""
    return EdgeDeviceManager()
