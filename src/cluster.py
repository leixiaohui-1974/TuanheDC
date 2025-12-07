#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.8 - Distributed Deployment & High Availability
团河渡槽自主运行系统 - 分布式部署与高可用模块

Features:
- Cluster node management
- Leader election (Raft-like)
- Load balancing
- Health monitoring
- Automatic failover
- State synchronization
- Service discovery
"""

import time
import json
import hashlib
import threading
import socket
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from collections import deque
import sqlite3
from pathlib import Path


class NodeRole(Enum):
    """Node roles in the cluster"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    OBSERVER = "observer"


class NodeStatus(Enum):
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    JOINING = "joining"
    LEAVING = "leaving"


class ServiceType(Enum):
    """Service types"""
    SIMULATION = "simulation"
    CONTROL = "control"
    API = "api"
    MONITORING = "monitoring"
    STORAGE = "storage"
    EDGE_GATEWAY = "edge_gateway"


@dataclass
class NodeInfo:
    """Cluster node information"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: NodeRole
    status: NodeStatus
    services: List[ServiceType]
    joined_at: datetime
    last_heartbeat: datetime
    load: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    connection_count: int = 0
    requests_per_second: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'port': self.port,
            'role': self.role.value,
            'status': self.status.value,
            'services': [s.value for s in self.services],
            'joined_at': self.joined_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'load': self.load,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'connection_count': self.connection_count,
            'requests_per_second': self.requests_per_second,
            'metadata': self.metadata
        }

    def is_alive(self, timeout_seconds: float = 30) -> bool:
        """Check if node is considered alive"""
        return (datetime.now() - self.last_heartbeat).total_seconds() < timeout_seconds


@dataclass
class ClusterState:
    """Cluster-wide state"""
    term: int = 0
    leader_id: Optional[str] = None
    commit_index: int = 0
    last_applied: int = 0
    state_hash: str = ""
    last_sync: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'leader_id': self.leader_id,
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'state_hash': self.state_hash,
            'last_sync': self.last_sync.isoformat()
        }


@dataclass
class LogEntry:
    """Replicated log entry"""
    index: int
    term: int
    command: str
    data: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'term': self.term,
            'command': self.command,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }


class LoadBalancer:
    """
    Load balancer for distributing requests across nodes
    """

    def __init__(self):
        self.strategies = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted': self._weighted,
            'random': self._random,
            'ip_hash': self._ip_hash
        }
        self.current_strategy = 'round_robin'
        self.rr_index = 0
        self.weights: Dict[str, float] = {}

    def select_node(self, nodes: List[NodeInfo], client_ip: str = None,
                    service: ServiceType = None) -> Optional[NodeInfo]:
        """Select a node based on the current strategy"""
        # Filter healthy nodes that provide the service
        available = [n for n in nodes
                    if n.status == NodeStatus.HEALTHY
                    and (service is None or service in n.services)]

        if not available:
            return None

        strategy = self.strategies.get(self.current_strategy, self._round_robin)
        return strategy(available, client_ip)

    def _round_robin(self, nodes: List[NodeInfo], client_ip: str = None) -> NodeInfo:
        """Round-robin selection"""
        node = nodes[self.rr_index % len(nodes)]
        self.rr_index += 1
        return node

    def _least_connections(self, nodes: List[NodeInfo], client_ip: str = None) -> NodeInfo:
        """Select node with least connections"""
        return min(nodes, key=lambda n: n.connection_count)

    def _weighted(self, nodes: List[NodeInfo], client_ip: str = None) -> NodeInfo:
        """Weighted selection based on node capacity"""
        # Calculate weights based on inverse load
        total_weight = sum(1.0 / (n.load + 0.1) for n in nodes)
        r = random.random() * total_weight

        cumulative = 0.0
        for node in nodes:
            cumulative += 1.0 / (node.load + 0.1)
            if r <= cumulative:
                return node
        return nodes[-1]

    def _random(self, nodes: List[NodeInfo], client_ip: str = None) -> NodeInfo:
        """Random selection"""
        return random.choice(nodes)

    def _ip_hash(self, nodes: List[NodeInfo], client_ip: str = None) -> NodeInfo:
        """Consistent hashing based on client IP"""
        if client_ip is None:
            return self._round_robin(nodes)

        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return nodes[hash_val % len(nodes)]

    def set_strategy(self, strategy: str):
        """Set the load balancing strategy"""
        if strategy in self.strategies:
            self.current_strategy = strategy


class HealthChecker:
    """
    Node health monitoring system
    """

    def __init__(self, check_interval: float = 5.0, timeout: float = 2.0):
        self.check_interval = check_interval
        self.timeout = timeout
        self.health_history: Dict[str, deque] = {}
        self.max_history = 100
        self.unhealthy_threshold = 3
        self.running = False
        self.thread = None
        self.check_functions: List[Callable] = []
        self.on_status_change: Optional[Callable] = None

    def start(self, nodes: Dict[str, NodeInfo]):
        """Start health checking"""
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, args=(nodes,), daemon=True)
        self.thread.start()

    def stop(self):
        """Stop health checking"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _check_loop(self, nodes: Dict[str, NodeInfo]):
        """Main health check loop"""
        while self.running:
            for node_id, node in list(nodes.items()):
                if node.role == NodeRole.OBSERVER:
                    continue

                old_status = node.status
                new_status = self._check_node(node)

                if old_status != new_status:
                    node.status = new_status
                    if self.on_status_change:
                        self.on_status_change(node, old_status, new_status)

            time.sleep(self.check_interval)

    def _check_node(self, node: NodeInfo) -> NodeStatus:
        """Check a single node's health"""
        if node_id := node.node_id not in self.health_history:
            self.health_history[node.node_id] = deque(maxlen=self.max_history)

        # Simulate health check (in production, would ping the node)
        is_healthy = self._ping_node(node)
        self.health_history[node.node_id].append({
            'timestamp': datetime.now(),
            'healthy': is_healthy,
            'latency': random.uniform(1, 50)  # Simulated latency
        })

        # Determine status based on recent history
        recent = list(self.health_history[node.node_id])[-self.unhealthy_threshold:]
        unhealthy_count = sum(1 for h in recent if not h['healthy'])

        if unhealthy_count == 0:
            return NodeStatus.HEALTHY
        elif unhealthy_count < self.unhealthy_threshold:
            return NodeStatus.DEGRADED
        else:
            return NodeStatus.UNHEALTHY

    def _ping_node(self, node: NodeInfo) -> bool:
        """Ping a node to check if it's alive"""
        # Check if heartbeat is recent
        return node.is_alive(self.timeout * 10)

    def get_node_health_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get health history for a node"""
        if node_id in self.health_history:
            return list(self.health_history[node_id])
        return []


class StateReplicator:
    """
    State replication across cluster nodes
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "cluster"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.state_snapshot: Dict[str, Any] = {}
        self.snapshot_interval = 100  # Log entries between snapshots
        self.lock = threading.Lock()

    def append_log(self, command: str, data: Dict[str, Any], term: int) -> LogEntry:
        """Append a new entry to the log"""
        with self.lock:
            index = len(self.log) + 1
            entry = LogEntry(
                index=index,
                term=term,
                command=command,
                data=data,
                timestamp=datetime.now()
            )
            self.log.append(entry)

            # Create snapshot if needed
            if len(self.log) % self.snapshot_interval == 0:
                self._create_snapshot()

            return entry

    def commit(self, index: int):
        """Commit log entries up to index"""
        with self.lock:
            if index > self.commit_index:
                self.commit_index = index
                self._apply_committed()

    def _apply_committed(self):
        """Apply committed entries to state"""
        for entry in self.log[self.commit_index-1:]:
            if entry.index <= self.commit_index:
                self._apply_entry(entry)

    def _apply_entry(self, entry: LogEntry):
        """Apply a single log entry to state"""
        if entry.command == 'SET':
            key = entry.data.get('key')
            value = entry.data.get('value')
            if key:
                self.state_snapshot[key] = value
        elif entry.command == 'DELETE':
            key = entry.data.get('key')
            if key in self.state_snapshot:
                del self.state_snapshot[key]

    def _create_snapshot(self):
        """Create a state snapshot"""
        snapshot_file = self.data_dir / f"snapshot_{self.commit_index}.json"
        with open(snapshot_file, 'w') as f:
            json.dump({
                'commit_index': self.commit_index,
                'state': self.state_snapshot,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

    def get_log_entries(self, from_index: int) -> List[LogEntry]:
        """Get log entries from index"""
        return [e for e in self.log if e.index >= from_index]

    def get_state_hash(self) -> str:
        """Get hash of current state"""
        state_str = json.dumps(self.state_snapshot, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class ClusterManager:
    """
    Main cluster management system for TAOS

    Implements:
    - Node registration and discovery
    - Leader election (simplified Raft)
    - State replication
    - Automatic failover
    - Load balancing
    """

    def __init__(self, node_id: str = None, data_dir: str = None):
        # This node's identity
        self.node_id = node_id or self._generate_node_id()
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        self.port = 5000

        # Cluster components
        self.nodes: Dict[str, NodeInfo] = {}
        self.cluster_state = ClusterState()
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker()
        self.replicator = StateReplicator(data_dir)

        # Election state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.votes_received: Set[str] = set()
        self.election_timeout = random.uniform(1.5, 3.0)
        self.last_heartbeat_time = time.time()

        # This node's info
        self.this_node = NodeInfo(
            node_id=self.node_id,
            hostname=self.hostname,
            ip_address=self.ip_address,
            port=self.port,
            role=NodeRole.FOLLOWER,
            status=NodeStatus.HEALTHY,
            services=[ServiceType.SIMULATION, ServiceType.CONTROL, ServiceType.API],
            joined_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        self.nodes[self.node_id] = self.this_node

        # Threading
        self.running = False
        self.election_thread = None
        self.heartbeat_thread = None
        self.lock = threading.Lock()

        # Callbacks
        self.on_leader_change: Optional[Callable] = None
        self.on_node_join: Optional[Callable] = None
        self.on_node_leave: Optional[Callable] = None

        # Data persistence
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "cluster"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import uuid
        return f"node-{uuid.uuid4().hex[:8]}"

    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def start(self):
        """Start the cluster manager"""
        self.running = True

        # Start health checker
        self.health_checker.on_status_change = self._on_node_status_change
        self.health_checker.start(self.nodes)

        # Start election timer
        self.election_thread = threading.Thread(target=self._election_loop, daemon=True)
        self.election_thread.start()

        # Start heartbeat sender (if leader)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        print(f"[Cluster] Node {self.node_id} started as {self.this_node.role.value}")

    def stop(self):
        """Stop the cluster manager"""
        self.running = False
        self.health_checker.stop()
        if self.election_thread:
            self.election_thread.join(timeout=5)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)

    def _election_loop(self):
        """Election timeout and candidate logic"""
        while self.running:
            time.sleep(0.1)

            if self.this_node.role == NodeRole.LEADER:
                continue

            # Check election timeout
            elapsed = time.time() - self.last_heartbeat_time
            if elapsed > self.election_timeout:
                self._start_election()

    def _start_election(self):
        """Start leader election"""
        with self.lock:
            self.current_term += 1
            self.this_node.role = NodeRole.CANDIDATE
            self.voted_for = self.node_id
            self.votes_received = {self.node_id}
            self.election_timeout = random.uniform(1.5, 3.0)
            self.last_heartbeat_time = time.time()

        print(f"[Cluster] Node {self.node_id} starting election for term {self.current_term}")

        # Request votes from other nodes
        self._request_votes()

    def _request_votes(self):
        """Request votes from other nodes (simulated)"""
        # In a real implementation, this would send RPCs to other nodes
        # For simulation, assume we get votes from all healthy nodes
        healthy_nodes = [n for n in self.nodes.values()
                        if n.status == NodeStatus.HEALTHY and n.node_id != self.node_id]

        # Simulate vote collection
        for node in healthy_nodes:
            # In production, would send RequestVote RPC
            # Here we simulate that followers grant votes
            self.votes_received.add(node.node_id)

        # Check if we have majority
        if len(self.votes_received) > len(self.nodes) / 2:
            self._become_leader()

    def _become_leader(self):
        """Transition to leader role"""
        with self.lock:
            self.this_node.role = NodeRole.LEADER
            self.cluster_state.leader_id = self.node_id
            self.cluster_state.term = self.current_term

        print(f"[Cluster] Node {self.node_id} became leader for term {self.current_term}")

        if self.on_leader_change:
            self.on_leader_change(self.node_id)

    def _heartbeat_loop(self):
        """Send heartbeats to followers (leader only)"""
        while self.running:
            if self.this_node.role == NodeRole.LEADER:
                self._send_heartbeats()
            time.sleep(0.5)

    def _send_heartbeats(self):
        """Send heartbeat to all followers"""
        self.this_node.last_heartbeat = datetime.now()

        # In production, would send AppendEntries RPCs
        # Here we update the cluster state
        self.cluster_state.last_sync = datetime.now()
        self.cluster_state.state_hash = self.replicator.get_state_hash()

    def receive_heartbeat(self, leader_id: str, term: int):
        """Receive heartbeat from leader"""
        with self.lock:
            if term >= self.current_term:
                self.current_term = term
                self.this_node.role = NodeRole.FOLLOWER
                self.cluster_state.leader_id = leader_id
                self.last_heartbeat_time = time.time()

    def _on_node_status_change(self, node: NodeInfo, old_status: NodeStatus,
                               new_status: NodeStatus):
        """Handle node status changes"""
        print(f"[Cluster] Node {node.node_id} status: {old_status.value} -> {new_status.value}")

        if new_status == NodeStatus.UNHEALTHY:
            # Check if leader went down
            if node.node_id == self.cluster_state.leader_id:
                print(f"[Cluster] Leader {node.node_id} is down, triggering election")
                self.last_heartbeat_time = 0  # Trigger election

    def join_cluster(self, seed_node: str) -> bool:
        """Join an existing cluster"""
        # In production, would contact seed node to get cluster info
        self.this_node.status = NodeStatus.JOINING

        # Simulate joining
        self.this_node.status = NodeStatus.HEALTHY
        self.this_node.role = NodeRole.FOLLOWER

        if self.on_node_join:
            self.on_node_join(self.this_node)

        return True

    def leave_cluster(self):
        """Leave the cluster gracefully"""
        self.this_node.status = NodeStatus.LEAVING

        # If we're the leader, trigger election
        if self.this_node.role == NodeRole.LEADER:
            self.this_node.role = NodeRole.FOLLOWER
            self.cluster_state.leader_id = None

        if self.on_node_leave:
            self.on_node_leave(self.this_node)

        self.stop()

    def register_node(self, node_info: Dict[str, Any]) -> bool:
        """Register a new node in the cluster"""
        node = NodeInfo(
            node_id=node_info['node_id'],
            hostname=node_info.get('hostname', 'unknown'),
            ip_address=node_info['ip_address'],
            port=node_info.get('port', 5000),
            role=NodeRole.FOLLOWER,
            status=NodeStatus.JOINING,
            services=[ServiceType(s) for s in node_info.get('services', [])],
            joined_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        self.nodes[node.node_id] = node
        node.status = NodeStatus.HEALTHY

        print(f"[Cluster] Registered new node: {node.node_id}")
        return True

    def unregister_node(self, node_id: str) -> bool:
        """Remove a node from the cluster"""
        if node_id in self.nodes and node_id != self.node_id:
            del self.nodes[node_id]
            print(f"[Cluster] Unregistered node: {node_id}")
            return True
        return False

    def get_leader(self) -> Optional[NodeInfo]:
        """Get current leader node"""
        leader_id = self.cluster_state.leader_id
        if leader_id and leader_id in self.nodes:
            return self.nodes[leader_id]
        return None

    def get_healthy_nodes(self) -> List[NodeInfo]:
        """Get all healthy nodes"""
        return [n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]

    def select_node_for_request(self, service: ServiceType = None,
                                 client_ip: str = None) -> Optional[NodeInfo]:
        """Select a node to handle a request"""
        nodes = self.get_healthy_nodes()
        return self.load_balancer.select_node(nodes, client_ip, service)

    def replicate_state(self, key: str, value: Any) -> bool:
        """Replicate state across the cluster"""
        if self.this_node.role != NodeRole.LEADER:
            # Forward to leader
            return False

        entry = self.replicator.append_log('SET', {'key': key, 'value': value},
                                          self.current_term)

        # In production, would wait for majority acknowledgment
        self.replicator.commit(entry.index)

        return True

    def get_state(self, key: str) -> Optional[Any]:
        """Get replicated state"""
        return self.replicator.state_snapshot.get(key)

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        return {
            'cluster_id': 'taos-cluster',
            'this_node': self.this_node.to_dict(),
            'cluster_state': self.cluster_state.to_dict(),
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'node_count': len(self.nodes),
            'healthy_count': len(self.get_healthy_nodes()),
            'leader': self.get_leader().to_dict() if self.get_leader() else None,
            'load_balancer': {
                'strategy': self.load_balancer.current_strategy
            },
            'replication': {
                'log_length': len(self.replicator.log),
                'commit_index': self.replicator.commit_index,
                'state_hash': self.replicator.get_state_hash()
            }
        }

    def get_node_metrics(self, node_id: str = None) -> Dict[str, Any]:
        """Get metrics for a node"""
        node = self.nodes.get(node_id or self.node_id)
        if not node:
            return {}

        return {
            'node_id': node.node_id,
            'role': node.role.value,
            'status': node.status.value,
            'load': node.load,
            'cpu_usage': node.cpu_usage,
            'memory_usage': node.memory_usage,
            'disk_usage': node.disk_usage,
            'connection_count': node.connection_count,
            'requests_per_second': node.requests_per_second,
            'health_history': self.health_checker.get_node_health_history(node.node_id)
        }

    def update_node_metrics(self, cpu: float = None, memory: float = None,
                           disk: float = None, connections: int = None,
                           rps: float = None):
        """Update this node's metrics"""
        if cpu is not None:
            self.this_node.cpu_usage = cpu
        if memory is not None:
            self.this_node.memory_usage = memory
        if disk is not None:
            self.this_node.disk_usage = disk
        if connections is not None:
            self.this_node.connection_count = connections
        if rps is not None:
            self.this_node.requests_per_second = rps

        # Calculate overall load
        self.this_node.load = (
            self.this_node.cpu_usage * 0.4 +
            self.this_node.memory_usage * 0.3 +
            self.this_node.disk_usage * 0.2 +
            min(self.this_node.connection_count / 100, 1.0) * 0.1
        )


class FailoverManager:
    """
    Automatic failover management
    """

    def __init__(self, cluster: ClusterManager):
        self.cluster = cluster
        self.failover_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self.running = False
        self.thread = None

        # Failover policies
        self.auto_failover_enabled = True
        self.min_healthy_nodes = 1
        self.failover_cooldown = 60  # seconds
        self.last_failover_time = 0

    def start(self):
        """Start failover monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop failover monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            self._check_failover_conditions()
            time.sleep(5)

    def _check_failover_conditions(self):
        """Check if failover is needed"""
        if not self.auto_failover_enabled:
            return

        # Check cooldown
        if time.time() - self.last_failover_time < self.failover_cooldown:
            return

        healthy_count = len(self.cluster.get_healthy_nodes())

        # Check if leader is healthy
        leader = self.cluster.get_leader()
        if leader and leader.status != NodeStatus.HEALTHY:
            self._trigger_failover('leader_unhealthy', leader.node_id)

        # Check minimum healthy nodes
        if healthy_count < self.min_healthy_nodes:
            self._trigger_failover('insufficient_nodes', None)

    def _trigger_failover(self, reason: str, failed_node: str):
        """Trigger failover process"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'failed_node': failed_node,
            'action': 'leader_election'
        }

        self.failover_history.append(event)
        if len(self.failover_history) > self.max_history:
            self.failover_history.pop(0)

        self.last_failover_time = time.time()

        print(f"[Failover] Triggered: {reason}, failed node: {failed_node}")

    def get_failover_history(self) -> List[Dict[str, Any]]:
        """Get failover history"""
        return self.failover_history


# Global instance
_cluster_manager = None


def get_cluster() -> ClusterManager:
    """Get global cluster manager"""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


if __name__ == "__main__":
    # Test cluster management
    print("=== Cluster Management Test ===")

    cluster = ClusterManager(node_id="node-001")
    cluster.start()

    # Simulate adding nodes
    print("\n1. Registering additional nodes...")
    cluster.register_node({
        'node_id': 'node-002',
        'ip_address': '192.168.1.102',
        'port': 5000,
        'services': ['simulation', 'control']
    })
    cluster.register_node({
        'node_id': 'node-003',
        'ip_address': '192.168.1.103',
        'port': 5000,
        'services': ['api', 'monitoring']
    })

    # Wait for election
    print("\n2. Waiting for leader election...")
    time.sleep(4)

    # Get cluster status
    print("\n3. Cluster Status:")
    status = cluster.get_cluster_status()
    print(f"   Nodes: {status['node_count']}")
    print(f"   Healthy: {status['healthy_count']}")
    print(f"   Leader: {status['leader']['node_id'] if status['leader'] else 'None'}")
    print(f"   This node role: {status['this_node']['role']}")

    # Test load balancing
    print("\n4. Load Balancing Test:")
    for i in range(5):
        node = cluster.select_node_for_request()
        if node:
            print(f"   Request {i+1} -> {node.node_id}")

    # Test state replication
    print("\n5. State Replication Test:")
    cluster.replicate_state('water_level', 4.5)
    cluster.replicate_state('flow_rate', 85.0)
    print(f"   water_level: {cluster.get_state('water_level')}")
    print(f"   flow_rate: {cluster.get_state('flow_rate')}")

    # Node metrics
    print("\n6. Node Metrics:")
    cluster.update_node_metrics(cpu=45.0, memory=60.0, disk=30.0, connections=15)
    metrics = cluster.get_node_metrics()
    print(f"   Load: {metrics['load']:.2f}")
    print(f"   CPU: {metrics['cpu_usage']}%")
    print(f"   Memory: {metrics['memory_usage']}%")

    cluster.stop()
    print("\nCluster management test completed!")
