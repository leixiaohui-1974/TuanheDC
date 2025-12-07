"""
TAOS V3.9 - Blockchain-based Data Audit & Traceability Module
区块链审计与数据追溯模块

Features:
- Blockchain-like immutable audit trail
- Hash chain for data integrity verification
- Comprehensive audit logging
- Data versioning and change tracking
- Compliance reporting
- Tamper detection and alerts
- Merkle tree for efficient verification
"""

import json
import os
import time
import hashlib
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import base64


class AuditEventType(Enum):
    """Audit event types"""
    # Data operations
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_READ = "data_read"

    # System operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_CONFIG_CHANGE = "system_config_change"

    # Control operations
    CONTROL_MODE_CHANGE = "control_mode_change"
    CONTROL_SETPOINT_CHANGE = "control_setpoint_change"
    CONTROL_COMMAND = "control_command"

    # Safety operations
    SAFETY_INTERLOCK = "safety_interlock"
    SAFETY_ALARM = "safety_alarm"
    SAFETY_OVERRIDE = "safety_override"

    # User operations
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_PERMISSION_CHANGE = "user_permission_change"

    # Sensor/Actuator operations
    SENSOR_CALIBRATION = "sensor_calibration"
    ACTUATOR_MAINTENANCE = "actuator_maintenance"

    # Scenario operations
    SCENARIO_INJECT = "scenario_inject"
    SCENARIO_CLEAR = "scenario_clear"


class AuditSeverity(Enum):
    """Audit event severity"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditBlock:
    """Blockchain audit block"""
    index: int
    timestamp: float
    event_type: str
    severity: str
    actor: str  # User or system component
    target: str  # Target resource/entity
    action: str  # Action description
    data: Dict[str, Any]  # Event data
    previous_hash: str
    hash: str = ""
    nonce: int = 0
    merkle_root: str = ""

    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "actor": self.actor,
            "target": self.target,
            "action": self.action,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "actor": self.actor,
            "target": self.target,
            "action": self.action,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "nonce": self.nonce,
            "merkle_root": self.merkle_root
        }


@dataclass
class DataVersion:
    """Data version tracking"""
    version_id: str
    entity_type: str
    entity_id: str
    version_number: int
    data: Dict[str, Any]
    changes: Dict[str, Any]  # What changed from previous version
    created_at: float
    created_by: str
    block_index: int  # Reference to audit block


class MerkleTree:
    """Merkle tree for efficient verification"""

    def __init__(self, leaves: List[str] = None):
        self.leaves = leaves or []
        self.tree = []
        if self.leaves:
            self._build_tree()

    def _hash(self, data: str) -> str:
        """Hash function"""
        return hashlib.sha256(data.encode()).hexdigest()

    def _build_tree(self):
        """Build the Merkle tree"""
        if not self.leaves:
            self.tree = []
            return

        # Hash all leaves
        current_level = [self._hash(leaf) for leaf in self.leaves]
        self.tree = [current_level]

        # Build tree levels
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self._hash(left + right))
            current_level = next_level
            self.tree.append(current_level)

    def get_root(self) -> str:
        """Get the Merkle root"""
        if not self.tree:
            return ""
        return self.tree[-1][0] if self.tree[-1] else ""

    def add_leaf(self, leaf: str):
        """Add a leaf and rebuild tree"""
        self.leaves.append(leaf)
        self._build_tree()

    def verify_leaf(self, leaf: str, index: int, proof: List[Tuple[str, str]]) -> bool:
        """Verify a leaf with proof"""
        current = self._hash(leaf)

        for sibling_hash, direction in proof:
            if direction == "left":
                current = self._hash(sibling_hash + current)
            else:
                current = self._hash(current + sibling_hash)

        return current == self.get_root()

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """Get proof for a leaf at index"""
        if index >= len(self.leaves):
            return []

        proof = []
        current_index = index

        for level in self.tree[:-1]:  # Exclude root level
            if current_index % 2 == 0:
                # Sibling is on the right
                sibling_index = current_index + 1
                if sibling_index < len(level):
                    proof.append((level[sibling_index], "right"))
            else:
                # Sibling is on the left
                sibling_index = current_index - 1
                proof.append((level[sibling_index], "left"))

            current_index //= 2

        return proof


class AuditChain:
    """Blockchain-based audit chain"""

    def __init__(self, data_dir: str = None, difficulty: int = 2):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'audit_data')
        self.difficulty = difficulty  # Proof of work difficulty
        self.chain: List[AuditBlock] = []
        self.pending_events: List[Dict[str, Any]] = []
        self.merkle_tree = MerkleTree()
        self._lock = threading.RLock()

        # Initialize chain
        self._init_chain()

    def _init_chain(self):
        """Initialize chain with genesis block"""
        if not self.chain:
            genesis = AuditBlock(
                index=0,
                timestamp=time.time(),
                event_type="genesis",
                severity=AuditSeverity.INFO.value,
                actor="system",
                target="audit_chain",
                action="Chain initialized",
                data={"version": "3.9.0", "system": "TAOS"},
                previous_hash="0"
            )
            genesis.hash = genesis.calculate_hash()
            self.chain.append(genesis)
            self.merkle_tree.add_leaf(genesis.hash)

    def _mine_block(self, block: AuditBlock) -> AuditBlock:
        """Mine block with proof of work"""
        prefix = "0" * self.difficulty
        block.nonce = 0

        while True:
            block.hash = block.calculate_hash()
            if block.hash.startswith(prefix):
                break
            block.nonce += 1

        return block

    def add_event(self, event_type: AuditEventType, actor: str, target: str,
                 action: str, data: Dict[str, Any] = None,
                 severity: AuditSeverity = AuditSeverity.INFO) -> AuditBlock:
        """Add an audit event to the chain"""
        with self._lock:
            previous_block = self.chain[-1]

            new_block = AuditBlock(
                index=len(self.chain),
                timestamp=time.time(),
                event_type=event_type.value,
                severity=severity.value,
                actor=actor,
                target=target,
                action=action,
                data=data or {},
                previous_hash=previous_block.hash
            )

            # Mine block
            new_block = self._mine_block(new_block)

            # Update Merkle tree
            self.merkle_tree.add_leaf(new_block.hash)
            new_block.merkle_root = self.merkle_tree.get_root()

            # Add to chain
            self.chain.append(new_block)

            return new_block

    def verify_chain(self) -> Tuple[bool, List[int]]:
        """
        Verify the integrity of the entire chain

        Returns:
            Tuple of (is_valid, list of invalid block indices)
        """
        invalid_blocks = []

        with self._lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]

                # Verify previous hash
                if current.previous_hash != previous.hash:
                    invalid_blocks.append(i)
                    continue

                # Verify current hash
                if current.hash != current.calculate_hash():
                    invalid_blocks.append(i)
                    continue

                # Verify proof of work
                if not current.hash.startswith("0" * self.difficulty):
                    invalid_blocks.append(i)

        return len(invalid_blocks) == 0, invalid_blocks

    def get_block(self, index: int) -> Optional[AuditBlock]:
        """Get block by index"""
        with self._lock:
            if 0 <= index < len(self.chain):
                return self.chain[index]
            return None

    def get_blocks_range(self, start: int, end: int) -> List[AuditBlock]:
        """Get blocks in range"""
        with self._lock:
            return self.chain[start:end]

    def query_by_actor(self, actor: str, limit: int = 100) -> List[AuditBlock]:
        """Query blocks by actor"""
        with self._lock:
            return [b for b in reversed(self.chain) if b.actor == actor][:limit]

    def query_by_target(self, target: str, limit: int = 100) -> List[AuditBlock]:
        """Query blocks by target"""
        with self._lock:
            return [b for b in reversed(self.chain) if b.target == target][:limit]

    def query_by_event_type(self, event_type: AuditEventType,
                           limit: int = 100) -> List[AuditBlock]:
        """Query blocks by event type"""
        with self._lock:
            return [b for b in reversed(self.chain)
                   if b.event_type == event_type.value][:limit]

    def query_by_time_range(self, start_time: float, end_time: float) -> List[AuditBlock]:
        """Query blocks by time range"""
        with self._lock:
            return [b for b in self.chain
                   if start_time <= b.timestamp <= end_time]

    def query_by_severity(self, severity: AuditSeverity,
                         limit: int = 100) -> List[AuditBlock]:
        """Query blocks by severity"""
        with self._lock:
            return [b for b in reversed(self.chain)
                   if b.severity == severity.value][:limit]

    def get_chain_stats(self) -> Dict[str, Any]:
        """Get chain statistics"""
        with self._lock:
            event_counts = {}
            severity_counts = {}
            actor_counts = {}

            for block in self.chain[1:]:  # Skip genesis
                event_counts[block.event_type] = event_counts.get(block.event_type, 0) + 1
                severity_counts[block.severity] = severity_counts.get(block.severity, 0) + 1
                actor_counts[block.actor] = actor_counts.get(block.actor, 0) + 1

            return {
                "total_blocks": len(self.chain),
                "first_block_time": datetime.fromtimestamp(self.chain[0].timestamp).isoformat(),
                "last_block_time": datetime.fromtimestamp(self.chain[-1].timestamp).isoformat(),
                "merkle_root": self.merkle_tree.get_root(),
                "event_counts": event_counts,
                "severity_counts": severity_counts,
                "top_actors": dict(sorted(actor_counts.items(),
                                         key=lambda x: x[1], reverse=True)[:10]),
                "difficulty": self.difficulty
            }

    def export_chain(self, filepath: str = None) -> str:
        """Export chain to JSON"""
        with self._lock:
            data = {
                "exported_at": datetime.now().isoformat(),
                "total_blocks": len(self.chain),
                "merkle_root": self.merkle_tree.get_root(),
                "blocks": [b.to_dict() for b in self.chain]
            }

            json_str = json.dumps(data, indent=2)

            if filepath:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(json_str)

            return json_str


class DataVersionManager:
    """Manage data versions for traceability"""

    def __init__(self, audit_chain: AuditChain):
        self.audit_chain = audit_chain
        self.versions: Dict[str, List[DataVersion]] = {}  # entity_key -> versions
        self._lock = threading.RLock()

    def _get_entity_key(self, entity_type: str, entity_id: str) -> str:
        """Get unique entity key"""
        return f"{entity_type}:{entity_id}"

    def _generate_version_id(self, entity_key: str, version_number: int) -> str:
        """Generate version ID"""
        data = f"{entity_key}:{version_number}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_changes(self, old_data: Dict[str, Any],
                          new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between versions"""
        changes = {
            "added": {},
            "removed": {},
            "modified": {}
        }

        old_keys = set(old_data.keys())
        new_keys = set(new_data.keys())

        # Added keys
        for key in new_keys - old_keys:
            changes["added"][key] = new_data[key]

        # Removed keys
        for key in old_keys - new_keys:
            changes["removed"][key] = old_data[key]

        # Modified keys
        for key in old_keys & new_keys:
            if old_data[key] != new_data[key]:
                changes["modified"][key] = {
                    "old": old_data[key],
                    "new": new_data[key]
                }

        return changes

    def create_version(self, entity_type: str, entity_id: str,
                      data: Dict[str, Any], actor: str) -> DataVersion:
        """Create a new version of an entity"""
        with self._lock:
            entity_key = self._get_entity_key(entity_type, entity_id)

            # Get existing versions
            existing_versions = self.versions.get(entity_key, [])
            version_number = len(existing_versions) + 1

            # Calculate changes
            if existing_versions:
                old_data = existing_versions[-1].data
                changes = self._calculate_changes(old_data, data)
                event_type = AuditEventType.DATA_UPDATE
            else:
                changes = {"added": data}
                event_type = AuditEventType.DATA_CREATE

            # Add audit event
            block = self.audit_chain.add_event(
                event_type=event_type,
                actor=actor,
                target=entity_key,
                action=f"Version {version_number} created",
                data={"changes_summary": {
                    "added": len(changes.get("added", {})),
                    "removed": len(changes.get("removed", {})),
                    "modified": len(changes.get("modified", {}))
                }}
            )

            # Create version
            version = DataVersion(
                version_id=self._generate_version_id(entity_key, version_number),
                entity_type=entity_type,
                entity_id=entity_id,
                version_number=version_number,
                data=data,
                changes=changes,
                created_at=time.time(),
                created_by=actor,
                block_index=block.index
            )

            # Store version
            if entity_key not in self.versions:
                self.versions[entity_key] = []
            self.versions[entity_key].append(version)

            return version

    def get_version(self, entity_type: str, entity_id: str,
                   version_number: int = None) -> Optional[DataVersion]:
        """Get a specific version (latest if version_number is None)"""
        with self._lock:
            entity_key = self._get_entity_key(entity_type, entity_id)
            versions = self.versions.get(entity_key, [])

            if not versions:
                return None

            if version_number is None:
                return versions[-1]

            for v in versions:
                if v.version_number == version_number:
                    return v
            return None

    def get_version_history(self, entity_type: str,
                           entity_id: str) -> List[Dict[str, Any]]:
        """Get version history for an entity"""
        with self._lock:
            entity_key = self._get_entity_key(entity_type, entity_id)
            versions = self.versions.get(entity_key, [])

            return [{
                "version_id": v.version_id,
                "version_number": v.version_number,
                "created_at": datetime.fromtimestamp(v.created_at).isoformat(),
                "created_by": v.created_by,
                "changes_summary": {
                    "added": len(v.changes.get("added", {})),
                    "removed": len(v.changes.get("removed", {})),
                    "modified": len(v.changes.get("modified", {}))
                },
                "block_index": v.block_index
            } for v in versions]

    def compare_versions(self, entity_type: str, entity_id: str,
                        version1: int, version2: int) -> Dict[str, Any]:
        """Compare two versions"""
        with self._lock:
            v1 = self.get_version(entity_type, entity_id, version1)
            v2 = self.get_version(entity_type, entity_id, version2)

            if not v1 or not v2:
                return {"error": "Version not found"}

            changes = self._calculate_changes(v1.data, v2.data)

            return {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "version1": version1,
                "version2": version2,
                "changes": changes
            }

    def rollback_to_version(self, entity_type: str, entity_id: str,
                           version_number: int, actor: str) -> Optional[DataVersion]:
        """Rollback to a specific version (creates new version with old data)"""
        target_version = self.get_version(entity_type, entity_id, version_number)
        if not target_version:
            return None

        # Create new version with the old data
        return self.create_version(
            entity_type=entity_type,
            entity_id=entity_id,
            data=target_version.data,
            actor=actor
        )


class ComplianceReporter:
    """Generate compliance reports"""

    def __init__(self, audit_chain: AuditChain):
        self.audit_chain = audit_chain

    def generate_access_report(self, start_time: float,
                              end_time: float) -> Dict[str, Any]:
        """Generate access audit report"""
        blocks = self.audit_chain.query_by_time_range(start_time, end_time)

        # Filter access events
        access_events = [
            b for b in blocks
            if b.event_type in [
                AuditEventType.USER_LOGIN.value,
                AuditEventType.USER_LOGOUT.value,
                AuditEventType.DATA_READ.value
            ]
        ]

        # Analyze by actor
        actor_summary = {}
        for block in access_events:
            if block.actor not in actor_summary:
                actor_summary[block.actor] = {
                    "logins": 0,
                    "logouts": 0,
                    "data_reads": 0,
                    "last_activity": None
                }

            if block.event_type == AuditEventType.USER_LOGIN.value:
                actor_summary[block.actor]["logins"] += 1
            elif block.event_type == AuditEventType.USER_LOGOUT.value:
                actor_summary[block.actor]["logouts"] += 1
            elif block.event_type == AuditEventType.DATA_READ.value:
                actor_summary[block.actor]["data_reads"] += 1

            actor_summary[block.actor]["last_activity"] = datetime.fromtimestamp(
                block.timestamp
            ).isoformat()

        return {
            "report_type": "access_audit",
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "total_events": len(access_events),
            "actor_summary": actor_summary,
            "generated_at": datetime.now().isoformat()
        }

    def generate_change_report(self, start_time: float,
                              end_time: float) -> Dict[str, Any]:
        """Generate data change audit report"""
        blocks = self.audit_chain.query_by_time_range(start_time, end_time)

        # Filter change events
        change_events = [
            b for b in blocks
            if b.event_type in [
                AuditEventType.DATA_CREATE.value,
                AuditEventType.DATA_UPDATE.value,
                AuditEventType.DATA_DELETE.value
            ]
        ]

        # Analyze by target
        target_summary = {}
        for block in change_events:
            if block.target not in target_summary:
                target_summary[block.target] = {
                    "creates": 0,
                    "updates": 0,
                    "deletes": 0,
                    "actors": set()
                }

            if block.event_type == AuditEventType.DATA_CREATE.value:
                target_summary[block.target]["creates"] += 1
            elif block.event_type == AuditEventType.DATA_UPDATE.value:
                target_summary[block.target]["updates"] += 1
            elif block.event_type == AuditEventType.DATA_DELETE.value:
                target_summary[block.target]["deletes"] += 1

            target_summary[block.target]["actors"].add(block.actor)

        # Convert sets to lists for JSON
        for target in target_summary:
            target_summary[target]["actors"] = list(target_summary[target]["actors"])

        return {
            "report_type": "change_audit",
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "total_changes": len(change_events),
            "target_summary": target_summary,
            "generated_at": datetime.now().isoformat()
        }

    def generate_security_report(self, start_time: float,
                                end_time: float) -> Dict[str, Any]:
        """Generate security audit report"""
        blocks = self.audit_chain.query_by_time_range(start_time, end_time)

        # Filter security events
        security_events = [
            b for b in blocks
            if b.event_type in [
                AuditEventType.SAFETY_INTERLOCK.value,
                AuditEventType.SAFETY_ALARM.value,
                AuditEventType.SAFETY_OVERRIDE.value,
                AuditEventType.USER_PERMISSION_CHANGE.value
            ] or b.severity in [AuditSeverity.WARNING.value, AuditSeverity.CRITICAL.value]
        ]

        # Group by severity
        severity_groups = {
            AuditSeverity.INFO.value: [],
            AuditSeverity.WARNING.value: [],
            AuditSeverity.CRITICAL.value: []
        }

        for block in security_events:
            severity_groups[block.severity].append({
                "timestamp": datetime.fromtimestamp(block.timestamp).isoformat(),
                "event_type": block.event_type,
                "actor": block.actor,
                "target": block.target,
                "action": block.action
            })

        return {
            "report_type": "security_audit",
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "total_events": len(security_events),
            "critical_count": len(severity_groups[AuditSeverity.CRITICAL.value]),
            "warning_count": len(severity_groups[AuditSeverity.WARNING.value]),
            "events_by_severity": severity_groups,
            "generated_at": datetime.now().isoformat()
        }

    def generate_control_report(self, start_time: float,
                               end_time: float) -> Dict[str, Any]:
        """Generate control operations audit report"""
        blocks = self.audit_chain.query_by_time_range(start_time, end_time)

        # Filter control events
        control_events = [
            b for b in blocks
            if b.event_type in [
                AuditEventType.CONTROL_MODE_CHANGE.value,
                AuditEventType.CONTROL_SETPOINT_CHANGE.value,
                AuditEventType.CONTROL_COMMAND.value,
                AuditEventType.SCENARIO_INJECT.value,
                AuditEventType.SCENARIO_CLEAR.value
            ]
        ]

        # Timeline of control changes
        timeline = []
        for block in control_events:
            timeline.append({
                "timestamp": datetime.fromtimestamp(block.timestamp).isoformat(),
                "event_type": block.event_type,
                "actor": block.actor,
                "action": block.action,
                "data": block.data
            })

        return {
            "report_type": "control_audit",
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "total_control_operations": len(control_events),
            "timeline": timeline,
            "generated_at": datetime.now().isoformat()
        }

    def generate_compliance_summary(self, start_time: float,
                                   end_time: float) -> Dict[str, Any]:
        """Generate comprehensive compliance summary"""
        is_valid, invalid_blocks = self.audit_chain.verify_chain()
        stats = self.audit_chain.get_chain_stats()

        blocks = self.audit_chain.query_by_time_range(start_time, end_time)

        return {
            "report_type": "compliance_summary",
            "period": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat()
            },
            "chain_integrity": {
                "is_valid": is_valid,
                "invalid_blocks": invalid_blocks,
                "total_blocks": stats["total_blocks"],
                "merkle_root": stats["merkle_root"]
            },
            "period_statistics": {
                "total_events": len(blocks),
                "event_types": self._count_by_field(blocks, "event_type"),
                "severity_distribution": self._count_by_field(blocks, "severity"),
                "active_actors": len(set(b.actor for b in blocks))
            },
            "generated_at": datetime.now().isoformat()
        }

    def _count_by_field(self, blocks: List[AuditBlock],
                       field: str) -> Dict[str, int]:
        """Count blocks by field value"""
        counts = {}
        for block in blocks:
            value = getattr(block, field)
            counts[value] = counts.get(value, 0) + 1
        return counts


class BlockchainAuditManager:
    """Main blockchain audit manager"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'audit_data')
        self.audit_chain = AuditChain(self.data_dir)
        self.version_manager = DataVersionManager(self.audit_chain)
        self.compliance_reporter = ComplianceReporter(self.audit_chain)
        self._lock = threading.RLock()

    # Event logging methods
    def log_data_create(self, actor: str, target: str, data: Dict[str, Any] = None):
        """Log data creation event"""
        return self.audit_chain.add_event(
            AuditEventType.DATA_CREATE, actor, target,
            f"Created {target}", data
        )

    def log_data_update(self, actor: str, target: str, data: Dict[str, Any] = None):
        """Log data update event"""
        return self.audit_chain.add_event(
            AuditEventType.DATA_UPDATE, actor, target,
            f"Updated {target}", data
        )

    def log_data_delete(self, actor: str, target: str, data: Dict[str, Any] = None):
        """Log data deletion event"""
        return self.audit_chain.add_event(
            AuditEventType.DATA_DELETE, actor, target,
            f"Deleted {target}", data,
            AuditSeverity.WARNING
        )

    def log_data_read(self, actor: str, target: str, data: Dict[str, Any] = None):
        """Log data read event"""
        return self.audit_chain.add_event(
            AuditEventType.DATA_READ, actor, target,
            f"Read {target}", data
        )

    def log_control_mode_change(self, actor: str, old_mode: str,
                               new_mode: str, data: Dict[str, Any] = None):
        """Log control mode change"""
        return self.audit_chain.add_event(
            AuditEventType.CONTROL_MODE_CHANGE, actor, "control_system",
            f"Mode changed: {old_mode} -> {new_mode}",
            {"old_mode": old_mode, "new_mode": new_mode, **(data or {})},
            AuditSeverity.WARNING
        )

    def log_control_setpoint(self, actor: str, parameter: str,
                            old_value: Any, new_value: Any):
        """Log control setpoint change"""
        return self.audit_chain.add_event(
            AuditEventType.CONTROL_SETPOINT_CHANGE, actor, parameter,
            f"Setpoint changed: {old_value} -> {new_value}",
            {"parameter": parameter, "old_value": old_value, "new_value": new_value}
        )

    def log_safety_event(self, actor: str, event_type: str,
                        target: str, data: Dict[str, Any] = None,
                        severity: AuditSeverity = AuditSeverity.WARNING):
        """Log safety event"""
        type_map = {
            "interlock": AuditEventType.SAFETY_INTERLOCK,
            "alarm": AuditEventType.SAFETY_ALARM,
            "override": AuditEventType.SAFETY_OVERRIDE
        }
        audit_type = type_map.get(event_type, AuditEventType.SAFETY_ALARM)

        return self.audit_chain.add_event(
            audit_type, actor, target,
            f"Safety {event_type}: {target}", data, severity
        )

    def log_user_login(self, user_id: str, ip_address: str = None,
                      data: Dict[str, Any] = None):
        """Log user login"""
        return self.audit_chain.add_event(
            AuditEventType.USER_LOGIN, user_id, "auth_system",
            f"User logged in from {ip_address or 'unknown'}",
            {"ip_address": ip_address, **(data or {})}
        )

    def log_user_logout(self, user_id: str, data: Dict[str, Any] = None):
        """Log user logout"""
        return self.audit_chain.add_event(
            AuditEventType.USER_LOGOUT, user_id, "auth_system",
            "User logged out", data
        )

    def log_scenario_inject(self, actor: str, scenario_type: str,
                           parameters: Dict[str, Any] = None):
        """Log scenario injection"""
        return self.audit_chain.add_event(
            AuditEventType.SCENARIO_INJECT, actor, "simulation",
            f"Scenario injected: {scenario_type}",
            {"scenario_type": scenario_type, "parameters": parameters},
            AuditSeverity.WARNING
        )

    def log_system_event(self, event_type: str, data: Dict[str, Any] = None):
        """Log system event"""
        type_map = {
            "start": AuditEventType.SYSTEM_START,
            "stop": AuditEventType.SYSTEM_STOP,
            "config_change": AuditEventType.SYSTEM_CONFIG_CHANGE
        }
        audit_type = type_map.get(event_type, AuditEventType.SYSTEM_CONFIG_CHANGE)

        return self.audit_chain.add_event(
            audit_type, "system", "taos",
            f"System {event_type}", data
        )

    # Version management
    def create_data_version(self, entity_type: str, entity_id: str,
                           data: Dict[str, Any], actor: str) -> DataVersion:
        """Create a new data version"""
        return self.version_manager.create_version(
            entity_type, entity_id, data, actor
        )

    def get_data_version(self, entity_type: str, entity_id: str,
                        version_number: int = None) -> Optional[DataVersion]:
        """Get a data version"""
        return self.version_manager.get_version(
            entity_type, entity_id, version_number
        )

    def get_version_history(self, entity_type: str,
                           entity_id: str) -> List[Dict[str, Any]]:
        """Get version history"""
        return self.version_manager.get_version_history(entity_type, entity_id)

    def compare_versions(self, entity_type: str, entity_id: str,
                        v1: int, v2: int) -> Dict[str, Any]:
        """Compare two versions"""
        return self.version_manager.compare_versions(entity_type, entity_id, v1, v2)

    # Compliance reports
    def generate_access_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate access report for last N hours"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.compliance_reporter.generate_access_report(start_time, end_time)

    def generate_change_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate change report for last N hours"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.compliance_reporter.generate_change_report(start_time, end_time)

    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for last N hours"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.compliance_reporter.generate_security_report(start_time, end_time)

    def generate_control_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate control report for last N hours"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.compliance_reporter.generate_control_report(start_time, end_time)

    def generate_compliance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate compliance summary for last N hours"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        return self.compliance_reporter.generate_compliance_summary(start_time, end_time)

    # Chain operations
    def verify_chain(self) -> Dict[str, Any]:
        """Verify chain integrity"""
        is_valid, invalid_blocks = self.audit_chain.verify_chain()
        return {
            "is_valid": is_valid,
            "invalid_blocks": invalid_blocks,
            "total_blocks": len(self.audit_chain.chain),
            "merkle_root": self.audit_chain.merkle_tree.get_root()
        }

    def get_chain_stats(self) -> Dict[str, Any]:
        """Get chain statistics"""
        return self.audit_chain.get_chain_stats()

    def query_events(self, actor: str = None, target: str = None,
                    event_type: str = None, severity: str = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit events"""
        if actor:
            blocks = self.audit_chain.query_by_actor(actor, limit)
        elif target:
            blocks = self.audit_chain.query_by_target(target, limit)
        elif event_type:
            try:
                et = AuditEventType(event_type)
                blocks = self.audit_chain.query_by_event_type(et, limit)
            except ValueError:
                blocks = []
        elif severity:
            try:
                sev = AuditSeverity(severity)
                blocks = self.audit_chain.query_by_severity(sev, limit)
            except ValueError:
                blocks = []
        else:
            blocks = self.audit_chain.get_blocks_range(
                max(0, len(self.audit_chain.chain) - limit),
                len(self.audit_chain.chain)
            )

        return [b.to_dict() for b in blocks]

    def export_chain(self, filepath: str = None) -> str:
        """Export audit chain"""
        return self.audit_chain.export_chain(filepath)

    def get_status(self) -> Dict[str, Any]:
        """Get audit system status"""
        verification = self.verify_chain()
        stats = self.get_chain_stats()

        return {
            "status": "healthy" if verification["is_valid"] else "integrity_warning",
            "chain_length": len(self.audit_chain.chain),
            "is_valid": verification["is_valid"],
            "merkle_root": verification["merkle_root"],
            "statistics": stats,
            "versioned_entities": len(self.version_manager.versions)
        }


# Global instance
_audit_manager: Optional[BlockchainAuditManager] = None


def get_audit_manager() -> BlockchainAuditManager:
    """Get global audit manager instance"""
    global _audit_manager
    if _audit_manager is None:
        _audit_manager = BlockchainAuditManager()
    return _audit_manager


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("TAOS V3.9 - Blockchain Audit Module Test")
    print("=" * 60)

    manager = get_audit_manager()

    # Test event logging
    print("\n1. Event Logging Tests:")
    print("-" * 40)

    # Log various events
    manager.log_system_event("start", {"version": "3.9.0"})
    manager.log_user_login("admin", "192.168.1.100")
    manager.log_data_create("admin", "sensor_config", {"sensor_id": "S001"})
    manager.log_data_update("admin", "sensor_config", {"calibration": 1.05})
    manager.log_control_mode_change("admin", "manual", "auto")
    manager.log_control_setpoint("admin", "water_level_target", 8.0, 8.5)
    manager.log_scenario_inject("admin", "hydraulic_jump", {"intensity": 0.7})
    manager.log_safety_event("system", "alarm", "water_level", {"level": 9.5})
    manager.log_user_logout("admin")

    print(f"  Total blocks in chain: {len(manager.audit_chain.chain)}")

    # Test chain verification
    print("\n2. Chain Verification:")
    print("-" * 40)

    verification = manager.verify_chain()
    print(f"  Chain valid: {verification['is_valid']}")
    print(f"  Total blocks: {verification['total_blocks']}")
    print(f"  Merkle root: {verification['merkle_root'][:32]}...")

    # Test version management
    print("\n3. Data Version Management:")
    print("-" * 40)

    # Create versions of a config entity
    config_v1 = {"name": "Config1", "value": 100, "enabled": True}
    config_v2 = {"name": "Config1", "value": 150, "enabled": True, "new_field": "added"}
    config_v3 = {"name": "Config1", "value": 200, "enabled": False, "new_field": "added"}

    manager.create_data_version("config", "CFG001", config_v1, "admin")
    manager.create_data_version("config", "CFG001", config_v2, "admin")
    manager.create_data_version("config", "CFG001", config_v3, "operator")

    history = manager.get_version_history("config", "CFG001")
    print(f"  Version history for config:CFG001:")
    for v in history:
        print(f"    v{v['version_number']}: {v['changes_summary']} by {v['created_by']}")

    # Compare versions
    comparison = manager.compare_versions("config", "CFG001", 1, 3)
    print(f"\n  Comparison v1 vs v3:")
    print(f"    Added: {list(comparison['changes']['added'].keys())}")
    print(f"    Modified: {list(comparison['changes']['modified'].keys())}")

    # Test event queries
    print("\n4. Event Query Tests:")
    print("-" * 40)

    admin_events = manager.query_events(actor="admin", limit=5)
    print(f"  Admin events: {len(admin_events)}")
    for event in admin_events[:3]:
        print(f"    - {event['event_type']}: {event['action']}")

    control_events = manager.query_events(event_type="control_mode_change")
    print(f"\n  Control mode change events: {len(control_events)}")

    # Test compliance reports
    print("\n5. Compliance Report Tests:")
    print("-" * 40)

    # Access report
    access_report = manager.generate_access_report(24)
    print(f"  Access Report:")
    print(f"    Period: {access_report['period']['start']} to {access_report['period']['end']}")
    print(f"    Total events: {access_report['total_events']}")

    # Change report
    change_report = manager.generate_change_report(24)
    print(f"\n  Change Report:")
    print(f"    Total changes: {change_report['total_changes']}")

    # Security report
    security_report = manager.generate_security_report(24)
    print(f"\n  Security Report:")
    print(f"    Total events: {security_report['total_events']}")
    print(f"    Critical: {security_report['critical_count']}")
    print(f"    Warning: {security_report['warning_count']}")

    # Control report
    control_report = manager.generate_control_report(24)
    print(f"\n  Control Report:")
    print(f"    Total operations: {control_report['total_control_operations']}")

    # Compliance summary
    summary = manager.generate_compliance_summary(24)
    print(f"\n  Compliance Summary:")
    print(f"    Chain integrity: {'Valid' if summary['chain_integrity']['is_valid'] else 'Invalid'}")
    print(f"    Total blocks: {summary['chain_integrity']['total_blocks']}")
    print(f"    Active actors: {summary['period_statistics']['active_actors']}")

    # Test chain statistics
    print("\n6. Chain Statistics:")
    print("-" * 40)

    stats = manager.get_chain_stats()
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  First block: {stats['first_block_time']}")
    print(f"  Last block: {stats['last_block_time']}")
    print(f"  Event types: {stats['event_counts']}")
    print(f"  Top actors: {stats['top_actors']}")

    # Test status
    print("\n7. System Status:")
    print("-" * 40)

    status = manager.get_status()
    print(f"  Status: {status['status']}")
    print(f"  Chain length: {status['chain_length']}")
    print(f"  Valid: {status['is_valid']}")
    print(f"  Versioned entities: {status['versioned_entities']}")

    print("\n" + "=" * 60)
    print("All blockchain audit tests completed successfully!")
    print("=" * 60)
