#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.6 - Backup and Restore System
团河渡槽自主运行系统 - 备份恢复模块

Features:
- Full system backup
- Incremental backup
- Database backup
- Configuration backup
- Automated backup scheduling
- Restore with validation
- Backup encryption (optional)
"""

import os
import json
import sqlite3
import shutil
import gzip
import tarfile
import hashlib
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pathlib import Path


class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DATABASE = "database"
    CONFIG = "config"
    LOGS = "logs"


class BackupStatus(Enum):
    """Backup status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupInfo:
    """Backup metadata"""
    backup_id: str
    backup_type: BackupType
    created_at: datetime
    size_bytes: int
    file_path: str
    checksum: str
    status: BackupStatus
    files_count: int = 0
    duration_seconds: float = 0
    description: str = ""
    parent_backup: Optional[str] = None  # For incremental backups
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type.value,
            'created_at': self.created_at.isoformat(),
            'size_bytes': self.size_bytes,
            'size_mb': round(self.size_bytes / (1024 * 1024), 2),
            'file_path': self.file_path,
            'checksum': self.checksum,
            'status': self.status.value,
            'files_count': self.files_count,
            'duration_seconds': self.duration_seconds,
            'description': self.description,
            'parent_backup': self.parent_backup,
            'metadata': self.metadata
        }


class BackupManager:
    """
    Backup and restore manager for TAOS
    """

    def __init__(self, backup_dir: str = None, data_dir: str = None):
        self.backup_dir = Path(backup_dir) if backup_dir else Path(__file__).parent.parent / "backups"
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.config_dir = Path(__file__).parent.parent / "config"
        self.logs_dir = Path(__file__).parent / "logs"

        # Create directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup catalog
        self.catalog_file = self.backup_dir / "catalog.json"
        self.backups: Dict[str, BackupInfo] = {}
        self._load_catalog()

        # Settings
        self.compression_level = 9
        self.max_backups = 30  # Maximum backups to keep
        self.lock = threading.Lock()

    def _load_catalog(self):
        """Load backup catalog"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    data = json.load(f)
                for backup_id, info in data.items():
                    self.backups[backup_id] = BackupInfo(
                        backup_id=info['backup_id'],
                        backup_type=BackupType(info['backup_type']),
                        created_at=datetime.fromisoformat(info['created_at']),
                        size_bytes=info['size_bytes'],
                        file_path=info['file_path'],
                        checksum=info['checksum'],
                        status=BackupStatus(info['status']),
                        files_count=info.get('files_count', 0),
                        duration_seconds=info.get('duration_seconds', 0),
                        description=info.get('description', ''),
                        parent_backup=info.get('parent_backup'),
                        metadata=info.get('metadata', {})
                    )
            except Exception as e:
                print(f"Failed to load backup catalog: {e}")

    def _save_catalog(self):
        """Save backup catalog"""
        with open(self.catalog_file, 'w') as f:
            data = {bid: info.to_dict() for bid, info in self.backups.items()}
            json.dump(data, f, indent=2)

    def create_backup(self, backup_type: BackupType = BackupType.FULL,
                     description: str = "") -> BackupInfo:
        """Create a backup"""
        with self.lock:
            start_time = time.time()
            backup_id = f"{backup_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_file = self.backup_dir / f"{backup_id}.tar.gz"

            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.now(),
                size_bytes=0,
                file_path=str(backup_file),
                checksum="",
                status=BackupStatus.IN_PROGRESS,
                description=description
            )

            try:
                if backup_type == BackupType.FULL:
                    self._create_full_backup(backup_file, backup_info)
                elif backup_type == BackupType.DATABASE:
                    self._create_database_backup(backup_file, backup_info)
                elif backup_type == BackupType.CONFIG:
                    self._create_config_backup(backup_file, backup_info)
                elif backup_type == BackupType.LOGS:
                    self._create_logs_backup(backup_file, backup_info)
                elif backup_type == BackupType.INCREMENTAL:
                    self._create_incremental_backup(backup_file, backup_info)

                # Calculate checksum
                backup_info.checksum = self._calculate_checksum(backup_file)
                backup_info.size_bytes = backup_file.stat().st_size
                backup_info.duration_seconds = time.time() - start_time
                backup_info.status = BackupStatus.COMPLETED

                # Add to catalog
                self.backups[backup_id] = backup_info
                self._save_catalog()

                # Cleanup old backups
                self._cleanup_old_backups()

                return backup_info

            except Exception as e:
                backup_info.status = BackupStatus.FAILED
                backup_info.metadata['error'] = str(e)
                self.backups[backup_id] = backup_info
                self._save_catalog()
                raise

    def _create_full_backup(self, backup_file: Path, info: BackupInfo):
        """Create full backup of all data"""
        files_count = 0

        with tarfile.open(backup_file, "w:gz", compresslevel=self.compression_level) as tar:
            # Backup databases
            if self.data_dir.exists():
                for db_file in self.data_dir.glob("*.db"):
                    tar.add(db_file, arcname=f"data/{db_file.name}")
                    files_count += 1

            # Backup config
            if self.config_dir.exists():
                for config_file in self.config_dir.glob("*"):
                    if config_file.is_file() and not config_file.name.startswith('.'):
                        tar.add(config_file, arcname=f"config/{config_file.name}")
                        files_count += 1

            # Backup metadata
            metadata = {
                'backup_type': 'full',
                'created_at': datetime.now().isoformat(),
                'taos_version': '3.6',
                'files_count': files_count
            }
            metadata_file = self.backup_dir / f"{info.backup_id}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            tar.add(metadata_file, arcname="metadata.json")
            metadata_file.unlink()

        info.files_count = files_count

    def _create_database_backup(self, backup_file: Path, info: BackupInfo):
        """Create database-only backup"""
        files_count = 0

        with tarfile.open(backup_file, "w:gz", compresslevel=self.compression_level) as tar:
            if self.data_dir.exists():
                for db_file in self.data_dir.glob("*.db"):
                    # Create a consistent snapshot
                    snapshot = db_file.with_suffix('.db.snapshot')
                    self._snapshot_database(db_file, snapshot)
                    tar.add(snapshot, arcname=f"data/{db_file.name}")
                    snapshot.unlink()
                    files_count += 1

        info.files_count = files_count

    def _create_config_backup(self, backup_file: Path, info: BackupInfo):
        """Create config-only backup"""
        files_count = 0

        with tarfile.open(backup_file, "w:gz", compresslevel=self.compression_level) as tar:
            if self.config_dir.exists():
                for config_file in self.config_dir.glob("*"):
                    if config_file.is_file() and not config_file.name.startswith('.'):
                        tar.add(config_file, arcname=f"config/{config_file.name}")
                        files_count += 1

        info.files_count = files_count

    def _create_logs_backup(self, backup_file: Path, info: BackupInfo):
        """Create logs backup"""
        files_count = 0

        with tarfile.open(backup_file, "w:gz", compresslevel=self.compression_level) as tar:
            if self.logs_dir.exists():
                for log_file in self.logs_dir.glob("*"):
                    if log_file.is_file() and not log_file.name.startswith('.'):
                        tar.add(log_file, arcname=f"logs/{log_file.name}")
                        files_count += 1

        info.files_count = files_count

    def _create_incremental_backup(self, backup_file: Path, info: BackupInfo):
        """Create incremental backup since last full backup"""
        # Find last full backup
        last_full = None
        for bid, binfo in sorted(self.backups.items(), key=lambda x: x[1].created_at, reverse=True):
            if binfo.backup_type == BackupType.FULL and binfo.status == BackupStatus.COMPLETED:
                last_full = binfo
                break

        if not last_full:
            # No full backup, create one instead
            return self._create_full_backup(backup_file, info)

        info.parent_backup = last_full.backup_id
        cutoff_time = last_full.created_at.timestamp()
        files_count = 0

        with tarfile.open(backup_file, "w:gz", compresslevel=self.compression_level) as tar:
            # Only backup files modified since last full backup
            for directory in [self.data_dir, self.config_dir]:
                if directory.exists():
                    for file_path in directory.glob("**/*"):
                        if file_path.is_file() and file_path.stat().st_mtime > cutoff_time:
                            rel_path = file_path.relative_to(file_path.parent.parent)
                            tar.add(file_path, arcname=str(rel_path))
                            files_count += 1

        info.files_count = files_count

    def _snapshot_database(self, source: Path, dest: Path):
        """Create consistent database snapshot"""
        # Use SQLite backup API for consistency
        source_conn = sqlite3.connect(str(source))
        dest_conn = sqlite3.connect(str(dest))

        source_conn.backup(dest_conn)

        source_conn.close()
        dest_conn.close()

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups"""
        if len(self.backups) <= self.max_backups:
            return

        # Sort by date, keep newest
        sorted_backups = sorted(
            self.backups.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )

        to_delete = sorted_backups[self.max_backups:]

        for backup_id, info in to_delete:
            try:
                if Path(info.file_path).exists():
                    Path(info.file_path).unlink()
                del self.backups[backup_id]
            except Exception as e:
                print(f"Failed to delete old backup {backup_id}: {e}")

        self._save_catalog()

    def restore(self, backup_id: str, target_dir: str = None,
               verify: bool = True) -> bool:
        """Restore from backup"""
        with self.lock:
            if backup_id not in self.backups:
                raise ValueError(f"Backup {backup_id} not found")

            info = self.backups[backup_id]
            backup_file = Path(info.file_path)

            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")

            # Verify checksum
            if verify:
                current_checksum = self._calculate_checksum(backup_file)
                if current_checksum != info.checksum:
                    raise ValueError("Backup checksum verification failed")

            # Determine target directories
            if target_dir:
                extract_dir = Path(target_dir)
            else:
                extract_dir = Path(__file__).parent.parent

            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                # For incremental, first restore parent
                if info.backup_type == BackupType.INCREMENTAL and info.parent_backup:
                    self.restore(info.parent_backup, str(extract_dir), verify=False)

                # Extract files
                tar.extractall(extract_dir)

            # Update status
            info.status = BackupStatus.VERIFIED
            self._save_catalog()

            return True

    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity"""
        if backup_id not in self.backups:
            return {'valid': False, 'error': 'Backup not found'}

        info = self.backups[backup_id]
        backup_file = Path(info.file_path)

        if not backup_file.exists():
            return {'valid': False, 'error': 'Backup file not found'}

        # Verify checksum
        current_checksum = self._calculate_checksum(backup_file)
        checksum_valid = current_checksum == info.checksum

        # Verify archive integrity
        archive_valid = True
        files_list = []
        try:
            with tarfile.open(backup_file, "r:gz") as tar:
                files_list = tar.getnames()
        except Exception as e:
            archive_valid = False

        return {
            'valid': checksum_valid and archive_valid,
            'checksum_valid': checksum_valid,
            'archive_valid': archive_valid,
            'expected_checksum': info.checksum,
            'actual_checksum': current_checksum,
            'files_count': len(files_list),
            'files': files_list[:20]  # First 20 files
        }

    def list_backups(self, backup_type: BackupType = None,
                    limit: int = 20) -> List[BackupInfo]:
        """List available backups"""
        backups = list(self.backups.values())

        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]

        return sorted(backups, key=lambda x: x.created_at, reverse=True)[:limit]

    def get_backup(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup info"""
        return self.backups.get(backup_id)

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        with self.lock:
            if backup_id not in self.backups:
                return False

            info = self.backups[backup_id]

            try:
                if Path(info.file_path).exists():
                    Path(info.file_path).unlink()
                del self.backups[backup_id]
                self._save_catalog()
                return True
            except Exception as e:
                print(f"Failed to delete backup: {e}")
                return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get backup statistics"""
        total_size = sum(b.size_bytes for b in self.backups.values())
        by_type = {}
        for b in self.backups.values():
            if b.backup_type.value not in by_type:
                by_type[b.backup_type.value] = {'count': 0, 'size': 0}
            by_type[b.backup_type.value]['count'] += 1
            by_type[b.backup_type.value]['size'] += b.size_bytes

        latest = None
        for b in sorted(self.backups.values(), key=lambda x: x.created_at, reverse=True):
            if b.status == BackupStatus.COMPLETED:
                latest = b
                break

        return {
            'total_backups': len(self.backups),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_type': by_type,
            'latest_backup': latest.to_dict() if latest else None,
            'backup_dir': str(self.backup_dir)
        }


class BackupScheduler:
    """Automated backup scheduler"""

    def __init__(self, manager: BackupManager):
        self.manager = manager
        self.schedules: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def add_daily_full(self, hour: int = 2):
        """Schedule daily full backup"""
        self.schedules.append({
            'type': BackupType.FULL,
            'hour': hour,
            'minute': 0,
            'last_run': None
        })

    def add_hourly_incremental(self):
        """Schedule hourly incremental backup"""
        self.schedules.append({
            'type': BackupType.INCREMENTAL,
            'interval_hours': 1,
            'last_run': None
        })

    def start(self):
        """Start scheduler"""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run_loop(self):
        """Scheduler loop"""
        while self.running:
            now = datetime.now()

            for schedule in self.schedules:
                should_run = False

                if 'hour' in schedule:
                    # Daily schedule
                    if now.hour == schedule['hour'] and now.minute == schedule['minute']:
                        last = schedule.get('last_run')
                        if not last or (now - last).total_seconds() > 3600:
                            should_run = True
                elif 'interval_hours' in schedule:
                    # Interval schedule
                    last = schedule.get('last_run')
                    if not last or (now - last).total_seconds() >= schedule['interval_hours'] * 3600:
                        should_run = True

                if should_run:
                    try:
                        self.manager.create_backup(schedule['type'], "Scheduled backup")
                        schedule['last_run'] = now
                    except Exception as e:
                        print(f"Scheduled backup failed: {e}")

            time.sleep(60)


# Global instance
_backup_manager = None


def get_backup_manager() -> BackupManager:
    """Get global backup manager"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


if __name__ == "__main__":
    # Test backup system
    print("=== Backup System Test ===")

    manager = BackupManager()

    # Create full backup
    print("\n1. Creating full backup...")
    try:
        backup = manager.create_backup(BackupType.FULL, "Test backup")
        print(f"   Backup ID: {backup.backup_id}")
        print(f"   Size: {backup.size_bytes} bytes")
        print(f"   Files: {backup.files_count}")
        print(f"   Duration: {backup.duration_seconds:.2f}s")
    except Exception as e:
        print(f"   Backup failed: {e}")

    # Create config backup
    print("\n2. Creating config backup...")
    try:
        config_backup = manager.create_backup(BackupType.CONFIG)
        print(f"   Backup ID: {config_backup.backup_id}")
    except Exception as e:
        print(f"   Backup failed: {e}")

    # List backups
    print("\n3. Listing backups:")
    for b in manager.list_backups():
        print(f"   - {b.backup_id}: {b.backup_type.value} ({b.size_bytes} bytes)")

    # Verify backup
    print("\n4. Verifying backup...")
    if backup:
        result = manager.verify_backup(backup.backup_id)
        print(f"   Valid: {result['valid']}")
        print(f"   Files: {result['files_count']}")

    # Statistics
    print("\n5. Statistics:")
    stats = manager.get_statistics()
    print(f"   Total backups: {stats['total_backups']}")
    print(f"   Total size: {stats['total_size_mb']} MB")

    print("\nBackup system test completed!")
