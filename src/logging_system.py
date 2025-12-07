#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - Logging and Audit Trail System
团河渡槽自主运行系统 - 日志与审计模块

Features:
- Structured logging with multiple handlers
- Audit trail for all operations
- Event categorization and filtering
- Log rotation and archiving
- Real-time log streaming
- Log analysis and statistics
"""

import os
import json
import logging
import threading
import queue
import gzip
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from enum import Enum
from collections import deque


class LogLevel(Enum):
    """Log levels"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class EventCategory(Enum):
    """Event categories for audit trail"""
    SYSTEM = "system"
    CONTROL = "control"
    SAFETY = "safety"
    SCENARIO = "scenario"
    ALARM = "alarm"
    USER = "user"
    CONFIG = "config"
    DATA = "data"
    NETWORK = "network"
    SECURITY = "security"


@dataclass
class LogEntry:
    """Log entry structure"""
    timestamp: str
    level: str
    category: str
    source: str
    message: str
    data: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class AuditEntry:
    """Audit trail entry"""
    timestamp: str
    category: str
    action: str
    actor: str
    target: str
    details: Dict[str, Any]
    result: str
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


class LogFormatter(logging.Formatter):
    """Custom log formatter for TAOS"""

    def __init__(self, include_data: bool = False):
        super().__init__()
        self.include_data = include_data

    def format(self, record: logging.LogRecord) -> str:
        # Base format
        timestamp = datetime.now().isoformat(timespec='milliseconds')
        level = record.levelname
        source = f"{record.name}:{record.funcName}:{record.lineno}"
        message = record.getMessage()

        # Basic log line
        log_line = f"{timestamp} [{level:8}] [{source}] {message}"

        # Add extra data if present
        if self.include_data and hasattr(record, 'data') and record.data:
            log_line += f" | data={json.dumps(record.data)}"

        # Add exception info
        if record.exc_info:
            import traceback
            log_line += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return log_line


class JsonFormatter(logging.Formatter):
    """JSON log formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "level": record.levelname,
            "logger": record.name,
            "source": f"{record.funcName}:{record.lineno}",
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, 'category'):
            log_dict['category'] = record.category
        if hasattr(record, 'data'):
            log_dict['data'] = record.data
        if hasattr(record, 'user'):
            log_dict['user'] = record.user
        if hasattr(record, 'trace_id'):
            log_dict['trace_id'] = record.trace_id

        # Add exception
        if record.exc_info:
            import traceback
            log_dict['exception'] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(log_dict, ensure_ascii=False)


class RotatingFileHandler(logging.Handler):
    """Custom rotating file handler with compression"""

    def __init__(self, filename: str, max_bytes: int = 10*1024*1024,
                 backup_count: int = 5, compress: bool = True):
        super().__init__()
        self.base_filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self.current_size = 0
        self.file = None
        self.lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        # Open file
        self._open_file()

    def _open_file(self):
        """Open log file"""
        if self.file:
            self.file.close()

        self.file = open(self.base_filename, 'a', encoding='utf-8')
        self.current_size = os.path.getsize(self.base_filename) if os.path.exists(self.base_filename) else 0

    def emit(self, record: logging.LogRecord):
        """Emit a log record"""
        try:
            msg = self.format(record) + '\n'
            msg_bytes = msg.encode('utf-8')

            with self.lock:
                if self.current_size + len(msg_bytes) > self.max_bytes:
                    self._rotate()

                self.file.write(msg)
                self.file.flush()
                self.current_size += len(msg_bytes)

        except Exception:
            self.handleError(record)

    def _rotate(self):
        """Rotate log files"""
        self.file.close()

        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.base_filename}.{i}"
            if self.compress:
                src += ".gz"
            dst = f"{self.base_filename}.{i + 1}"
            if self.compress:
                dst += ".gz"

            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

        # Compress current file
        if self.compress:
            with open(self.base_filename, 'rb') as f_in:
                with gzip.open(f"{self.base_filename}.1.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
        else:
            os.rename(self.base_filename, f"{self.base_filename}.1")

        # Open new file
        self._open_file()

    def close(self):
        """Close handler"""
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None
        super().close()


class DatabaseLogHandler(logging.Handler):
    """SQLite database log handler"""

    def __init__(self, db_path: str, table_name: str = "logs"):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.queue = queue.Queue()
        self.running = True
        self.batch_size = 100
        self.flush_interval = 5.0

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

        # Initialize database
        self._init_db()

        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                category TEXT,
                source TEXT,
                message TEXT NOT NULL,
                data TEXT,
                user TEXT,
                trace_id TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp
            ON {self.table_name}(timestamp)
        """)

        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_level
            ON {self.table_name}(level)
        """)

        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_category
            ON {self.table_name}(category)
        """)

        conn.commit()
        conn.close()

    def emit(self, record: logging.LogRecord):
        """Emit log record to queue"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(timespec='milliseconds'),
                "level": record.levelname,
                "category": getattr(record, 'category', 'system'),
                "source": f"{record.name}:{record.funcName}:{record.lineno}",
                "message": record.getMessage(),
                "data": json.dumps(getattr(record, 'data', None)) if hasattr(record, 'data') and record.data else None,
                "user": getattr(record, 'user', None),
                "trace_id": getattr(record, 'trace_id', None),
            }
            self.queue.put(entry)
        except Exception:
            self.handleError(record)

    def _writer_loop(self):
        """Background writer loop"""
        import time

        while self.running:
            entries = []

            # Collect entries
            try:
                while len(entries) < self.batch_size:
                    entry = self.queue.get(timeout=self.flush_interval)
                    entries.append(entry)
            except queue.Empty:
                pass

            # Write batch
            if entries:
                self._write_batch(entries)

    def _write_batch(self, entries: List[Dict]):
        """Write batch of entries to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.executemany(f"""
                INSERT INTO {self.table_name}
                (timestamp, level, category, source, message, data, user, trace_id)
                VALUES (:timestamp, :level, :category, :source, :message, :data, :user, :trace_id)
            """, entries)

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Database log write error: {e}")

    def close(self):
        """Close handler"""
        self.running = False
        self.writer_thread.join(timeout=self.flush_interval + 1)
        super().close()


class RealtimeLogBuffer:
    """Circular buffer for real-time log viewing"""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.subscribers: List[Callable[[LogEntry], None]] = []

    def add(self, entry: LogEntry):
        """Add log entry"""
        with self.lock:
            self.buffer.append(entry)

        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(entry)
            except Exception:
                pass

    def get_recent(self, count: int = 100, level: Optional[str] = None,
                   category: Optional[str] = None) -> List[LogEntry]:
        """Get recent log entries with optional filtering"""
        with self.lock:
            entries = list(self.buffer)

        # Filter
        if level:
            entries = [e for e in entries if e.level == level]
        if category:
            entries = [e for e in entries if e.category == category]

        return entries[-count:]

    def subscribe(self, callback: Callable[[LogEntry], None]):
        """Subscribe to log entries"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[LogEntry], None]):
        """Unsubscribe from log entries"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class AuditTrail:
    """Audit trail manager"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                target TEXT,
                details TEXT,
                result TEXT,
                ip_address TEXT,
                session_id TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_trail(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_trail(category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_trail(actor)
        """)

        conn.commit()
        conn.close()

    def log(self, category: str, action: str, actor: str, target: str = "",
            details: Dict[str, Any] = None, result: str = "success",
            ip_address: str = None, session_id: str = None):
        """Log audit entry"""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(timespec='milliseconds'),
            category=category,
            action=action,
            actor=actor,
            target=target,
            details=details or {},
            result=result,
            ip_address=ip_address,
            session_id=session_id
        )

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO audit_trail
                    (timestamp, category, action, actor, target, details, result, ip_address, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.timestamp,
                    entry.category,
                    entry.action,
                    entry.actor,
                    entry.target,
                    json.dumps(entry.details),
                    entry.result,
                    entry.ip_address,
                    entry.session_id
                ))

                conn.commit()
                conn.close()

            except Exception as e:
                print(f"Audit log error: {e}")

    def query(self, start_time: datetime = None, end_time: datetime = None,
              category: str = None, actor: str = None, action: str = None,
              limit: int = 100) -> List[AuditEntry]:
        """Query audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM audit_trail WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if category:
            query += " AND category = ?"
            params.append(category)
        if actor:
            query += " AND actor = ?"
            params.append(actor)
        if action:
            query += " AND action LIKE ?"
            params.append(f"%{action}%")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            entries.append(AuditEntry(
                timestamp=row[1],
                category=row[2],
                action=row[3],
                actor=row[4],
                target=row[5] or "",
                details=json.loads(row[6]) if row[6] else {},
                result=row[7] or "",
                ip_address=row[8],
                session_id=row[9]
            ))

        return entries

    def get_statistics(self, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """Get audit statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM audit_trail WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        # Total count
        cursor.execute(f"SELECT COUNT(*) FROM ({query})", params)
        total = cursor.fetchone()[0]

        # By category
        cursor.execute(f"""
            SELECT category, COUNT(*) FROM ({query}) GROUP BY category
        """, params)
        by_category = dict(cursor.fetchall())

        # By result
        cursor.execute(f"""
            SELECT result, COUNT(*) FROM ({query}) GROUP BY result
        """, params)
        by_result = dict(cursor.fetchall())

        # By actor
        cursor.execute(f"""
            SELECT actor, COUNT(*) FROM ({query}) GROUP BY actor ORDER BY COUNT(*) DESC LIMIT 10
        """, params)
        by_actor = dict(cursor.fetchall())

        conn.close()

        return {
            "total": total,
            "by_category": by_category,
            "by_result": by_result,
            "top_actors": by_actor
        }


class TAOSLogger:
    """Main logger for TAOS system"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Paths
        self.log_dir = Path(__file__).parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Main logger
        self.logger = logging.getLogger("TAOS")
        self.logger.setLevel(logging.DEBUG)

        # Realtime buffer
        self.realtime_buffer = RealtimeLogBuffer(max_size=2000)

        # Audit trail
        self.audit = AuditTrail(str(self.log_dir / "audit.db"))

        # Setup handlers
        self._setup_handlers()

        # Category loggers
        self.category_loggers: Dict[str, logging.Logger] = {}

    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(LogFormatter())
        self.logger.addHandler(console_handler)

        # File handler (text)
        file_handler = RotatingFileHandler(
            str(self.log_dir / "taos.log"),
            max_bytes=10*1024*1024,
            backup_count=10,
            compress=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LogFormatter(include_data=True))
        self.logger.addHandler(file_handler)

        # JSON file handler
        json_handler = RotatingFileHandler(
            str(self.log_dir / "taos.json.log"),
            max_bytes=10*1024*1024,
            backup_count=5,
            compress=True
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(json_handler)

        # Database handler
        db_handler = DatabaseLogHandler(str(self.log_dir / "logs.db"))
        db_handler.setLevel(logging.INFO)
        self.logger.addHandler(db_handler)

        # Realtime handler
        class RealtimeHandler(logging.Handler):
            def __init__(self, buffer: RealtimeLogBuffer):
                super().__init__()
                self.buffer = buffer

            def emit(self, record: logging.LogRecord):
                entry = LogEntry(
                    timestamp=datetime.now().isoformat(timespec='milliseconds'),
                    level=record.levelname,
                    category=getattr(record, 'category', 'system'),
                    source=f"{record.name}:{record.funcName}",
                    message=record.getMessage(),
                    data=getattr(record, 'data', None),
                    user=getattr(record, 'user', None),
                    trace_id=getattr(record, 'trace_id', None)
                )
                self.buffer.add(entry)

        realtime_handler = RealtimeHandler(self.realtime_buffer)
        realtime_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(realtime_handler)

    def get_logger(self, category: str) -> logging.Logger:
        """Get category-specific logger"""
        if category not in self.category_loggers:
            logger = self.logger.getChild(category)
            self.category_loggers[category] = logger
        return self.category_loggers[category]

    def log(self, level: str, message: str, category: str = "system",
            data: Dict[str, Any] = None, user: str = None, trace_id: str = None):
        """Log message with extra fields"""
        logger = self.get_logger(category)
        log_level = getattr(logging, level.upper(), logging.INFO)

        extra = {
            'category': category,
            'data': data,
            'user': user,
            'trace_id': trace_id
        }

        logger.log(log_level, message, extra=extra)

    def debug(self, message: str, category: str = "system", **kwargs):
        self.log("DEBUG", message, category, **kwargs)

    def info(self, message: str, category: str = "system", **kwargs):
        self.log("INFO", message, category, **kwargs)

    def warning(self, message: str, category: str = "system", **kwargs):
        self.log("WARNING", message, category, **kwargs)

    def error(self, message: str, category: str = "system", **kwargs):
        self.log("ERROR", message, category, **kwargs)

    def critical(self, message: str, category: str = "system", **kwargs):
        self.log("CRITICAL", message, category, **kwargs)

    # Convenience methods for specific categories
    def log_control(self, message: str, level: str = "INFO", **kwargs):
        self.log(level, message, "control", **kwargs)

    def log_safety(self, message: str, level: str = "INFO", **kwargs):
        self.log(level, message, "safety", **kwargs)

    def log_scenario(self, message: str, level: str = "INFO", **kwargs):
        self.log(level, message, "scenario", **kwargs)

    def log_alarm(self, message: str, level: str = "WARNING", **kwargs):
        self.log(level, message, "alarm", **kwargs)

    def log_user_action(self, message: str, user: str, **kwargs):
        self.log("INFO", message, "user", user=user, **kwargs)

    def log_config_change(self, message: str, data: Dict[str, Any] = None, **kwargs):
        self.log("INFO", message, "config", data=data, **kwargs)

    # Audit methods
    def audit_action(self, action: str, actor: str, target: str = "",
                    details: Dict[str, Any] = None, result: str = "success",
                    category: str = "user", **kwargs):
        """Log audit action"""
        self.audit.log(category, action, actor, target, details, result, **kwargs)

    # Query methods
    def get_recent_logs(self, count: int = 100, level: str = None,
                       category: str = None) -> List[Dict[str, Any]]:
        """Get recent logs from realtime buffer"""
        entries = self.realtime_buffer.get_recent(count, level, category)
        return [asdict(e) for e in entries]

    def query_logs(self, start_time: datetime = None, end_time: datetime = None,
                  level: str = None, category: str = None, search: str = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs from database"""
        db_path = str(self.log_dir / "logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM logs WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if level:
            query += " AND level = ?"
            params.append(level)
        if category:
            query += " AND category = ?"
            params.append(category)
        if search:
            query += " AND message LIKE ?"
            params.append(f"%{search}%")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "timestamp": row[1],
                "level": row[2],
                "category": row[3],
                "source": row[4],
                "message": row[5],
                "data": json.loads(row[6]) if row[6] else None,
                "user": row[7],
                "trace_id": row[8]
            })

        return logs

    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get log statistics"""
        start_time = datetime.now() - timedelta(hours=hours)

        db_path = str(self.log_dir / "logs.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Total count
        cursor.execute("""
            SELECT COUNT(*) FROM logs WHERE timestamp >= ?
        """, (start_time.isoformat(),))
        total = cursor.fetchone()[0]

        # By level
        cursor.execute("""
            SELECT level, COUNT(*) FROM logs
            WHERE timestamp >= ?
            GROUP BY level
        """, (start_time.isoformat(),))
        by_level = dict(cursor.fetchall())

        # By category
        cursor.execute("""
            SELECT category, COUNT(*) FROM logs
            WHERE timestamp >= ?
            GROUP BY category
        """, (start_time.isoformat(),))
        by_category = dict(cursor.fetchall())

        # Error rate
        cursor.execute("""
            SELECT COUNT(*) FROM logs
            WHERE timestamp >= ? AND level IN ('ERROR', 'CRITICAL')
        """, (start_time.isoformat(),))
        error_count = cursor.fetchone()[0]

        conn.close()

        return {
            "period_hours": hours,
            "total_logs": total,
            "by_level": by_level,
            "by_category": by_category,
            "error_count": error_count,
            "error_rate": round(error_count / total * 100, 2) if total > 0 else 0
        }

    def get_audit_trail(self, **kwargs) -> List[Dict[str, Any]]:
        """Get audit trail entries"""
        entries = self.audit.query(**kwargs)
        return [asdict(e) for e in entries]

    def get_audit_statistics(self, **kwargs) -> Dict[str, Any]:
        """Get audit statistics"""
        return self.audit.get_statistics(**kwargs)


# Global logger instance
_logger = None


def get_logger() -> TAOSLogger:
    """Get global TAOS logger"""
    global _logger
    if _logger is None:
        _logger = TAOSLogger()
    return _logger


# Convenience functions
def log_info(message: str, category: str = "system", **kwargs):
    get_logger().info(message, category, **kwargs)


def log_warning(message: str, category: str = "system", **kwargs):
    get_logger().warning(message, category, **kwargs)


def log_error(message: str, category: str = "system", **kwargs):
    get_logger().error(message, category, **kwargs)


def log_debug(message: str, category: str = "system", **kwargs):
    get_logger().debug(message, category, **kwargs)


def audit(action: str, actor: str, **kwargs):
    get_logger().audit_action(action, actor, **kwargs)


if __name__ == "__main__":
    # Test logging system
    logger = get_logger()

    print("=== Logging System Test ===")

    # Test various log levels
    logger.info("System started", category="system")
    logger.debug("Debug message", category="system", data={"key": "value"})
    logger.warning("Warning: High water level", category="safety")
    logger.error("Error: Sensor timeout", category="sensor")

    # Test control logging
    logger.log_control("Gate position adjusted to 45%", data={"gate_id": 1, "position": 45})

    # Test safety logging
    logger.log_safety("Interlock triggered: HIGH_LEVEL", level="WARNING")

    # Test scenario logging
    logger.log_scenario("Scenario S1.1 activated")

    # Test alarm logging
    logger.log_alarm("ALARM: Water level exceeded 6.5m")

    # Test audit trail
    logger.audit_action(
        action="config_change",
        actor="admin",
        target="control.target_level",
        details={"old_value": 4.0, "new_value": 4.5},
        result="success",
        category="config"
    )

    logger.audit_action(
        action="emergency_stop",
        actor="operator",
        target="system",
        details={"reason": "High vibration detected"},
        result="success",
        category="safety"
    )

    # Test queries
    print("\n=== Recent Logs ===")
    recent = logger.get_recent_logs(count=5)
    for log in recent:
        print(f"[{log['level']}] {log['message']}")

    print("\n=== Log Statistics ===")
    stats = logger.get_log_statistics(hours=1)
    print(f"Total logs: {stats['total_logs']}")
    print(f"By level: {stats['by_level']}")
    print(f"Error rate: {stats['error_rate']}%")

    print("\n=== Audit Trail ===")
    audits = logger.get_audit_trail(limit=5)
    for audit_entry in audits:
        print(f"[{audit_entry['category']}] {audit_entry['action']} by {audit_entry['actor']}")

    print("\nLogging system test completed!")
