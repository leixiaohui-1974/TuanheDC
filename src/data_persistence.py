"""
TAOS Data Persistence Module
数据持久化模块 - 时序数据存储、统计分析、数据导出

Features:
- SQLite database for time series storage
- Automatic data aggregation (1min, 5min, 1hour, 1day)
- Scenario event logging
- System statistics computation
- Data export (CSV, JSON, Parquet)
- Data retention policies
"""

import sqlite3
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import csv
import gzip


class AggregationType(Enum):
    """数据聚合类型"""
    RAW = "raw"              # 原始数据
    MINUTE_1 = "1min"        # 1分钟聚合
    MINUTE_5 = "5min"        # 5分钟聚合
    HOUR_1 = "1hour"         # 1小时聚合
    DAY_1 = "1day"           # 1天聚合


class ExportFormat(Enum):
    """导出格式"""
    CSV = "csv"
    JSON = "json"
    GZIP_CSV = "csv.gz"


@dataclass
class StateRecord:
    """状态记录"""
    timestamp: datetime
    sim_time: float
    h: float                 # 水位 (m)
    Q_in: float              # 进水流量 (m³/s)
    Q_out: float             # 出水流量 (m³/s)
    fr: float                # 弗劳德数
    v: float                 # 流速 (m/s)
    T_sun: float             # 阳面温度 (°C)
    T_shade: float           # 阴面温度 (°C)
    vib_amp: float           # 振动幅度 (mm)
    joint_gap: float         # 伸缩缝间隙 (mm)
    ground_accel: float      # 地面加速度 (g)
    active_scenarios: List[str]
    risk_level: str
    control_status: str
    mpc_method: str


@dataclass
class ScenarioEvent:
    """场景事件记录"""
    event_id: str
    timestamp: datetime
    scenario_id: str
    severity: str
    duration: float          # 持续时间 (s)
    peak_values: Dict[str, float]
    control_actions: Dict[str, Any]
    outcome: str             # resolved, ongoing, escalated


@dataclass
class SystemStatistics:
    """系统统计数据"""
    period_start: datetime
    period_end: datetime
    total_samples: int
    h_mean: float
    h_std: float
    h_min: float
    h_max: float
    fr_mean: float
    fr_max: float
    T_delta_mean: float
    T_delta_max: float
    vib_mean: float
    vib_max: float
    scenario_counts: Dict[str, int]
    risk_level_distribution: Dict[str, int]
    control_status_distribution: Dict[str, int]
    mpc_fallback_rate: float
    safe_operation_ratio: float


class DataPersistence:
    """数据持久化管理器"""

    def __init__(self, db_path: str = "data/taos.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_db_directory()
        self._init_database()

        # 缓冲区 - 批量写入优化
        self.write_buffer: List[StateRecord] = []
        self.buffer_size = 100
        self.last_flush = datetime.now()
        self.flush_interval = timedelta(seconds=30)

        # 聚合缓存
        self.aggregation_cache: Dict[str, List[StateRecord]] = {
            "1min": [],
            "5min": [],
            "1hour": [],
            "1day": []
        }

    def _ensure_db_directory(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def _init_database(self):
        """初始化数据库表结构"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 原始状态数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sim_time REAL,
                    h REAL,
                    Q_in REAL,
                    Q_out REAL,
                    fr REAL,
                    v REAL,
                    T_sun REAL,
                    T_shade REAL,
                    vib_amp REAL,
                    joint_gap REAL,
                    ground_accel REAL,
                    active_scenarios TEXT,
                    risk_level TEXT,
                    control_status TEXT,
                    mpc_method TEXT
                )
            ''')

            # 1分钟聚合表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_1min (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    h_mean REAL, h_min REAL, h_max REAL, h_std REAL,
                    Q_in_mean REAL, Q_out_mean REAL,
                    fr_mean REAL, fr_max REAL,
                    T_delta_mean REAL, T_delta_max REAL,
                    vib_mean REAL, vib_max REAL,
                    sample_count INTEGER
                )
            ''')

            # 5分钟聚合表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_5min (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    h_mean REAL, h_min REAL, h_max REAL, h_std REAL,
                    Q_in_mean REAL, Q_out_mean REAL,
                    fr_mean REAL, fr_max REAL,
                    T_delta_mean REAL, T_delta_max REAL,
                    vib_mean REAL, vib_max REAL,
                    sample_count INTEGER
                )
            ''')

            # 1小时聚合表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_1hour (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    h_mean REAL, h_min REAL, h_max REAL, h_std REAL,
                    Q_in_mean REAL, Q_out_mean REAL,
                    fr_mean REAL, fr_max REAL,
                    T_delta_mean REAL, T_delta_max REAL,
                    vib_mean REAL, vib_max REAL,
                    sample_count INTEGER,
                    scenario_counts TEXT,
                    risk_distribution TEXT
                )
            ''')

            # 1天聚合表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_1day (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    h_mean REAL, h_min REAL, h_max REAL, h_std REAL,
                    Q_in_mean REAL, Q_out_mean REAL,
                    fr_mean REAL, fr_max REAL,
                    T_delta_mean REAL, T_delta_max REAL,
                    vib_mean REAL, vib_max REAL,
                    sample_count INTEGER,
                    scenario_counts TEXT,
                    risk_distribution TEXT,
                    safe_operation_ratio REAL
                )
            ''')

            # 场景事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scenario_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    scenario_id TEXT,
                    severity TEXT,
                    duration REAL,
                    peak_values TEXT,
                    control_actions TEXT,
                    outcome TEXT
                )
            ''')

            # 系统告警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            ''')

            # 操作日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT,
                    operator TEXT,
                    parameters TEXT,
                    result TEXT
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_raw_timestamp ON state_raw(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_1min_timestamp ON state_1min(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_5min_timestamp ON state_5min(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state_1hour_timestamp ON state_1hour(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_scenario_events_timestamp ON scenario_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_alerts_timestamp ON system_alerts(timestamp)')

            conn.commit()
            conn.close()

    def record_state(self, state: Dict[str, Any]):
        """记录状态数据"""
        record = StateRecord(
            timestamp=datetime.now(),
            sim_time=state.get('sim_time', state.get('time', 0.0)),
            h=state.get('h', 0.0),
            Q_in=state.get('Q_in', state.get('Q_in_cmd', 80.0)),
            Q_out=state.get('Q_out', state.get('Q_out_cmd', 80.0)),
            fr=state.get('fr', 0.0),
            v=state.get('v', 0.0),
            T_sun=state.get('T_sun', 25.0),
            T_shade=state.get('T_shade', 25.0),
            vib_amp=state.get('vib_amp', 0.0),
            joint_gap=state.get('joint_gap', 20.0),
            ground_accel=state.get('ground_accel', 0.0),
            active_scenarios=state.get('active_scenarios', []),
            risk_level=state.get('risk_level', 'INFO'),
            control_status=state.get('status', 'NORMAL'),
            mpc_method=state.get('mpc_method', 'UNKNOWN')
        )

        self.write_buffer.append(record)

        # 检查是否需要刷新缓冲区
        if len(self.write_buffer) >= self.buffer_size or \
           datetime.now() - self.last_flush > self.flush_interval:
            self._flush_buffer()

    def _flush_buffer(self):
        """刷新写入缓冲区到数据库"""
        if not self.write_buffer:
            return

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for record in self.write_buffer:
                cursor.execute('''
                    INSERT INTO state_raw (
                        timestamp, sim_time, h, Q_in, Q_out, fr, v,
                        T_sun, T_shade, vib_amp, joint_gap, ground_accel,
                        active_scenarios, risk_level, control_status, mpc_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.timestamp.isoformat(),
                    record.sim_time,
                    record.h,
                    record.Q_in,
                    record.Q_out,
                    record.fr,
                    record.v,
                    record.T_sun,
                    record.T_shade,
                    record.vib_amp,
                    record.joint_gap,
                    record.ground_accel,
                    json.dumps(record.active_scenarios),
                    record.risk_level,
                    record.control_status,
                    record.mpc_method
                ))

            conn.commit()
            conn.close()

            # 更新聚合缓存
            self._update_aggregation_cache(self.write_buffer)

            self.write_buffer = []
            self.last_flush = datetime.now()

    def _update_aggregation_cache(self, records: List[StateRecord]):
        """更新聚合缓存并触发聚合"""
        for record in records:
            self.aggregation_cache["1min"].append(record)

        # 检查1分钟聚合
        if len(self.aggregation_cache["1min"]) >= 120:  # ~1分钟 @ 0.5s采样
            self._aggregate_and_store("1min", self.aggregation_cache["1min"])
            self.aggregation_cache["1min"] = []

    def _aggregate_and_store(self, level: str, records: List[StateRecord]):
        """计算聚合并存储"""
        if not records:
            return

        # 计算统计值
        h_values = [r.h for r in records]
        fr_values = [r.fr for r in records]
        T_delta_values = [r.T_sun - r.T_shade for r in records]
        vib_values = [r.vib_amp for r in records]
        Q_in_values = [r.Q_in for r in records]
        Q_out_values = [r.Q_out for r in records]

        import statistics as stats

        aggregated = {
            'timestamp': records[0].timestamp.isoformat(),
            'h_mean': stats.mean(h_values),
            'h_min': min(h_values),
            'h_max': max(h_values),
            'h_std': stats.stdev(h_values) if len(h_values) > 1 else 0.0,
            'Q_in_mean': stats.mean(Q_in_values),
            'Q_out_mean': stats.mean(Q_out_values),
            'fr_mean': stats.mean(fr_values),
            'fr_max': max(fr_values),
            'T_delta_mean': stats.mean(T_delta_values),
            'T_delta_max': max(T_delta_values),
            'vib_mean': stats.mean(vib_values),
            'vib_max': max(vib_values),
            'sample_count': len(records)
        }

        # 存储聚合数据
        table_name = f"state_{level}"
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {table_name} (
                    timestamp, h_mean, h_min, h_max, h_std,
                    Q_in_mean, Q_out_mean, fr_mean, fr_max,
                    T_delta_mean, T_delta_max, vib_mean, vib_max, sample_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                aggregated['timestamp'],
                aggregated['h_mean'], aggregated['h_min'],
                aggregated['h_max'], aggregated['h_std'],
                aggregated['Q_in_mean'], aggregated['Q_out_mean'],
                aggregated['fr_mean'], aggregated['fr_max'],
                aggregated['T_delta_mean'], aggregated['T_delta_max'],
                aggregated['vib_mean'], aggregated['vib_max'],
                aggregated['sample_count']
            ))
            conn.commit()
            conn.close()

    def record_scenario_event(self, event: ScenarioEvent):
        """记录场景事件"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO scenario_events (
                    event_id, timestamp, scenario_id, severity,
                    duration, peak_values, control_actions, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.timestamp.isoformat(),
                event.scenario_id,
                event.severity,
                event.duration,
                json.dumps(event.peak_values),
                json.dumps(event.control_actions),
                event.outcome
            ))
            conn.commit()
            conn.close()

    def record_alert(self, alert_type: str, severity: str,
                     message: str, details: Dict[str, Any] = None):
        """记录系统告警"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_alerts (
                    timestamp, alert_type, severity, message, details
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                alert_type,
                severity,
                message,
                json.dumps(details or {})
            ))
            conn.commit()
            conn.close()

    def record_operation(self, operation_type: str, operator: str,
                        parameters: Dict[str, Any], result: str):
        """记录操作日志"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO operation_logs (
                    timestamp, operation_type, operator, parameters, result
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                operation_type,
                operator,
                json.dumps(parameters),
                result
            ))
            conn.commit()
            conn.close()

    def query_state_history(self, start_time: datetime, end_time: datetime,
                           aggregation: AggregationType = AggregationType.RAW,
                           limit: int = 10000) -> List[Dict[str, Any]]:
        """查询状态历史"""
        table_name = f"state_{aggregation.value}" if aggregation != AggregationType.RAW else "state_raw"

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(f'''
                SELECT * FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (start_time.isoformat(), end_time.isoformat(), limit))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def query_scenario_events(self, start_time: datetime = None,
                             end_time: datetime = None,
                             scenario_id: str = None) -> List[Dict[str, Any]]:
        """查询场景事件"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM scenario_events WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            if scenario_id:
                query += " AND scenario_id = ?"
                params.append(scenario_id)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def get_statistics(self, start_time: datetime,
                       end_time: datetime) -> SystemStatistics:
        """获取系统统计数据"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 基础统计
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    AVG(h) as h_mean,
                    MIN(h) as h_min,
                    MAX(h) as h_max,
                    AVG(fr) as fr_mean,
                    MAX(fr) as fr_max,
                    AVG(T_sun - T_shade) as T_delta_mean,
                    MAX(T_sun - T_shade) as T_delta_max,
                    AVG(vib_amp) as vib_mean,
                    MAX(vib_amp) as vib_max
                FROM state_raw
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_time.isoformat(), end_time.isoformat()))

            row = cursor.fetchone()

            # 计算标准差
            cursor.execute('''
                SELECT h FROM state_raw
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_time.isoformat(), end_time.isoformat()))
            h_values = [r[0] for r in cursor.fetchall()]

            import statistics as stats
            h_std = stats.stdev(h_values) if len(h_values) > 1 else 0.0

            # 场景统计
            cursor.execute('''
                SELECT active_scenarios FROM state_raw
                WHERE timestamp BETWEEN ? AND ?
            ''', (start_time.isoformat(), end_time.isoformat()))

            scenario_counts = defaultdict(int)
            for (scenarios_json,) in cursor.fetchall():
                scenarios = json.loads(scenarios_json) if scenarios_json else []
                for s in scenarios:
                    scenario_counts[s] += 1

            # 风险级别分布
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM state_raw
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY risk_level
            ''', (start_time.isoformat(), end_time.isoformat()))
            risk_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            # 控制状态分布
            cursor.execute('''
                SELECT control_status, COUNT(*) as count
                FROM state_raw
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY control_status
            ''', (start_time.isoformat(), end_time.isoformat()))
            status_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            # 计算安全运行比例
            total = row[0] if row[0] else 1
            safe_count = risk_distribution.get('INFO', 0) + risk_distribution.get('LOW', 0)
            safe_ratio = safe_count / total

            conn.close()

            return SystemStatistics(
                period_start=start_time,
                period_end=end_time,
                total_samples=row[0] or 0,
                h_mean=row[1] or 0.0,
                h_std=h_std,
                h_min=row[2] or 0.0,
                h_max=row[3] or 0.0,
                fr_mean=row[4] or 0.0,
                fr_max=row[5] or 0.0,
                T_delta_mean=row[6] or 0.0,
                T_delta_max=row[7] or 0.0,
                vib_mean=row[8] or 0.0,
                vib_max=row[9] or 0.0,
                scenario_counts=dict(scenario_counts),
                risk_level_distribution=risk_distribution,
                control_status_distribution=status_distribution,
                mpc_fallback_rate=0.0,  # 需要从MPC状态计算
                safe_operation_ratio=safe_ratio
            )

    def export_data(self, start_time: datetime, end_time: datetime,
                    format: ExportFormat = ExportFormat.CSV,
                    output_path: str = None) -> str:
        """导出数据"""
        data = self.query_state_history(start_time, end_time, AggregationType.RAW)

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/export_{timestamp}.{format.value}"

        # 确保导出目录存在
        export_dir = os.path.dirname(output_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir)

        if format == ExportFormat.CSV:
            self._export_csv(data, output_path)
        elif format == ExportFormat.JSON:
            self._export_json(data, output_path)
        elif format == ExportFormat.GZIP_CSV:
            self._export_gzip_csv(data, output_path)

        return output_path

    def _export_csv(self, data: List[Dict], path: str):
        """导出CSV格式"""
        if not data:
            return

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def _export_json(self, data: List[Dict], path: str):
        """导出JSON格式"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_gzip_csv(self, data: List[Dict], path: str):
        """导出压缩CSV格式"""
        if not data:
            return

        with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def cleanup_old_data(self, retention_days: int = 30):
        """清理过期数据"""
        cutoff = datetime.now() - timedelta(days=retention_days)

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 清理原始数据 (保留更短时间)
            raw_cutoff = datetime.now() - timedelta(days=7)
            cursor.execute(
                'DELETE FROM state_raw WHERE timestamp < ?',
                (raw_cutoff.isoformat(),)
            )

            # 清理1分钟聚合 (保留14天)
            min1_cutoff = datetime.now() - timedelta(days=14)
            cursor.execute(
                'DELETE FROM state_1min WHERE timestamp < ?',
                (min1_cutoff.isoformat(),)
            )

            # 清理5分钟聚合 (保留30天)
            cursor.execute(
                'DELETE FROM state_5min WHERE timestamp < ?',
                (cutoff.isoformat(),)
            )

            # 1小时和1天聚合保留更长时间 (365天)
            year_cutoff = datetime.now() - timedelta(days=365)
            cursor.execute(
                'DELETE FROM state_1hour WHERE timestamp < ?',
                (year_cutoff.isoformat(),)
            )

            conn.commit()
            conn.close()

    def get_recent_alerts(self, limit: int = 50,
                          unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """获取最近告警"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM system_alerts"
            if unacknowledged_only:
                query += " WHERE acknowledged = 0"
            query += " ORDER BY timestamp DESC LIMIT ?"

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

    def acknowledge_alert(self, alert_id: int):
        """确认告警"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE system_alerts SET acknowledged = 1 WHERE id = ?',
                (alert_id,)
            )
            conn.commit()
            conn.close()

    def resolve_alert(self, alert_id: int):
        """解决告警"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''UPDATE system_alerts
                   SET resolved = 1, resolved_at = ?
                   WHERE id = ?''',
                (datetime.now().isoformat(), alert_id)
            )
            conn.commit()
            conn.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            tables = ['state_raw', 'state_1min', 'state_5min',
                      'state_1hour', 'state_1day', 'scenario_events',
                      'system_alerts', 'operation_logs']

            stats = {}
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]
                stats[table] = count

            # 数据库文件大小
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            conn.close()

            return {
                'table_counts': stats,
                'db_size_bytes': db_size,
                'db_size_mb': round(db_size / (1024 * 1024), 2),
                'buffer_size': len(self.write_buffer),
                'last_flush': self.last_flush.isoformat()
            }

    def close(self):
        """关闭并刷新缓冲区"""
        self._flush_buffer()


class DataAnalytics:
    """数据分析模块"""

    def __init__(self, persistence: DataPersistence):
        self.persistence = persistence

    def trend_analysis(self, variable: str, hours: int = 24) -> Dict[str, Any]:
        """趋势分析"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        data = self.persistence.query_state_history(
            start_time, end_time, AggregationType.MINUTE_5
        )

        if not data:
            return {'error': 'No data available'}

        values = []
        timestamps = []
        for record in data:
            if variable in record:
                values.append(record[variable])
                timestamps.append(record['timestamp'])

        if not values:
            return {'error': f'Variable {variable} not found'}

        import statistics as stats

        # 计算趋势
        n = len(values)
        if n > 1:
            x_mean = (n - 1) / 2
            y_mean = stats.mean(values)
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0
        else:
            slope = 0

        return {
            'variable': variable,
            'period_hours': hours,
            'sample_count': n,
            'current': values[-1] if values else 0,
            'mean': stats.mean(values),
            'std': stats.stdev(values) if n > 1 else 0,
            'min': min(values),
            'max': max(values),
            'trend_slope': slope,
            'trend_direction': 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
        }

    def anomaly_detection_report(self, hours: int = 24) -> Dict[str, Any]:
        """异常检测报告"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        data = self.persistence.query_state_history(start_time, end_time)

        anomalies = []
        for record in data:
            reasons = []

            # 检查水位异常
            h = record.get('h', 4.0)
            if h < 2.0 or h > 6.0:
                reasons.append(f"水位异常: {h:.2f}m")

            # 检查弗劳德数异常
            fr = record.get('fr', 0.0)
            if fr > 0.8:
                reasons.append(f"弗劳德数过高: {fr:.2f}")

            # 检查温差异常
            T_sun = record.get('T_sun', 25.0)
            T_shade = record.get('T_shade', 25.0)
            T_delta = abs(T_sun - T_shade)
            if T_delta > 15:
                reasons.append(f"温差过大: {T_delta:.1f}°C")

            # 检查振动异常
            vib = record.get('vib_amp', 0.0)
            if vib > 50:
                reasons.append(f"振动过大: {vib:.1f}mm")

            if reasons:
                anomalies.append({
                    'timestamp': record.get('timestamp'),
                    'reasons': reasons,
                    'risk_level': record.get('risk_level', 'UNKNOWN')
                })

        return {
            'period_hours': hours,
            'total_records': len(data),
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / max(len(data), 1),
            'anomalies': anomalies[-100:]  # 最近100条
        }

    def scenario_frequency_analysis(self, days: int = 7) -> Dict[str, Any]:
        """场景频率分析"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        stats_data = self.persistence.get_statistics(start_time, end_time)

        total = sum(stats_data.scenario_counts.values())

        return {
            'period_days': days,
            'total_samples': stats_data.total_samples,
            'scenario_counts': stats_data.scenario_counts,
            'scenario_frequencies': {
                k: v / max(total, 1) for k, v in stats_data.scenario_counts.items()
            },
            'most_frequent': max(stats_data.scenario_counts.items(),
                                key=lambda x: x[1]) if stats_data.scenario_counts else None,
            'risk_distribution': stats_data.risk_level_distribution,
            'safe_operation_ratio': stats_data.safe_operation_ratio
        }


# 全局实例
_persistence_instance: Optional[DataPersistence] = None


def get_persistence() -> DataPersistence:
    """获取全局持久化实例"""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = DataPersistence()
    return _persistence_instance
