#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.6 - Performance Monitoring System
团河渡槽自主运行系统 - 性能监控模块

Features:
- Prometheus-compatible metrics
- System resource monitoring
- Application metrics collection
- Custom metric registration
- Metric export (Prometheus format)
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from collections import deque
import json


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Metric value with timestamp"""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    help: str
    labels: List[str] = field(default_factory=list)
    values: Dict[str, MetricValue] = field(default_factory=dict)

    # For histograms
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
    bucket_counts: Dict[str, Dict[float, int]] = field(default_factory=dict)
    sum_value: Dict[str, float] = field(default_factory=dict)
    count_value: Dict[str, int] = field(default_factory=dict)


class Counter:
    """Counter metric - only increases"""

    def __init__(self, metric: Metric):
        self.metric = metric
        self.lock = threading.Lock()

    def inc(self, value: float = 1, labels: Dict[str, str] = None):
        """Increment counter"""
        key = self._labels_key(labels or {})
        with self.lock:
            if key not in self.metric.values:
                self.metric.values[key] = MetricValue(0, time.time(), labels or {})
            self.metric.values[key].value += value
            self.metric.values[key].timestamp = time.time()

    def _labels_key(self, labels: Dict[str, str]) -> str:
        return json.dumps(labels, sort_keys=True)


class Gauge:
    """Gauge metric - can increase or decrease"""

    def __init__(self, metric: Metric):
        self.metric = metric
        self.lock = threading.Lock()

    def set(self, value: float, labels: Dict[str, str] = None):
        """Set gauge value"""
        key = self._labels_key(labels or {})
        with self.lock:
            self.metric.values[key] = MetricValue(value, time.time(), labels or {})

    def inc(self, value: float = 1, labels: Dict[str, str] = None):
        """Increment gauge"""
        key = self._labels_key(labels or {})
        with self.lock:
            if key not in self.metric.values:
                self.metric.values[key] = MetricValue(0, time.time(), labels or {})
            self.metric.values[key].value += value
            self.metric.values[key].timestamp = time.time()

    def dec(self, value: float = 1, labels: Dict[str, str] = None):
        """Decrement gauge"""
        self.inc(-value, labels)

    def _labels_key(self, labels: Dict[str, str]) -> str:
        return json.dumps(labels, sort_keys=True)


class Histogram:
    """Histogram metric - measures distributions"""

    def __init__(self, metric: Metric):
        self.metric = metric
        self.lock = threading.Lock()

    def observe(self, value: float, labels: Dict[str, str] = None):
        """Observe a value"""
        key = self._labels_key(labels or {})
        with self.lock:
            if key not in self.metric.bucket_counts:
                self.metric.bucket_counts[key] = {b: 0 for b in self.metric.buckets}
                self.metric.bucket_counts[key][float('inf')] = 0
                self.metric.sum_value[key] = 0
                self.metric.count_value[key] = 0

            for bucket in self.metric.buckets:
                if value <= bucket:
                    self.metric.bucket_counts[key][bucket] += 1
            self.metric.bucket_counts[key][float('inf')] += 1
            self.metric.sum_value[key] += value
            self.metric.count_value[key] += 1

    def _labels_key(self, labels: Dict[str, str]) -> str:
        return json.dumps(labels, sort_keys=True)


class MetricsRegistry:
    """Registry for all metrics"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()

    def counter(self, name: str, help: str, labels: List[str] = None) -> Counter:
        """Create or get counter"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, MetricType.COUNTER, help, labels or [])
            return Counter(self.metrics[name])

    def gauge(self, name: str, help: str, labels: List[str] = None) -> Gauge:
        """Create or get gauge"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, MetricType.GAUGE, help, labels or [])
            return Gauge(self.metrics[name])

    def histogram(self, name: str, help: str, labels: List[str] = None,
                  buckets: List[float] = None) -> Histogram:
        """Create or get histogram"""
        with self.lock:
            if name not in self.metrics:
                metric = Metric(name, MetricType.HISTOGRAM, help, labels or [])
                if buckets:
                    metric.buckets = buckets
                self.metrics[name] = metric
            return Histogram(self.metrics[name])

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        with self.lock:
            for name, metric in self.metrics.items():
                # Add help and type
                lines.append(f"# HELP {name} {metric.help}")
                lines.append(f"# TYPE {name} {metric.type.value}")

                if metric.type == MetricType.COUNTER or metric.type == MetricType.GAUGE:
                    for key, mv in metric.values.items():
                        labels_str = self._format_labels(mv.labels)
                        lines.append(f"{name}{labels_str} {mv.value}")

                elif metric.type == MetricType.HISTOGRAM:
                    for key, buckets in metric.bucket_counts.items():
                        labels = json.loads(key) if key else {}
                        cumulative = 0
                        for bucket in sorted(b for b in buckets.keys() if b != float('inf')):
                            cumulative += buckets[bucket]
                            bucket_labels = {**labels, 'le': str(bucket)}
                            labels_str = self._format_labels(bucket_labels)
                            lines.append(f"{name}_bucket{labels_str} {cumulative}")

                        # +Inf bucket
                        inf_labels = {**labels, 'le': '+Inf'}
                        labels_str = self._format_labels(inf_labels)
                        lines.append(f"{name}_bucket{labels_str} {buckets[float('inf')]}")

                        # Sum and count
                        labels_str = self._format_labels(labels)
                        lines.append(f"{name}_sum{labels_str} {metric.sum_value.get(key, 0)}")
                        lines.append(f"{name}_count{labels_str} {metric.count_value.get(key, 0)}")

                lines.append("")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON"""
        result = {}

        with self.lock:
            for name, metric in self.metrics.items():
                result[name] = {
                    'type': metric.type.value,
                    'help': metric.help,
                    'values': {}
                }

                if metric.type == MetricType.COUNTER or metric.type == MetricType.GAUGE:
                    for key, mv in metric.values.items():
                        labels = json.loads(key) if key else {}
                        result[name]['values'][key] = {
                            'value': mv.value,
                            'labels': labels,
                            'timestamp': mv.timestamp
                        }

                elif metric.type == MetricType.HISTOGRAM:
                    for key in metric.bucket_counts:
                        result[name]['values'][key] = {
                            'buckets': {str(k): v for k, v in metric.bucket_counts[key].items()},
                            'sum': metric.sum_value.get(key, 0),
                            'count': metric.count_value.get(key, 0)
                        }

        return result

    def _format_labels(self, labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


class SystemMetricsCollector:
    """Collects system metrics"""

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry

        # Define system metrics
        self.cpu_usage = registry.gauge(
            "taos_system_cpu_percent",
            "CPU usage percentage"
        )
        self.memory_usage = registry.gauge(
            "taos_system_memory_bytes",
            "Memory usage in bytes",
            ["type"]
        )
        self.memory_percent = registry.gauge(
            "taos_system_memory_percent",
            "Memory usage percentage"
        )
        self.thread_count = registry.gauge(
            "taos_system_threads",
            "Number of active threads"
        )
        self.uptime = registry.gauge(
            "taos_system_uptime_seconds",
            "System uptime in seconds"
        )

        self.start_time = time.time()

    def collect(self):
        """Collect system metrics"""
        try:
            import os

            # Thread count
            self.thread_count.set(threading.active_count())

            # Uptime
            self.uptime.set(time.time() - self.start_time)

            # Try to get more detailed metrics if psutil is available
            try:
                import psutil
                process = psutil.Process(os.getpid())

                self.cpu_usage.set(process.cpu_percent())

                mem_info = process.memory_info()
                self.memory_usage.set(mem_info.rss, {'type': 'rss'})
                self.memory_usage.set(mem_info.vms, {'type': 'vms'})

                self.memory_percent.set(process.memory_percent())

            except ImportError:
                pass

        except Exception as e:
            print(f"System metrics collection error: {e}")


class ApplicationMetricsCollector:
    """Collects TAOS application metrics"""

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry

        # HTTP metrics
        self.http_requests = registry.counter(
            "taos_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        self.http_duration = registry.histogram(
            "taos_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        )

        # Simulation metrics
        self.simulation_steps = registry.counter(
            "taos_simulation_steps_total",
            "Total simulation steps"
        )
        self.simulation_time = registry.gauge(
            "taos_simulation_time_seconds",
            "Current simulation time"
        )

        # State metrics
        self.water_level = registry.gauge(
            "taos_state_water_level_meters",
            "Current water level in meters"
        )
        self.flow_velocity = registry.gauge(
            "taos_state_flow_velocity_mps",
            "Current flow velocity in m/s"
        )
        self.froude_number = registry.gauge(
            "taos_state_froude_number",
            "Current Froude number"
        )
        self.temperature_sun = registry.gauge(
            "taos_state_temperature_celsius",
            "Temperature in Celsius",
            ["side"]
        )
        self.vibration = registry.gauge(
            "taos_state_vibration_mm",
            "Vibration amplitude in mm"
        )
        self.joint_gap = registry.gauge(
            "taos_state_joint_gap_mm",
            "Expansion joint gap in mm"
        )

        # Safety metrics
        self.safety_level = registry.gauge(
            "taos_safety_level",
            "Current safety level (1-5)"
        )
        self.active_alerts = registry.gauge(
            "taos_alerts_active",
            "Number of active alerts",
            ["severity"]
        )
        self.alert_total = registry.counter(
            "taos_alerts_total",
            "Total alerts generated",
            ["severity", "category"]
        )

        # Control metrics
        self.control_actions = registry.counter(
            "taos_control_actions_total",
            "Total control actions",
            ["type"]
        )
        self.scenarios_detected = registry.counter(
            "taos_scenarios_detected_total",
            "Total scenarios detected",
            ["scenario"]
        )

        # Database metrics
        self.db_queries = registry.counter(
            "taos_db_queries_total",
            "Total database queries",
            ["operation"]
        )
        self.db_query_duration = registry.histogram(
            "taos_db_query_duration_seconds",
            "Database query duration",
            ["operation"]
        )

    def update_state(self, state: Dict[str, Any]):
        """Update state metrics from simulation state"""
        self.simulation_time.set(state.get('time', 0))
        self.water_level.set(state.get('h', 0))
        self.flow_velocity.set(state.get('v', 0))
        self.froude_number.set(state.get('fr', 0))
        self.temperature_sun.set(state.get('T_sun', 0), {'side': 'sun'})
        self.temperature_sun.set(state.get('T_shade', 0), {'side': 'shade'})
        self.vibration.set(state.get('vib_amp', 0))
        self.joint_gap.set(state.get('joint_gap', 0))

    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests.inc(labels={'method': method, 'endpoint': endpoint, 'status': str(status)})
        self.http_duration.observe(duration, labels={'method': method, 'endpoint': endpoint})


class PerformanceMonitor:
    """
    Main performance monitor for TAOS
    """

    def __init__(self):
        self.registry = MetricsRegistry()
        self.system_collector = SystemMetricsCollector(self.registry)
        self.app_collector = ApplicationMetricsCollector(self.registry)

        self.running = False
        self.collection_interval = 5.0  # seconds
        self.thread: Optional[threading.Thread] = None

        # History for graphs
        self.history_size = 1000
        self.metric_history: Dict[str, deque] = {}

    def start(self):
        """Start background collection"""
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop background collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.collection_interval + 1)

    def _collection_loop(self):
        """Background collection loop"""
        while self.running:
            try:
                self.system_collector.collect()
                self._record_history()
            except Exception as e:
                print(f"Metrics collection error: {e}")

            time.sleep(self.collection_interval)

    def _record_history(self):
        """Record metrics history for graphs"""
        timestamp = time.time()

        for name, metric in self.registry.metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = deque(maxlen=self.history_size)

            if metric.type in (MetricType.COUNTER, MetricType.GAUGE):
                for key, mv in metric.values.items():
                    self.metric_history[name].append({
                        'timestamp': timestamp,
                        'value': mv.value,
                        'labels': mv.labels
                    })

    def update_state(self, state: Dict[str, Any]):
        """Update application metrics from state"""
        self.app_collector.update_state(state)
        self.app_collector.simulation_steps.inc()

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return self.registry.export_prometheus()

    def get_json_metrics(self) -> Dict[str, Any]:
        """Get metrics as JSON"""
        return self.registry.export_json()

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        metrics = self.registry.export_json()

        summary = {
            'timestamp': datetime.now().isoformat(),
            'metric_count': len(metrics),
            'metrics': {}
        }

        for name, data in metrics.items():
            if data['values']:
                if data['type'] in ('counter', 'gauge'):
                    values = [v['value'] for v in data['values'].values()]
                    summary['metrics'][name] = {
                        'type': data['type'],
                        'current': values[0] if len(values) == 1 else values
                    }
                elif data['type'] == 'histogram':
                    for key, v in data['values'].items():
                        summary['metrics'][name] = {
                            'type': 'histogram',
                            'count': v['count'],
                            'sum': v['sum'],
                            'avg': v['sum'] / v['count'] if v['count'] > 0 else 0
                        }

        return summary

    def get_history(self, metric_name: str, minutes: int = 10) -> List[Dict]:
        """Get metric history"""
        if metric_name not in self.metric_history:
            return []

        cutoff = time.time() - (minutes * 60)
        return [
            h for h in self.metric_history[metric_name]
            if h['timestamp'] >= cutoff
        ]


# Global instance
_monitor = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


# Flask middleware for request tracking
def create_request_tracker(app):
    """Create Flask request tracking middleware"""
    monitor = get_monitor()

    @app.before_request
    def before_request():
        from flask import request, g
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        from flask import request, g
        duration = time.time() - getattr(g, 'start_time', time.time())

        # Simplify endpoint for metrics
        endpoint = request.endpoint or request.path
        if len(endpoint) > 50:
            endpoint = endpoint[:50]

        monitor.app_collector.record_http_request(
            request.method,
            endpoint,
            response.status_code,
            duration
        )

        return response

    return monitor


if __name__ == "__main__":
    # Test monitoring system
    print("=== Performance Monitoring Test ===")

    monitor = PerformanceMonitor()
    monitor.start()

    # Simulate some metrics
    print("\n1. Recording metrics...")
    for i in range(5):
        monitor.update_state({
            'time': i * 0.5,
            'h': 4.0 + i * 0.1,
            'v': 2.0,
            'fr': 0.32,
            'T_sun': 30 + i,
            'T_shade': 25,
            'vib_amp': 1.0,
            'joint_gap': 20
        })
        monitor.app_collector.http_requests.inc(
            labels={'method': 'GET', 'endpoint': '/api/state', 'status': '200'}
        )
        time.sleep(0.1)

    # Get Prometheus format
    print("\n2. Prometheus format (sample):")
    prom = monitor.get_prometheus_metrics()
    print(prom[:1000] + "...")

    # Get JSON format
    print("\n3. JSON summary:")
    summary = monitor.get_summary()
    print(f"   Metrics count: {summary['metric_count']}")
    for name, data in list(summary['metrics'].items())[:5]:
        print(f"   - {name}: {data}")

    monitor.stop()
    print("\nPerformance monitoring test completed!")
