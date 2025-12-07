"""
TAOS Visualization Enhancement Module
可视化增强模块 - 实时仪表盘、图表生成、SVG可视化

Features:
- Real-time dashboard data API
- Time series chart data generation
- Gauge visualization data
- Scenario timeline visualization
- Risk heatmap generation
- System topology visualization
- Alert notification management
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class ChartType(Enum):
    """图表类型"""
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    PIE = "pie"
    SCATTER = "scatter"
    TIMELINE = "timeline"


class ColorScheme(Enum):
    """颜色方案"""
    NORMAL = "#22c55e"      # 绿色
    WARNING = "#f59e0b"     # 橙色
    DANGER = "#ef4444"      # 红色
    INFO = "#3b82f6"        # 蓝色
    NEUTRAL = "#6b7280"     # 灰色


@dataclass
class GaugeConfig:
    """仪表盘配置"""
    name: str
    unit: str
    min_value: float
    max_value: float
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    danger_low: Optional[float] = None
    danger_high: Optional[float] = None
    decimals: int = 2


@dataclass
class ChartSeries:
    """图表数据系列"""
    name: str
    data: List[Tuple[float, float]]  # [(timestamp, value), ...]
    color: str = ColorScheme.INFO.value
    type: str = "line"
    yAxis: int = 0


@dataclass
class AlertNotification:
    """告警通知"""
    id: str
    timestamp: datetime
    level: str           # info, warning, danger
    title: str
    message: str
    source: str
    acknowledged: bool = False


class DashboardDataGenerator:
    """仪表盘数据生成器"""

    def __init__(self):
        # 仪表盘配置
        self.gauge_configs = {
            'h': GaugeConfig(
                name="水位",
                unit="m",
                min_value=0,
                max_value=8,
                warning_low=2.5,
                warning_high=5.5,
                danger_low=1.5,
                danger_high=6.5
            ),
            'fr': GaugeConfig(
                name="弗劳德数",
                unit="",
                min_value=0,
                max_value=1.2,
                warning_high=0.7,
                danger_high=0.85
            ),
            'Q_in': GaugeConfig(
                name="进水流量",
                unit="m³/s",
                min_value=0,
                max_value=150,
                warning_low=40,
                warning_high=120,
                danger_high=140
            ),
            'Q_out': GaugeConfig(
                name="出水流量",
                unit="m³/s",
                min_value=0,
                max_value=150,
                warning_low=40,
                warning_high=120,
                danger_high=140
            ),
            'T_sun': GaugeConfig(
                name="阳面温度",
                unit="°C",
                min_value=-10,
                max_value=60,
                warning_high=45,
                danger_high=55
            ),
            'T_shade': GaugeConfig(
                name="阴面温度",
                unit="°C",
                min_value=-10,
                max_value=50,
                warning_high=40,
                danger_high=45
            ),
            'T_delta': GaugeConfig(
                name="温差",
                unit="°C",
                min_value=0,
                max_value=30,
                warning_high=12,
                danger_high=18
            ),
            'vib_amp': GaugeConfig(
                name="振动幅度",
                unit="mm",
                min_value=0,
                max_value=100,
                warning_high=40,
                danger_high=60
            ),
            'joint_gap': GaugeConfig(
                name="伸缩缝间隙",
                unit="mm",
                min_value=0,
                max_value=50,
                warning_low=8,
                warning_high=35,
                danger_low=5,
                danger_high=40
            ),
            'ground_accel': GaugeConfig(
                name="地面加速度",
                unit="g",
                min_value=0,
                max_value=1.0,
                warning_high=0.2,
                danger_high=0.4
            )
        }

        # 历史数据缓存
        self.history_cache: Dict[str, deque] = {}
        self.cache_size = 500

        # 告警队列
        self.alerts: deque = deque(maxlen=100)

    def get_gauge_data(self, variable: str, value: float) -> Dict[str, Any]:
        """生成仪表盘数据"""
        config = self.gauge_configs.get(variable)
        if not config:
            return {'error': f'Unknown variable: {variable}'}

        # 计算状态颜色
        color = self._get_value_color(value, config)
        status = self._get_value_status(value, config)

        # 计算百分比
        range_val = config.max_value - config.min_value
        percentage = (value - config.min_value) / range_val * 100 if range_val > 0 else 0
        percentage = max(0, min(100, percentage))

        return {
            'variable': variable,
            'name': config.name,
            'value': round(value, config.decimals),
            'unit': config.unit,
            'percentage': round(percentage, 1),
            'color': color,
            'status': status,
            'min': config.min_value,
            'max': config.max_value,
            'warning_low': config.warning_low,
            'warning_high': config.warning_high,
            'danger_low': config.danger_low,
            'danger_high': config.danger_high,
            'zones': self._get_gauge_zones(config)
        }

    def _get_value_color(self, value: float, config: GaugeConfig) -> str:
        """获取值对应的颜色"""
        if config.danger_low and value < config.danger_low:
            return ColorScheme.DANGER.value
        if config.danger_high and value > config.danger_high:
            return ColorScheme.DANGER.value
        if config.warning_low and value < config.warning_low:
            return ColorScheme.WARNING.value
        if config.warning_high and value > config.warning_high:
            return ColorScheme.WARNING.value
        return ColorScheme.NORMAL.value

    def _get_value_status(self, value: float, config: GaugeConfig) -> str:
        """获取值状态"""
        if config.danger_low and value < config.danger_low:
            return 'danger'
        if config.danger_high and value > config.danger_high:
            return 'danger'
        if config.warning_low and value < config.warning_low:
            return 'warning'
        if config.warning_high and value > config.warning_high:
            return 'warning'
        return 'normal'

    def _get_gauge_zones(self, config: GaugeConfig) -> List[Dict[str, Any]]:
        """获取仪表盘区域划分"""
        zones = []
        range_val = config.max_value - config.min_value

        # 危险区 (低)
        if config.danger_low:
            zones.append({
                'from': config.min_value,
                'to': config.danger_low,
                'color': ColorScheme.DANGER.value
            })

        # 警告区 (低)
        if config.warning_low:
            from_val = config.danger_low or config.min_value
            zones.append({
                'from': from_val,
                'to': config.warning_low,
                'color': ColorScheme.WARNING.value
            })

        # 正常区
        normal_from = config.warning_low or config.danger_low or config.min_value
        normal_to = config.warning_high or config.danger_high or config.max_value
        zones.append({
            'from': normal_from,
            'to': normal_to,
            'color': ColorScheme.NORMAL.value
        })

        # 警告区 (高)
        if config.warning_high:
            to_val = config.danger_high or config.max_value
            zones.append({
                'from': config.warning_high,
                'to': to_val,
                'color': ColorScheme.WARNING.value
            })

        # 危险区 (高)
        if config.danger_high:
            zones.append({
                'from': config.danger_high,
                'to': config.max_value,
                'color': ColorScheme.DANGER.value
            })

        return zones

    def update_history(self, state: Dict[str, Any], timestamp: datetime = None):
        """更新历史数据缓存"""
        ts = timestamp or datetime.now()
        ts_ms = ts.timestamp() * 1000  # JavaScript时间戳

        for var in ['h', 'Q_in', 'Q_out', 'fr', 'v', 'T_sun', 'T_shade',
                    'vib_amp', 'joint_gap', 'ground_accel']:
            if var in state:
                if var not in self.history_cache:
                    self.history_cache[var] = deque(maxlen=self.cache_size)
                self.history_cache[var].append((ts_ms, state[var]))

        # 计算派生量
        if 'T_sun' in state and 'T_shade' in state:
            T_delta = abs(state['T_sun'] - state['T_shade'])
            if 'T_delta' not in self.history_cache:
                self.history_cache['T_delta'] = deque(maxlen=self.cache_size)
            self.history_cache['T_delta'].append((ts_ms, T_delta))

    def get_chart_data(self, variables: List[str],
                       minutes: int = 10) -> Dict[str, Any]:
        """获取图表数据"""
        cutoff = (datetime.now() - timedelta(minutes=minutes)).timestamp() * 1000
        series = []

        colors = [
            "#3b82f6", "#22c55e", "#f59e0b", "#ef4444",
            "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"
        ]

        for i, var in enumerate(variables):
            if var not in self.history_cache:
                continue

            data = [(ts, val) for ts, val in self.history_cache[var] if ts >= cutoff]

            config = self.gauge_configs.get(var)
            name = config.name if config else var
            unit = config.unit if config else ""

            series.append({
                'name': f"{name} ({unit})" if unit else name,
                'data': data,
                'color': colors[i % len(colors)],
                'variable': var
            })

        return {
            'series': series,
            'timeRange': {
                'from': cutoff,
                'to': datetime.now().timestamp() * 1000
            },
            'updateInterval': 500
        }

    def get_multi_axis_chart(self, groups: List[List[str]],
                              minutes: int = 10) -> Dict[str, Any]:
        """获取多轴图表数据"""
        cutoff = (datetime.now() - timedelta(minutes=minutes)).timestamp() * 1000
        series = []
        yAxes = []

        colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]

        for axis_idx, group in enumerate(groups):
            axis_vars = []
            for var in group:
                if var in self.history_cache:
                    config = self.gauge_configs.get(var)
                    data = [(ts, val) for ts, val in self.history_cache[var] if ts >= cutoff]
                    series.append({
                        'name': config.name if config else var,
                        'data': data,
                        'color': colors[axis_idx % len(colors)],
                        'yAxis': axis_idx
                    })
                    axis_vars.append(var)

            if axis_vars:
                config = self.gauge_configs.get(axis_vars[0])
                yAxes.append({
                    'title': config.name if config else axis_vars[0],
                    'unit': config.unit if config else "",
                    'opposite': axis_idx > 0
                })

        return {
            'series': series,
            'yAxes': yAxes,
            'timeRange': {
                'from': cutoff,
                'to': datetime.now().timestamp() * 1000
            }
        }

    def get_scenario_timeline(self, scenario_history: List[Dict[str, Any]],
                               hours: int = 24) -> Dict[str, Any]:
        """生成场景时间线数据"""
        cutoff = datetime.now() - timedelta(hours=hours)
        events = []

        scenario_colors = {
            'S1.1': '#3b82f6', 'S1.2': '#60a5fa', 'S1.3': '#93c5fd',
            'S2.1': '#22c55e', 'S2.2': '#4ade80',
            'S3.1': '#f59e0b', 'S3.2': '#fbbf24', 'S3.3': '#fcd34d', 'S3.4': '#fde68a',
            'S4.1': '#ef4444', 'S4.2': '#f87171', 'S4.3': '#fca5a5',
            'S5.1': '#8b5cf6', 'S5.2': '#a78bfa',
            'S6.1': '#ec4899', 'S6.2': '#f472b6', 'S6.3': '#f9a8d4', 'S6.4': '#fbcfe8',
            'NORMAL': '#6b7280',
            'MULTI_PHYSICS': '#0ea5e9'
        }

        for entry in scenario_history:
            timestamp = entry.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            if timestamp and timestamp >= cutoff:
                scenarios = entry.get('active_scenarios', [])
                for scenario in scenarios:
                    events.append({
                        'time': timestamp.timestamp() * 1000,
                        'scenario': scenario,
                        'color': scenario_colors.get(scenario, '#6b7280'),
                        'risk_level': entry.get('risk_level', 'INFO')
                    })

        return {
            'events': events,
            'timeRange': {
                'from': cutoff.timestamp() * 1000,
                'to': datetime.now().timestamp() * 1000
            },
            'scenarioColors': scenario_colors
        }

    def get_risk_heatmap(self, history: List[Dict[str, Any]],
                          hours: int = 24) -> Dict[str, Any]:
        """生成风险热力图数据"""
        # 按小时聚合
        hour_buckets: Dict[int, Dict[str, int]] = {}
        cutoff = datetime.now() - timedelta(hours=hours)

        for entry in history:
            timestamp = entry.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            if timestamp and timestamp >= cutoff:
                hour = timestamp.hour
                if hour not in hour_buckets:
                    hour_buckets[hour] = {'INFO': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}

                risk = entry.get('risk_level', 'INFO')
                if risk in hour_buckets[hour]:
                    hour_buckets[hour][risk] += 1

        # 转换为热力图格式
        heatmap_data = []
        risk_weights = {'INFO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}

        for hour in range(24):
            if hour in hour_buckets:
                total = sum(hour_buckets[hour].values())
                if total > 0:
                    weighted_sum = sum(
                        count * risk_weights[risk]
                        for risk, count in hour_buckets[hour].items()
                    )
                    intensity = weighted_sum / (total * 4)  # 归一化到0-1
                else:
                    intensity = 0
            else:
                intensity = 0

            heatmap_data.append({
                'hour': hour,
                'intensity': intensity,
                'distribution': hour_buckets.get(hour, {})
            })

        return {
            'data': heatmap_data,
            'maxIntensity': 1.0,
            'colorScale': [
                '#22c55e',  # 0 - 绿色
                '#84cc16',  # 0.25
                '#fbbf24',  # 0.5 - 黄色
                '#f97316',  # 0.75 - 橙色
                '#ef4444'   # 1.0 - 红色
            ]
        }

    def get_system_topology(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成系统拓扑可视化数据"""
        nodes = [
            {
                'id': 'upstream',
                'label': '上游水源',
                'type': 'source',
                'x': 50, 'y': 100,
                'status': 'normal'
            },
            {
                'id': 'inlet',
                'label': '进水口',
                'type': 'control',
                'x': 150, 'y': 100,
                'status': self._get_flow_status(state.get('Q_in', 80)),
                'value': f"{state.get('Q_in', 80):.1f} m³/s"
            },
            {
                'id': 'aqueduct_1',
                'label': '渠段1',
                'type': 'channel',
                'x': 250, 'y': 100,
                'status': self._get_channel_status(state),
                'value': f"h={state.get('h', 4):.2f}m"
            },
            {
                'id': 'aqueduct_2',
                'label': '渠段2',
                'type': 'channel',
                'x': 350, 'y': 100,
                'status': self._get_channel_status(state)
            },
            {
                'id': 'aqueduct_3',
                'label': '渠段3',
                'type': 'channel',
                'x': 450, 'y': 100,
                'status': self._get_channel_status(state)
            },
            {
                'id': 'outlet',
                'label': '出水口',
                'type': 'control',
                'x': 550, 'y': 100,
                'status': self._get_flow_status(state.get('Q_out', 80)),
                'value': f"{state.get('Q_out', 80):.1f} m³/s"
            },
            {
                'id': 'downstream',
                'label': '下游',
                'type': 'sink',
                'x': 650, 'y': 100,
                'status': 'normal'
            },
            {
                'id': 'sensor_h',
                'label': '水位传感器',
                'type': 'sensor',
                'x': 250, 'y': 50,
                'status': 'active',
                'value': f"{state.get('h', 4):.2f}m"
            },
            {
                'id': 'sensor_t',
                'label': '温度传感器',
                'type': 'sensor',
                'x': 350, 'y': 50,
                'status': 'active',
                'value': f"ΔT={abs(state.get('T_sun', 25) - state.get('T_shade', 25)):.1f}°C"
            },
            {
                'id': 'sensor_v',
                'label': '振动传感器',
                'type': 'sensor',
                'x': 450, 'y': 50,
                'status': 'active',
                'value': f"{state.get('vib_amp', 0):.1f}mm"
            }
        ]

        edges = [
            {'from': 'upstream', 'to': 'inlet'},
            {'from': 'inlet', 'to': 'aqueduct_1'},
            {'from': 'aqueduct_1', 'to': 'aqueduct_2'},
            {'from': 'aqueduct_2', 'to': 'aqueduct_3'},
            {'from': 'aqueduct_3', 'to': 'outlet'},
            {'from': 'outlet', 'to': 'downstream'},
            {'from': 'sensor_h', 'to': 'aqueduct_1', 'type': 'monitor'},
            {'from': 'sensor_t', 'to': 'aqueduct_2', 'type': 'monitor'},
            {'from': 'sensor_v', 'to': 'aqueduct_3', 'type': 'monitor'}
        ]

        return {
            'nodes': nodes,
            'edges': edges,
            'timestamp': datetime.now().isoformat()
        }

    def _get_flow_status(self, flow: float) -> str:
        """获取流量状态"""
        if flow > 120 or flow < 40:
            return 'danger'
        elif flow > 100 or flow < 60:
            return 'warning'
        return 'normal'

    def _get_channel_status(self, state: Dict[str, Any]) -> str:
        """获取渠道状态"""
        h = state.get('h', 4.0)
        fr = state.get('fr', 0.5)
        vib = state.get('vib_amp', 0)

        if h < 2 or h > 6 or fr > 0.85 or vib > 60:
            return 'danger'
        elif h < 3 or h > 5 or fr > 0.7 or vib > 40:
            return 'warning'
        return 'normal'

    def add_alert(self, level: str, title: str, message: str, source: str):
        """添加告警通知"""
        alert = AlertNotification(
            id=f"alert_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            source=source
        )
        self.alerts.appendleft(alert)

    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取告警列表"""
        return [
            {
                'id': a.id,
                'timestamp': a.timestamp.isoformat(),
                'level': a.level,
                'title': a.title,
                'message': a.message,
                'source': a.source,
                'acknowledged': a.acknowledged
            }
            for a in list(self.alerts)[:limit]
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False


class DashboardLayout:
    """仪表盘布局配置"""

    @staticmethod
    def get_default_layout() -> Dict[str, Any]:
        """获取默认布局配置"""
        return {
            'rows': [
                {
                    'height': '30%',
                    'widgets': [
                        {'type': 'gauge', 'variable': 'h', 'width': '16.66%'},
                        {'type': 'gauge', 'variable': 'fr', 'width': '16.66%'},
                        {'type': 'gauge', 'variable': 'Q_in', 'width': '16.66%'},
                        {'type': 'gauge', 'variable': 'Q_out', 'width': '16.66%'},
                        {'type': 'gauge', 'variable': 'T_delta', 'width': '16.66%'},
                        {'type': 'gauge', 'variable': 'vib_amp', 'width': '16.66%'}
                    ]
                },
                {
                    'height': '40%',
                    'widgets': [
                        {
                            'type': 'multi_chart',
                            'groups': [['h', 'fr'], ['Q_in', 'Q_out']],
                            'width': '60%',
                            'minutes': 10
                        },
                        {
                            'type': 'topology',
                            'width': '40%'
                        }
                    ]
                },
                {
                    'height': '30%',
                    'widgets': [
                        {'type': 'scenario_timeline', 'width': '50%', 'hours': 1},
                        {'type': 'alerts', 'width': '25%'},
                        {'type': 'status_summary', 'width': '25%'}
                    ]
                }
            ],
            'refreshInterval': 500,
            'theme': 'dark'
        }

    @staticmethod
    def get_monitoring_layout() -> Dict[str, Any]:
        """获取监控专用布局"""
        return {
            'rows': [
                {
                    'height': '50%',
                    'widgets': [
                        {
                            'type': 'multi_chart',
                            'groups': [['h'], ['Q_in', 'Q_out']],
                            'width': '50%',
                            'minutes': 30
                        },
                        {
                            'type': 'multi_chart',
                            'groups': [['T_sun', 'T_shade'], ['vib_amp']],
                            'width': '50%',
                            'minutes': 30
                        }
                    ]
                },
                {
                    'height': '50%',
                    'widgets': [
                        {'type': 'scenario_timeline', 'width': '40%', 'hours': 24},
                        {'type': 'risk_heatmap', 'width': '30%', 'hours': 24},
                        {'type': 'alerts', 'width': '30%'}
                    ]
                }
            ],
            'refreshInterval': 1000,
            'theme': 'dark'
        }


class SVGGenerator:
    """SVG可视化生成器"""

    @staticmethod
    def generate_gauge_svg(value: float, config: GaugeConfig,
                           size: int = 200) -> str:
        """生成仪表盘SVG"""
        center = size / 2
        radius = size / 2 - 20
        start_angle = 135
        end_angle = 405
        total_angle = end_angle - start_angle

        # 计算指针角度
        range_val = config.max_value - config.min_value
        percentage = (value - config.min_value) / range_val if range_val > 0 else 0
        percentage = max(0, min(1, percentage))
        pointer_angle = start_angle + percentage * total_angle

        # 获取颜色
        if config.danger_high and value > config.danger_high:
            color = ColorScheme.DANGER.value
        elif config.danger_low and value < config.danger_low:
            color = ColorScheme.DANGER.value
        elif config.warning_high and value > config.warning_high:
            color = ColorScheme.WARNING.value
        elif config.warning_low and value < config.warning_low:
            color = ColorScheme.WARNING.value
        else:
            color = ColorScheme.NORMAL.value

        # 生成SVG
        svg = f'''<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <!-- 背景弧 -->
  <path d="{SVGGenerator._arc_path(center, center, radius, start_angle, end_angle)}"
        fill="none" stroke="#374151" stroke-width="15" stroke-linecap="round"/>

  <!-- 数值弧 -->
  <path d="{SVGGenerator._arc_path(center, center, radius, start_angle, pointer_angle)}"
        fill="none" stroke="{color}" stroke-width="15" stroke-linecap="round"/>

  <!-- 指针 -->
  <line x1="{center}" y1="{center}"
        x2="{center + (radius - 30) * math.cos(math.radians(pointer_angle - 90))}"
        y2="{center + (radius - 30) * math.sin(math.radians(pointer_angle - 90))}"
        stroke="{color}" stroke-width="3"/>

  <!-- 中心圆 -->
  <circle cx="{center}" cy="{center}" r="8" fill="{color}"/>

  <!-- 数值文本 -->
  <text x="{center}" y="{center + 35}" text-anchor="middle"
        font-size="24" font-weight="bold" fill="{color}">
    {value:.{config.decimals}f}
  </text>
  <text x="{center}" y="{center + 55}" text-anchor="middle"
        font-size="14" fill="#9ca3af">
    {config.unit}
  </text>

  <!-- 名称 -->
  <text x="{center}" y="{size - 10}" text-anchor="middle"
        font-size="12" fill="#d1d5db">
    {config.name}
  </text>
</svg>'''
        return svg

    @staticmethod
    def _arc_path(cx: float, cy: float, r: float,
                  start_deg: float, end_deg: float) -> str:
        """生成SVG弧形路径"""
        start_rad = math.radians(start_deg - 90)
        end_rad = math.radians(end_deg - 90)

        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)

        large_arc = 1 if (end_deg - start_deg) > 180 else 0

        return f"M {x1} {y1} A {r} {r} 0 {large_arc} 1 {x2} {y2}"

    @staticmethod
    def generate_mini_chart_svg(data: List[Tuple[float, float]],
                                width: int = 150, height: int = 50,
                                color: str = "#3b82f6") -> str:
        """生成迷你图表SVG"""
        if not data or len(data) < 2:
            return f'<svg width="{width}" height="{height}"></svg>'

        values = [v for _, v in data]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val or 1

        points = []
        for i, (_, val) in enumerate(data):
            x = (i / (len(data) - 1)) * width
            y = height - ((val - min_val) / range_val) * (height - 4) - 2
            points.append(f"{x},{y}")

        path = "M " + " L ".join(points)

        return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
</svg>'''


# 全局实例
_dashboard_instance: Optional[DashboardDataGenerator] = None


def get_dashboard() -> DashboardDataGenerator:
    """获取全局仪表盘实例"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = DashboardDataGenerator()
    return _dashboard_instance
