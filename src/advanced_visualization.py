#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.9 - Advanced Visualization System
团河渡槽自主运行系统 - 高级可视化模块

Features:
- Interactive 3D visualization data
- Real-time animated charts
- Heat maps and contour plots
- Sankey flow diagrams
- Geographic mapping (GIS)
- Custom dashboard builder
- Export to various formats
- Responsive chart configurations
"""

import time
import json
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from enum import Enum
from collections import deque
from pathlib import Path


class ChartType(Enum):
    """Chart types"""
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    DONUT = "donut"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SANKEY = "sankey"
    TREEMAP = "treemap"
    RADAR = "radar"
    CANDLESTICK = "candlestick"
    WATERFALL = "waterfall"
    GEO = "geo"
    THREE_D = "3d"


class ColorScheme(Enum):
    """Color schemes for visualization"""
    DEFAULT = "default"
    COOL = "cool"
    WARM = "warm"
    RAINBOW = "rainbow"
    GRAYSCALE = "grayscale"
    SAFETY = "safety"  # Green-Yellow-Red
    WATER = "water"    # Blue gradients
    THERMAL = "thermal"  # Blue-Red


class ExportFormat(Enum):
    """Export formats"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class ChartConfig:
    """Chart configuration"""
    chart_id: str
    chart_type: ChartType
    title: str
    subtitle: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    width: int = 800
    height: int = 400
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    animate: bool = True
    legend_position: str = "bottom"
    show_grid: bool = True
    show_tooltip: bool = True
    responsive: bool = True
    theme: str = "light"
    custom_colors: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chart_id': self.chart_id,
            'chart_type': self.chart_type.value,
            'title': self.title,
            'subtitle': self.subtitle,
            'x_axis_label': self.x_axis_label,
            'y_axis_label': self.y_axis_label,
            'width': self.width,
            'height': self.height,
            'color_scheme': self.color_scheme.value,
            'animate': self.animate,
            'legend_position': self.legend_position,
            'show_grid': self.show_grid,
            'show_tooltip': self.show_tooltip,
            'responsive': self.responsive,
            'theme': self.theme,
            'custom_colors': self.custom_colors,
            'options': self.options
        }


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str  # chart, stat, table, map, etc.
    title: str
    row: int
    col: int
    width: int  # Grid units
    height: int  # Grid units
    refresh_interval: int = 5  # seconds
    data_source: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'widget_id': self.widget_id,
            'widget_type': self.widget_type,
            'title': self.title,
            'row': self.row,
            'col': self.col,
            'width': self.width,
            'height': self.height,
            'refresh_interval': self.refresh_interval,
            'data_source': self.data_source,
            'config': self.config
        }


class ColorPalettes:
    """Predefined color palettes"""

    PALETTES = {
        ColorScheme.DEFAULT: ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272'],
        ColorScheme.COOL: ['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8', '#48cae4', '#023e8a'],
        ColorScheme.WARM: ['#ffbe0b', '#fb5607', '#ff006e', '#8338ec', '#3a86ff', '#e63946'],
        ColorScheme.RAINBOW: ['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#0000ff', '#4b0082'],
        ColorScheme.GRAYSCALE: ['#212529', '#495057', '#6c757d', '#adb5bd', '#ced4da', '#dee2e6'],
        ColorScheme.SAFETY: ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6c757d', '#17a2b8'],
        ColorScheme.WATER: ['#03045e', '#0077b6', '#00b4d8', '#90e0ef', '#caf0f8', '#48cae4'],
        ColorScheme.THERMAL: ['#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff8000', '#ff0000']
    }

    @classmethod
    def get_palette(cls, scheme: ColorScheme) -> List[str]:
        return cls.PALETTES.get(scheme, cls.PALETTES[ColorScheme.DEFAULT])

    @classmethod
    def get_gradient(cls, scheme: ColorScheme, steps: int = 10) -> List[str]:
        """Generate gradient colors"""
        palette = cls.get_palette(scheme)
        if len(palette) < 2:
            return palette * steps

        gradient = []
        segments = len(palette) - 1
        per_segment = steps // segments

        for i in range(segments):
            start_color = palette[i]
            end_color = palette[i + 1]

            for j in range(per_segment):
                ratio = j / per_segment
                color = cls._interpolate_color(start_color, end_color, ratio)
                gradient.append(color)

        return gradient[:steps]

    @staticmethod
    def _interpolate_color(color1: str, color2: str, ratio: float) -> str:
        """Interpolate between two hex colors"""
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)

        return f"#{r:02x}{g:02x}{b:02x}"


class ChartDataGenerator:
    """
    Generate chart data in various formats
    """

    @staticmethod
    def generate_line_chart(data: List[Dict[str, Any]], x_field: str,
                           y_fields: List[str], config: ChartConfig) -> Dict[str, Any]:
        """Generate line chart data"""
        series = []
        for y_field in y_fields:
            series.append({
                'name': y_field,
                'type': 'line',
                'data': [[d.get(x_field), d.get(y_field)] for d in data],
                'smooth': True
            })

        return {
            'config': config.to_dict(),
            'xAxis': {'type': 'value' if isinstance(data[0].get(x_field), (int, float)) else 'category'},
            'yAxis': {'type': 'value', 'name': config.y_axis_label},
            'series': series,
            'legend': {'data': y_fields}
        }

    @staticmethod
    def generate_area_chart(data: List[Dict[str, Any]], x_field: str,
                           y_fields: List[str], config: ChartConfig,
                           stacked: bool = False) -> Dict[str, Any]:
        """Generate area chart data"""
        series = []
        for i, y_field in enumerate(y_fields):
            series_data = {
                'name': y_field,
                'type': 'line',
                'areaStyle': {'opacity': 0.7 - i * 0.1},
                'data': [[d.get(x_field), d.get(y_field)] for d in data],
                'smooth': True
            }
            if stacked:
                series_data['stack'] = 'total'
            series.append(series_data)

        return {
            'config': config.to_dict(),
            'xAxis': {'type': 'category', 'data': [d.get(x_field) for d in data]},
            'yAxis': {'type': 'value'},
            'series': series
        }

    @staticmethod
    def generate_gauge_chart(value: float, min_val: float, max_val: float,
                            config: ChartConfig, thresholds: List[Tuple[float, str]] = None) -> Dict[str, Any]:
        """Generate gauge chart data"""
        if thresholds is None:
            thresholds = [(0.3, '#67e0e3'), (0.7, '#37a2da'), (1, '#fd666d')]

        return {
            'config': config.to_dict(),
            'series': [{
                'type': 'gauge',
                'min': min_val,
                'max': max_val,
                'data': [{'value': value, 'name': config.title}],
                'axisLine': {
                    'lineStyle': {
                        'color': thresholds
                    }
                },
                'pointer': {
                    'itemStyle': {
                        'color': 'auto'
                    }
                },
                'detail': {
                    'formatter': '{value}',
                    'fontSize': 20
                }
            }]
        }

    @staticmethod
    def generate_heatmap(data: List[List[float]], x_labels: List[str],
                        y_labels: List[str], config: ChartConfig) -> Dict[str, Any]:
        """Generate heatmap data"""
        # Convert to ECharts format
        heatmap_data = []
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                heatmap_data.append([j, i, value])

        # Get min/max for color mapping
        flat_data = [item[2] for item in heatmap_data]
        min_val = min(flat_data) if flat_data else 0
        max_val = max(flat_data) if flat_data else 1

        return {
            'config': config.to_dict(),
            'xAxis': {'type': 'category', 'data': x_labels},
            'yAxis': {'type': 'category', 'data': y_labels},
            'visualMap': {
                'min': min_val,
                'max': max_val,
                'calculable': True,
                'inRange': {
                    'color': ColorPalettes.get_gradient(config.color_scheme)
                }
            },
            'series': [{
                'type': 'heatmap',
                'data': heatmap_data,
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }

    @staticmethod
    def generate_sankey(nodes: List[Dict[str, Any]], links: List[Dict[str, Any]],
                       config: ChartConfig) -> Dict[str, Any]:
        """Generate Sankey diagram data"""
        return {
            'config': config.to_dict(),
            'series': [{
                'type': 'sankey',
                'data': nodes,
                'links': links,
                'emphasis': {
                    'focus': 'adjacency'
                },
                'lineStyle': {
                    'color': 'gradient',
                    'curveness': 0.5
                }
            }]
        }

    @staticmethod
    def generate_radar(indicators: List[Dict[str, Any]], data: List[Dict[str, Any]],
                      config: ChartConfig) -> Dict[str, Any]:
        """Generate radar chart data"""
        return {
            'config': config.to_dict(),
            'radar': {
                'indicator': indicators
            },
            'series': [{
                'type': 'radar',
                'data': data
            }]
        }

    @staticmethod
    def generate_3d_surface(x: List[float], y: List[float], z: List[List[float]],
                           config: ChartConfig) -> Dict[str, Any]:
        """Generate 3D surface data"""
        # Convert to ECharts GL format
        surface_data = []
        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                surface_data.append([xj, yi, z[i][j]])

        return {
            'config': config.to_dict(),
            'grid3D': {},
            'xAxis3D': {'type': 'value', 'name': config.x_axis_label},
            'yAxis3D': {'type': 'value', 'name': config.y_axis_label},
            'zAxis3D': {'type': 'value'},
            'series': [{
                'type': 'surface',
                'data': surface_data,
                'shading': 'color',
                'wireframe': {
                    'show': True
                }
            }]
        }


class GISVisualization:
    """
    Geographic Information System visualization
    """

    def __init__(self):
        # Tuanhe Aqueduct coordinates (approximate)
        self.aqueduct_path = [
            {'lng': 116.35, 'lat': 39.75, 'name': '起点'},
            {'lng': 116.36, 'lat': 39.76, 'name': '监测点1'},
            {'lng': 116.37, 'lat': 39.77, 'name': '监测点2'},
            {'lng': 116.38, 'lat': 39.78, 'name': '监测点3'},
            {'lng': 116.39, 'lat': 39.79, 'name': '终点'}
        ]

        # Monitoring stations
        self.stations = [
            {'id': 'S001', 'name': '上游监测站', 'lng': 116.35, 'lat': 39.75, 'type': 'flow'},
            {'id': 'S002', 'name': '中游监测站', 'lng': 116.37, 'lat': 39.77, 'type': 'multi'},
            {'id': 'S003', 'name': '下游监测站', 'lng': 116.39, 'lat': 39.79, 'type': 'flow'},
            {'id': 'S004', 'name': '气象站', 'lng': 116.36, 'lat': 39.76, 'type': 'weather'},
            {'id': 'S005', 'name': '地震监测点', 'lng': 116.38, 'lat': 39.78, 'type': 'seismic'}
        ]

    def get_map_data(self, include_realtime: bool = True,
                    state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get GIS map data"""
        map_data = {
            'center': {'lng': 116.37, 'lat': 39.77},
            'zoom': 14,
            'aqueduct_path': self.aqueduct_path,
            'stations': self.stations,
            'layers': {
                'aqueduct': True,
                'stations': True,
                'alerts': True,
                'weather': False
            }
        }

        if include_realtime and state:
            # Add real-time status to stations
            for station in map_data['stations']:
                if station['type'] == 'flow':
                    station['current_value'] = state.get('Q_in', 85)
                    station['status'] = 'normal' if state.get('risk_level') == 'LOW' else 'warning'
                elif station['type'] == 'weather':
                    station['temperature'] = state.get('T_sun', 25)
                    station['status'] = 'normal'

        return map_data

    def get_flow_animation(self, flow_rate: float) -> Dict[str, Any]:
        """Get flow animation data for map"""
        # Calculate animation parameters based on flow rate
        speed = flow_rate / 100  # Normalize to 0-1

        return {
            'path': [[p['lng'], p['lat']] for p in self.aqueduct_path],
            'speed': speed,
            'color': '#00b4d8',
            'width': 3 + flow_rate / 50,
            'trail_length': 100 + int(flow_rate * 2)
        }

    def get_station_markers(self, station_status: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Get station markers with status"""
        markers = []
        status_colors = {
            'normal': '#28a745',
            'warning': '#ffc107',
            'alert': '#dc3545',
            'offline': '#6c757d'
        }

        for station in self.stations:
            status = 'normal'
            if station_status and station['id'] in station_status:
                status = station_status[station['id']]

            markers.append({
                'position': [station['lng'], station['lat']],
                'name': station['name'],
                'id': station['id'],
                'type': station['type'],
                'status': status,
                'color': status_colors.get(status, '#6c757d'),
                'icon': self._get_station_icon(station['type'])
            })

        return markers

    def _get_station_icon(self, station_type: str) -> str:
        """Get icon name for station type"""
        icons = {
            'flow': 'water',
            'multi': 'dashboard',
            'weather': 'sun',
            'seismic': 'activity'
        }
        return icons.get(station_type, 'marker')


class DashboardBuilder:
    """
    Custom dashboard builder
    """

    def __init__(self):
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.widget_templates: Dict[str, DashboardWidget] = {}
        self._init_templates()

    def _init_templates(self):
        """Initialize widget templates"""
        self.widget_templates = {
            'water_level_gauge': DashboardWidget(
                widget_id='template_water_level',
                widget_type='gauge',
                title='水位',
                row=0, col=0, width=2, height=2,
                data_source='state.h',
                config={'min': 0, 'max': 8, 'unit': 'm'}
            ),
            'flow_chart': DashboardWidget(
                widget_id='template_flow',
                widget_type='line_chart',
                title='流量趋势',
                row=0, col=2, width=4, height=2,
                data_source='history',
                config={'fields': ['Q_in', 'Q_out'], 'duration': 300}
            ),
            'temperature_heatmap': DashboardWidget(
                widget_id='template_temp',
                widget_type='heatmap',
                title='温度分布',
                row=2, col=0, width=3, height=2,
                data_source='twin.temperature',
                config={'color_scheme': 'thermal'}
            ),
            'risk_indicator': DashboardWidget(
                widget_id='template_risk',
                widget_type='stat',
                title='风险等级',
                row=0, col=6, width=1, height=1,
                data_source='state.risk_level',
                config={'colors': {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'red'}}
            ),
            'scenario_radar': DashboardWidget(
                widget_id='template_scenario',
                widget_type='radar',
                title='场景分析',
                row=2, col=3, width=3, height=2,
                data_source='prediction.scenarios',
                config={'indicators': ['hydraulic', 'thermal', 'structural', 'seismic']}
            ),
            'map_widget': DashboardWidget(
                widget_id='template_map',
                widget_type='map',
                title='渡槽地图',
                row=4, col=0, width=6, height=3,
                data_source='gis',
                config={'show_flow': True, 'show_stations': True}
            )
        }

    def create_dashboard(self, dashboard_id: str, name: str,
                        layout: str = 'grid') -> Dict[str, Any]:
        """Create a new dashboard"""
        dashboard = {
            'dashboard_id': dashboard_id,
            'name': name,
            'layout': layout,
            'grid_cols': 8,
            'grid_rows': 8,
            'widgets': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        self.dashboards[dashboard_id] = dashboard
        return dashboard

    def add_widget(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            return False

        self.dashboards[dashboard_id]['widgets'].append(widget.to_dict())
        self.dashboards[dashboard_id]['updated_at'] = datetime.now().isoformat()
        return True

    def add_template_widget(self, dashboard_id: str, template_id: str,
                           row: int = None, col: int = None) -> bool:
        """Add widget from template"""
        if template_id not in self.widget_templates:
            return False

        template = self.widget_templates[template_id]
        widget = DashboardWidget(
            widget_id=f"{template_id}_{int(time.time())}",
            widget_type=template.widget_type,
            title=template.title,
            row=row if row is not None else template.row,
            col=col if col is not None else template.col,
            width=template.width,
            height=template.height,
            refresh_interval=template.refresh_interval,
            data_source=template.data_source,
            config=dict(template.config)
        )

        return self.add_widget(dashboard_id, widget)

    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard configuration"""
        return self.dashboards.get(dashboard_id)

    def get_default_dashboard(self) -> Dict[str, Any]:
        """Get default dashboard configuration"""
        dashboard = self.create_dashboard('default', '团河渡槽监控仪表盘')

        # Add default widgets
        for template_id in ['water_level_gauge', 'flow_chart', 'risk_indicator',
                           'scenario_radar', 'map_widget']:
            self.add_template_widget('default', template_id)

        return self.dashboards['default']

    def export_dashboard(self, dashboard_id: str, format: str = 'json') -> Optional[str]:
        """Export dashboard configuration"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None

        if format == 'json':
            return json.dumps(dashboard, indent=2, ensure_ascii=False)

        return None

    def import_dashboard(self, config_str: str) -> bool:
        """Import dashboard configuration"""
        try:
            config = json.loads(config_str)
            dashboard_id = config.get('dashboard_id')
            if dashboard_id:
                self.dashboards[dashboard_id] = config
                return True
        except:
            pass
        return False


class AdvancedVisualizationManager:
    """
    Advanced visualization management system
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "viz"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.chart_generator = ChartDataGenerator()
        self.gis = GISVisualization()
        self.dashboard_builder = DashboardBuilder()

        # Data cache
        self.data_cache: Dict[str, deque] = {}
        self.cache_size = 1000

    def cache_state(self, state: Dict[str, Any]):
        """Cache state data for visualization"""
        timestamp = datetime.now().isoformat()
        entry = {'timestamp': timestamp, **state}

        for key in ['h', 'Q_in', 'Q_out', 'T_sun', 'T_shade', 'vib_amp', 'fr']:
            if key not in self.data_cache:
                self.data_cache[key] = deque(maxlen=self.cache_size)
            self.data_cache[key].append({
                'timestamp': timestamp,
                'value': state.get(key, 0)
            })

    def get_time_series_chart(self, variables: List[str], duration_minutes: int = 10,
                             chart_type: ChartType = ChartType.LINE) -> Dict[str, Any]:
        """Get time series chart data"""
        config = ChartConfig(
            chart_id=f"ts_{int(time.time())}",
            chart_type=chart_type,
            title='时间序列数据',
            x_axis_label='时间',
            y_axis_label='值',
            color_scheme=ColorScheme.DEFAULT
        )

        # Collect data
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        data = []

        if variables and variables[0] in self.data_cache:
            cache = self.data_cache[variables[0]]
            for entry in cache:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time > cutoff:
                    data_point = {'timestamp': entry['timestamp']}
                    for var in variables:
                        var_cache = self.data_cache.get(var, [])
                        # Find matching timestamp
                        for ve in var_cache:
                            if ve['timestamp'] == entry['timestamp']:
                                data_point[var] = ve['value']
                                break
                    data.append(data_point)

        return self.chart_generator.generate_line_chart(data, 'timestamp', variables, config)

    def get_water_level_gauge(self, current_level: float) -> Dict[str, Any]:
        """Get water level gauge chart"""
        config = ChartConfig(
            chart_id='water_level_gauge',
            chart_type=ChartType.GAUGE,
            title='水位',
            color_scheme=ColorScheme.WATER
        )

        thresholds = [
            (0.25, '#28a745'),  # Low - Green
            (0.5, '#ffc107'),   # Normal - Yellow
            (0.75, '#fd7e14'),  # High - Orange
            (1, '#dc3545')      # Critical - Red
        ]

        return self.chart_generator.generate_gauge_chart(
            current_level, 0, 8, config, thresholds
        )

    def get_thermal_heatmap(self, thermal_data: List[List[float]] = None) -> Dict[str, Any]:
        """Get thermal distribution heatmap"""
        config = ChartConfig(
            chart_id='thermal_heatmap',
            chart_type=ChartType.HEATMAP,
            title='温度分布',
            color_scheme=ColorScheme.THERMAL
        )

        # Default thermal data if not provided
        if thermal_data is None:
            # Simulate temperature distribution along aqueduct sections
            thermal_data = [
                [25, 26, 27, 28, 29, 30, 29, 28, 27, 26],
                [24, 25, 26, 27, 28, 29, 28, 27, 26, 25],
                [23, 24, 25, 26, 27, 28, 27, 26, 25, 24]
            ]

        x_labels = [f'{i*130}m' for i in range(10)]
        y_labels = ['顶面', '侧面', '底面']

        return self.chart_generator.generate_heatmap(thermal_data, x_labels, y_labels, config)

    def get_flow_sankey(self, flow_data: Dict[str, float] = None) -> Dict[str, Any]:
        """Get flow Sankey diagram"""
        config = ChartConfig(
            chart_id='flow_sankey',
            chart_type=ChartType.SANKEY,
            title='流量分布',
            color_scheme=ColorScheme.WATER
        )

        if flow_data is None:
            flow_data = {'Q_in': 85, 'Q_out': 82, 'loss': 3}

        nodes = [
            {'name': '上游来水'},
            {'name': '渡槽输水'},
            {'name': '下游出水'},
            {'name': '蒸发损失'},
            {'name': '渗漏损失'}
        ]

        links = [
            {'source': '上游来水', 'target': '渡槽输水', 'value': flow_data['Q_in']},
            {'source': '渡槽输水', 'target': '下游出水', 'value': flow_data['Q_out']},
            {'source': '渡槽输水', 'target': '蒸发损失', 'value': flow_data['loss'] * 0.3},
            {'source': '渡槽输水', 'target': '渗漏损失', 'value': flow_data['loss'] * 0.7}
        ]

        return self.chart_generator.generate_sankey(nodes, links, config)

    def get_risk_radar(self, risk_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Get risk assessment radar chart"""
        config = ChartConfig(
            chart_id='risk_radar',
            chart_type=ChartType.RADAR,
            title='风险评估雷达图',
            color_scheme=ColorScheme.SAFETY
        )

        indicators = [
            {'name': '水力风险', 'max': 100},
            {'name': '热力风险', 'max': 100},
            {'name': '结构风险', 'max': 100},
            {'name': '地震风险', 'max': 100},
            {'name': '设备风险', 'max': 100}
        ]

        if risk_scores is None:
            risk_scores = {
                'hydraulic': 30,
                'thermal': 45,
                'structural': 20,
                'seismic': 10,
                'equipment': 25
            }

        data = [{
            'name': '当前风险',
            'value': [
                risk_scores.get('hydraulic', 0),
                risk_scores.get('thermal', 0),
                risk_scores.get('structural', 0),
                risk_scores.get('seismic', 0),
                risk_scores.get('equipment', 0)
            ]
        }]

        return self.chart_generator.generate_radar(indicators, data, config)

    def get_3d_water_surface(self, water_levels: List[List[float]] = None) -> Dict[str, Any]:
        """Get 3D water surface visualization"""
        config = ChartConfig(
            chart_id='water_surface_3d',
            chart_type=ChartType.THREE_D,
            title='水面三维分布',
            x_axis_label='长度 (m)',
            y_axis_label='宽度 (m)',
            color_scheme=ColorScheme.WATER
        )

        # Generate sample water surface data
        x = [i * 100 for i in range(13)]  # 0 to 1200m
        y = [i * 0.5 for i in range(15)]  # 0 to 7m width

        if water_levels is None:
            # Simulate water surface with slight variations
            water_levels = []
            for yi in range(15):
                row = []
                for xi in range(13):
                    # Base level with wave pattern
                    level = 4.5 + 0.1 * math.sin(xi * 0.5) + 0.05 * math.cos(yi * 0.3)
                    row.append(level)
                water_levels.append(row)

        return self.chart_generator.generate_3d_surface(x, y, water_levels, config)

    def get_gis_map_data(self, state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get GIS map visualization data"""
        return self.gis.get_map_data(include_realtime=True, state=state)

    def get_dashboard_config(self, dashboard_id: str = 'default') -> Dict[str, Any]:
        """Get dashboard configuration"""
        if dashboard_id == 'default':
            return self.dashboard_builder.get_default_dashboard()
        return self.dashboard_builder.get_dashboard(dashboard_id)

    def get_status(self) -> Dict[str, Any]:
        """Get visualization system status"""
        return {
            'cached_variables': list(self.data_cache.keys()),
            'cache_sizes': {k: len(v) for k, v in self.data_cache.items()},
            'dashboards': list(self.dashboard_builder.dashboards.keys()),
            'widget_templates': list(self.dashboard_builder.widget_templates.keys()),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_viz_manager = None


def get_viz_manager() -> AdvancedVisualizationManager:
    """Get global visualization manager"""
    global _viz_manager
    if _viz_manager is None:
        _viz_manager = AdvancedVisualizationManager()
    return _viz_manager


if __name__ == "__main__":
    # Test advanced visualization
    print("=== Advanced Visualization Test ===")

    manager = AdvancedVisualizationManager()

    # Cache some test data
    print("\n1. Caching test data...")
    for i in range(20):
        manager.cache_state({
            'h': 4.5 + 0.1 * math.sin(i * 0.3),
            'Q_in': 85 + 5 * math.cos(i * 0.2),
            'Q_out': 83 + 4 * math.cos(i * 0.2),
            'T_sun': 28 + 2 * math.sin(i * 0.1),
            'T_shade': 22 + 1 * math.sin(i * 0.1),
            'vib_amp': 2 + 0.5 * math.sin(i * 0.5),
            'fr': 0.32 + 0.02 * math.sin(i * 0.4)
        })
        time.sleep(0.01)

    # Get time series chart
    print("\n2. Time series chart:")
    ts_chart = manager.get_time_series_chart(['h', 'Q_in', 'Q_out'], duration_minutes=60)
    print(f"   Series count: {len(ts_chart['series'])}")

    # Get gauge
    print("\n3. Water level gauge:")
    gauge = manager.get_water_level_gauge(4.65)
    print(f"   Value: {gauge['series'][0]['data'][0]['value']}")

    # Get heatmap
    print("\n4. Thermal heatmap:")
    heatmap = manager.get_thermal_heatmap()
    print(f"   Data points: {len(heatmap['series'][0]['data'])}")

    # Get Sankey
    print("\n5. Flow Sankey diagram:")
    sankey = manager.get_flow_sankey({'Q_in': 85, 'Q_out': 82, 'loss': 3})
    print(f"   Nodes: {len(sankey['series'][0]['data'])}")
    print(f"   Links: {len(sankey['series'][0]['links'])}")

    # Get radar
    print("\n6. Risk radar chart:")
    radar = manager.get_risk_radar()
    print(f"   Indicators: {len(radar['radar']['indicator'])}")

    # Get GIS data
    print("\n7. GIS map data:")
    gis = manager.get_gis_map_data({'Q_in': 85, 'risk_level': 'LOW'})
    print(f"   Stations: {len(gis['stations'])}")
    print(f"   Center: {gis['center']}")

    # Get dashboard
    print("\n8. Dashboard configuration:")
    dashboard = manager.get_dashboard_config('default')
    print(f"   Dashboard: {dashboard['name']}")
    print(f"   Widgets: {len(dashboard['widgets'])}")

    # Status
    print("\n9. System status:")
    status = manager.get_status()
    print(f"   Cached variables: {status['cached_variables']}")
    print(f"   Dashboards: {status['dashboards']}")

    print("\nAdvanced Visualization test completed!")
