"""
TAOS V3.10 Reporting and Visualization Module
报表与可视化模块

Features:
- Automated report generation (daily, weekly, monthly)
- Custom report templates
- Data aggregation and statistics
- Chart generation and export
- Dashboard configuration
- Export formats (PDF, Excel, CSV, JSON)
"""

import json
import csv
import io
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import logging
import uuid
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Report types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"
    ON_DEMAND = "on_demand"


class AggregationType(Enum):
    """Data aggregation types"""
    AVERAGE = "average"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    RANGE = "range"
    STD_DEV = "std_dev"
    MEDIAN = "median"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"


class ChartType(Enum):
    """Chart types for visualization"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"


class ExportFormat(Enum):
    """Export formats"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class TimeResolution(Enum):
    """Time resolution for data aggregation"""
    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class DataPoint:
    """Data point for reporting"""
    timestamp: datetime
    value: Any
    tag_id: str = ""
    quality: str = "good"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedData:
    """Aggregated data result"""
    tag_id: str
    start_time: datetime
    end_time: datetime
    resolution: TimeResolution
    aggregation: AggregationType
    values: List[Dict[str, Any]]  # [{timestamp, value}, ...]
    statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChartConfig:
    """Chart configuration"""
    chart_id: str
    title: str
    chart_type: ChartType
    data_sources: List[str]  # Tag IDs
    aggregation: AggregationType = AggregationType.AVERAGE
    resolution: TimeResolution = TimeResolution.HOUR
    # Display options
    show_legend: bool = True
    show_grid: bool = True
    x_axis_label: str = ""
    y_axis_label: str = ""
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    colors: List[str] = field(default_factory=list)
    # For gauge/single value
    thresholds: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReportSection:
    """Report section configuration"""
    section_id: str
    title: str
    section_type: str  # "text", "chart", "table", "kpi", "summary"
    content: Any = None  # Text content or data config
    charts: List[ChartConfig] = field(default_factory=list)
    order: int = 0


@dataclass
class ReportTemplate:
    """Report template definition"""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    sections: List[ReportSection] = field(default_factory=list)
    schedule: Optional[Dict[str, Any]] = None  # Cron-like schedule
    recipients: List[str] = field(default_factory=list)
    export_formats: List[ExportFormat] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Report:
    """Generated report"""
    report_id: str
    template_id: str
    name: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    sections: List[Dict[str, Any]]
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'template_id': self.template_id,
            'name': self.name,
            'generated_at': self.generated_at.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'sections': self.sections,
            'summary': self.summary,
            'metadata': self.metadata
        }


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # "chart", "gauge", "value", "table", "alarm_list", "map"
    config: Dict[str, Any]
    position: Dict[str, int]  # {x, y, width, height}
    refresh_interval: int = 60  # seconds
    data_source: Optional[str] = None


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: str = "grid"  # "grid", "free"
    refresh_interval: int = 60
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_default: bool = False


class DataAggregator:
    """Data aggregation engine"""

    def __init__(self):
        self._cache: Dict[str, List[DataPoint]] = {}
        self._lock = threading.Lock()

    def add_data(self, tag_id: str, data_point: DataPoint):
        """Add data point to aggregation cache"""
        with self._lock:
            if tag_id not in self._cache:
                self._cache[tag_id] = []
            self._cache[tag_id].append(data_point)

    def add_batch_data(self, tag_id: str, data_points: List[DataPoint]):
        """Add multiple data points"""
        with self._lock:
            if tag_id not in self._cache:
                self._cache[tag_id] = []
            self._cache[tag_id].extend(data_points)

    def clear_cache(self, tag_id: Optional[str] = None):
        """Clear aggregation cache"""
        with self._lock:
            if tag_id:
                self._cache[tag_id] = []
            else:
                self._cache.clear()

    def aggregate(self, tag_id: str, start: datetime, end: datetime,
                  resolution: TimeResolution,
                  aggregation: AggregationType) -> AggregatedData:
        """Aggregate data for a tag"""
        with self._lock:
            raw_data = self._cache.get(tag_id, [])

        # Filter by time range
        filtered = [
            dp for dp in raw_data
            if start <= dp.timestamp <= end
        ]

        # Group by time bucket
        buckets = self._bucket_data(filtered, resolution)

        # Aggregate each bucket
        values = []
        all_values = []

        for bucket_time, bucket_data in sorted(buckets.items()):
            bucket_values = [dp.value for dp in bucket_data if isinstance(dp.value, (int, float))]
            if bucket_values:
                agg_value = self._calculate_aggregation(bucket_values, aggregation)
                values.append({
                    'timestamp': bucket_time.isoformat(),
                    'value': agg_value,
                    'count': len(bucket_values)
                })
                all_values.extend(bucket_values)

        # Calculate overall statistics
        stats = {}
        if all_values:
            stats = {
                'min': min(all_values),
                'max': max(all_values),
                'avg': sum(all_values) / len(all_values),
                'count': len(all_values),
                'range': max(all_values) - min(all_values)
            }
            if len(all_values) > 1:
                stats['std_dev'] = statistics.stdev(all_values)
                stats['median'] = statistics.median(all_values)

        return AggregatedData(
            tag_id=tag_id,
            start_time=start,
            end_time=end,
            resolution=resolution,
            aggregation=aggregation,
            values=values,
            statistics=stats
        )

    def _bucket_data(self, data: List[DataPoint],
                      resolution: TimeResolution) -> Dict[datetime, List[DataPoint]]:
        """Group data points into time buckets"""
        buckets = defaultdict(list)

        for dp in data:
            bucket_time = self._get_bucket_time(dp.timestamp, resolution)
            buckets[bucket_time].append(dp)

        return dict(buckets)

    def _get_bucket_time(self, timestamp: datetime,
                          resolution: TimeResolution) -> datetime:
        """Get bucket start time for a timestamp"""
        if resolution == TimeResolution.SECOND:
            return timestamp.replace(microsecond=0)
        elif resolution == TimeResolution.MINUTE:
            return timestamp.replace(second=0, microsecond=0)
        elif resolution == TimeResolution.FIVE_MINUTES:
            minute = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif resolution == TimeResolution.FIFTEEN_MINUTES:
            minute = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minute, second=0, microsecond=0)
        elif resolution == TimeResolution.HOUR:
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif resolution == TimeResolution.DAY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif resolution == TimeResolution.WEEK:
            days_since_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_since_monday)
            return monday.replace(hour=0, minute=0, second=0, microsecond=0)
        elif resolution == TimeResolution.MONTH:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp

    def _calculate_aggregation(self, values: List[float],
                                aggregation: AggregationType) -> float:
        """Calculate aggregation for a list of values"""
        if not values:
            return 0.0

        if aggregation == AggregationType.AVERAGE:
            return sum(values) / len(values)
        elif aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.FIRST:
            return values[0]
        elif aggregation == AggregationType.LAST:
            return values[-1]
        elif aggregation == AggregationType.RANGE:
            return max(values) - min(values)
        elif aggregation == AggregationType.STD_DEV:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        elif aggregation == AggregationType.MEDIAN:
            return statistics.median(values)
        elif aggregation == AggregationType.PERCENTILE_95:
            return self._percentile(values, 95)
        elif aggregation == AggregationType.PERCENTILE_99:
            return self._percentile(values, 99)
        else:
            return sum(values) / len(values)

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class ChartGenerator:
    """Generate chart data for visualization"""

    def generate_line_chart(self, data: List[AggregatedData],
                            config: ChartConfig) -> Dict[str, Any]:
        """Generate line chart data"""
        series = []
        for agg_data in data:
            series.append({
                'name': agg_data.tag_id,
                'data': [
                    {'x': v['timestamp'], 'y': v['value']}
                    for v in agg_data.values
                ]
            })

        return {
            'chart_type': 'line',
            'title': config.title,
            'series': series,
            'options': {
                'show_legend': config.show_legend,
                'show_grid': config.show_grid,
                'x_axis': config.x_axis_label,
                'y_axis': config.y_axis_label,
                'y_min': config.y_axis_min,
                'y_max': config.y_axis_max,
                'colors': config.colors
            }
        }

    def generate_bar_chart(self, data: List[AggregatedData],
                           config: ChartConfig) -> Dict[str, Any]:
        """Generate bar chart data"""
        categories = []
        series_data = defaultdict(list)

        for agg_data in data:
            for v in agg_data.values:
                if v['timestamp'] not in categories:
                    categories.append(v['timestamp'])
                series_data[agg_data.tag_id].append(v['value'])

        return {
            'chart_type': 'bar',
            'title': config.title,
            'categories': categories,
            'series': [
                {'name': name, 'data': values}
                for name, values in series_data.items()
            ],
            'options': {
                'show_legend': config.show_legend,
                'colors': config.colors
            }
        }

    def generate_pie_chart(self, data: Dict[str, float],
                           config: ChartConfig) -> Dict[str, Any]:
        """Generate pie chart data"""
        return {
            'chart_type': 'pie',
            'title': config.title,
            'data': [
                {'name': name, 'value': value}
                for name, value in data.items()
            ],
            'options': {
                'show_legend': config.show_legend,
                'colors': config.colors
            }
        }

    def generate_gauge(self, value: float, config: ChartConfig) -> Dict[str, Any]:
        """Generate gauge chart data"""
        return {
            'chart_type': 'gauge',
            'title': config.title,
            'value': value,
            'thresholds': config.thresholds,
            'options': {
                'min': config.y_axis_min or 0,
                'max': config.y_axis_max or 100
            }
        }

    def generate_table(self, data: List[Dict[str, Any]],
                       columns: List[Dict[str, str]],
                       config: ChartConfig) -> Dict[str, Any]:
        """Generate table data"""
        return {
            'chart_type': 'table',
            'title': config.title,
            'columns': columns,
            'data': data
        }


class ReportExporter:
    """Export reports to various formats"""

    def export_csv(self, data: List[Dict[str, Any]],
                   columns: List[str]) -> str:
        """Export to CSV format"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k, '') for k in columns})
        return output.getvalue()

    def export_json(self, report: Report) -> str:
        """Export to JSON format"""
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    def export_html(self, report: Report) -> str:
        """Export to HTML format"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .section-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; }}
        .kpi {{ display: inline-block; margin: 10px; padding: 20px; background: #e3f2fd; }}
        .kpi-value {{ font-size: 24px; font-weight: bold; }}
        .kpi-label {{ color: #666; }}
    </style>
</head>
<body>
    <h1>{report.name}</h1>
    <p>报告周期: {report.period_start.strftime('%Y-%m-%d')} 至 {report.period_end.strftime('%Y-%m-%d')}</p>
    <p>生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>摘要</h2>
        {"".join(f'<div class="kpi"><div class="kpi-value">{v}</div><div class="kpi-label">{k}</div></div>' for k, v in report.summary.items())}
    </div>
"""
        for section in report.sections:
            html += f"""
    <div class="section">
        <div class="section-title">{section.get('title', '')}</div>
        <div class="section-content">{self._render_section_content(section)}</div>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _render_section_content(self, section: Dict) -> str:
        """Render section content to HTML"""
        section_type = section.get('section_type', 'text')

        if section_type == 'text':
            return f"<p>{section.get('content', '')}</p>"
        elif section_type == 'table':
            return self._render_table_html(section.get('data', []), section.get('columns', []))
        elif section_type == 'kpi':
            kpis = section.get('kpis', {})
            return "".join(
                f'<div class="kpi"><div class="kpi-value">{v}</div><div class="kpi-label">{k}</div></div>'
                for k, v in kpis.items()
            )
        else:
            return json.dumps(section.get('content', {}))

    def _render_table_html(self, data: List[Dict], columns: List[str]) -> str:
        """Render table as HTML"""
        if not data or not columns:
            return "<p>无数据</p>"

        html = "<table><tr>"
        for col in columns:
            html += f"<th>{col}</th>"
        html += "</tr>"

        for row in data:
            html += "<tr>"
            for col in columns:
                html += f"<td>{row.get(col, '')}</td>"
            html += "</tr>"

        html += "</table>"
        return html


class ReportGenerator:
    """Report generation engine"""

    def __init__(self, aggregator: DataAggregator):
        self.aggregator = aggregator
        self.chart_generator = ChartGenerator()
        self.exporter = ReportExporter()
        self.templates: Dict[str, ReportTemplate] = {}

    def add_template(self, template: ReportTemplate):
        """Add report template"""
        self.templates[template.template_id] = template

    def generate_report(self, template_id: str,
                        period_start: datetime,
                        period_end: datetime,
                        parameters: Dict[str, Any] = None) -> Report:
        """Generate report from template"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        parameters = parameters or {}
        sections = []
        summary = {}

        for section_config in sorted(template.sections, key=lambda s: s.order):
            section_data = self._generate_section(
                section_config, period_start, period_end, parameters
            )
            sections.append(section_data)

            # Extract KPIs for summary
            if section_config.section_type == 'kpi':
                summary.update(section_data.get('kpis', {}))

        report = Report(
            report_id=str(uuid.uuid4()),
            template_id=template_id,
            name=template.name,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            sections=sections,
            summary=summary
        )

        logger.info(f"Generated report: {template.name}")
        return report

    def _generate_section(self, section: ReportSection,
                           period_start: datetime,
                           period_end: datetime,
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report section"""
        result = {
            'section_id': section.section_id,
            'title': section.title,
            'section_type': section.section_type
        }

        if section.section_type == 'text':
            result['content'] = section.content
        elif section.section_type == 'chart':
            result['charts'] = [
                self._generate_chart(chart, period_start, period_end)
                for chart in section.charts
            ]
        elif section.section_type == 'table':
            result['data'] = self._generate_table_data(
                section.content, period_start, period_end
            )
            result['columns'] = section.content.get('columns', [])
        elif section.section_type == 'kpi':
            result['kpis'] = self._calculate_kpis(
                section.content, period_start, period_end
            )
        elif section.section_type == 'summary':
            result['summary'] = self._generate_summary(period_start, period_end)

        return result

    def _generate_chart(self, config: ChartConfig,
                         period_start: datetime,
                         period_end: datetime) -> Dict[str, Any]:
        """Generate chart data"""
        aggregated_data = []
        for tag_id in config.data_sources:
            agg = self.aggregator.aggregate(
                tag_id, period_start, period_end,
                config.resolution, config.aggregation
            )
            aggregated_data.append(agg)

        if config.chart_type == ChartType.LINE:
            return self.chart_generator.generate_line_chart(aggregated_data, config)
        elif config.chart_type == ChartType.BAR:
            return self.chart_generator.generate_bar_chart(aggregated_data, config)
        elif config.chart_type == ChartType.GAUGE:
            # Use last value for gauge
            value = aggregated_data[0].values[-1]['value'] if aggregated_data and aggregated_data[0].values else 0
            return self.chart_generator.generate_gauge(value, config)
        else:
            return {'chart_type': config.chart_type.value, 'data': []}

    def _generate_table_data(self, config: Dict,
                              period_start: datetime,
                              period_end: datetime) -> List[Dict]:
        """Generate table data"""
        tag_ids = config.get('data_sources', [])
        resolution = TimeResolution(config.get('resolution', 'hour'))
        aggregation = AggregationType(config.get('aggregation', 'average'))

        data = []
        for tag_id in tag_ids:
            agg = self.aggregator.aggregate(
                tag_id, period_start, period_end, resolution, aggregation
            )
            for v in agg.values:
                data.append({
                    'tag_id': tag_id,
                    'timestamp': v['timestamp'],
                    'value': round(v['value'], 2) if isinstance(v['value'], float) else v['value'],
                    'count': v['count']
                })

        return data

    def _calculate_kpis(self, config: Dict,
                         period_start: datetime,
                         period_end: datetime) -> Dict[str, Any]:
        """Calculate KPIs"""
        kpis = {}
        kpi_definitions = config.get('kpis', [])

        for kpi_def in kpi_definitions:
            tag_id = kpi_def.get('tag_id')
            kpi_name = kpi_def.get('name')
            aggregation = AggregationType(kpi_def.get('aggregation', 'average'))

            agg = self.aggregator.aggregate(
                tag_id, period_start, period_end,
                TimeResolution.DAY, aggregation
            )

            if agg.statistics:
                value = agg.statistics.get('avg', 0)
                unit = kpi_def.get('unit', '')
                kpis[kpi_name] = f"{value:.2f} {unit}"

        return kpis

    def _generate_summary(self, period_start: datetime,
                           period_end: datetime) -> Dict[str, Any]:
        """Generate report summary"""
        return {
            'period': f"{period_start.strftime('%Y-%m-%d')} - {period_end.strftime('%Y-%m-%d')}",
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def export(self, report: Report, format: ExportFormat) -> str:
        """Export report to specified format"""
        if format == ExportFormat.JSON:
            return self.exporter.export_json(report)
        elif format == ExportFormat.HTML:
            return self.exporter.export_html(report)
        elif format == ExportFormat.CSV:
            # Flatten report data for CSV
            all_data = []
            for section in report.sections:
                if 'data' in section:
                    all_data.extend(section['data'])
            columns = ['tag_id', 'timestamp', 'value', 'count']
            return self.exporter.export_csv(all_data, columns)
        else:
            return self.exporter.export_json(report)


class DashboardManager:
    """Dashboard management"""

    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self._lock = threading.Lock()

    def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create a new dashboard"""
        with self._lock:
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Created dashboard: {dashboard.name}")
            return dashboard.dashboard_id

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)

    def update_dashboard(self, dashboard_id: str,
                         updates: Dict[str, Any]) -> bool:
        """Update dashboard"""
        with self._lock:
            if dashboard_id not in self.dashboards:
                return False

            dashboard = self.dashboards[dashboard_id]
            for key, value in updates.items():
                if hasattr(dashboard, key):
                    setattr(dashboard, key, value)
            return True

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete dashboard"""
        with self._lock:
            if dashboard_id in self.dashboards:
                del self.dashboards[dashboard_id]
                return True
            return False

    def add_widget(self, dashboard_id: str, widget: DashboardWidget) -> bool:
        """Add widget to dashboard"""
        with self._lock:
            if dashboard_id not in self.dashboards:
                return False

            self.dashboards[dashboard_id].widgets.append(widget)
            return True

    def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove widget from dashboard"""
        with self._lock:
            if dashboard_id not in self.dashboards:
                return False

            dashboard = self.dashboards[dashboard_id]
            dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
            return True

    def get_widget_data(self, widget: DashboardWidget,
                        aggregator: DataAggregator) -> Dict[str, Any]:
        """Get data for a dashboard widget"""
        config = widget.config
        now = datetime.now()

        # Determine time range
        time_range = config.get('time_range', '1h')
        time_ranges = {
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }
        delta = time_ranges.get(time_range, timedelta(hours=1))
        start = now - delta
        end = now

        if widget.widget_type == 'value':
            # Single value widget
            tag_id = config.get('tag_id')
            agg = aggregator.aggregate(
                tag_id, start, end,
                TimeResolution.MINUTE, AggregationType.LAST
            )
            return {
                'value': agg.values[-1]['value'] if agg.values else None,
                'statistics': agg.statistics
            }

        elif widget.widget_type == 'chart':
            # Chart widget
            tag_ids = config.get('tag_ids', [])
            chart_type = ChartType(config.get('chart_type', 'line'))
            resolution = TimeResolution(config.get('resolution', 'minute'))

            data = []
            for tag_id in tag_ids:
                agg = aggregator.aggregate(
                    tag_id, start, end,
                    resolution, AggregationType.AVERAGE
                )
                data.append({
                    'tag_id': tag_id,
                    'values': agg.values
                })

            return {'series': data, 'chart_type': chart_type.value}

        elif widget.widget_type == 'gauge':
            # Gauge widget
            tag_id = config.get('tag_id')
            agg = aggregator.aggregate(
                tag_id, start, end,
                TimeResolution.MINUTE, AggregationType.LAST
            )
            return {
                'value': agg.values[-1]['value'] if agg.values else 0,
                'min': config.get('min', 0),
                'max': config.get('max', 100),
                'thresholds': config.get('thresholds', [])
            }

        elif widget.widget_type == 'table':
            # Table widget
            tag_ids = config.get('tag_ids', [])
            data = []
            for tag_id in tag_ids:
                agg = aggregator.aggregate(
                    tag_id, start, end,
                    TimeResolution.HOUR, AggregationType.AVERAGE
                )
                if agg.statistics:
                    data.append({
                        'tag_id': tag_id,
                        'current': agg.values[-1]['value'] if agg.values else None,
                        'average': agg.statistics.get('avg'),
                        'min': agg.statistics.get('min'),
                        'max': agg.statistics.get('max')
                    })
            return {'rows': data}

        return {}

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all dashboards"""
        return [
            {
                'dashboard_id': d.dashboard_id,
                'name': d.name,
                'description': d.description,
                'widget_count': len(d.widgets),
                'is_default': d.is_default
            }
            for d in self.dashboards.values()
        ]


class ReportingManager:
    """
    Central reporting and visualization manager
    报表与可视化管理中心
    """

    def __init__(self):
        self.aggregator = DataAggregator()
        self.report_generator = ReportGenerator(self.aggregator)
        self.dashboard_manager = DashboardManager()
        self.scheduled_reports: Dict[str, Dict] = {}
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

    def add_data(self, tag_id: str, value: Any,
                 timestamp: Optional[datetime] = None):
        """Add data point for reporting"""
        dp = DataPoint(
            timestamp=timestamp or datetime.now(),
            value=value,
            tag_id=tag_id
        )
        self.aggregator.add_data(tag_id, dp)

    def add_batch_data(self, tag_id: str, data: List[tuple]):
        """Add batch data (timestamp, value) tuples"""
        data_points = [
            DataPoint(timestamp=ts, value=val, tag_id=tag_id)
            for ts, val in data
        ]
        self.aggregator.add_batch_data(tag_id, data_points)

    def add_report_template(self, template: ReportTemplate):
        """Add report template"""
        self.report_generator.add_template(template)

    def generate_report(self, template_id: str,
                        period_start: datetime,
                        period_end: datetime,
                        parameters: Dict[str, Any] = None) -> Report:
        """Generate report"""
        return self.report_generator.generate_report(
            template_id, period_start, period_end, parameters
        )

    def export_report(self, report: Report, format: ExportFormat) -> str:
        """Export report to format"""
        return self.report_generator.export(report, format)

    def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create dashboard"""
        return self.dashboard_manager.create_dashboard(dashboard)

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard"""
        return self.dashboard_manager.get_dashboard(dashboard_id)

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get all widget data for a dashboard"""
        dashboard = self.dashboard_manager.get_dashboard(dashboard_id)
        if not dashboard:
            return {}

        widget_data = {}
        for widget in dashboard.widgets:
            widget_data[widget.widget_id] = self.dashboard_manager.get_widget_data(
                widget, self.aggregator
            )

        return {
            'dashboard_id': dashboard_id,
            'name': dashboard.name,
            'widgets': widget_data
        }

    def schedule_report(self, template_id: str, schedule: Dict[str, Any],
                        recipients: List[str] = None):
        """Schedule automatic report generation"""
        self.scheduled_reports[template_id] = {
            'schedule': schedule,
            'recipients': recipients or [],
            'last_run': None,
            'enabled': True
        }

    def get_status(self) -> Dict[str, Any]:
        """Get reporting system status"""
        return {
            'templates': len(self.report_generator.templates),
            'dashboards': len(self.dashboard_manager.dashboards),
            'scheduled_reports': len(self.scheduled_reports),
            'aggregator_tags': len(self.aggregator._cache)
        }


# Singleton instance
_reporting_manager: Optional[ReportingManager] = None


def get_reporting_manager() -> ReportingManager:
    """Get singleton instance of ReportingManager"""
    global _reporting_manager
    if _reporting_manager is None:
        _reporting_manager = ReportingManager()
    return _reporting_manager


# Helper function to create common report templates
def create_daily_operations_template() -> ReportTemplate:
    """Create daily operations report template"""
    return ReportTemplate(
        template_id="daily_ops",
        name="每日运行报告",
        description="水渠系统每日运行状态报告",
        report_type=ReportType.DAILY,
        sections=[
            ReportSection(
                section_id="summary",
                title="运行概况",
                section_type="kpi",
                content={
                    'kpis': [
                        {'tag_id': 'water_level', 'name': '平均水位', 'aggregation': 'average', 'unit': 'm'},
                        {'tag_id': 'flow_rate', 'name': '总流量', 'aggregation': 'sum', 'unit': 'm³'},
                        {'tag_id': 'gate_operations', 'name': '闸门操作次数', 'aggregation': 'count', 'unit': '次'}
                    ]
                },
                order=1
            ),
            ReportSection(
                section_id="water_level_chart",
                title="水位趋势",
                section_type="chart",
                charts=[
                    ChartConfig(
                        chart_id="water_level_trend",
                        title="24小时水位趋势",
                        chart_type=ChartType.LINE,
                        data_sources=["water_level"],
                        resolution=TimeResolution.HOUR
                    )
                ],
                order=2
            ),
            ReportSection(
                section_id="flow_chart",
                title="流量趋势",
                section_type="chart",
                charts=[
                    ChartConfig(
                        chart_id="flow_trend",
                        title="24小时流量趋势",
                        chart_type=ChartType.LINE,
                        data_sources=["flow_rate"],
                        resolution=TimeResolution.HOUR
                    )
                ],
                order=3
            )
        ],
        export_formats=[ExportFormat.HTML, ExportFormat.PDF]
    )
