#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - Report Generation System
团河渡槽自主运行系统 - 报表生成模块

Features:
- Multiple report types (daily, weekly, monthly, custom)
- Various formats (HTML, PDF, CSV, JSON)
- Automated scheduling
- Template-based generation
- Chart and graph integration
"""

import json
import csv
import io
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pathlib import Path
import threading
import time


class ReportType(Enum):
    """Report types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SHIFT = "shift"
    INCIDENT = "incident"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats"""
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    TEXT = "text"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Report:
    """Report data structure"""
    report_id: str
    report_type: ReportType
    title: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    sections: List[ReportSection]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'title': self.title,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'summary': self.summary,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'data': s.data,
                    'charts': s.charts,
                    'tables': s.tables
                }
                for s in self.sections
            ],
            'metadata': self.metadata
        }


class DataCollector:
    """Collects data for reports from various sources"""

    def __init__(self):
        self.data_sources: Dict[str, Callable] = {}

    def register_source(self, name: str, collector: Callable):
        """Register a data source"""
        self.data_sources[name] = collector

    def collect(self, source: str, start_time: datetime, end_time: datetime,
               **kwargs) -> Dict[str, Any]:
        """Collect data from a source"""
        if source in self.data_sources:
            try:
                return self.data_sources[source](start_time, end_time, **kwargs)
            except Exception as e:
                return {'error': str(e)}
        return {}

    def collect_all(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Collect data from all sources"""
        data = {}
        for name, collector in self.data_sources.items():
            try:
                data[name] = collector(start_time, end_time)
            except Exception as e:
                data[name] = {'error': str(e)}
        return data


class ReportTemplate:
    """Base report template"""

    def __init__(self, template_id: str, name: str):
        self.template_id = template_id
        self.name = name

    def generate_sections(self, data: Dict[str, Any],
                         start_time: datetime, end_time: datetime) -> List[ReportSection]:
        """Generate report sections from data"""
        raise NotImplementedError

    def generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate report summary"""
        raise NotImplementedError


class DailyReportTemplate(ReportTemplate):
    """Daily operation report template"""

    def __init__(self):
        super().__init__("daily", "日运行报表")

    def generate_sections(self, data: Dict[str, Any],
                         start_time: datetime, end_time: datetime) -> List[ReportSection]:
        sections = []

        # 1. Overview Section
        overview_data = data.get('state_summary', {})
        sections.append(ReportSection(
            title="运行概况",
            content=self._generate_overview_content(overview_data),
            data=overview_data,
            tables=[{
                'title': '关键参数统计',
                'headers': ['参数', '平均值', '最大值', '最小值', '标准差'],
                'rows': self._generate_stats_rows(overview_data)
            }]
        ))

        # 2. Safety Section
        safety_data = data.get('safety', {})
        sections.append(ReportSection(
            title="安全状况",
            content=self._generate_safety_content(safety_data),
            data=safety_data,
            charts=[{
                'type': 'pie',
                'title': '安全等级分布',
                'data': safety_data.get('level_distribution', {})
            }]
        ))

        # 3. Alerts Section
        alert_data = data.get('alerts', {})
        sections.append(ReportSection(
            title="告警统计",
            content=self._generate_alert_content(alert_data),
            data=alert_data,
            tables=[{
                'title': '告警明细',
                'headers': ['时间', '级别', '类别', '标题', '状态'],
                'rows': alert_data.get('alert_list', [])
            }]
        ))

        # 4. Control Actions Section
        control_data = data.get('control', {})
        sections.append(ReportSection(
            title="控制动作",
            content=self._generate_control_content(control_data),
            data=control_data
        ))

        # 5. Scenarios Section
        scenario_data = data.get('scenarios', {})
        sections.append(ReportSection(
            title="场景记录",
            content=self._generate_scenario_content(scenario_data),
            data=scenario_data
        ))

        return sections

    def generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate daily summary"""
        state = data.get('state_summary', {})
        alerts = data.get('alerts', {})
        safety = data.get('safety', {})

        summary_parts = []
        summary_parts.append(f"本日系统运行{state.get('uptime_hours', 0):.1f}小时")
        summary_parts.append(f"水位波动范围：{state.get('h_min', 0):.2f}m - {state.get('h_max', 0):.2f}m")
        summary_parts.append(f"告警总数：{alerts.get('total', 0)}次")
        summary_parts.append(f"安全运行率：{safety.get('safe_ratio', 100):.1f}%")

        return "。".join(summary_parts) + "。"

    def _generate_overview_content(self, data: Dict) -> str:
        return f"""
本日系统持续运行，主要运行参数保持稳定。
平均水位 {data.get('h_mean', 4.0):.2f}m，平均流速 {data.get('v_mean', 2.0):.2f}m/s。
最高温度出现在向阳侧，达到 {data.get('T_sun_max', 25):.1f}°C。
        """.strip()

    def _generate_safety_content(self, data: Dict) -> str:
        return f"""
安全系统运行正常，安全运行时间占比 {data.get('safe_ratio', 100):.1f}%。
触发安全联锁 {data.get('interlock_count', 0)} 次。
        """.strip()

    def _generate_alert_content(self, data: Dict) -> str:
        return f"""
本日共产生告警 {data.get('total', 0)} 条。
其中紧急告警 {data.get('emergency', 0)} 条，严重告警 {data.get('critical', 0)} 条，
一般告警 {data.get('alarm', 0)} 条，警告 {data.get('warning', 0)} 条。
        """.strip()

    def _generate_control_content(self, data: Dict) -> str:
        return f"""
控制系统执行自动控制动作 {data.get('auto_actions', 0)} 次。
手动干预 {data.get('manual_actions', 0)} 次。
MPC控制器回退率 {data.get('fallback_rate', 0):.1f}%。
        """.strip()

    def _generate_scenario_content(self, data: Dict) -> str:
        scenarios = data.get('detected_scenarios', [])
        if scenarios:
            return f"检测到 {len(scenarios)} 个场景：{', '.join(scenarios)}"
        return "本日无异常场景检测。"

    def _generate_stats_rows(self, data: Dict) -> List[List[str]]:
        rows = []
        params = [
            ('水位 h (m)', 'h'),
            ('流速 v (m/s)', 'v'),
            ('弗劳德数 Fr', 'fr'),
            ('向阳侧温度 (°C)', 'T_sun'),
            ('背阴侧温度 (°C)', 'T_shade'),
            ('伸缩缝间隙 (mm)', 'joint_gap'),
            ('振动幅度 (mm)', 'vib_amp')
        ]

        for name, key in params:
            rows.append([
                name,
                f"{data.get(f'{key}_mean', 0):.2f}",
                f"{data.get(f'{key}_max', 0):.2f}",
                f"{data.get(f'{key}_min', 0):.2f}",
                f"{data.get(f'{key}_std', 0):.3f}"
            ])

        return rows


class SafetyReportTemplate(ReportTemplate):
    """Safety report template"""

    def __init__(self):
        super().__init__("safety", "安全分析报表")

    def generate_sections(self, data: Dict[str, Any],
                         start_time: datetime, end_time: datetime) -> List[ReportSection]:
        sections = []

        # Safety overview
        sections.append(ReportSection(
            title="安全概述",
            content=self._generate_safety_overview(data),
            data=data.get('safety_overview', {})
        ))

        # Fault analysis
        sections.append(ReportSection(
            title="故障分析",
            content=self._generate_fault_analysis(data),
            data=data.get('faults', {}),
            tables=[{
                'title': '故障记录',
                'headers': ['时间', '类型', '严重度', '描述', '处理'],
                'rows': data.get('fault_list', [])
            }]
        ))

        # Interlock status
        sections.append(ReportSection(
            title="联锁动作",
            content=self._generate_interlock_content(data),
            data=data.get('interlocks', {})
        ))

        # Risk assessment
        sections.append(ReportSection(
            title="风险评估",
            content=self._generate_risk_content(data),
            data=data.get('risks', {})
        ))

        return sections

    def generate_summary(self, data: Dict) -> str:
        safety = data.get('safety_overview', {})
        return f"安全运行率{safety.get('safe_ratio', 100):.1f}%，共发生{safety.get('fault_count', 0)}次故障，{safety.get('interlock_count', 0)}次联锁动作。"

    def _generate_safety_overview(self, data: Dict) -> str:
        overview = data.get('safety_overview', {})
        return f"报告期内安全等级分布：正常{overview.get('normal_ratio', 0):.1f}%，警告{overview.get('warning_ratio', 0):.1f}%，危险{overview.get('danger_ratio', 0):.1f}%。"

    def _generate_fault_analysis(self, data: Dict) -> str:
        faults = data.get('faults', {})
        return f"共检测到{faults.get('total', 0)}次故障，其中传感器故障{faults.get('sensor', 0)}次，控制故障{faults.get('control', 0)}次。"

    def _generate_interlock_content(self, data: Dict) -> str:
        interlocks = data.get('interlocks', {})
        return f"联锁系统触发{interlocks.get('triggered_count', 0)}次，成功保护{interlocks.get('protection_count', 0)}次。"

    def _generate_risk_content(self, data: Dict) -> str:
        risks = data.get('risks', {})
        return f"当前风险等级：{risks.get('current_level', '低')}。主要风险因素：{', '.join(risks.get('factors', ['无']))}"


class HTMLReportRenderer:
    """Render report as HTML"""

    def __init__(self):
        self.css = self._default_css()

    def _default_css(self) -> str:
        return """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        .report { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }
        .header { border-bottom: 2px solid #1976d2; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #1976d2; margin: 0; font-size: 28px; }
        .header .subtitle { color: #666; margin-top: 10px; }
        .meta { display: flex; gap: 30px; margin-top: 15px; font-size: 14px; color: #888; }
        .summary { background: #e3f2fd; padding: 20px; border-radius: 6px; margin-bottom: 30px; }
        .summary h3 { margin-top: 0; color: #1976d2; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #333; border-left: 4px solid #1976d2; padding-left: 15px; }
        .section-content { padding: 15px 0; line-height: 1.8; color: #555; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #f5f5f5; padding: 12px; text-align: left; border-bottom: 2px solid #ddd; }
        td { padding: 10px 12px; border-bottom: 1px solid #eee; }
        tr:hover { background: #fafafa; }
        .chart-container { background: #fafafa; padding: 20px; border-radius: 6px; margin: 20px 0; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #888; font-size: 12px; text-align: center; }
        """

    def render(self, report: Report) -> str:
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>{self.css}</style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>{report.title}</h1>
            <div class="subtitle">报表类型：{report.report_type.value}</div>
            <div class="meta">
                <span>报告期间：{report.period_start.strftime('%Y-%m-%d %H:%M')} - {report.period_end.strftime('%Y-%m-%d %H:%M')}</span>
                <span>生成时间：{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>报表ID：{report.report_id}</span>
            </div>
        </div>

        <div class="summary">
            <h3>摘要</h3>
            <p>{report.summary}</p>
        </div>

        {self._render_sections(report.sections)}

        <div class="footer">
            <p>TAOS - 团河渡槽自主运行系统 | 自动生成报表</p>
            <p>报表ID: {report.report_id}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _render_sections(self, sections: List[ReportSection]) -> str:
        html_parts = []
        for section in sections:
            html_parts.append(f"""
        <div class="section">
            <h2>{section.title}</h2>
            <div class="section-content">{section.content}</div>
            {self._render_tables(section.tables)}
            {self._render_charts(section.charts)}
        </div>""")
        return "\n".join(html_parts)

    def _render_tables(self, tables: List[Dict]) -> str:
        if not tables:
            return ""

        html_parts = []
        for table in tables:
            headers = "".join(f"<th>{h}</th>" for h in table.get('headers', []))
            rows = ""
            for row in table.get('rows', []):
                cells = "".join(f"<td>{c}</td>" for c in row)
                rows += f"<tr>{cells}</tr>"

            html_parts.append(f"""
            <div class="table-container">
                <h4>{table.get('title', '')}</h4>
                <table>
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>""")

        return "\n".join(html_parts)

    def _render_charts(self, charts: List[Dict]) -> str:
        if not charts:
            return ""

        # In a real implementation, this would generate actual charts
        # For now, just show placeholders
        html_parts = []
        for chart in charts:
            html_parts.append(f"""
            <div class="chart-container">
                <h4>{chart.get('title', 'Chart')}</h4>
                <p style="color: #999;">[图表: {chart.get('type', 'chart')}]</p>
                <pre>{json.dumps(chart.get('data', {}), indent=2, ensure_ascii=False)}</pre>
            </div>""")

        return "\n".join(html_parts)


class ReportGenerator:
    """
    Main report generator for TAOS
    """

    def __init__(self):
        self.templates: Dict[str, ReportTemplate] = {}
        self.renderers: Dict[ReportFormat, Any] = {}
        self.collector = DataCollector()
        self.generated_reports: List[Report] = []

        # Register default templates
        self._register_default_templates()

        # Register default renderers
        self._register_default_renderers()

        # Register default data collectors
        self._register_default_collectors()

    def _register_default_templates(self):
        """Register default report templates"""
        self.templates['daily'] = DailyReportTemplate()
        self.templates['safety'] = SafetyReportTemplate()

    def _register_default_renderers(self):
        """Register default renderers"""
        self.renderers[ReportFormat.HTML] = HTMLReportRenderer()

    def _register_default_collectors(self):
        """Register default data collectors"""
        # State summary collector
        def collect_state_summary(start: datetime, end: datetime) -> Dict:
            # In real implementation, this would query actual data
            return {
                'uptime_hours': (end - start).total_seconds() / 3600,
                'h_mean': 4.0, 'h_max': 4.5, 'h_min': 3.5, 'h_std': 0.15,
                'v_mean': 2.0, 'v_max': 2.3, 'v_min': 1.8, 'v_std': 0.1,
                'fr_mean': 0.32, 'fr_max': 0.4, 'fr_min': 0.28, 'fr_std': 0.02,
                'T_sun_mean': 30, 'T_sun_max': 45, 'T_sun_min': 20, 'T_sun_std': 5,
                'T_shade_mean': 25, 'T_shade_max': 30, 'T_shade_min': 18, 'T_shade_std': 3,
                'joint_gap_mean': 20, 'joint_gap_max': 22, 'joint_gap_min': 18, 'joint_gap_std': 1,
                'vib_amp_mean': 1, 'vib_amp_max': 3, 'vib_amp_min': 0, 'vib_amp_std': 0.5
            }

        def collect_safety(start: datetime, end: datetime) -> Dict:
            return {
                'safe_ratio': 98.5,
                'interlock_count': 2,
                'level_distribution': {'NORMAL': 95, 'WARNING': 4, 'ALARM': 1}
            }

        def collect_alerts(start: datetime, end: datetime) -> Dict:
            return {
                'total': 5,
                'emergency': 0,
                'critical': 1,
                'alarm': 2,
                'warning': 2,
                'alert_list': [
                    ['2025-12-07 10:30', 'WARNING', '水力', '水位偏高', '已解除'],
                    ['2025-12-07 14:20', 'ALARM', '热力', '温差过大', '已解除']
                ]
            }

        def collect_control(start: datetime, end: datetime) -> Dict:
            return {
                'auto_actions': 150,
                'manual_actions': 3,
                'fallback_rate': 2.5
            }

        def collect_scenarios(start: datetime, end: datetime) -> Dict:
            return {
                'detected_scenarios': ['S3.1 日照弯曲']
            }

        self.collector.register_source('state_summary', collect_state_summary)
        self.collector.register_source('safety', collect_safety)
        self.collector.register_source('alerts', collect_alerts)
        self.collector.register_source('control', collect_control)
        self.collector.register_source('scenarios', collect_scenarios)

    def generate(self, report_type: ReportType, start_time: datetime = None,
                end_time: datetime = None, template_id: str = None) -> Report:
        """Generate a report"""
        # Set default time range
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            if report_type == ReportType.DAILY:
                start_time = end_time - timedelta(days=1)
            elif report_type == ReportType.WEEKLY:
                start_time = end_time - timedelta(weeks=1)
            elif report_type == ReportType.MONTHLY:
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(hours=8)  # Default shift

        # Get template
        if template_id is None:
            template_id = report_type.value
        template = self.templates.get(template_id, self.templates.get('daily'))

        # Collect data
        data = self.collector.collect_all(start_time, end_time)

        # Generate report
        report_id = f"{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        report = Report(
            report_id=report_id,
            report_type=report_type,
            title=f"TAOS {template.name} - {start_time.strftime('%Y-%m-%d')}",
            period_start=start_time,
            period_end=end_time,
            generated_at=datetime.now(),
            sections=template.generate_sections(data, start_time, end_time),
            summary=template.generate_summary(data),
            metadata={'template': template_id, 'data_sources': list(data.keys())}
        )

        self.generated_reports.append(report)
        return report

    def render(self, report: Report, format: ReportFormat = ReportFormat.HTML) -> str:
        """Render report to specified format"""
        if format == ReportFormat.HTML:
            renderer = self.renderers.get(ReportFormat.HTML)
            if renderer:
                return renderer.render(report)

        elif format == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

        elif format == ReportFormat.CSV:
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(['Report ID', report.report_id])
            writer.writerow(['Type', report.report_type.value])
            writer.writerow(['Period', f"{report.period_start} - {report.period_end}"])
            writer.writerow(['Generated', report.generated_at.isoformat()])
            writer.writerow([])
            writer.writerow(['Summary', report.summary])
            writer.writerow([])

            # Write sections
            for section in report.sections:
                writer.writerow([f"=== {section.title} ==="])
                writer.writerow([section.content])
                for table in section.tables:
                    writer.writerow([])
                    writer.writerow([table.get('title', '')])
                    writer.writerow(table.get('headers', []))
                    for row in table.get('rows', []):
                        writer.writerow(row)
                writer.writerow([])

            return output.getvalue()

        elif format == ReportFormat.TEXT:
            lines = []
            lines.append("=" * 60)
            lines.append(report.title)
            lines.append("=" * 60)
            lines.append(f"报告期间: {report.period_start} - {report.period_end}")
            lines.append(f"生成时间: {report.generated_at}")
            lines.append("")
            lines.append("摘要:")
            lines.append(report.summary)
            lines.append("")

            for section in report.sections:
                lines.append("-" * 40)
                lines.append(section.title)
                lines.append("-" * 40)
                lines.append(section.content)
                lines.append("")

            return "\n".join(lines)

        return ""

    def register_template(self, template: ReportTemplate):
        """Register a custom template"""
        self.templates[template.template_id] = template

    def register_collector(self, name: str, collector: Callable):
        """Register a data collector"""
        self.collector.register_source(name, collector)

    def get_recent_reports(self, limit: int = 10) -> List[Report]:
        """Get recently generated reports"""
        return self.generated_reports[-limit:]

    def schedule_report(self, report_type: ReportType, schedule: str,
                       callback: Callable[[Report], None] = None):
        """Schedule automated report generation"""
        # This would integrate with a scheduler like APScheduler
        # For now, just a placeholder
        pass


class ReportScheduler:
    """Automated report scheduler"""

    def __init__(self, generator: ReportGenerator):
        self.generator = generator
        self.schedules: List[Dict] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def add_schedule(self, report_type: ReportType, hour: int, minute: int = 0,
                    callback: Callable = None):
        """Add a daily schedule"""
        self.schedules.append({
            'type': report_type,
            'hour': hour,
            'minute': minute,
            'callback': callback,
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
                if self._should_run(schedule, now):
                    try:
                        report = self.generator.generate(schedule['type'])
                        schedule['last_run'] = now

                        if schedule['callback']:
                            schedule['callback'](report)

                    except Exception as e:
                        print(f"Scheduled report error: {e}")

            time.sleep(60)  # Check every minute

    def _should_run(self, schedule: Dict, now: datetime) -> bool:
        """Check if schedule should run"""
        if now.hour != schedule['hour'] or now.minute != schedule['minute']:
            return False

        last_run = schedule['last_run']
        if last_run is None:
            return True

        return (now - last_run).total_seconds() > 3600


# Global instance
_generator = None


def get_report_generator() -> ReportGenerator:
    """Get global report generator"""
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator


if __name__ == "__main__":
    # Test report generator
    print("=== Report Generator Test ===")

    generator = ReportGenerator()

    # Generate daily report
    print("\n1. Generating daily report...")
    report = generator.generate(ReportType.DAILY)
    print(f"   Report ID: {report.report_id}")
    print(f"   Sections: {len(report.sections)}")
    print(f"   Summary: {report.summary[:100]}...")

    # Render as HTML
    print("\n2. Rendering as HTML...")
    html = generator.render(report, ReportFormat.HTML)
    print(f"   HTML length: {len(html)} chars")

    # Render as JSON
    print("\n3. Rendering as JSON...")
    json_output = generator.render(report, ReportFormat.JSON)
    print(f"   JSON length: {len(json_output)} chars")

    # Render as text
    print("\n4. Rendering as TEXT...")
    text = generator.render(report, ReportFormat.TEXT)
    print(text[:500] + "...")

    # Generate safety report
    print("\n5. Generating safety report...")
    safety_report = generator.generate(ReportType.SAFETY, template_id='safety')
    print(f"   Report ID: {safety_report.report_id}")

    print("\nReport generator test completed!")
