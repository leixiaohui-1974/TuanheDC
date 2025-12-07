#!/usr/bin/env python3
"""
预测与计划信息系统 for TAOS V3.3
Tuanhe Aqueduct Autonomous Operation System

实现前瞻性场景预测和计划管理:
- 天气预报集成（温度、风速、降水）
- 流量预测（上游来水、下游需求）
- 地震预警（主震预警、余震概率）
- 维护计划（设备检修、传感器校准）
- 运行调度（闸门操作、流量目标）
- 应急演练（场景模拟、响应测试）

预测驱动的场景生成:
- 基于预测提前生成预期场景
- 计算场景发生概率
- 优化控制策略准备
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json


class ForecastSource(Enum):
    """预报来源"""
    WEATHER_BUREAU = "weather_bureau"      # 气象局
    HYDROLOGY_CENTER = "hydrology_center"  # 水文中心
    SEISMIC_NETWORK = "seismic_network"    # 地震台网
    INTERNAL_MODEL = "internal_model"       # 内部模型


class PlanType(Enum):
    """计划类型"""
    MAINTENANCE = "maintenance"      # 维护计划
    OPERATION = "operation"          # 运行调度
    EMERGENCY_DRILL = "drill"        # 应急演练
    INSPECTION = "inspection"        # 巡检计划
    CALIBRATION = "calibration"      # 校准计划


class AlertLevel(Enum):
    """预警级别"""
    BLUE = 1    # 关注
    YELLOW = 2  # 预警
    ORANGE = 3  # 警报
    RED = 4     # 紧急


@dataclass
class WeatherForecast:
    """天气预报"""
    timestamp: datetime
    valid_hours: int  # 预报有效时长

    # 温度预报
    T_max: float      # 最高温度
    T_min: float      # 最低温度
    T_trend: str      # 趋势: rising, falling, stable
    cooling_rate: float = 0.0  # 预计降温速率

    # 风速预报
    wind_speed_max: float = 0.0
    wind_direction: str = "N"
    gust_speed: float = 0.0

    # 降水预报
    precipitation: float = 0.0   # mm
    precipitation_prob: float = 0.0
    storm_warning: bool = False

    # 日照
    solar_hours: float = 0.0
    uv_index: int = 0

    # 预报置信度
    confidence: float = 0.8
    source: ForecastSource = ForecastSource.WEATHER_BUREAU


@dataclass
class FlowForecast:
    """流量预报"""
    timestamp: datetime
    valid_hours: int

    # 上游来水预测
    Q_upstream: float       # 预计上游流量
    Q_upstream_max: float   # 可能最大值
    Q_upstream_min: float   # 可能最小值

    # 下游需求预测
    Q_downstream_demand: float  # 下游需水

    # 水库调度信息
    reservoir_release: float = 0.0  # 计划泄水
    release_schedule: List[Tuple[datetime, float]] = field(default_factory=list)

    # 异常预警
    surge_probability: float = 0.0  # 浪涌概率
    low_flow_risk: bool = False     # 低流量风险

    confidence: float = 0.85
    source: ForecastSource = ForecastSource.HYDROLOGY_CENTER


@dataclass
class SeismicAlert:
    """地震预警"""
    timestamp: datetime
    alert_level: AlertLevel

    # 地震信息
    magnitude: float = 0.0          # 震级
    epicenter_distance: float = 0.0  # 震中距离(km)
    estimated_arrival: float = 0.0   # 预计到达时间(s)

    # 地面运动预测
    expected_pga: float = 0.0       # 预计峰值加速度(g)
    expected_duration: float = 0.0   # 预计持续时间(s)

    # 余震预测
    aftershock_probability: float = 0.0
    aftershock_magnitude_range: Tuple[float, float] = (0.0, 0.0)

    # 响应建议
    recommended_action: str = "MONITOR"

    source: ForecastSource = ForecastSource.SEISMIC_NETWORK


@dataclass
class MaintenancePlan:
    """维护计划"""
    plan_id: str
    plan_type: PlanType
    start_time: datetime
    end_time: datetime

    # 影响范围
    affected_systems: List[str] = field(default_factory=list)
    affected_sensors: List[str] = field(default_factory=list)
    affected_actuators: List[str] = field(default_factory=list)

    # 能力降级
    sensor_availability: float = 1.0   # 传感器可用率
    actuator_availability: float = 1.0  # 执行器可用率
    control_capacity: float = 1.0       # 控制能力

    # 计划状态
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    description: str = ""


@dataclass
class OperationSchedule:
    """运行调度计划"""
    schedule_id: str
    start_time: datetime
    end_time: datetime

    # 目标设定
    target_flow: float = 80.0      # 目标流量
    target_level: float = 4.0      # 目标水位
    flow_ramp_rate: float = 5.0    # 流量变化速率

    # 闸门操作
    gate_operations: List[Dict[str, Any]] = field(default_factory=list)

    # 特殊要求
    priority: int = 1              # 优先级
    constraints: Dict[str, Any] = field(default_factory=dict)

    status: str = "scheduled"


class PredictionSystem:
    """
    预测信息系统

    整合各类预测信息，为场景预判提供数据支持
    """

    def __init__(self):
        # 当前预报
        self.weather_forecast: Optional[WeatherForecast] = None
        self.flow_forecast: Optional[FlowForecast] = None
        self.seismic_alert: Optional[SeismicAlert] = None

        # 预报历史
        self.forecast_history: List[Dict] = []

        # 场景概率预测
        self.scenario_probabilities: Dict[str, float] = {}

        # 预测模型参数
        self.thermal_sensitivity = 0.5   # 热响应敏感度
        self.hydraulic_lag = 2.0         # 水力响应滞后(小时)
        self.seismic_response_time = 10  # 地震响应时间(秒)

    def update_weather_forecast(self, forecast: WeatherForecast):
        """更新天气预报"""
        self.weather_forecast = forecast
        self._update_scenario_probabilities()
        self.forecast_history.append({
            'type': 'weather',
            'timestamp': datetime.now().isoformat(),
            'data': self._forecast_to_dict(forecast)
        })

    def update_flow_forecast(self, forecast: FlowForecast):
        """更新流量预报"""
        self.flow_forecast = forecast
        self._update_scenario_probabilities()
        self.forecast_history.append({
            'type': 'flow',
            'timestamp': datetime.now().isoformat(),
            'data': self._forecast_to_dict(forecast)
        })

    def update_seismic_alert(self, alert: SeismicAlert):
        """更新地震预警"""
        self.seismic_alert = alert
        self._update_scenario_probabilities()
        self.forecast_history.append({
            'type': 'seismic',
            'timestamp': datetime.now().isoformat(),
            'data': self._alert_to_dict(alert)
        })

    def _update_scenario_probabilities(self):
        """基于预测信息更新场景发生概率"""
        probs = {
            'NORMAL': 1.0,
            'S1.1': 0.0, 'S1.2': 0.0, 'S1.3': 0.0,
            'S2.1': 0.0, 'S2.2': 0.0,
            'S3.1': 0.0, 'S3.2': 0.0, 'S3.3': 0.0, 'S3.4': 0.0,
            'S4.1': 0.0, 'S4.2': 0.0, 'S4.3': 0.0,
            'S5.1': 0.0, 'S5.2': 0.0,
            'S6.1': 0.0, 'S6.2': 0.0, 'S6.3': 0.0, 'S6.4': 0.0,
        }

        # 基于天气预报计算场景概率
        if self.weather_forecast:
            wf = self.weather_forecast

            # 热力场景概率
            delta_T_expected = wf.T_max - wf.T_min
            if delta_T_expected > 15:
                probs['S3.1'] = min(0.9, delta_T_expected / 25)  # 热弯曲
            if wf.cooling_rate > 2.0:
                probs['S3.2'] = min(0.8, wf.cooling_rate / 5)  # 快速冷却
            if wf.T_max > 38:
                probs['S3.4'] = min(0.7, (wf.T_max - 35) / 15)  # 热胀应力

            # 接缝场景概率
            if wf.T_min < -5:
                probs['S4.1'] = min(0.8, (-wf.T_min) / 20)  # 接缝膨胀
            if wf.T_max > 40:
                probs['S4.2'] = min(0.7, (wf.T_max - 35) / 15)  # 接缝压缩

            # 风振场景概率
            if wf.wind_speed_max > 10:
                probs['S2.1'] = min(0.8, wf.wind_speed_max / 25)  # VIV
            if wf.gust_speed > 15:
                probs['S2.2'] = min(0.7, wf.gust_speed / 30)  # 抖振

        # 基于流量预报计算场景概率
        if self.flow_forecast:
            ff = self.flow_forecast

            if ff.surge_probability > 0.3:
                probs['S1.2'] = ff.surge_probability  # 浪涌
            if ff.Q_upstream_max > 150:
                probs['S1.1'] = min(0.8, (ff.Q_upstream_max - 100) / 100)  # 水力跳跃
            if ff.low_flow_risk:
                probs['S1.3'] = 0.6  # 低水位

        # 基于地震预警计算场景概率
        if self.seismic_alert:
            sa = self.seismic_alert

            if sa.alert_level.value >= AlertLevel.YELLOW.value:
                probs['S5.1'] = min(0.95, 0.3 + sa.expected_pga * 2)  # 主震
                probs['S5.2'] = sa.aftershock_probability  # 余震

        # 更新NORMAL概率（无异常场景时）
        max_risk_prob = max(probs[k] for k in probs if k != 'NORMAL')
        probs['NORMAL'] = max(0.0, 1.0 - max_risk_prob)

        self.scenario_probabilities = probs

    def get_predicted_scenarios(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """获取高概率预测场景"""
        return [
            (s, p) for s, p in self.scenario_probabilities.items()
            if p >= threshold and s != 'NORMAL'
        ]

    def get_forecast_summary(self) -> Dict[str, Any]:
        """获取预报摘要"""
        return {
            'weather': self._forecast_to_dict(self.weather_forecast) if self.weather_forecast else None,
            'flow': self._forecast_to_dict(self.flow_forecast) if self.flow_forecast else None,
            'seismic': self._alert_to_dict(self.seismic_alert) if self.seismic_alert else None,
            'scenario_probabilities': self.scenario_probabilities,
            'high_probability_scenarios': self.get_predicted_scenarios(0.3),
            'forecast_confidence': self._calculate_overall_confidence()
        }

    def _calculate_overall_confidence(self) -> float:
        """计算整体预报置信度"""
        confidences = []
        if self.weather_forecast:
            confidences.append(self.weather_forecast.confidence)
        if self.flow_forecast:
            confidences.append(self.flow_forecast.confidence)
        if self.seismic_alert:
            confidences.append(0.9)  # 地震预警置信度较高
        return sum(confidences) / len(confidences) if confidences else 0.5

    def _forecast_to_dict(self, forecast) -> Dict:
        """将预报对象转为字典"""
        if forecast is None:
            return {}
        result = {}
        for key, value in forecast.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list):
                result[key] = [
                    (t.isoformat(), v) if isinstance(t, datetime) else (t, v)
                    for t, v in value
                ] if value and isinstance(value[0], tuple) else value
            else:
                result[key] = value
        return result

    def _alert_to_dict(self, alert) -> Dict:
        """将预警对象转为字典"""
        if alert is None:
            return {}
        result = {}
        for key, value in alert.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result


class PlanningSystem:
    """
    计划信息系统

    管理各类运行和维护计划
    """

    def __init__(self):
        # 维护计划
        self.maintenance_plans: List[MaintenancePlan] = []

        # 运行调度
        self.operation_schedules: List[OperationSchedule] = []

        # 应急演练计划
        self.drill_schedules: List[Dict] = []

        # 当前活动计划
        self.active_plans: List[str] = []

    def add_maintenance_plan(self, plan: MaintenancePlan):
        """添加维护计划"""
        self.maintenance_plans.append(plan)
        self._sort_plans()

    def add_operation_schedule(self, schedule: OperationSchedule):
        """添加运行调度"""
        self.operation_schedules.append(schedule)
        self._sort_schedules()

    def add_drill_schedule(self, drill: Dict):
        """添加演练计划"""
        self.drill_schedules.append(drill)

    def get_active_plans(self, current_time: datetime = None) -> List[Dict]:
        """获取当前活动的计划"""
        if current_time is None:
            current_time = datetime.now()

        active = []

        # 检查维护计划
        for plan in self.maintenance_plans:
            if plan.start_time <= current_time <= plan.end_time:
                if plan.status == 'scheduled':
                    plan.status = 'in_progress'
                active.append({
                    'type': 'maintenance',
                    'plan_id': plan.plan_id,
                    'plan_type': plan.plan_type.value,
                    'affected_systems': plan.affected_systems,
                    'sensor_availability': plan.sensor_availability,
                    'actuator_availability': plan.actuator_availability,
                    'control_capacity': plan.control_capacity,
                    'description': plan.description
                })

        # 检查运行调度
        for schedule in self.operation_schedules:
            if schedule.start_time <= current_time <= schedule.end_time:
                if schedule.status == 'scheduled':
                    schedule.status = 'in_progress'
                active.append({
                    'type': 'operation',
                    'schedule_id': schedule.schedule_id,
                    'target_flow': schedule.target_flow,
                    'target_level': schedule.target_level,
                    'gate_operations': schedule.gate_operations,
                    'priority': schedule.priority
                })

        return active

    def get_upcoming_plans(self, hours: int = 24) -> List[Dict]:
        """获取未来指定时间内的计划"""
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=hours)

        upcoming = []

        for plan in self.maintenance_plans:
            if current_time < plan.start_time <= future_time:
                upcoming.append({
                    'type': 'maintenance',
                    'plan_id': plan.plan_id,
                    'start_time': plan.start_time.isoformat(),
                    'end_time': plan.end_time.isoformat(),
                    'description': plan.description
                })

        for schedule in self.operation_schedules:
            if current_time < schedule.start_time <= future_time:
                upcoming.append({
                    'type': 'operation',
                    'schedule_id': schedule.schedule_id,
                    'start_time': schedule.start_time.isoformat(),
                    'end_time': schedule.end_time.isoformat(),
                    'target_flow': schedule.target_flow
                })

        return upcoming

    def get_plan_impacts(self) -> Dict[str, Any]:
        """获取计划对系统的影响"""
        active_plans = self.get_active_plans()

        impact = {
            'sensor_availability': 1.0,
            'actuator_availability': 1.0,
            'control_capacity': 1.0,
            'target_overrides': {},
            'active_plan_count': len(active_plans)
        }

        for plan in active_plans:
            if plan['type'] == 'maintenance':
                impact['sensor_availability'] = min(
                    impact['sensor_availability'],
                    plan.get('sensor_availability', 1.0)
                )
                impact['actuator_availability'] = min(
                    impact['actuator_availability'],
                    plan.get('actuator_availability', 1.0)
                )
                impact['control_capacity'] = min(
                    impact['control_capacity'],
                    plan.get('control_capacity', 1.0)
                )
            elif plan['type'] == 'operation':
                impact['target_overrides']['target_flow'] = plan.get('target_flow')
                impact['target_overrides']['target_level'] = plan.get('target_level')

        return impact

    def _sort_plans(self):
        """按开始时间排序维护计划"""
        self.maintenance_plans.sort(key=lambda x: x.start_time)

    def _sort_schedules(self):
        """按优先级和开始时间排序运行调度"""
        self.operation_schedules.sort(key=lambda x: (-x.priority, x.start_time))


class PredictiveScenarioManager:
    """
    预测性场景管理器

    整合预测和计划信息，实现前瞻性场景管理
    """

    def __init__(self):
        self.prediction = PredictionSystem()
        self.planning = PlanningSystem()

        # 场景预警阈值
        self.warning_threshold = 0.5
        self.alert_threshold = 0.7

        # 预判状态
        self.predicted_scenarios: List[str] = []
        self.scenario_timeline: List[Dict] = []

    def update_predictions(self, weather: WeatherForecast = None,
                          flow: FlowForecast = None,
                          seismic: SeismicAlert = None):
        """更新所有预测信息"""
        if weather:
            self.prediction.update_weather_forecast(weather)
        if flow:
            self.prediction.update_flow_forecast(flow)
        if seismic:
            self.prediction.update_seismic_alert(seismic)

        self._update_predicted_scenarios()

    def _update_predicted_scenarios(self):
        """更新预测场景列表"""
        high_prob = self.prediction.get_predicted_scenarios(self.warning_threshold)
        self.predicted_scenarios = [s for s, _ in high_prob]

        # 考虑计划影响
        plan_impact = self.planning.get_plan_impacts()

        # 如果传感器可用率低，增加S6.1概率
        if plan_impact['sensor_availability'] < 0.8:
            if 'S6.1' not in self.predicted_scenarios:
                self.predicted_scenarios.append('S6.1')

        # 如果执行器可用率低，增加S6.2概率
        if plan_impact['actuator_availability'] < 0.8:
            if 'S6.2' not in self.predicted_scenarios:
                self.predicted_scenarios.append('S6.2')

        # 如果控制能力降低，增加S6.4概率
        if plan_impact['control_capacity'] < 0.7:
            if 'S6.4' not in self.predicted_scenarios:
                self.predicted_scenarios.append('S6.4')

    def get_scenario_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """获取场景预报"""
        return {
            'current_time': datetime.now().isoformat(),
            'forecast_horizon_hours': hours,
            'predicted_scenarios': self.predicted_scenarios,
            'scenario_probabilities': self.prediction.scenario_probabilities,
            'active_plans': self.planning.get_active_plans(),
            'upcoming_plans': self.planning.get_upcoming_plans(hours),
            'plan_impacts': self.planning.get_plan_impacts(),
            'forecast_summary': self.prediction.get_forecast_summary(),
            'alerts': self._generate_alerts()
        }

    def _generate_alerts(self) -> List[Dict]:
        """生成预警信息"""
        alerts = []

        for scenario, prob in self.prediction.scenario_probabilities.items():
            if prob >= self.alert_threshold and scenario != 'NORMAL':
                alerts.append({
                    'level': 'ALERT',
                    'scenario': scenario,
                    'probability': prob,
                    'message': f'High probability of {scenario}: {prob:.0%}'
                })
            elif prob >= self.warning_threshold and scenario != 'NORMAL':
                alerts.append({
                    'level': 'WARNING',
                    'scenario': scenario,
                    'probability': prob,
                    'message': f'Elevated probability of {scenario}: {prob:.0%}'
                })

        # 检查地震预警
        if self.prediction.seismic_alert:
            sa = self.prediction.seismic_alert
            if sa.alert_level.value >= AlertLevel.ORANGE.value:
                alerts.append({
                    'level': 'EMERGENCY',
                    'scenario': 'S5.1',
                    'message': f'Seismic alert: M{sa.magnitude} expected in {sa.estimated_arrival}s',
                    'recommended_action': sa.recommended_action
                })

        return sorted(alerts, key=lambda x: {'EMERGENCY': 0, 'ALERT': 1, 'WARNING': 2}.get(x['level'], 3))

    def get_recommended_preparations(self) -> List[Dict]:
        """获取建议的准备措施"""
        preparations = []
        probs = self.prediction.scenario_probabilities

        # 基于预测场景推荐准备措施
        if probs.get('S1.1', 0) > 0.3 or probs.get('S1.2', 0) > 0.3:
            preparations.append({
                'scenario': 'Hydraulic',
                'action': 'Pre-adjust water level to target 6-7m',
                'reason': 'High probability of hydraulic events',
                'priority': 'HIGH' if max(probs.get('S1.1', 0), probs.get('S1.2', 0)) > 0.6 else 'MEDIUM'
            })

        if probs.get('S3.1', 0) > 0.4 or probs.get('S3.2', 0) > 0.4:
            preparations.append({
                'scenario': 'Thermal',
                'action': 'Increase flow for thermal buffering',
                'reason': 'Significant temperature variations expected',
                'priority': 'HIGH' if max(probs.get('S3.1', 0), probs.get('S3.2', 0)) > 0.6 else 'MEDIUM'
            })

        if probs.get('S5.1', 0) > 0.2:
            preparations.append({
                'scenario': 'Seismic',
                'action': 'Lower water level to 2.5-3m, prepare emergency drainage',
                'reason': 'Seismic activity warning',
                'priority': 'CRITICAL' if probs.get('S5.1', 0) > 0.5 else 'HIGH'
            })

        if probs.get('S2.1', 0) > 0.3:
            preparations.append({
                'scenario': 'Wind',
                'action': 'Increase water level for mass damping',
                'reason': 'High wind conditions expected',
                'priority': 'MEDIUM'
            })

        # 基于计划影响推荐措施
        impact = self.planning.get_plan_impacts()
        if impact['sensor_availability'] < 1.0:
            preparations.append({
                'scenario': 'Maintenance',
                'action': 'Enable conservative control mode during maintenance',
                'reason': f'Sensor availability reduced to {impact["sensor_availability"]:.0%}',
                'priority': 'HIGH' if impact['sensor_availability'] < 0.7 else 'MEDIUM'
            })

        return sorted(preparations, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(x['priority'], 4))


def create_sample_forecasts() -> Tuple[WeatherForecast, FlowForecast, Optional[SeismicAlert]]:
    """创建示例预报数据用于测试"""
    now = datetime.now()

    weather = WeatherForecast(
        timestamp=now,
        valid_hours=24,
        T_max=38.0,
        T_min=22.0,
        T_trend='rising',
        cooling_rate=0.0,
        wind_speed_max=15.0,
        wind_direction='NW',
        gust_speed=22.0,
        precipitation=0.0,
        precipitation_prob=0.1,
        storm_warning=False,
        solar_hours=10.0,
        uv_index=8,
        confidence=0.85
    )

    flow = FlowForecast(
        timestamp=now,
        valid_hours=24,
        Q_upstream=120.0,
        Q_upstream_max=160.0,
        Q_upstream_min=90.0,
        Q_downstream_demand=100.0,
        reservoir_release=30.0,
        release_schedule=[(now + timedelta(hours=6), 50.0)],
        surge_probability=0.35,
        low_flow_risk=False,
        confidence=0.8
    )

    # 无地震预警
    seismic = None

    return weather, flow, seismic


if __name__ == '__main__':
    # 测试预测与计划系统
    print("=" * 60)
    print("TAOS V3.3 预测与计划信息系统测试")
    print("=" * 60)

    manager = PredictiveScenarioManager()

    # 创建示例预报
    weather, flow, seismic = create_sample_forecasts()

    # 更新预测
    manager.update_predictions(weather=weather, flow=flow, seismic=seismic)

    # 添加维护计划
    now = datetime.now()
    maintenance = MaintenancePlan(
        plan_id="M001",
        plan_type=PlanType.CALIBRATION,
        start_time=now,
        end_time=now + timedelta(hours=2),
        affected_sensors=['h_sensor_1', 'T_sensor_2'],
        sensor_availability=0.8,
        description="水位和温度传感器校准"
    )
    manager.planning.add_maintenance_plan(maintenance)

    # 获取场景预报
    forecast = manager.get_scenario_forecast(24)

    print("\n场景概率预测:")
    for scenario, prob in sorted(forecast['scenario_probabilities'].items(),
                                  key=lambda x: -x[1]):
        if prob > 0.1:
            print(f"  {scenario}: {prob:.1%}")

    print("\n预警信息:")
    for alert in forecast['alerts']:
        print(f"  [{alert['level']}] {alert['message']}")

    print("\n建议准备措施:")
    for prep in manager.get_recommended_preparations():
        print(f"  [{prep['priority']}] {prep['scenario']}: {prep['action']}")

    print("\n活动计划影响:")
    impact = forecast['plan_impacts']
    print(f"  传感器可用率: {impact['sensor_availability']:.0%}")
    print(f"  执行器可用率: {impact['actuator_availability']:.0%}")
    print(f"  控制能力: {impact['control_capacity']:.0%}")

    print("\n" + "=" * 60)
