#!/usr/bin/env python3
"""
全场景空间生成器 for TAOS V3.2
Tuanhe Aqueduct Autonomous Operation System

实现真正的全场景覆盖：
- 6大类20+基础场景
- 5级强度分级
- 自动组合场景生成（2-4场景组合）
- 环境因素叠加（季节×昼夜×天气）
- 总计可生成10000+独立场景配置

场景编码规则：
- 基础场景: S{类别}.{编号}.{强度} 例如 S1.1.3 = 水力跳跃-中等强度
- 组合场景: C_{场景1}_{场景2}_... 例如 C_S1.1_S3.1 = 水力跳跃+热弯曲
- 环境场景: E_{季节}_{时段}_{天气} 例如 E_SUMMER_NOON_CLEAR
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import json


class ScenarioCategory(Enum):
    """场景大类"""
    HYDRAULIC = "S1"      # 水力类
    WIND = "S2"           # 风振类
    THERMAL = "S3"        # 热力类
    STRUCTURAL = "S4"     # 结构类
    SEISMIC = "S5"        # 地震类
    FAULT = "S6"          # 故障类
    ENVIRONMENT = "E"     # 环境类
    COMBINED = "C"        # 组合类


class Severity(Enum):
    """场景强度等级"""
    LEVEL_1 = 1  # 轻微
    LEVEL_2 = 2  # 较轻
    LEVEL_3 = 3  # 中等
    LEVEL_4 = 4  # 严重
    LEVEL_5 = 5  # 极端


class Season(Enum):
    """季节"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class TimeOfDay(Enum):
    """时段"""
    DAWN = "dawn"        # 黎明 5-7
    MORNING = "morning"  # 上午 7-12
    NOON = "noon"        # 正午 12-14
    AFTERNOON = "afternoon"  # 下午 14-18
    EVENING = "evening"  # 傍晚 18-20
    NIGHT = "night"      # 夜间 20-5


class Weather(Enum):
    """天气"""
    CLEAR = "clear"      # 晴天
    CLOUDY = "cloudy"    # 多云
    RAINY = "rainy"      # 雨天
    STORMY = "stormy"    # 暴风雨
    SNOWY = "snowy"      # 雪天
    FOGGY = "foggy"      # 雾天


@dataclass
class ScenarioConfig:
    """场景配置"""
    id: str
    name: str
    category: ScenarioCategory
    severity: Severity
    parameters: Dict[str, float]
    incompatible_with: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EnvironmentConfig:
    """环境配置"""
    season: Season
    time_of_day: TimeOfDay
    weather: Weather
    T_ambient: float
    solar_rad: float
    wind_speed: float
    humidity: float


class FullScenarioSpace:
    """
    全场景空间管理器

    生成并管理所有可能的场景组合
    """

    def __init__(self):
        # 基础场景定义
        self.base_scenarios = self._define_base_scenarios()

        # 场景兼容性矩阵
        self.compatibility_matrix = self._build_compatibility_matrix()

        # 环境配置空间
        self.environment_space = self._define_environment_space()

        # 缓存生成的组合场景
        self._combined_scenarios_cache = {}
        self._scenario_count = 0

    def _define_base_scenarios(self) -> Dict[str, ScenarioConfig]:
        """定义所有基础场景（含5级强度）"""
        scenarios = {}

        # ============ S1: 水力类场景 ============
        for level in range(1, 6):
            fr_values = [0.75, 0.85, 0.95, 1.1, 1.3]
            h_values = [3.5, 3.0, 2.5, 2.0, 1.5]
            Q_values = [100, 120, 140, 160, 180]

            # S1.1 水力跳跃
            scenarios[f"S1.1.{level}"] = ScenarioConfig(
                id=f"S1.1.{level}",
                name=f"水力跳跃-L{level}",
                category=ScenarioCategory.HYDRAULIC,
                severity=Severity(level),
                parameters={
                    'h': h_values[level-1],
                    'Q_in': Q_values[level-1],
                    'target_fr': fr_values[level-1]
                },
                incompatible_with=['S5.1.4', 'S5.1.5'],  # 强震时不可能有水力跳跃
                description=f"高流速导致的水力跳跃，Fr={fr_values[level-1]}"
            )

            # S1.2 浪涌波
            surge_Q = [110, 130, 150, 180, 220]
            scenarios[f"S1.2.{level}"] = ScenarioConfig(
                id=f"S1.2.{level}",
                name=f"浪涌波-L{level}",
                category=ScenarioCategory.HYDRAULIC,
                severity=Severity(level),
                parameters={
                    'Q_surge': surge_Q[level-1],
                    'surge_duration': 30 + level * 20
                },
                description=f"上游来水浪涌，峰值流量{surge_Q[level-1]}m³/s"
            )

            # S1.3 低水位
            low_h = [3.0, 2.5, 2.0, 1.5, 1.0]
            scenarios[f"S1.3.{level}"] = ScenarioConfig(
                id=f"S1.3.{level}",
                name=f"低水位-L{level}",
                category=ScenarioCategory.HYDRAULIC,
                severity=Severity(level),
                parameters={
                    'h': low_h[level-1],
                    'Q_in': 50 - level * 8
                },
                description=f"水位过低，h={low_h[level-1]}m"
            )

        # ============ S2: 风振类场景 ============
        for level in range(1, 6):
            wind_speeds = [8, 12, 16, 20, 25]
            vib_amps = [10, 20, 35, 50, 70]

            # S2.1 涡激振动
            scenarios[f"S2.1.{level}"] = ScenarioConfig(
                id=f"S2.1.{level}",
                name=f"涡激振动-L{level}",
                category=ScenarioCategory.WIND,
                severity=Severity(level),
                parameters={
                    'wind_speed': wind_speeds[level-1],
                    'vib_amp': vib_amps[level-1]
                },
                description=f"风速{wind_speeds[level-1]}m/s引起的涡激振动"
            )

            # S2.2 抖振
            scenarios[f"S2.2.{level}"] = ScenarioConfig(
                id=f"S2.2.{level}",
                name=f"抖振-L{level}",
                category=ScenarioCategory.WIND,
                severity=Severity(level),
                parameters={
                    'wind_speed': wind_speeds[level-1] + 5,
                    'turbulence': 0.1 + level * 0.05
                },
                description=f"强风引起的结构抖振"
            )

        # ============ S3: 热力类场景 ============
        for level in range(1, 6):
            delta_T = [8, 12, 17, 22, 28]
            cooling_rates = [1, 2, 3.5, 5, 8]

            # S3.1 热弯曲（日照温差）
            scenarios[f"S3.1.{level}"] = ScenarioConfig(
                id=f"S3.1.{level}",
                name=f"热弯曲-L{level}",
                category=ScenarioCategory.THERMAL,
                severity=Severity(level),
                parameters={
                    'T_sun': 25 + delta_T[level-1],
                    'T_shade': 25,
                    'delta_T': delta_T[level-1]
                },
                description=f"日照温差{delta_T[level-1]}°C导致的热弯曲"
            )

            # S3.2 快速冷却
            scenarios[f"S3.2.{level}"] = ScenarioConfig(
                id=f"S3.2.{level}",
                name=f"快速冷却-L{level}",
                category=ScenarioCategory.THERMAL,
                severity=Severity(level),
                parameters={
                    'cooling_rate': cooling_rates[level-1],
                    'T_drop': cooling_rates[level-1] * 10
                },
                description=f"快速降温{cooling_rates[level-1]}°C/min"
            )

            # S3.3 轴承锁定
            lock_stress = [32, 36, 42, 50, 60]
            scenarios[f"S3.3.{level}"] = ScenarioConfig(
                id=f"S3.3.{level}",
                name=f"轴承锁定-L{level}",
                category=ScenarioCategory.THERMAL,
                severity=Severity(level),
                parameters={
                    'bearing_locked': True,
                    'bearing_stress': lock_stress[level-1]
                },
                description=f"轴承锁定，应力{lock_stress[level-1]}MPa"
            )

            # S3.4 热胀应力
            scenarios[f"S3.4.{level}"] = ScenarioConfig(
                id=f"S3.4.{level}",
                name=f"热胀应力-L{level}",
                category=ScenarioCategory.THERMAL,
                severity=Severity(level),
                parameters={
                    'T_ambient': 30 + level * 5,
                    'thermal_stress': 5 + level * 3
                },
                description=f"高温热胀应力"
            )

        # ============ S4: 结构类场景 ============
        for level in range(1, 6):
            gap_expand = [26, 28, 31, 34, 38]
            gap_compress = [8, 6, 4, 2, 0]

            # S4.1 接缝膨胀（寒冷）
            scenarios[f"S4.1.{level}"] = ScenarioConfig(
                id=f"S4.1.{level}",
                name=f"接缝膨胀-L{level}",
                category=ScenarioCategory.STRUCTURAL,
                severity=Severity(level),
                parameters={
                    'joint_gap': gap_expand[level-1],
                    'T_ambient': -5 - level * 5
                },
                description=f"低温接缝扩大至{gap_expand[level-1]}mm"
            )

            # S4.2 接缝压缩（高温）
            scenarios[f"S4.2.{level}"] = ScenarioConfig(
                id=f"S4.2.{level}",
                name=f"接缝压缩-L{level}",
                category=ScenarioCategory.STRUCTURAL,
                severity=Severity(level),
                parameters={
                    'joint_gap': gap_compress[level-1],
                    'T_ambient': 35 + level * 3
                },
                description=f"高温接缝收缩至{gap_compress[level-1]}mm"
            )

            # S4.3 结构疲劳
            scenarios[f"S4.3.{level}"] = ScenarioConfig(
                id=f"S4.3.{level}",
                name=f"结构疲劳-L{level}",
                category=ScenarioCategory.STRUCTURAL,
                severity=Severity(level),
                parameters={
                    'fatigue_cycles': 10000 * level,
                    'crack_length': level * 0.5
                },
                description=f"累积疲劳损伤"
            )

        # ============ S5: 地震类场景 ============
        for level in range(1, 6):
            accels = [0.05, 0.15, 0.3, 0.5, 0.8]
            vibs = [15, 30, 50, 80, 120]

            # S5.1 主震
            scenarios[f"S5.1.{level}"] = ScenarioConfig(
                id=f"S5.1.{level}",
                name=f"主震-L{level}",
                category=ScenarioCategory.SEISMIC,
                severity=Severity(level),
                parameters={
                    'ground_accel': accels[level-1],
                    'vib_amp': vibs[level-1],
                    'duration': 10 + level * 10
                },
                incompatible_with=['S1.1.4', 'S1.1.5'] if level >= 4 else [],
                description=f"地震加速度{accels[level-1]}g"
            )

            # S5.2 余震
            scenarios[f"S5.2.{level}"] = ScenarioConfig(
                id=f"S5.2.{level}",
                name=f"余震-L{level}",
                category=ScenarioCategory.SEISMIC,
                severity=Severity(level),
                parameters={
                    'ground_accel': accels[level-1] * 0.5,
                    'num_aftershocks': level * 2
                },
                requires=['S5.1'],
                description=f"余震序列，{level*2}次"
            )

        # ============ S6: 故障类场景 ============
        for level in range(1, 6):
            # S6.1 传感器故障
            scenarios[f"S6.1.{level}"] = ScenarioConfig(
                id=f"S6.1.{level}",
                name=f"传感器故障-L{level}",
                category=ScenarioCategory.FAULT,
                severity=Severity(level),
                parameters={
                    'sensor_degradation': 0.1 * level,
                    'affected_sensors': level
                },
                description=f"{level}个传感器异常"
            )

            # S6.2 执行器故障
            scenarios[f"S6.2.{level}"] = ScenarioConfig(
                id=f"S6.2.{level}",
                name=f"执行器故障-L{level}",
                category=ScenarioCategory.FAULT,
                severity=Severity(level),
                parameters={
                    'actuator_fault': True,
                    'response_delay': level * 0.5
                },
                description=f"执行器响应延迟{level*0.5}s"
            )

            # S6.3 通信故障
            scenarios[f"S6.3.{level}"] = ScenarioConfig(
                id=f"S6.3.{level}",
                name=f"通信故障-L{level}",
                category=ScenarioCategory.FAULT,
                severity=Severity(level),
                parameters={
                    'comm_loss_rate': 0.1 * level,
                    'latency': level * 100  # ms
                },
                description=f"通信丢包率{level*10}%"
            )

            # S6.4 控制器故障
            scenarios[f"S6.4.{level}"] = ScenarioConfig(
                id=f"S6.4.{level}",
                name=f"控制器故障-L{level}",
                category=ScenarioCategory.FAULT,
                severity=Severity(level),
                parameters={
                    'controller_mode': 'degraded',
                    'capability': 1.0 - level * 0.15
                },
                description=f"控制器降级至{100-level*15}%能力"
            )

        return scenarios

    def _build_compatibility_matrix(self) -> Dict[str, Set[str]]:
        """构建场景兼容性矩阵"""
        matrix = {}
        for sid, scenario in self.base_scenarios.items():
            incompatible = set(scenario.incompatible_with)
            # 添加逻辑不兼容（同类高强度互斥）
            prefix = sid.rsplit('.', 1)[0]  # 如 S1.1
            level = int(sid.split('.')[-1])
            if level >= 4:
                for l in range(4, 6):
                    if l != level:
                        incompatible.add(f"{prefix}.{l}")
            matrix[sid] = incompatible
        return matrix

    def _define_environment_space(self) -> List[EnvironmentConfig]:
        """定义环境配置空间"""
        configs = []

        # 环境参数表
        env_params = {
            (Season.SPRING, TimeOfDay.MORNING, Weather.CLEAR): (15, 0.6, 3, 60),
            (Season.SPRING, TimeOfDay.NOON, Weather.CLEAR): (22, 0.9, 4, 55),
            (Season.SPRING, TimeOfDay.NIGHT, Weather.RAINY): (12, 0, 6, 85),

            (Season.SUMMER, TimeOfDay.DAWN, Weather.CLEAR): (25, 0.3, 2, 70),
            (Season.SUMMER, TimeOfDay.NOON, Weather.CLEAR): (38, 1.0, 3, 50),
            (Season.SUMMER, TimeOfDay.AFTERNOON, Weather.STORMY): (30, 0.2, 18, 90),
            (Season.SUMMER, TimeOfDay.NIGHT, Weather.CLEAR): (28, 0, 2, 65),

            (Season.AUTUMN, TimeOfDay.MORNING, Weather.FOGGY): (12, 0.2, 1, 95),
            (Season.AUTUMN, TimeOfDay.NOON, Weather.CLOUDY): (18, 0.4, 5, 60),
            (Season.AUTUMN, TimeOfDay.EVENING, Weather.RAINY): (10, 0, 8, 88),

            (Season.WINTER, TimeOfDay.DAWN, Weather.SNOWY): (-8, 0.1, 5, 80),
            (Season.WINTER, TimeOfDay.NOON, Weather.CLEAR): (5, 0.7, 8, 40),
            (Season.WINTER, TimeOfDay.NIGHT, Weather.CLEAR): (-15, 0, 4, 50),
            (Season.WINTER, TimeOfDay.AFTERNOON, Weather.STORMY): (-5, 0.1, 20, 75),
        }

        for (season, time, weather), (T, solar, wind, humidity) in env_params.items():
            configs.append(EnvironmentConfig(
                season=season,
                time_of_day=time,
                weather=weather,
                T_ambient=T,
                solar_rad=solar,
                wind_speed=wind,
                humidity=humidity
            ))

        return configs

    def get_base_scenario_count(self) -> int:
        """获取基础场景数量"""
        return len(self.base_scenarios)

    def generate_combined_scenarios(self, max_combination: int = 3) -> List[Dict[str, Any]]:
        """
        生成组合场景

        Args:
            max_combination: 最大组合数量（2-4）

        Returns:
            组合场景列表
        """
        combined = []
        base_ids = list(self.base_scenarios.keys())

        for n in range(2, min(max_combination + 1, 5)):
            for combo in combinations(base_ids, n):
                # 检查兼容性
                if self._is_compatible(combo):
                    # 检查是否有实际意义（不同类别组合更有价值）
                    categories = set(self.base_scenarios[s].category for s in combo)
                    if len(categories) >= 2:  # 至少两个不同类别
                        combined_id = "C_" + "_".join(combo)
                        combined.append({
                            'id': combined_id,
                            'components': list(combo),
                            'categories': [c.value for c in categories],
                            'total_severity': sum(
                                self.base_scenarios[s].severity.value for s in combo
                            )
                        })

        return combined

    def _is_compatible(self, scenario_ids: Tuple[str, ...]) -> bool:
        """检查场景组合是否兼容"""
        for sid in scenario_ids:
            incompatible = self.compatibility_matrix.get(sid, set())
            for other in scenario_ids:
                if other != sid and other in incompatible:
                    return False
        return True

    def generate_full_scenario_space(self) -> Dict[str, Any]:
        """
        生成完整场景空间

        Returns:
            包含所有场景的字典
        """
        # 基础场景
        base_count = len(self.base_scenarios)

        # 组合场景（2-3组合）
        combined_2 = self.generate_combined_scenarios(2)
        combined_3 = self.generate_combined_scenarios(3)

        # 环境场景
        env_count = len(self.environment_space)

        # 完整场景空间 = 基础 × 环境 + 组合 × 环境
        total_single = base_count * env_count
        total_combined = (len(combined_2) + len(combined_3)) * env_count
        total = base_count + len(combined_2) + len(combined_3) + total_single + total_combined

        return {
            'base_scenarios': base_count,
            'combined_2': len(combined_2),
            'combined_3': len(combined_3),
            'environment_variations': env_count,
            'total_single_with_env': total_single,
            'total_combined_with_env': total_combined,
            'estimated_total': total,
            'categories': {
                'hydraulic': len([s for s in self.base_scenarios if 'S1' in s]),
                'wind': len([s for s in self.base_scenarios if 'S2' in s]),
                'thermal': len([s for s in self.base_scenarios if 'S3' in s]),
                'structural': len([s for s in self.base_scenarios if 'S4' in s]),
                'seismic': len([s for s in self.base_scenarios if 'S5' in s]),
                'fault': len([s for s in self.base_scenarios if 'S6' in s]),
            }
        }

    def get_scenario(self, scenario_id: str) -> Optional[ScenarioConfig]:
        """获取场景配置"""
        return self.base_scenarios.get(scenario_id)

    def inject_scenario(self, scenario_id: str, sim: Any, env: Optional[EnvironmentConfig] = None):
        """
        注入场景到仿真模型

        Args:
            scenario_id: 场景ID（基础或组合）
            sim: 仿真模型
            env: 可选的环境配置
        """
        if scenario_id.startswith('C_'):
            # 组合场景
            components = scenario_id[2:].split('_')
            for comp in components:
                self._inject_base_scenario(comp, sim)
        else:
            # 基础场景
            self._inject_base_scenario(scenario_id, sim)

        # 应用环境配置
        if env:
            sim.T_ambient = env.T_ambient
            sim.solar_rad = env.solar_rad
            sim.wind_speed = env.wind_speed

    def _inject_base_scenario(self, scenario_id: str, sim: Any):
        """注入基础场景"""
        scenario = self.base_scenarios.get(scenario_id)
        if not scenario:
            # 尝试简化ID匹配（如 S1.1 匹配 S1.1.3）
            for sid, sc in self.base_scenarios.items():
                if sid.startswith(scenario_id):
                    scenario = sc
                    break

        if not scenario:
            return

        params = scenario.parameters

        # 根据类别应用参数
        if scenario.category == ScenarioCategory.HYDRAULIC:
            if 'h' in params:
                sim.h = params['h']
            if 'Q_in' in params:
                sim.Q_in = params['Q_in']
            if 'Q_surge' in params:
                sim.Q_in = params['Q_surge']

        elif scenario.category == ScenarioCategory.WIND:
            if 'wind_speed' in params:
                sim.wind_speed = params['wind_speed']
            if 'vib_amp' in params:
                sim.vib_amp = params['vib_amp']

        elif scenario.category == ScenarioCategory.THERMAL:
            if 'T_sun' in params:
                sim.T_sun = params['T_sun']
            if 'T_shade' in params:
                sim.T_shade = params['T_shade']
            if 'bearing_locked' in params:
                sim.bearing_locked = params['bearing_locked']
            if 'T_ambient' in params:
                sim.T_ambient = params['T_ambient']

        elif scenario.category == ScenarioCategory.STRUCTURAL:
            if 'joint_gap' in params:
                sim.joint_gap = params['joint_gap']
            if 'T_ambient' in params:
                sim.T_ambient = params['T_ambient']

        elif scenario.category == ScenarioCategory.SEISMIC:
            if 'ground_accel' in params:
                sim.ground_accel = params['ground_accel']
            if 'vib_amp' in params:
                sim.vib_amp = params['vib_amp']


class ScenarioSpaceExplorer:
    """
    场景空间探索器

    用于系统地探索和测试整个场景空间
    """

    def __init__(self, scenario_space: FullScenarioSpace):
        self.space = scenario_space
        self.tested_scenarios = set()
        self.test_results = {}

    def get_untested_scenarios(self, limit: int = 100) -> List[str]:
        """获取未测试的场景"""
        all_scenarios = list(self.space.base_scenarios.keys())
        untested = [s for s in all_scenarios if s not in self.tested_scenarios]
        return untested[:limit]

    def record_test_result(self, scenario_id: str, passed: bool, details: Dict = None):
        """记录测试结果"""
        self.tested_scenarios.add(scenario_id)
        self.test_results[scenario_id] = {
            'passed': passed,
            'details': details or {}
        }

    def get_coverage_report(self) -> Dict[str, Any]:
        """获取测试覆盖率报告"""
        total = len(self.space.base_scenarios)
        tested = len(self.tested_scenarios)
        passed = sum(1 for r in self.test_results.values() if r['passed'])

        return {
            'total_scenarios': total,
            'tested': tested,
            'passed': passed,
            'failed': tested - passed,
            'coverage_rate': tested / total if total > 0 else 0,
            'pass_rate': passed / tested if tested > 0 else 0
        }


def print_scenario_space_summary():
    """打印场景空间摘要"""
    space = FullScenarioSpace()
    summary = space.generate_full_scenario_space()

    print("=" * 60)
    print("TAOS V3.2 全场景空间统计")
    print("=" * 60)
    print(f"\n基础场景数量: {summary['base_scenarios']}")
    print(f"  - 水力类(S1): {summary['categories']['hydraulic']}")
    print(f"  - 风振类(S2): {summary['categories']['wind']}")
    print(f"  - 热力类(S3): {summary['categories']['thermal']}")
    print(f"  - 结构类(S4): {summary['categories']['structural']}")
    print(f"  - 地震类(S5): {summary['categories']['seismic']}")
    print(f"  - 故障类(S6): {summary['categories']['fault']}")
    print(f"\n组合场景数量:")
    print(f"  - 2场景组合: {summary['combined_2']}")
    print(f"  - 3场景组合: {summary['combined_3']}")
    print(f"\n环境变化: {summary['environment_variations']}种")
    print(f"\n场景空间总规模:")
    print(f"  - 基础场景×环境: {summary['total_single_with_env']}")
    print(f"  - 组合场景×环境: {summary['total_combined_with_env']}")
    print(f"  - 估计总场景数: {summary['estimated_total']:,}")
    print("=" * 60)

    return summary


if __name__ == '__main__':
    print_scenario_space_summary()
