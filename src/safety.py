"""
TAOS Safety Enhancement Module
安全增强模块 - 故障诊断、冗余控制、安全保护、网络安全

Features:
- Fault detection and diagnosis (FDD)
- Redundant control architecture
- Safety interlocks
- Emergency response automation
- Cybersecurity monitoring
- Fail-safe mechanisms
"""

import hashlib
import hmac
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random
import string


class FaultType(Enum):
    """故障类型"""
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    COMMUNICATION_LOSS = "communication_loss"
    POWER_FAILURE = "power_failure"
    CONTROL_DEVIATION = "control_deviation"
    STRUCTURAL_DAMAGE = "structural_damage"
    HYDRAULIC_ANOMALY = "hydraulic_anomaly"
    THERMAL_STRESS = "thermal_stress"


class FaultSeverity(Enum):
    """故障严重程度"""
    MINOR = 1        # 轻微 - 可继续运行
    MODERATE = 2     # 中等 - 需要关注
    MAJOR = 3        # 重大 - 需要干预
    CRITICAL = 4     # 严重 - 需要紧急响应
    CATASTROPHIC = 5 # 灾难性 - 需要立即停机


class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    ALARM = "alarm"
    EMERGENCY = "emergency"


@dataclass
class FaultEvent:
    """故障事件"""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    timestamp: datetime
    location: str
    description: str
    symptoms: List[str]
    probable_causes: List[str]
    recommended_actions: List[str]
    auto_response: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SafetyInterlock:
    """安全联锁"""
    interlock_id: str
    name: str
    condition: str           # 触发条件描述
    action: str              # 触发动作
    priority: int
    enabled: bool = True
    triggered: bool = False
    last_trigger_time: Optional[datetime] = None


@dataclass
class RedundantChannel:
    """冗余通道"""
    channel_id: str
    channel_type: str        # primary, secondary, backup
    status: str              # active, standby, failed
    last_health_check: datetime
    health_score: float      # 0-1
    data_source: str


class FaultDiagnosisEngine:
    """故障诊断引擎"""

    def __init__(self):
        self.fault_history: deque = deque(maxlen=1000)
        self.active_faults: Dict[str, FaultEvent] = {}

        # 故障检测规则
        self.detection_rules: Dict[str, Callable] = {}
        self._init_detection_rules()

        # 症状-故障映射
        self.symptom_fault_map = {
            ('h_stuck', 'Q_in_normal'): (FaultType.SENSOR_FAILURE, '水位传感器故障'),
            ('h_oscillation', 'fr_high'): (FaultType.HYDRAULIC_ANOMALY, '水力不稳定'),
            ('T_delta_high', 'joint_gap_large'): (FaultType.THERMAL_STRESS, '热应力过大'),
            ('vib_high', 'ground_accel_normal'): (FaultType.STRUCTURAL_DAMAGE, '结构异常振动'),
            ('control_unresponsive',): (FaultType.ACTUATOR_FAILURE, '执行器故障'),
            ('data_gap',): (FaultType.COMMUNICATION_LOSS, '通信中断')
        }

        # 历史状态用于检测
        self.state_history: deque = deque(maxlen=100)

    def _init_detection_rules(self):
        """初始化故障检测规则"""
        self.detection_rules = {
            'sensor_stuck': self._detect_sensor_stuck,
            'sensor_drift': self._detect_sensor_drift,
            'actuator_response': self._detect_actuator_failure,
            'hydraulic_instability': self._detect_hydraulic_instability,
            'thermal_stress': self._detect_thermal_stress,
            'structural_vibration': self._detect_structural_vibration,
            'communication': self._detect_communication_failure
        }

    def diagnose(self, state: Dict[str, Any]) -> List[FaultEvent]:
        """执行故障诊断"""
        self.state_history.append({
            'timestamp': datetime.now(),
            'state': state.copy()
        })

        detected_faults = []

        for rule_name, rule_func in self.detection_rules.items():
            fault = rule_func(state)
            if fault:
                if fault.fault_id not in self.active_faults:
                    self.active_faults[fault.fault_id] = fault
                    self.fault_history.append(fault)
                detected_faults.append(fault)

        return detected_faults

    def _detect_sensor_stuck(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测传感器僵值故障"""
        if len(self.state_history) < 30:
            return None

        for var in ['h', 'T_sun', 'T_shade', 'vib_amp']:
            values = [s['state'].get(var, 0) for s in list(self.state_history)[-30:]]

            # 检查是否所有值相同
            if len(set(values)) == 1 and values[0] != 0:
                return FaultEvent(
                    fault_id=f"SENSOR_STUCK_{var}_{datetime.now().timestamp()}",
                    fault_type=FaultType.SENSOR_FAILURE,
                    severity=FaultSeverity.MAJOR,
                    timestamp=datetime.now(),
                    location=f"{var}_sensor",
                    description=f"{var}传感器数据僵值",
                    symptoms=[f'{var}数据30个采样点无变化'],
                    probable_causes=['传感器故障', '信号线断开', '采集卡故障'],
                    recommended_actions=['检查传感器连接', '切换到备用传感器', '人工核实数据'],
                    auto_response='启用备用传感器'
                )
        return None

    def _detect_sensor_drift(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测传感器漂移故障"""
        if len(self.state_history) < 50:
            return None

        # 检测水位传感器漂移
        h_values = [s['state'].get('h', 4) for s in list(self.state_history)[-50:]]
        Q_in = [s['state'].get('Q_in', 80) for s in list(self.state_history)[-50:]]
        Q_out = [s['state'].get('Q_out', 80) for s in list(self.state_history)[-50:]]

        # 如果流入流出平衡但水位持续变化
        avg_Q_diff = sum(qi - qo for qi, qo in zip(Q_in, Q_out)) / len(Q_in)
        h_trend = (h_values[-1] - h_values[0]) / len(h_values)

        # 流量差异小但水位变化大 -> 可能是传感器漂移
        if abs(avg_Q_diff) < 5 and abs(h_trend) > 0.02:
            return FaultEvent(
                fault_id=f"SENSOR_DRIFT_h_{datetime.now().timestamp()}",
                fault_type=FaultType.SENSOR_FAILURE,
                severity=FaultSeverity.MODERATE,
                timestamp=datetime.now(),
                location="h_sensor",
                description="水位传感器可能存在漂移",
                symptoms=['水位持续变化但流量平衡', f'漂移速率: {h_trend:.4f}m/sample'],
                probable_causes=['传感器老化', '温度影响', '零点漂移'],
                recommended_actions=['校准传感器', '检查环境温度', '对比其他测点']
            )
        return None

    def _detect_actuator_failure(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测执行器故障"""
        if len(self.state_history) < 20:
            return None

        # 检查流量控制响应
        recent = list(self.state_history)[-20:]
        Q_in_cmd = [s['state'].get('Q_in', 80) for s in recent]
        h_values = [s['state'].get('h', 4) for s in recent]

        # 如果命令变化但水位无响应
        cmd_change = max(Q_in_cmd) - min(Q_in_cmd)
        h_change = max(h_values) - min(h_values)

        if cmd_change > 20 and h_change < 0.1:
            return FaultEvent(
                fault_id=f"ACTUATOR_FAIL_{datetime.now().timestamp()}",
                fault_type=FaultType.ACTUATOR_FAILURE,
                severity=FaultSeverity.MAJOR,
                timestamp=datetime.now(),
                location="inlet_valve",
                description="进水阀门可能失效",
                symptoms=[f'流量命令变化{cmd_change:.1f}但水位变化仅{h_change:.2f}'],
                probable_causes=['阀门卡滞', '液压系统故障', '控制信号断开'],
                recommended_actions=['检查阀门位置', '检查液压油位', '手动操作测试'],
                auto_response='切换到备用控制通道'
            )
        return None

    def _detect_hydraulic_instability(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测水力不稳定"""
        fr = state.get('fr', 0.5)
        h = state.get('h', 4.0)
        v = state.get('v', 2.0)

        if fr > 0.85:
            return FaultEvent(
                fault_id=f"HYDRAULIC_INSTAB_{datetime.now().timestamp()}",
                fault_type=FaultType.HYDRAULIC_ANOMALY,
                severity=FaultSeverity.CRITICAL if fr > 0.95 else FaultSeverity.MAJOR,
                timestamp=datetime.now(),
                location="aqueduct_main",
                description="检测到水力不稳定",
                symptoms=[f'弗劳德数={fr:.2f}', f'水位={h:.2f}m', f'流速={v:.2f}m/s'],
                probable_causes=['流量过大', '水位过低', '下游堵塞'],
                recommended_actions=['减小进水流量', '检查下游出口', '降低流速'],
                auto_response='自动减流至安全流量'
            )
        return None

    def _detect_thermal_stress(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测热应力问题"""
        T_sun = state.get('T_sun', 25)
        T_shade = state.get('T_shade', 25)
        T_delta = abs(T_sun - T_shade)
        joint_gap = state.get('joint_gap', 20)

        if T_delta > 18:
            severity = FaultSeverity.CRITICAL if T_delta > 22 else FaultSeverity.MAJOR
            return FaultEvent(
                fault_id=f"THERMAL_STRESS_{datetime.now().timestamp()}",
                fault_type=FaultType.THERMAL_STRESS,
                severity=severity,
                timestamp=datetime.now(),
                location="aqueduct_structure",
                description="热应力超限",
                symptoms=[f'阳阴面温差={T_delta:.1f}°C', f'伸缩缝间隙={joint_gap:.1f}mm'],
                probable_causes=['强日照', '局部遮挡不足', '材料老化'],
                recommended_actions=['启动喷淋降温', '监测结构变形', '检查伸缩缝'],
                auto_response='启动温度调节措施'
            )
        return None

    def _detect_structural_vibration(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测结构振动异常"""
        vib_amp = state.get('vib_amp', 0)
        ground_accel = state.get('ground_accel', 0)

        # 无地震但振动大
        if vib_amp > 50 and ground_accel < 0.05:
            return FaultEvent(
                fault_id=f"STRUCT_VIB_{datetime.now().timestamp()}",
                fault_type=FaultType.STRUCTURAL_DAMAGE,
                severity=FaultSeverity.MAJOR,
                timestamp=datetime.now(),
                location="aqueduct_structure",
                description="结构异常振动",
                symptoms=[f'振动幅度={vib_amp:.1f}mm', f'地面加速度={ground_accel:.3f}g'],
                probable_causes=['水流脉动', '结构松动', '共振现象'],
                recommended_actions=['检查支座', '调整流量', '结构检测'],
                auto_response='降低流速减小激励'
            )
        return None

    def _detect_communication_failure(self, state: Dict[str, Any]) -> Optional[FaultEvent]:
        """检测通信故障"""
        # 检查数据时间戳间隙
        if len(self.state_history) < 5:
            return None

        recent = list(self.state_history)[-5:]
        timestamps = [s['timestamp'] for s in recent]

        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if gap > 5:  # 超过5秒的间隙
                return FaultEvent(
                    fault_id=f"COMM_FAIL_{datetime.now().timestamp()}",
                    fault_type=FaultType.COMMUNICATION_LOSS,
                    severity=FaultSeverity.MODERATE,
                    timestamp=datetime.now(),
                    location="data_link",
                    description="检测到通信中断",
                    symptoms=[f'数据间隙={gap:.1f}秒'],
                    probable_causes=['网络故障', '设备重启', '电源问题'],
                    recommended_actions=['检查网络连接', '检查设备状态', '启用备用链路']
                )
        return None

    def resolve_fault(self, fault_id: str):
        """解决故障"""
        if fault_id in self.active_faults:
            fault = self.active_faults[fault_id]
            fault.resolved = True
            fault.resolution_time = datetime.now()
            del self.active_faults[fault_id]

    def get_active_faults(self) -> List[FaultEvent]:
        """获取活动故障列表"""
        return list(self.active_faults.values())

    def get_fault_statistics(self) -> Dict[str, Any]:
        """获取故障统计"""
        faults = list(self.fault_history)

        type_counts = {}
        severity_counts = {}

        for fault in faults:
            t = fault.fault_type.value
            s = fault.severity.name
            type_counts[t] = type_counts.get(t, 0) + 1
            severity_counts[s] = severity_counts.get(s, 0) + 1

        return {
            'total_faults': len(faults),
            'active_faults': len(self.active_faults),
            'type_distribution': type_counts,
            'severity_distribution': severity_counts
        }


class RedundantController:
    """冗余控制器"""

    def __init__(self):
        self.channels: Dict[str, RedundantChannel] = {}
        self.active_channel: Optional[str] = None
        self.voting_threshold = 2  # 三取二表决

        self._init_channels()

    def _init_channels(self):
        """初始化冗余通道"""
        self.channels = {
            'primary': RedundantChannel(
                channel_id='primary',
                channel_type='primary',
                status='active',
                last_health_check=datetime.now(),
                health_score=1.0,
                data_source='sensor_group_A'
            ),
            'secondary': RedundantChannel(
                channel_id='secondary',
                channel_type='secondary',
                status='standby',
                last_health_check=datetime.now(),
                health_score=1.0,
                data_source='sensor_group_B'
            ),
            'backup': RedundantChannel(
                channel_id='backup',
                channel_type='backup',
                status='standby',
                last_health_check=datetime.now(),
                health_score=1.0,
                data_source='sensor_group_C'
            )
        }
        self.active_channel = 'primary'

    def health_check(self, channel_id: str, test_value: float,
                     expected_value: float) -> bool:
        """执行通道健康检查"""
        if channel_id not in self.channels:
            return False

        channel = self.channels[channel_id]
        error = abs(test_value - expected_value)
        relative_error = error / max(abs(expected_value), 0.01)

        # 更新健康分数
        if relative_error < 0.05:
            channel.health_score = min(1.0, channel.health_score + 0.1)
        elif relative_error < 0.1:
            channel.health_score = max(0, channel.health_score - 0.1)
        else:
            channel.health_score = max(0, channel.health_score - 0.3)

        channel.last_health_check = datetime.now()

        # 更新状态
        if channel.health_score < 0.3:
            channel.status = 'failed'
        elif channel.health_score < 0.7:
            channel.status = 'degraded'
        else:
            channel.status = 'active' if channel_id == self.active_channel else 'standby'

        return channel.health_score >= 0.5

    def vote(self, values: Dict[str, float]) -> Tuple[float, float]:
        """三取二表决"""
        valid_values = []

        for channel_id, value in values.items():
            if channel_id in self.channels:
                channel = self.channels[channel_id]
                if channel.status != 'failed' and channel.health_score >= 0.5:
                    valid_values.append((value, channel.health_score))

        if not valid_values:
            return (0.0, 0.0)  # 无有效值

        if len(valid_values) == 1:
            return (valid_values[0][0], valid_values[0][1] * 0.5)

        # 加权平均
        total_weight = sum(h for _, h in valid_values)
        weighted_avg = sum(v * h for v, h in valid_values) / total_weight
        confidence = min(1.0, total_weight / 3)

        return (weighted_avg, confidence)

    def switch_channel(self, target_channel: str) -> bool:
        """切换活动通道"""
        if target_channel not in self.channels:
            return False

        channel = self.channels[target_channel]
        if channel.status == 'failed':
            return False

        # 将当前通道设为备用
        if self.active_channel and self.active_channel in self.channels:
            old_channel = self.channels[self.active_channel]
            if old_channel.status == 'active':
                old_channel.status = 'standby'

        # 激活新通道
        channel.status = 'active'
        self.active_channel = target_channel
        return True

    def get_best_channel(self) -> str:
        """获取最佳通道"""
        best_channel = None
        best_score = -1

        for channel_id, channel in self.channels.items():
            if channel.status != 'failed' and channel.health_score > best_score:
                best_score = channel.health_score
                best_channel = channel_id

        return best_channel or 'primary'

    def get_channel_status(self) -> Dict[str, Any]:
        """获取通道状态"""
        return {
            'active_channel': self.active_channel,
            'channels': {
                cid: {
                    'type': c.channel_type,
                    'status': c.status,
                    'health_score': c.health_score,
                    'last_check': c.last_health_check.isoformat()
                }
                for cid, c in self.channels.items()
            }
        }


class SafetyInterlockSystem:
    """安全联锁系统"""

    def __init__(self):
        self.interlocks: Dict[str, SafetyInterlock] = {}
        self.trigger_history: deque = deque(maxlen=500)

        self._init_interlocks()

    def _init_interlocks(self):
        """初始化安全联锁"""
        self.interlocks = {
            'IL_HIGH_WATER': SafetyInterlock(
                interlock_id='IL_HIGH_WATER',
                name='高水位联锁',
                condition='h > 6.0m',
                action='紧急开启出水阀',
                priority=1
            ),
            'IL_LOW_WATER': SafetyInterlock(
                interlock_id='IL_LOW_WATER',
                name='低水位联锁',
                condition='h < 1.5m',
                action='紧急关闭出水阀',
                priority=1
            ),
            'IL_HIGH_FROUDE': SafetyInterlock(
                interlock_id='IL_HIGH_FROUDE',
                name='高弗劳德数联锁',
                condition='fr > 0.95',
                action='减小进水流量至60%',
                priority=2
            ),
            'IL_SEISMIC': SafetyInterlock(
                interlock_id='IL_SEISMIC',
                name='地震联锁',
                condition='ground_accel > 0.3g',
                action='关闭所有阀门',
                priority=1
            ),
            'IL_THERMAL': SafetyInterlock(
                interlock_id='IL_THERMAL',
                name='热应力联锁',
                condition='T_delta > 20°C',
                action='启动喷淋冷却',
                priority=3
            ),
            'IL_VIBRATION': SafetyInterlock(
                interlock_id='IL_VIBRATION',
                name='振动联锁',
                condition='vib_amp > 80mm',
                action='减小流量并告警',
                priority=2
            ),
            'IL_JOINT_GAP_HIGH': SafetyInterlock(
                interlock_id='IL_JOINT_GAP_HIGH',
                name='伸缩缝过大联锁',
                condition='joint_gap > 40mm',
                action='结构安全告警',
                priority=2
            ),
            'IL_JOINT_GAP_LOW': SafetyInterlock(
                interlock_id='IL_JOINT_GAP_LOW',
                name='伸缩缝过小联锁',
                condition='joint_gap < 5mm',
                action='结构安全告警',
                priority=2
            ),
            'IL_COMM_LOSS': SafetyInterlock(
                interlock_id='IL_COMM_LOSS',
                name='通信中断联锁',
                condition='通信中断>10s',
                action='切换备用通道，保持当前状态',
                priority=2
            ),
            'IL_POWER_FAIL': SafetyInterlock(
                interlock_id='IL_POWER_FAIL',
                name='电源故障联锁',
                condition='主电源失电',
                action='切换备用电源，安全停机',
                priority=1
            )
        }

    def check_interlocks(self, state: Dict[str, Any]) -> List[Tuple[SafetyInterlock, str]]:
        """检查所有联锁条件"""
        triggered = []

        h = state.get('h', 4.0)
        fr = state.get('fr', 0.5)
        T_sun = state.get('T_sun', 25)
        T_shade = state.get('T_shade', 25)
        T_delta = abs(T_sun - T_shade)
        vib_amp = state.get('vib_amp', 0)
        joint_gap = state.get('joint_gap', 20)
        ground_accel = state.get('ground_accel', 0)

        checks = {
            'IL_HIGH_WATER': h > 6.0,
            'IL_LOW_WATER': h < 1.5,
            'IL_HIGH_FROUDE': fr > 0.95,
            'IL_SEISMIC': ground_accel > 0.3,
            'IL_THERMAL': T_delta > 20,
            'IL_VIBRATION': vib_amp > 80,
            'IL_JOINT_GAP_HIGH': joint_gap > 40,
            'IL_JOINT_GAP_LOW': joint_gap < 5
        }

        for interlock_id, condition_met in checks.items():
            interlock = self.interlocks.get(interlock_id)
            if interlock and interlock.enabled and condition_met:
                if not interlock.triggered:
                    interlock.triggered = True
                    interlock.last_trigger_time = datetime.now()
                    self.trigger_history.append({
                        'timestamp': datetime.now(),
                        'interlock_id': interlock_id,
                        'action': interlock.action
                    })
                triggered.append((interlock, interlock.action))
            elif interlock and interlock.triggered and not condition_met:
                interlock.triggered = False

        # 按优先级排序
        triggered.sort(key=lambda x: x[0].priority)

        return triggered

    def enable_interlock(self, interlock_id: str) -> bool:
        """启用联锁"""
        if interlock_id in self.interlocks:
            self.interlocks[interlock_id].enabled = True
            return True
        return False

    def disable_interlock(self, interlock_id: str, reason: str = "") -> bool:
        """禁用联锁 (需要记录原因)"""
        if interlock_id in self.interlocks:
            self.interlocks[interlock_id].enabled = False
            self.trigger_history.append({
                'timestamp': datetime.now(),
                'interlock_id': interlock_id,
                'action': f'DISABLED: {reason}'
            })
            return True
        return False

    def get_interlock_status(self) -> Dict[str, Any]:
        """获取联锁状态"""
        return {
            interlock_id: {
                'name': il.name,
                'condition': il.condition,
                'action': il.action,
                'enabled': il.enabled,
                'triggered': il.triggered,
                'priority': il.priority,
                'last_trigger': il.last_trigger_time.isoformat() if il.last_trigger_time else None
            }
            for interlock_id, il in self.interlocks.items()
        }


class EmergencyResponseSystem:
    """应急响应系统"""

    def __init__(self):
        self.response_procedures: Dict[str, List[str]] = {}
        self.active_emergency: Optional[str] = None
        self.emergency_history: deque = deque(maxlen=100)

        self._init_procedures()

    def _init_procedures(self):
        """初始化应急程序"""
        self.response_procedures = {
            'FLOOD': [
                '1. 立即打开所有出水阀至最大',
                '2. 关闭进水阀至最小安全流量',
                '3. 启动溢流通道',
                '4. 通知下游用户',
                '5. 派遣巡检人员现场确认'
            ],
            'DROUGHT': [
                '1. 减小出水流量',
                '2. 通知下游用户限水',
                '3. 协调上游增加放水',
                '4. 监测水位变化趋势'
            ],
            'EARTHQUAKE': [
                '1. 自动关闭所有阀门',
                '2. 启动结构监测加密',
                '3. 派遣巡检人员检查',
                '4. 评估结构安全后恢复运行'
            ],
            'STRUCTURAL_DAMAGE': [
                '1. 减小流量至最小',
                '2. 隔离受损区段',
                '3. 紧急工程评估',
                '4. 准备临时修复方案'
            ],
            'CYBER_ATTACK': [
                '1. 切换到本地手动控制',
                '2. 断开网络连接',
                '3. 通知网络安全团队',
                '4. 启动备用通信',
                '5. 记录攻击痕迹'
            ],
            'POWER_FAILURE': [
                '1. 自动切换备用电源',
                '2. 确认关键设备供电',
                '3. 减小非关键负载',
                '4. 通知电力部门'
            ]
        }

    def activate_emergency(self, emergency_type: str) -> Dict[str, Any]:
        """激活应急响应"""
        if emergency_type not in self.response_procedures:
            return {'error': f'Unknown emergency type: {emergency_type}'}

        self.active_emergency = emergency_type
        procedures = self.response_procedures[emergency_type]

        event = {
            'timestamp': datetime.now(),
            'type': emergency_type,
            'procedures': procedures,
            'status': 'activated'
        }
        self.emergency_history.append(event)

        return {
            'emergency_type': emergency_type,
            'procedures': procedures,
            'activated_at': datetime.now().isoformat()
        }

    def deactivate_emergency(self, reason: str = "resolved") -> bool:
        """解除应急状态"""
        if self.active_emergency:
            self.emergency_history.append({
                'timestamp': datetime.now(),
                'type': self.active_emergency,
                'status': 'deactivated',
                'reason': reason
            })
            self.active_emergency = None
            return True
        return False

    def get_emergency_status(self) -> Dict[str, Any]:
        """获取应急状态"""
        return {
            'active_emergency': self.active_emergency,
            'procedures': self.response_procedures.get(self.active_emergency, []) if self.active_emergency else [],
            'available_types': list(self.response_procedures.keys()),
            'recent_history': list(self.emergency_history)[-10:]
        }


class CybersecurityMonitor:
    """网络安全监控"""

    def __init__(self):
        self.api_key = self._generate_api_key()
        self.access_log: deque = deque(maxlen=1000)
        self.failed_attempts: Dict[str, int] = {}
        self.blocked_ips: set = set()
        self.rate_limits: Dict[str, deque] = {}

        # 安全配置
        self.max_failed_attempts = 5
        self.rate_limit_window = 60  # 秒
        self.rate_limit_max = 100    # 每分钟最大请求数

    def _generate_api_key(self) -> str:
        """生成API密钥"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

    def verify_request(self, request_info: Dict[str, Any]) -> Tuple[bool, str]:
        """验证请求"""
        ip = request_info.get('ip', 'unknown')
        api_key = request_info.get('api_key', '')
        endpoint = request_info.get('endpoint', '')

        # 检查是否被封禁
        if ip in self.blocked_ips:
            self._log_access(ip, endpoint, 'BLOCKED')
            return False, 'IP blocked'

        # 检查速率限制
        if not self._check_rate_limit(ip):
            self._log_access(ip, endpoint, 'RATE_LIMITED')
            return False, 'Rate limit exceeded'

        # 验证API密钥 (对于需要认证的端点)
        if request_info.get('requires_auth', False):
            if not self._verify_api_key(api_key):
                self._record_failed_attempt(ip)
                self._log_access(ip, endpoint, 'AUTH_FAILED')
                return False, 'Invalid API key'

        self._log_access(ip, endpoint, 'ALLOWED')
        return True, 'OK'

    def _verify_api_key(self, key: str) -> bool:
        """验证API密钥"""
        return hmac.compare_digest(key, self.api_key)

    def _check_rate_limit(self, ip: str) -> bool:
        """检查速率限制"""
        now = time.time()

        if ip not in self.rate_limits:
            self.rate_limits[ip] = deque(maxlen=self.rate_limit_max * 2)

        # 清理过期记录
        while self.rate_limits[ip] and now - self.rate_limits[ip][0] > self.rate_limit_window:
            self.rate_limits[ip].popleft()

        if len(self.rate_limits[ip]) >= self.rate_limit_max:
            return False

        self.rate_limits[ip].append(now)
        return True

    def _record_failed_attempt(self, ip: str):
        """记录失败尝试"""
        self.failed_attempts[ip] = self.failed_attempts.get(ip, 0) + 1

        if self.failed_attempts[ip] >= self.max_failed_attempts:
            self.blocked_ips.add(ip)

    def _log_access(self, ip: str, endpoint: str, result: str):
        """记录访问日志"""
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'ip': ip,
            'endpoint': endpoint,
            'result': result
        })

    def unblock_ip(self, ip: str) -> bool:
        """解除IP封禁"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            self.failed_attempts[ip] = 0
            return True
        return False

    def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        recent_logs = list(self.access_log)[-50:]
        blocked_count = len([l for l in recent_logs if l['result'] == 'BLOCKED'])
        auth_failed_count = len([l for l in recent_logs if l['result'] == 'AUTH_FAILED'])

        return {
            'blocked_ips': list(self.blocked_ips),
            'failed_attempts': dict(self.failed_attempts),
            'recent_blocked': blocked_count,
            'recent_auth_failures': auth_failed_count,
            'total_requests': len(self.access_log),
            'api_key_hint': self.api_key[:8] + '...'
        }

    def generate_new_api_key(self) -> str:
        """生成新的API密钥"""
        self.api_key = self._generate_api_key()
        return self.api_key


class SafetyManager:
    """安全管理器 - 整合所有安全组件"""

    def __init__(self):
        self.fault_diagnosis = FaultDiagnosisEngine()
        self.redundant_control = RedundantController()
        self.interlocks = SafetyInterlockSystem()
        self.emergency = EmergencyResponseSystem()
        self.cybersecurity = CybersecurityMonitor()

        self.safety_level = SafetyLevel.NORMAL
        self.last_check = datetime.now()

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理状态更新，返回安全评估结果"""
        results = {
            'safety_level': SafetyLevel.NORMAL,
            'faults': [],
            'interlocks_triggered': [],
            'actions': [],
            'warnings': []
        }

        # 故障诊断
        faults = self.fault_diagnosis.diagnose(state)
        results['faults'] = [
            {
                'id': f.fault_id,
                'type': f.fault_type.value,
                'severity': f.severity.name,
                'description': f.description
            }
            for f in faults
        ]

        # 检查联锁
        triggered = self.interlocks.check_interlocks(state)
        results['interlocks_triggered'] = [
            {'id': il.interlock_id, 'action': action}
            for il, action in triggered
        ]

        # 确定安全级别
        if any(f.severity == FaultSeverity.CATASTROPHIC for f in faults):
            results['safety_level'] = SafetyLevel.EMERGENCY
        elif any(f.severity == FaultSeverity.CRITICAL for f in faults) or \
             any(il.priority == 1 for il, _ in triggered):
            results['safety_level'] = SafetyLevel.ALARM
        elif any(f.severity == FaultSeverity.MAJOR for f in faults):
            results['safety_level'] = SafetyLevel.WARNING
        elif any(f.severity == FaultSeverity.MODERATE for f in faults):
            results['safety_level'] = SafetyLevel.CAUTION

        self.safety_level = results['safety_level']
        self.last_check = datetime.now()

        # 生成建议动作
        for il, action in triggered:
            results['actions'].append(action)

        return results

    def get_safety_summary(self) -> Dict[str, Any]:
        """获取安全状态摘要"""
        return {
            'safety_level': self.safety_level.value,
            'last_check': self.last_check.isoformat(),
            'active_faults': len(self.fault_diagnosis.active_faults),
            'fault_statistics': self.fault_diagnosis.get_fault_statistics(),
            'channel_status': self.redundant_control.get_channel_status(),
            'interlock_status': self.interlocks.get_interlock_status(),
            'emergency_status': self.emergency.get_emergency_status(),
            'security_status': self.cybersecurity.get_security_status()
        }


# 全局实例
_safety_instance: Optional[SafetyManager] = None


def get_safety_manager() -> SafetyManager:
    """获取全局安全管理器实例"""
    global _safety_instance
    if _safety_instance is None:
        _safety_instance = SafetyManager()
    return _safety_instance
