"""
TAOS Engineering Integration Module
工程对接模块 - SCADA接口、OPC-UA协议、Modbus通信

Features:
- SCADA system integration
- OPC-UA server/client
- Modbus TCP/RTU communication
- Real-time data exchange
- Command and control interface
- Historical data archival
"""

import struct
import socket
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json


class CommunicationProtocol(Enum):
    """通信协议"""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    DNP3 = "dnp3"
    IEC61850 = "iec61850"


class DataQuality(Enum):
    """数据质量"""
    GOOD = 0
    UNCERTAIN = 1
    BAD = 2
    SUBSTITUTED = 3
    MANUAL = 4


class AlarmPriority(Enum):
    """告警优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class SCADAPoint:
    """SCADA数据点"""
    tag_name: str
    description: str
    address: str
    data_type: str          # REAL, INT, BOOL, STRING
    unit: str
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    alarm_low: Optional[float] = None
    alarm_high: Optional[float] = None
    current_value: Any = None
    quality: DataQuality = DataQuality.GOOD
    timestamp: datetime = field(default_factory=datetime.now)
    read_only: bool = True


@dataclass
class SCADAAlarm:
    """SCADA告警"""
    alarm_id: str
    tag_name: str
    priority: AlarmPriority
    message: str
    timestamp: datetime
    acknowledged: bool = False
    ack_time: Optional[datetime] = None
    ack_user: Optional[str] = None
    cleared: bool = False
    clear_time: Optional[datetime] = None


@dataclass
class OPCUANode:
    """OPC-UA节点"""
    node_id: str
    browse_name: str
    display_name: str
    data_type: str
    value: Any
    access_level: str       # Read, Write, ReadWrite
    timestamp: datetime
    status_code: int = 0


class SCADAInterface:
    """SCADA系统接口"""

    def __init__(self):
        self.points: Dict[str, SCADAPoint] = {}
        self.alarms: Dict[str, SCADAAlarm] = {}
        self.alarm_history: deque = deque(maxlen=10000)

        # 数据变更回调
        self.on_value_change: Optional[Callable] = None
        self.on_alarm: Optional[Callable] = None

        self._init_points()

    def _init_points(self):
        """初始化SCADA数据点"""
        points_config = [
            # 水力参数
            ('AI_H_LEVEL', '水位', 'DB1.DBD0', 'REAL', 'm', 0, 8, 2.0, 6.0),
            ('AI_Q_IN', '进水流量', 'DB1.DBD4', 'REAL', 'm³/s', 0, 150, 40, 120),
            ('AI_Q_OUT', '出水流量', 'DB1.DBD8', 'REAL', 'm³/s', 0, 150, 40, 120),
            ('AI_FROUDE', '弗劳德数', 'DB1.DBD12', 'REAL', '', 0, 1.5, None, 0.85),
            ('AI_VELOCITY', '流速', 'DB1.DBD16', 'REAL', 'm/s', 0, 5, None, 3.5),

            # 温度参数
            ('AI_T_SUN', '阳面温度', 'DB2.DBD0', 'REAL', '°C', -20, 60, None, 45),
            ('AI_T_SHADE', '阴面温度', 'DB2.DBD4', 'REAL', '°C', -20, 50, None, 40),
            ('AI_T_DELTA', '温差', 'DB2.DBD8', 'REAL', '°C', 0, 30, None, 18),

            # 结构参数
            ('AI_VIB_AMP', '振动幅度', 'DB3.DBD0', 'REAL', 'mm', 0, 100, None, 60),
            ('AI_JOINT_GAP', '伸缩缝间隙', 'DB3.DBD4', 'REAL', 'mm', 0, 50, 5, 40),
            ('AI_GROUND_ACC', '地面加速度', 'DB3.DBD8', 'REAL', 'g', 0, 1, None, 0.3),

            # 控制参数 (可写)
            ('AO_Q_IN_SP', '进水流量设定', 'DB10.DBD0', 'REAL', 'm³/s', 0, 150, None, None),
            ('AO_Q_OUT_SP', '出水流量设定', 'DB10.DBD4', 'REAL', 'm³/s', 0, 150, None, None),
            ('AO_H_TARGET', '目标水位', 'DB10.DBD8', 'REAL', 'm', 2, 6, None, None),

            # 状态点
            ('DI_INLET_OPEN', '进水阀开启', 'DB20.DBX0.0', 'BOOL', '', None, None, None, None),
            ('DI_OUTLET_OPEN', '出水阀开启', 'DB20.DBX0.1', 'BOOL', '', None, None, None, None),
            ('DI_PUMP_RUN', '泵运行状态', 'DB20.DBX0.2', 'BOOL', '', None, None, None, None),
            ('DI_EMERGENCY', '紧急停止', 'DB20.DBX0.3', 'BOOL', '', None, None, None, None),

            # 控制命令 (可写)
            ('DO_INLET_CMD', '进水阀控制', 'DB21.DBX0.0', 'BOOL', '', None, None, None, None),
            ('DO_OUTLET_CMD', '出水阀控制', 'DB21.DBX0.1', 'BOOL', '', None, None, None, None),
            ('DO_PUMP_CMD', '泵启停控制', 'DB21.DBX0.2', 'BOOL', '', None, None, None, None),
        ]

        for cfg in points_config:
            tag, desc, addr, dtype, unit, low, high, alarm_low, alarm_high = cfg
            is_writable = tag.startswith('AO_') or tag.startswith('DO_')
            self.points[tag] = SCADAPoint(
                tag_name=tag,
                description=desc,
                address=addr,
                data_type=dtype,
                unit=unit,
                low_limit=low,
                high_limit=high,
                alarm_low=alarm_low,
                alarm_high=alarm_high,
                read_only=not is_writable
            )

    def update_value(self, tag_name: str, value: Any,
                     quality: DataQuality = DataQuality.GOOD):
        """更新数据点值"""
        if tag_name not in self.points:
            return False

        point = self.points[tag_name]
        old_value = point.current_value
        point.current_value = value
        point.quality = quality
        point.timestamp = datetime.now()

        # 检查告警
        self._check_alarms(point)

        # 触发回调
        if self.on_value_change and old_value != value:
            self.on_value_change(tag_name, value, old_value)

        return True

    def _check_alarms(self, point: SCADAPoint):
        """检查告警条件"""
        if point.data_type != 'REAL' or point.current_value is None:
            return

        value = point.current_value

        # 低限告警
        if point.alarm_low is not None and value < point.alarm_low:
            self._raise_alarm(
                point.tag_name,
                AlarmPriority.HIGH,
                f"{point.description}低于告警下限: {value:.2f} < {point.alarm_low}"
            )
        # 高限告警
        elif point.alarm_high is not None and value > point.alarm_high:
            self._raise_alarm(
                point.tag_name,
                AlarmPriority.HIGH,
                f"{point.description}超过告警上限: {value:.2f} > {point.alarm_high}"
            )
        else:
            # 清除现有告警
            self._clear_alarm(point.tag_name)

    def _raise_alarm(self, tag_name: str, priority: AlarmPriority, message: str):
        """产生告警"""
        alarm_id = f"ALM_{tag_name}_{datetime.now().timestamp()}"

        alarm = SCADAAlarm(
            alarm_id=alarm_id,
            tag_name=tag_name,
            priority=priority,
            message=message,
            timestamp=datetime.now()
        )

        self.alarms[tag_name] = alarm
        self.alarm_history.append(alarm)

        if self.on_alarm:
            self.on_alarm(alarm)

    def _clear_alarm(self, tag_name: str):
        """清除告警"""
        if tag_name in self.alarms:
            alarm = self.alarms[tag_name]
            alarm.cleared = True
            alarm.clear_time = datetime.now()
            del self.alarms[tag_name]

    def write_value(self, tag_name: str, value: Any) -> bool:
        """写入数据点值"""
        if tag_name not in self.points:
            return False

        point = self.points[tag_name]
        if point.read_only:
            return False

        # 范围检查
        if point.low_limit is not None and value < point.low_limit:
            return False
        if point.high_limit is not None and value > point.high_limit:
            return False

        point.current_value = value
        point.timestamp = datetime.now()
        return True

    def acknowledge_alarm(self, tag_name: str, user: str) -> bool:
        """确认告警"""
        if tag_name not in self.alarms:
            return False

        alarm = self.alarms[tag_name]
        alarm.acknowledged = True
        alarm.ack_time = datetime.now()
        alarm.ack_user = user
        return True

    def get_all_values(self) -> Dict[str, Any]:
        """获取所有数据点值"""
        return {
            tag: {
                'value': point.current_value,
                'quality': point.quality.name,
                'timestamp': point.timestamp.isoformat(),
                'unit': point.unit,
                'description': point.description
            }
            for tag, point in self.points.items()
        }

    def get_active_alarms(self) -> List[Dict[str, Any]]:
        """获取活动告警"""
        return [
            {
                'alarm_id': alarm.alarm_id,
                'tag': alarm.tag_name,
                'priority': alarm.priority.name,
                'message': alarm.message,
                'timestamp': alarm.timestamp.isoformat(),
                'acknowledged': alarm.acknowledged
            }
            for alarm in self.alarms.values()
        ]

    def update_from_state(self, state: Dict[str, Any]):
        """从系统状态更新SCADA数据点"""
        mappings = {
            'h': 'AI_H_LEVEL',
            'Q_in': 'AI_Q_IN',
            'Q_out': 'AI_Q_OUT',
            'fr': 'AI_FROUDE',
            'v': 'AI_VELOCITY',
            'T_sun': 'AI_T_SUN',
            'T_shade': 'AI_T_SHADE',
            'vib_amp': 'AI_VIB_AMP',
            'joint_gap': 'AI_JOINT_GAP',
            'ground_accel': 'AI_GROUND_ACC'
        }

        for state_key, tag_name in mappings.items():
            if state_key in state:
                self.update_value(tag_name, state[state_key])

        # 计算温差
        if 'T_sun' in state and 'T_shade' in state:
            T_delta = abs(state['T_sun'] - state['T_shade'])
            self.update_value('AI_T_DELTA', T_delta)


class OPCUAServer:
    """OPC-UA服务器模拟"""

    def __init__(self, endpoint: str = "opc.tcp://localhost:4840"):
        self.endpoint = endpoint
        self.nodes: Dict[str, OPCUANode] = {}
        self.namespace_uri = "urn:taos:aqueduct"
        self.namespace_index = 2

        self._init_nodes()
        self.running = False

    def _init_nodes(self):
        """初始化OPC-UA节点"""
        # 创建节点结构
        node_configs = [
            # 对象节点
            ('Objects/Aqueduct', 'Aqueduct', 'Aqueduct', 'Object', None, 'Read'),
            ('Objects/Aqueduct/Hydraulics', 'Hydraulics', 'Hydraulics', 'Object', None, 'Read'),
            ('Objects/Aqueduct/Thermal', 'Thermal', 'Thermal', 'Object', None, 'Read'),
            ('Objects/Aqueduct/Structural', 'Structural', 'Structural', 'Object', None, 'Read'),
            ('Objects/Aqueduct/Control', 'Control', 'Control', 'Object', None, 'Read'),

            # 水力变量
            ('Objects/Aqueduct/Hydraulics/WaterLevel', 'WaterLevel', 'Water Level', 'Double', 4.0, 'Read'),
            ('Objects/Aqueduct/Hydraulics/InletFlow', 'InletFlow', 'Inlet Flow', 'Double', 80.0, 'Read'),
            ('Objects/Aqueduct/Hydraulics/OutletFlow', 'OutletFlow', 'Outlet Flow', 'Double', 80.0, 'Read'),
            ('Objects/Aqueduct/Hydraulics/FroudeNumber', 'FroudeNumber', 'Froude Number', 'Double', 0.5, 'Read'),
            ('Objects/Aqueduct/Hydraulics/Velocity', 'Velocity', 'Flow Velocity', 'Double', 2.0, 'Read'),

            # 温度变量
            ('Objects/Aqueduct/Thermal/SunSideTemp', 'SunSideTemp', 'Sun Side Temperature', 'Double', 25.0, 'Read'),
            ('Objects/Aqueduct/Thermal/ShadeSideTemp', 'ShadeSideTemp', 'Shade Side Temperature', 'Double', 25.0, 'Read'),
            ('Objects/Aqueduct/Thermal/TempDelta', 'TempDelta', 'Temperature Difference', 'Double', 0.0, 'Read'),

            # 结构变量
            ('Objects/Aqueduct/Structural/VibrationAmp', 'VibrationAmp', 'Vibration Amplitude', 'Double', 0.0, 'Read'),
            ('Objects/Aqueduct/Structural/JointGap', 'JointGap', 'Expansion Joint Gap', 'Double', 20.0, 'Read'),
            ('Objects/Aqueduct/Structural/GroundAccel', 'GroundAccel', 'Ground Acceleration', 'Double', 0.0, 'Read'),

            # 控制变量
            ('Objects/Aqueduct/Control/InletFlowSP', 'InletFlowSP', 'Inlet Flow Setpoint', 'Double', 80.0, 'ReadWrite'),
            ('Objects/Aqueduct/Control/OutletFlowSP', 'OutletFlowSP', 'Outlet Flow Setpoint', 'Double', 80.0, 'ReadWrite'),
            ('Objects/Aqueduct/Control/TargetLevel', 'TargetLevel', 'Target Water Level', 'Double', 4.0, 'ReadWrite'),
            ('Objects/Aqueduct/Control/Mode', 'Mode', 'Control Mode', 'String', 'AUTO', 'ReadWrite'),
            ('Objects/Aqueduct/Control/Status', 'Status', 'System Status', 'String', 'NORMAL', 'Read'),
        ]

        for cfg in node_configs:
            node_id, browse_name, display_name, data_type, value, access = cfg
            self.nodes[node_id] = OPCUANode(
                node_id=f"ns={self.namespace_index};s={node_id}",
                browse_name=browse_name,
                display_name=display_name,
                data_type=data_type,
                value=value,
                access_level=access,
                timestamp=datetime.now()
            )

    def read_value(self, node_id: str) -> Tuple[Any, int]:
        """读取节点值"""
        # 简化node_id格式
        simple_id = node_id.split(';s=')[-1] if ';s=' in node_id else node_id

        if simple_id in self.nodes:
            node = self.nodes[simple_id]
            return (node.value, node.status_code)
        return (None, 0x80000000)  # BadNodeIdUnknown

    def write_value(self, node_id: str, value: Any) -> int:
        """写入节点值"""
        simple_id = node_id.split(';s=')[-1] if ';s=' in node_id else node_id

        if simple_id not in self.nodes:
            return 0x80000000  # BadNodeIdUnknown

        node = self.nodes[simple_id]
        if node.access_level not in ['Write', 'ReadWrite']:
            return 0x803B0000  # BadNotWritable

        node.value = value
        node.timestamp = datetime.now()
        return 0  # Good

    def update_from_state(self, state: Dict[str, Any]):
        """从系统状态更新OPC-UA节点"""
        mappings = {
            'h': 'Objects/Aqueduct/Hydraulics/WaterLevel',
            'Q_in': 'Objects/Aqueduct/Hydraulics/InletFlow',
            'Q_out': 'Objects/Aqueduct/Hydraulics/OutletFlow',
            'fr': 'Objects/Aqueduct/Hydraulics/FroudeNumber',
            'v': 'Objects/Aqueduct/Hydraulics/Velocity',
            'T_sun': 'Objects/Aqueduct/Thermal/SunSideTemp',
            'T_shade': 'Objects/Aqueduct/Thermal/ShadeSideTemp',
            'vib_amp': 'Objects/Aqueduct/Structural/VibrationAmp',
            'joint_gap': 'Objects/Aqueduct/Structural/JointGap',
            'ground_accel': 'Objects/Aqueduct/Structural/GroundAccel'
        }

        for state_key, node_id in mappings.items():
            if state_key in state and node_id in self.nodes:
                self.nodes[node_id].value = state[state_key]
                self.nodes[node_id].timestamp = datetime.now()

        # 温差
        if 'T_sun' in state and 'T_shade' in state:
            T_delta = abs(state['T_sun'] - state['T_shade'])
            node_id = 'Objects/Aqueduct/Thermal/TempDelta'
            if node_id in self.nodes:
                self.nodes[node_id].value = T_delta
                self.nodes[node_id].timestamp = datetime.now()

    def get_all_nodes(self) -> Dict[str, Any]:
        """获取所有节点"""
        return {
            node_id: {
                'browse_name': node.browse_name,
                'display_name': node.display_name,
                'data_type': node.data_type,
                'value': node.value,
                'access_level': node.access_level,
                'timestamp': node.timestamp.isoformat()
            }
            for node_id, node in self.nodes.items()
        }


class ModbusInterface:
    """Modbus通信接口"""

    def __init__(self):
        # 寄存器映射
        self.holding_registers: Dict[int, float] = {}  # 保持寄存器 (可读写)
        self.input_registers: Dict[int, float] = {}    # 输入寄存器 (只读)
        self.coils: Dict[int, bool] = {}               # 线圈 (可读写)
        self.discrete_inputs: Dict[int, bool] = {}     # 离散输入 (只读)

        self._init_registers()

    def _init_registers(self):
        """初始化寄存器映射"""
        # 输入寄存器 (只读) - 测量值
        # 每个浮点数占用2个寄存器 (32位)
        self.register_map = {
            # 输入寄存器 (30001-30999)
            'h': (0, 'input', 'float'),
            'Q_in': (2, 'input', 'float'),
            'Q_out': (4, 'input', 'float'),
            'fr': (6, 'input', 'float'),
            'v': (8, 'input', 'float'),
            'T_sun': (10, 'input', 'float'),
            'T_shade': (12, 'input', 'float'),
            'vib_amp': (14, 'input', 'float'),
            'joint_gap': (16, 'input', 'float'),
            'ground_accel': (18, 'input', 'float'),

            # 保持寄存器 (40001-40999) - 设定值
            'Q_in_sp': (0, 'holding', 'float'),
            'Q_out_sp': (2, 'holding', 'float'),
            'h_target': (4, 'holding', 'float'),
            'control_mode': (6, 'holding', 'int'),
        }

        # 线圈 (00001-09999) - 控制命令
        self.coil_map = {
            'inlet_valve_open': 0,
            'outlet_valve_open': 1,
            'pump_run': 2,
            'emergency_stop': 3,
            'auto_mode': 4,
            'manual_mode': 5,
        }

        # 离散输入 (10001-19999) - 状态
        self.discrete_map = {
            'inlet_valve_status': 0,
            'outlet_valve_status': 1,
            'pump_status': 2,
            'system_fault': 3,
            'high_level_alarm': 4,
            'low_level_alarm': 5,
        }

        # 初始化值
        for addr in range(0, 40, 2):
            self.input_registers[addr] = 0.0
            self.input_registers[addr + 1] = 0.0

        for addr in range(0, 20, 2):
            self.holding_registers[addr] = 0.0
            self.holding_registers[addr + 1] = 0.0

        for addr in range(10):
            self.coils[addr] = False
            self.discrete_inputs[addr] = False

    def _float_to_registers(self, value: float) -> Tuple[int, int]:
        """将浮点数转换为两个16位寄存器"""
        packed = struct.pack('>f', value)
        high, low = struct.unpack('>HH', packed)
        return high, low

    def _registers_to_float(self, high: int, low: int) -> float:
        """将两个16位寄存器转换为浮点数"""
        packed = struct.pack('>HH', high, low)
        return struct.unpack('>f', packed)[0]

    def read_input_registers(self, address: int, count: int) -> List[int]:
        """读取输入寄存器 (功能码04)"""
        result = []
        for i in range(count):
            addr = address + i
            if addr in self.input_registers:
                # 返回16位整数形式
                result.append(int(self.input_registers[addr]) & 0xFFFF)
            else:
                result.append(0)
        return result

    def read_holding_registers(self, address: int, count: int) -> List[int]:
        """读取保持寄存器 (功能码03)"""
        result = []
        for i in range(count):
            addr = address + i
            if addr in self.holding_registers:
                result.append(int(self.holding_registers[addr]) & 0xFFFF)
            else:
                result.append(0)
        return result

    def write_single_register(self, address: int, value: int) -> bool:
        """写单个寄存器 (功能码06)"""
        self.holding_registers[address] = value
        return True

    def write_multiple_registers(self, address: int, values: List[int]) -> bool:
        """写多个寄存器 (功能码16)"""
        for i, value in enumerate(values):
            self.holding_registers[address + i] = value
        return True

    def read_coils(self, address: int, count: int) -> List[bool]:
        """读取线圈 (功能码01)"""
        return [self.coils.get(address + i, False) for i in range(count)]

    def write_single_coil(self, address: int, value: bool) -> bool:
        """写单个线圈 (功能码05)"""
        self.coils[address] = value
        return True

    def read_discrete_inputs(self, address: int, count: int) -> List[bool]:
        """读取离散输入 (功能码02)"""
        return [self.discrete_inputs.get(address + i, False) for i in range(count)]

    def update_from_state(self, state: Dict[str, Any]):
        """从系统状态更新Modbus寄存器"""
        for var_name, (addr, reg_type, data_type) in self.register_map.items():
            if var_name in state and reg_type == 'input':
                value = state[var_name]
                if data_type == 'float':
                    high, low = self._float_to_registers(value)
                    self.input_registers[addr] = high
                    self.input_registers[addr + 1] = low
                else:
                    self.input_registers[addr] = int(value)

        # 更新离散输入
        self.discrete_inputs[self.discrete_map['high_level_alarm']] = state.get('h', 4.0) > 5.5
        self.discrete_inputs[self.discrete_map['low_level_alarm']] = state.get('h', 4.0) < 2.5


class DataHistorian:
    """数据历史记录器"""

    def __init__(self, storage_path: str = "data/historian"):
        self.storage_path = storage_path
        self.buffer: Dict[str, deque] = {}
        self.buffer_size = 1000

        # 压缩配置
        self.compression_enabled = True
        self.deadband = 0.01  # 1%死区

    def record(self, tag_name: str, value: float, quality: int = 0):
        """记录数据点"""
        if tag_name not in self.buffer:
            self.buffer[tag_name] = deque(maxlen=self.buffer_size)

        timestamp = datetime.now()

        # 死区压缩
        if self.compression_enabled and len(self.buffer[tag_name]) > 0:
            last_value = self.buffer[tag_name][-1]['value']
            if abs(value - last_value) / max(abs(last_value), 0.001) < self.deadband:
                return  # 在死区内，不记录

        self.buffer[tag_name].append({
            'timestamp': timestamp.isoformat(),
            'value': value,
            'quality': quality
        })

    def query(self, tag_name: str, start_time: datetime,
              end_time: datetime) -> List[Dict[str, Any]]:
        """查询历史数据"""
        if tag_name not in self.buffer:
            return []

        result = []
        for record in self.buffer[tag_name]:
            ts = datetime.fromisoformat(record['timestamp'])
            if start_time <= ts <= end_time:
                result.append(record)

        return result

    def get_statistics(self, tag_name: str, start_time: datetime,
                       end_time: datetime) -> Dict[str, Any]:
        """获取统计数据"""
        data = self.query(tag_name, start_time, end_time)
        if not data:
            return {'error': 'No data'}

        values = [d['value'] for d in data]

        return {
            'tag': tag_name,
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'first': values[0],
            'last': values[-1]
        }


class EngineeringIntegrationManager:
    """工程集成管理器"""

    def __init__(self):
        self.scada = SCADAInterface()
        self.opcua = OPCUAServer()
        self.modbus = ModbusInterface()
        self.historian = DataHistorian()

        # 设置回调
        self.scada.on_alarm = self._handle_alarm

    def _handle_alarm(self, alarm: SCADAAlarm):
        """处理SCADA告警"""
        # 可以在这里添加告警处理逻辑
        pass

    def update_all(self, state: Dict[str, Any]):
        """更新所有接口"""
        self.scada.update_from_state(state)
        self.opcua.update_from_state(state)
        self.modbus.update_from_state(state)

        # 记录历史
        for key, value in state.items():
            if isinstance(value, (int, float)):
                self.historian.record(key, float(value))

    def get_status(self) -> Dict[str, Any]:
        """获取所有接口状态"""
        return {
            'scada': {
                'points_count': len(self.scada.points),
                'active_alarms': len(self.scada.alarms),
                'alarms': self.scada.get_active_alarms()
            },
            'opcua': {
                'endpoint': self.opcua.endpoint,
                'nodes_count': len(self.opcua.nodes),
                'namespace': self.opcua.namespace_uri
            },
            'modbus': {
                'input_registers': len(self.modbus.input_registers),
                'holding_registers': len(self.modbus.holding_registers),
                'coils': len(self.modbus.coils)
            },
            'historian': {
                'tags_count': len(self.historian.buffer),
                'compression': self.historian.compression_enabled
            }
        }

    def get_scada_values(self) -> Dict[str, Any]:
        """获取SCADA数据"""
        return self.scada.get_all_values()

    def get_opcua_nodes(self) -> Dict[str, Any]:
        """获取OPC-UA节点"""
        return self.opcua.get_all_nodes()

    def write_setpoint(self, parameter: str, value: float) -> bool:
        """写入设定值"""
        success = True

        # 写入SCADA
        tag_map = {
            'Q_in': 'AO_Q_IN_SP',
            'Q_out': 'AO_Q_OUT_SP',
            'h_target': 'AO_H_TARGET'
        }
        if parameter in tag_map:
            success = success and self.scada.write_value(tag_map[parameter], value)

        # 写入OPC-UA
        node_map = {
            'Q_in': 'Objects/Aqueduct/Control/InletFlowSP',
            'Q_out': 'Objects/Aqueduct/Control/OutletFlowSP',
            'h_target': 'Objects/Aqueduct/Control/TargetLevel'
        }
        if parameter in node_map:
            status = self.opcua.write_value(node_map[parameter], value)
            success = success and (status == 0)

        return success


# 全局实例
_integration_instance: Optional[EngineeringIntegrationManager] = None


def get_integration() -> EngineeringIntegrationManager:
    """获取全局工程集成管理器实例"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = EngineeringIntegrationManager()
    return _integration_instance
