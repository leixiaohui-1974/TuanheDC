"""
TAOS V3.10 Phase 1 Modules Comprehensive Tests
Phase 1 模块综合测试

Tests for:
- Real-time Data Interface (OPC-UA, Modbus, MQTT)
- Alarm and Event Management
- Reporting and Visualization
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import Phase 1 modules
from realtime_data_interface import (
    ProtocolType, DataQuality, ModbusDataType, DataPoint,
    ConnectionConfig, OPCUAConfig, ModbusConfig, MQTTConfig, TagConfig,
    ProtocolAdapter, OPCUAAdapter, ModbusAdapter, MQTTAdapter,
    DataBuffer, RealtimeDataManager, create_adapter, get_realtime_data_manager
)

from alarm_event_management import (
    AlarmSeverity, AlarmState, AlarmCategory, ComparisonOperator,
    NotificationType, AlarmCondition, AlarmRule, AlarmInstance, Event,
    NotificationConfig, EventCorrelator, AlarmManager,
    get_alarm_manager, create_high_limit_rule, create_low_limit_rule, create_range_rule
)

from reporting_visualization import (
    ReportType, AggregationType, ChartType, ExportFormat, TimeResolution,
    DataPoint as ReportDataPoint, AggregatedData, ChartConfig, ReportSection,
    ReportTemplate, Report, DashboardWidget, Dashboard,
    DataAggregator, ChartGenerator, ReportExporter, ReportGenerator,
    DashboardManager, ReportingManager, get_reporting_manager,
    create_daily_operations_template
)


# ============================================================
# Real-time Data Interface Tests
# ============================================================

class TestRealtimeDataInterface:
    """Tests for real-time data interface module"""

    def test_protocol_types(self):
        """Test protocol type enumeration"""
        assert ProtocolType.OPC_UA.value == "opc_ua"
        assert ProtocolType.MODBUS_TCP.value == "modbus_tcp"
        assert ProtocolType.MODBUS_RTU.value == "modbus_rtu"
        assert ProtocolType.MQTT.value == "mqtt"

    def test_data_quality(self):
        """Test data quality enumeration"""
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.BAD.value == "bad"
        assert DataQuality.COMM_ERROR.value == "comm_error"

    def test_data_point_creation(self):
        """Test DataPoint creation and serialization"""
        dp = DataPoint(
            tag_id="test_tag",
            value=42.5,
            timestamp=datetime.now(),
            quality=DataQuality.GOOD,
            source="test_source",
            unit="m³/s"
        )

        assert dp.tag_id == "test_tag"
        assert dp.value == 42.5
        assert dp.quality == DataQuality.GOOD

        # Test serialization
        dp_dict = dp.to_dict()
        assert dp_dict['tag_id'] == "test_tag"
        assert dp_dict['value'] == 42.5
        assert dp_dict['quality'] == "good"

    def test_opcua_config(self):
        """Test OPC-UA configuration"""
        config = OPCUAConfig(
            name="test_opcua",
            host="192.168.1.100",
            port=4840,
            namespace_index=2
        )

        assert config.protocol == ProtocolType.OPC_UA
        assert config.endpoint_url == "opc.tcp://192.168.1.100:4840"
        assert config.namespace_index == 2

    def test_modbus_config(self):
        """Test Modbus configuration"""
        # TCP config
        tcp_config = ModbusConfig(
            name="test_modbus_tcp",
            host="192.168.1.101",
            port=502,
            slave_id=1,
            is_rtu=False
        )
        assert tcp_config.protocol == ProtocolType.MODBUS_TCP

        # RTU config
        rtu_config = ModbusConfig(
            name="test_modbus_rtu",
            host="/dev/ttyUSB0",
            port=0,
            slave_id=1,
            is_rtu=True,
            baudrate=9600
        )
        assert rtu_config.protocol == ProtocolType.MODBUS_RTU

    def test_mqtt_config(self):
        """Test MQTT configuration"""
        config = MQTTConfig(
            name="test_mqtt",
            host="mqtt.example.com",
            port=1883,
            username="user",
            password="pass"
        )

        assert config.protocol == ProtocolType.MQTT
        assert "taos_client" in config.client_id

    def test_tag_config(self):
        """Test tag configuration"""
        tag = TagConfig(
            tag_id="water_level_1",
            name="Water Level Sensor 1",
            description="Main channel water level",
            unit="m",
            address="ns=2;s=WaterLevel.Sensor1",
            scale_factor=0.001,
            offset=0.0,
            sample_interval=1.0
        )

        assert tag.tag_id == "water_level_1"
        assert tag.scale_factor == 0.001


class TestOPCUAAdapter:
    """Tests for OPC-UA adapter"""

    def test_adapter_creation(self):
        """Test OPC-UA adapter creation"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)

        assert adapter.config.name == "test_opcua"
        assert not adapter.connected

    def test_connect_disconnect(self):
        """Test connection and disconnection"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)

        # Connect
        result = adapter.connect()
        assert result == True
        assert adapter.is_connected()

        # Disconnect
        result = adapter.disconnect()
        assert result == True
        assert not adapter.is_connected()

    def test_read_tag(self):
        """Test reading a tag"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)
        adapter.connect()

        tag = TagConfig(
            tag_id="test_tag",
            name="Test Tag",
            address="ns=2;s=TestTag",
            scale_factor=1.0
        )

        dp = adapter.read_tag(tag)

        assert dp.tag_id == "test_tag"
        assert dp.quality == DataQuality.GOOD
        assert dp.value is not None

        adapter.disconnect()

    def test_read_tags_batch(self):
        """Test batch reading"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)
        adapter.connect()

        tags = [
            TagConfig(tag_id=f"tag_{i}", name=f"Tag {i}", address=f"ns=2;s=Tag{i}")
            for i in range(5)
        ]

        data_points = adapter.read_tags(tags)

        assert len(data_points) == 5
        for dp in data_points:
            assert dp.quality == DataQuality.GOOD

        adapter.disconnect()

    def test_write_tag(self):
        """Test writing to a tag"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)
        adapter.connect()

        tag = TagConfig(
            tag_id="test_tag",
            name="Test Tag",
            address="ns=2;s=TestTag"
        )

        result = adapter.write_tag(tag, 50.0)
        assert result == True

        adapter.disconnect()

    def test_subscription(self):
        """Test tag subscription"""
        config = OPCUAConfig(
            name="test_opcua",
            host="localhost",
            port=4840
        )
        adapter = OPCUAAdapter(config)
        adapter.connect()

        tags = [
            TagConfig(tag_id="sub_tag", name="Sub Tag", address="ns=2;s=SubTag")
        ]

        received_data = []
        def callback(dp):
            received_data.append(dp)

        sub_id = adapter.subscribe(tags, callback, interval=1.0)
        assert sub_id is not None

        # Unsubscribe
        result = adapter.unsubscribe(sub_id)
        assert result == True

        adapter.disconnect()


class TestModbusAdapter:
    """Tests for Modbus adapter"""

    def test_adapter_creation(self):
        """Test Modbus adapter creation"""
        config = ModbusConfig(
            name="test_modbus",
            host="localhost",
            port=502,
            slave_id=1
        )
        adapter = ModbusAdapter(config)

        assert adapter.config.slave_id == 1

    def test_address_parsing(self):
        """Test Modbus address parsing"""
        config = ModbusConfig(
            name="test_modbus",
            host="localhost",
            port=502
        )
        adapter = ModbusAdapter(config)

        # Test holding register
        data_type, register = adapter._parse_address("HR:100")
        assert data_type == ModbusDataType.HOLDING_REGISTER
        assert register == 100

        # Test coil
        data_type, register = adapter._parse_address("C:50")
        assert data_type == ModbusDataType.COIL
        assert register == 50

        # Test input register
        data_type, register = adapter._parse_address("IR:200")
        assert data_type == ModbusDataType.INPUT_REGISTER
        assert register == 200

    def test_read_tag(self):
        """Test reading Modbus tag"""
        config = ModbusConfig(
            name="test_modbus",
            host="localhost",
            port=502,
            slave_id=1
        )
        adapter = ModbusAdapter(config)
        adapter.connect()

        tag = TagConfig(
            tag_id="modbus_tag",
            name="Modbus Tag",
            address="HR:100",
            scale_factor=0.01
        )

        dp = adapter.read_tag(tag)

        assert dp.tag_id == "modbus_tag"
        assert dp.quality == DataQuality.GOOD
        assert 'register' in dp.metadata

        adapter.disconnect()

    def test_batch_register_read(self):
        """Test batch register reading"""
        config = ModbusConfig(
            name="test_modbus",
            host="localhost",
            port=502
        )
        adapter = ModbusAdapter(config)
        adapter.connect()

        values = adapter.read_registers_batch(100, 10)

        assert len(values) == 10
        for v in values:
            assert 0 <= v <= 65535

        adapter.disconnect()


class TestMQTTAdapter:
    """Tests for MQTT adapter"""

    def test_adapter_creation(self):
        """Test MQTT adapter creation"""
        config = MQTTConfig(
            name="test_mqtt",
            host="localhost",
            port=1883
        )
        adapter = MQTTAdapter(config)

        assert "taos_client" in adapter.config.client_id

    def test_connect_disconnect(self):
        """Test MQTT connection"""
        config = MQTTConfig(
            name="test_mqtt",
            host="localhost",
            port=1883
        )
        adapter = MQTTAdapter(config)

        result = adapter.connect()
        assert result == True
        assert adapter.is_connected()

        result = adapter.disconnect()
        assert result == True

    def test_subscribe_publish(self):
        """Test MQTT subscribe and publish"""
        config = MQTTConfig(
            name="test_mqtt",
            host="localhost",
            port=1883
        )
        adapter = MQTTAdapter(config)
        adapter.connect()

        received = []
        def callback(topic, payload):
            received.append((topic, payload))

        result = adapter.subscribe_topic("test/topic", callback)
        assert result == True

        result = adapter.publish("test/topic", {"value": 42})
        assert result == True

        adapter.disconnect()

    def test_unsubscribe(self):
        """Test MQTT unsubscribe"""
        config = MQTTConfig(
            name="test_mqtt",
            host="localhost",
            port=1883
        )
        adapter = MQTTAdapter(config)
        adapter.connect()

        adapter.subscribe_topic("test/topic", lambda t, p: None)
        result = adapter.unsubscribe_topic("test/topic")
        assert result == True

        adapter.disconnect()


class TestDataBuffer:
    """Tests for data buffer"""

    def test_buffer_creation(self):
        """Test buffer creation"""
        buffer = DataBuffer(max_size=1000)
        assert buffer.max_size == 1000

    def test_add_and_get(self):
        """Test adding and getting data"""
        buffer = DataBuffer()

        dp = DataPoint(
            tag_id="test_tag",
            value=42.0,
            timestamp=datetime.now()
        )
        buffer.add(dp)

        latest = buffer.get_latest("test_tag")
        assert latest is not None
        assert latest.value == 42.0

    def test_get_history(self):
        """Test getting historical data"""
        buffer = DataBuffer()

        for i in range(10):
            dp = DataPoint(
                tag_id="test_tag",
                value=float(i),
                timestamp=datetime.now()
            )
            buffer.add(dp)

        history = buffer.get_history("test_tag", count=5)
        assert len(history) == 5

    def test_time_range_query(self):
        """Test time range query"""
        buffer = DataBuffer()

        now = datetime.now()
        for i in range(10):
            dp = DataPoint(
                tag_id="test_tag",
                value=float(i),
                timestamp=now + timedelta(minutes=i)
            )
            buffer.add(dp)

        start = now + timedelta(minutes=2)
        end = now + timedelta(minutes=7)

        result = buffer.get_time_range("test_tag", start, end)
        assert len(result) == 6

    def test_buffer_statistics(self):
        """Test buffer statistics"""
        buffer = DataBuffer()

        for i in range(100):
            buffer.add(DataPoint(
                tag_id=f"tag_{i % 5}",
                value=float(i),
                timestamp=datetime.now()
            ))

        stats = buffer.get_statistics()
        assert stats['tag_count'] == 5
        assert stats['total_points'] == 100

    def test_buffer_clear(self):
        """Test buffer clearing"""
        buffer = DataBuffer()

        for i in range(10):
            buffer.add(DataPoint(
                tag_id="test_tag",
                value=float(i),
                timestamp=datetime.now()
            ))

        buffer.clear("test_tag")
        assert buffer.get_latest("test_tag") is None

        buffer.add(DataPoint(tag_id="tag1", value=1, timestamp=datetime.now()))
        buffer.add(DataPoint(tag_id="tag2", value=2, timestamp=datetime.now()))
        buffer.clear()

        stats = buffer.get_statistics()
        assert stats['tag_count'] == 0


class TestRealtimeDataManager:
    """Tests for real-time data manager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = RealtimeDataManager()
        assert len(manager.adapters) == 0
        assert len(manager.tags) == 0

    def test_add_adapter(self):
        """Test adding adapter"""
        manager = RealtimeDataManager()

        config = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)

        result = manager.add_adapter("opcua1", adapter)
        assert result == True
        assert "opcua1" in manager.adapters

    def test_add_tag(self):
        """Test adding tag to adapter"""
        manager = RealtimeDataManager()

        config = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)
        manager.add_adapter("opcua1", adapter)

        tag = TagConfig(
            tag_id="water_level",
            name="Water Level",
            address="ns=2;s=WL"
        )

        result = manager.add_tag("opcua1", tag)
        assert result == True
        assert "water_level" in manager.tags

    def test_connect_all(self):
        """Test connecting all adapters"""
        manager = RealtimeDataManager()

        config1 = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        config2 = ModbusConfig(name="modbus1", host="localhost", port=502)

        manager.add_adapter("opcua1", OPCUAAdapter(config1))
        manager.add_adapter("modbus1", ModbusAdapter(config2))

        results = manager.connect_all()

        assert results["opcua1"] == True
        assert results["modbus1"] == True

    def test_read_all_tags(self):
        """Test reading all tags"""
        manager = RealtimeDataManager()

        config = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)
        manager.add_adapter("opcua1", adapter)

        for i in range(3):
            tag = TagConfig(
                tag_id=f"tag_{i}",
                name=f"Tag {i}",
                address=f"ns=2;s=Tag{i}"
            )
            manager.add_tag("opcua1", tag)

        manager.connect_all()

        data_points = manager.read_all_tags()
        assert len(data_points) == 3

    def test_callback_registration(self):
        """Test callback registration"""
        manager = RealtimeDataManager()

        config = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)
        manager.add_adapter("opcua1", adapter)

        tag = TagConfig(tag_id="test", name="Test", address="ns=2;s=Test")
        manager.add_tag("opcua1", tag)

        received = []
        def callback(dp):
            received.append(dp)

        manager.register_callback(callback)
        manager.connect_all()
        manager.read_tag("test")

        assert len(received) == 1

    def test_manager_status(self):
        """Test getting manager status"""
        manager = RealtimeDataManager()

        config = OPCUAConfig(name="opcua1", host="localhost", port=4840)
        manager.add_adapter("opcua1", OPCUAAdapter(config))
        manager.connect_all()

        status = manager.get_status()

        assert "adapters" in status
        assert "opcua1" in status["adapters"]
        assert status["adapters"]["opcua1"]["connected"] == True

    def test_create_adapter_factory(self):
        """Test adapter factory function"""
        opcua_config = OPCUAConfig(name="test", host="localhost", port=4840)
        opcua_adapter = create_adapter(opcua_config)
        assert isinstance(opcua_adapter, OPCUAAdapter)

        modbus_config = ModbusConfig(name="test", host="localhost", port=502)
        modbus_adapter = create_adapter(modbus_config)
        assert isinstance(modbus_adapter, ModbusAdapter)

        mqtt_config = MQTTConfig(name="test", host="localhost", port=1883)
        mqtt_adapter = create_adapter(mqtt_config)
        assert isinstance(mqtt_adapter, MQTTAdapter)


# ============================================================
# Alarm and Event Management Tests
# ============================================================

class TestAlarmEnums:
    """Tests for alarm enumerations"""

    def test_alarm_severity(self):
        """Test alarm severity levels"""
        assert AlarmSeverity.INFO.value == 1
        assert AlarmSeverity.EMERGENCY.value == 5
        assert AlarmSeverity.CRITICAL.display_name == "严重"

    def test_alarm_state(self):
        """Test alarm states"""
        assert AlarmState.ACTIVE.value == "active"
        assert AlarmState.ACKNOWLEDGED.value == "acknowledged"
        assert AlarmState.CLEARED.value == "cleared"

    def test_comparison_operators(self):
        """Test comparison operators"""
        assert ComparisonOperator.GREATER.value == ">"
        assert ComparisonOperator.IN_RANGE.value == "in_range"


class TestAlarmCondition:
    """Tests for alarm conditions"""

    def test_greater_than_condition(self):
        """Test greater than condition"""
        condition = AlarmCondition(
            tag_id="water_level",
            operator=ComparisonOperator.GREATER,
            threshold=10.0
        )

        assert condition.evaluate(11.0) == True
        assert condition.evaluate(9.0) == False
        assert condition.evaluate(10.0) == False

    def test_less_than_condition(self):
        """Test less than condition"""
        condition = AlarmCondition(
            tag_id="pressure",
            operator=ComparisonOperator.LESS,
            threshold=5.0
        )

        assert condition.evaluate(4.0) == True
        assert condition.evaluate(6.0) == False

    def test_in_range_condition(self):
        """Test in range condition"""
        condition = AlarmCondition(
            tag_id="temperature",
            operator=ComparisonOperator.IN_RANGE,
            threshold=20.0,
            threshold_high=30.0
        )

        assert condition.evaluate(25.0) == True
        assert condition.evaluate(15.0) == False
        assert condition.evaluate(35.0) == False

    def test_out_of_range_condition(self):
        """Test out of range condition"""
        condition = AlarmCondition(
            tag_id="flow",
            operator=ComparisonOperator.OUT_OF_RANGE,
            threshold=10.0,
            threshold_high=100.0
        )

        assert condition.evaluate(5.0) == True
        assert condition.evaluate(110.0) == True
        assert condition.evaluate(50.0) == False

    def test_deadband(self):
        """Test deadband in conditions"""
        condition = AlarmCondition(
            tag_id="level",
            operator=ComparisonOperator.GREATER,
            threshold=10.0,
            deadband=0.5
        )

        assert condition.evaluate(10.6) == True
        assert condition.evaluate(10.4) == False

    def test_deviation_condition(self):
        """Test deviation condition"""
        condition = AlarmCondition(
            tag_id="setpoint",
            operator=ComparisonOperator.DEVIATION,
            threshold=5.0,
            reference_value=100.0
        )

        assert condition.evaluate(110.0) == True
        assert condition.evaluate(103.0) == False


class TestAlarmRule:
    """Tests for alarm rules"""

    def test_rule_creation(self):
        """Test alarm rule creation"""
        rule = AlarmRule(
            rule_id="high_level_001",
            name="High Water Level",
            description="Water level exceeds maximum",
            severity=AlarmSeverity.ALERT,
            category=AlarmCategory.PROCESS,
            conditions=[
                AlarmCondition(
                    tag_id="water_level",
                    operator=ComparisonOperator.GREATER,
                    threshold=10.0
                )
            ]
        )

        assert rule.rule_id == "high_level_001"
        assert rule.severity == AlarmSeverity.ALERT

    def test_rule_evaluation_and_logic(self):
        """Test rule evaluation with AND logic"""
        rule = AlarmRule(
            rule_id="complex_rule",
            name="Complex Alarm",
            description="Multiple conditions",
            severity=AlarmSeverity.CRITICAL,
            category=AlarmCategory.SAFETY,
            conditions=[
                AlarmCondition(
                    tag_id="level",
                    operator=ComparisonOperator.GREATER,
                    threshold=10.0
                ),
                AlarmCondition(
                    tag_id="flow",
                    operator=ComparisonOperator.GREATER,
                    threshold=50.0
                )
            ],
            condition_logic="AND"
        )

        # Both conditions met
        assert rule.evaluate({"level": 15.0, "flow": 60.0}) == True

        # Only one condition met
        assert rule.evaluate({"level": 15.0, "flow": 40.0}) == False

    def test_rule_evaluation_or_logic(self):
        """Test rule evaluation with OR logic"""
        rule = AlarmRule(
            rule_id="or_rule",
            name="OR Alarm",
            description="Either condition",
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.PROCESS,
            conditions=[
                AlarmCondition(
                    tag_id="sensor1",
                    operator=ComparisonOperator.GREATER,
                    threshold=100.0
                ),
                AlarmCondition(
                    tag_id="sensor2",
                    operator=ComparisonOperator.GREATER,
                    threshold=100.0
                )
            ],
            condition_logic="OR"
        )

        assert rule.evaluate({"sensor1": 110.0, "sensor2": 50.0}) == True
        assert rule.evaluate({"sensor1": 50.0, "sensor2": 110.0}) == True
        assert rule.evaluate({"sensor1": 50.0, "sensor2": 50.0}) == False

    def test_disabled_rule(self):
        """Test disabled rule"""
        rule = AlarmRule(
            rule_id="disabled_rule",
            name="Disabled",
            description="Should not trigger",
            severity=AlarmSeverity.WARNING,
            category=AlarmCategory.PROCESS,
            conditions=[
                AlarmCondition(
                    tag_id="value",
                    operator=ComparisonOperator.GREATER,
                    threshold=0
                )
            ],
            enabled=False
        )

        assert rule.evaluate({"value": 100}) == False


class TestAlarmInstance:
    """Tests for alarm instances"""

    def test_alarm_instance_creation(self):
        """Test alarm instance creation"""
        alarm = AlarmInstance(
            alarm_id="alarm_001",
            rule_id="rule_001",
            name="Test Alarm",
            description="Test description",
            severity=AlarmSeverity.ALERT,
            category=AlarmCategory.PROCESS,
            state=AlarmState.ACTIVE,
            timestamp=datetime.now(),
            source_tag="water_level",
            trigger_value=12.5,
            threshold=10.0
        )

        assert alarm.alarm_id == "alarm_001"
        assert alarm.state == AlarmState.ACTIVE

    def test_alarm_serialization(self):
        """Test alarm serialization"""
        alarm = AlarmInstance(
            alarm_id="alarm_001",
            rule_id="rule_001",
            name="Test Alarm",
            description="Test",
            severity=AlarmSeverity.CRITICAL,
            category=AlarmCategory.SAFETY,
            state=AlarmState.ACTIVE,
            timestamp=datetime.now(),
            source_tag="sensor",
            trigger_value=100,
            threshold=90
        )

        data = alarm.to_dict()

        assert data['alarm_id'] == "alarm_001"
        assert data['severity'] == 4
        assert data['severity_name'] == "严重"
        assert data['state'] == "active"


class TestEventCorrelator:
    """Tests for event correlation"""

    def test_correlator_creation(self):
        """Test correlator creation"""
        correlator = EventCorrelator(correlation_window=60.0)
        assert correlator.correlation_window == 60.0

    def test_add_event(self):
        """Test adding events"""
        correlator = EventCorrelator()

        event = Event(
            event_id="evt_001",
            event_type="comm_error",
            timestamp=datetime.now(),
            source="plc_1",
            message="Communication timeout"
        )

        correlator.add_event(event)
        assert len(correlator.event_buffer) == 1

    def test_find_correlations(self):
        """Test finding correlated events"""
        correlator = EventCorrelator(correlation_window=60.0)

        now = datetime.now()

        # Add some events
        correlator.add_event(Event(
            event_id="evt_001",
            event_type="comm_error",
            timestamp=now - timedelta(seconds=30),
            source="plc_1",
            message="Comm error",
            category="communication"
        ))

        # Create alarm to correlate
        alarm = AlarmInstance(
            alarm_id="alm_001",
            rule_id="rule_001",
            name="Process Alarm",
            description="Process alarm after comm error",
            severity=AlarmSeverity.ALERT,
            category=AlarmCategory.PROCESS,
            state=AlarmState.ACTIVE,
            timestamp=now,
            source_tag="sensor",
            trigger_value=100,
            threshold=90
        )

        correlations = correlator.find_correlations(alarm)

        assert len(correlations) > 0
        assert correlations[0]['type'] == 'temporal'


class TestAlarmManager:
    """Tests for alarm manager"""

    def test_manager_creation(self):
        """Test alarm manager creation"""
        manager = AlarmManager()
        assert len(manager.rules) == 0
        assert len(manager.active_alarms) == 0

    def test_add_rule(self):
        """Test adding alarm rule"""
        manager = AlarmManager()

        rule = create_high_limit_rule(
            "high_level",
            "High Level Alarm",
            "water_level",
            10.0,
            AlarmSeverity.ALERT
        )

        result = manager.add_rule(rule)
        assert result == True
        assert "high_level" in manager.rules

    def test_process_values_trigger_alarm(self):
        """Test processing values that trigger alarm"""
        manager = AlarmManager()

        rule = create_high_limit_rule(
            "high_level",
            "High Level",
            "level",
            10.0,
            AlarmSeverity.ALERT
        )
        manager.add_rule(rule)

        # Process value above threshold
        manager.process_values({"level": 15.0})

        assert len(manager.active_alarms) == 1

    def test_auto_clear_alarm(self):
        """Test automatic alarm clearing"""
        manager = AlarmManager()

        rule = create_high_limit_rule(
            "high_level",
            "High Level",
            "level",
            10.0
        )
        manager.add_rule(rule)

        # Trigger alarm
        manager.process_values({"level": 15.0})
        assert len(manager.active_alarms) == 1

        # Value returns to normal
        manager.process_values({"level": 5.0})
        assert len(manager.active_alarms) == 0

    def test_acknowledge_alarm(self):
        """Test acknowledging alarm"""
        manager = AlarmManager()

        rule = create_high_limit_rule(
            "high_level",
            "High Level",
            "level",
            10.0
        )
        manager.add_rule(rule)

        manager.process_values({"level": 15.0})
        alarm_id = list(manager.active_alarms.keys())[0]

        result = manager.acknowledge_alarm(alarm_id, "operator1", "Acknowledged")

        assert result == True
        alarm = manager.active_alarms[alarm_id]
        assert alarm.state == AlarmState.ACKNOWLEDGED
        assert alarm.acknowledged_by == "operator1"

    def test_shelve_alarm(self):
        """Test shelving alarm"""
        manager = AlarmManager()

        rule = create_high_limit_rule("r1", "Rule 1", "tag", 10.0)
        manager.add_rule(rule)

        manager.process_values({"tag": 15.0})
        alarm_id = list(manager.active_alarms.keys())[0]

        result = manager.shelve_alarm(alarm_id, 30, "operator1", "Maintenance")

        assert result == True
        alarm = manager.active_alarms[alarm_id]
        assert alarm.state == AlarmState.SHELVED

    def test_get_active_alarms_filtered(self):
        """Test getting filtered active alarms"""
        manager = AlarmManager()

        manager.add_rule(create_high_limit_rule(
            "warning_rule", "Warning", "tag1", 10.0, AlarmSeverity.WARNING
        ))
        manager.add_rule(create_high_limit_rule(
            "critical_rule", "Critical", "tag2", 10.0, AlarmSeverity.CRITICAL
        ))

        manager.process_values({"tag1": 15.0, "tag2": 15.0})

        # Get all
        all_alarms = manager.get_active_alarms()
        assert len(all_alarms) == 2

        # Filter by severity
        critical_only = manager.get_active_alarms(severity=AlarmSeverity.CRITICAL)
        assert len(critical_only) == 1

    def test_alarm_statistics(self):
        """Test alarm statistics"""
        manager = AlarmManager()

        rule = create_high_limit_rule("r1", "Rule 1", "tag", 10.0)
        manager.add_rule(rule)

        # Trigger some alarms
        for i in range(3):
            manager.process_values({"tag": 15.0 + i})
            time.sleep(0.01)

        stats = manager.get_statistics(period_hours=1)

        assert 'active_count' in stats
        assert 'active_by_severity' in stats

    def test_log_event(self):
        """Test logging events"""
        manager = AlarmManager()

        event = manager.log_event(
            event_type="operator_action",
            source="console",
            message="Operator logged in",
            user="operator1"
        )

        assert event.event_id is not None
        assert event.event_type == "operator_action"

    def test_helper_functions(self):
        """Test helper functions for creating rules"""
        high_rule = create_high_limit_rule(
            "high", "High Limit", "tag", 100.0, AlarmSeverity.ALERT
        )
        assert high_rule.conditions[0].operator == ComparisonOperator.GREATER

        low_rule = create_low_limit_rule(
            "low", "Low Limit", "tag", 10.0, AlarmSeverity.WARNING
        )
        assert low_rule.conditions[0].operator == ComparisonOperator.LESS

        range_rule = create_range_rule(
            "range", "Range", "tag", 10.0, 100.0
        )
        assert range_rule.conditions[0].operator == ComparisonOperator.OUT_OF_RANGE


# ============================================================
# Reporting and Visualization Tests
# ============================================================

class TestDataAggregator:
    """Tests for data aggregator"""

    def test_aggregator_creation(self):
        """Test aggregator creation"""
        aggregator = DataAggregator()
        assert aggregator is not None

    def test_add_data(self):
        """Test adding data points"""
        aggregator = DataAggregator()

        dp = ReportDataPoint(
            timestamp=datetime.now(),
            value=42.0,
            tag_id="test_tag"
        )

        aggregator.add_data("test_tag", dp)
        assert "test_tag" in aggregator._cache

    def test_aggregate_average(self):
        """Test average aggregation"""
        aggregator = DataAggregator()

        now = datetime.now()
        for i in range(10):
            dp = ReportDataPoint(
                timestamp=now + timedelta(minutes=i),
                value=float(i * 10),
                tag_id="test"
            )
            aggregator.add_data("test", dp)

        result = aggregator.aggregate(
            "test",
            now - timedelta(minutes=1),
            now + timedelta(minutes=15),
            TimeResolution.HOUR,
            AggregationType.AVERAGE
        )

        assert result.tag_id == "test"
        assert len(result.values) >= 1
        assert 'avg' in result.statistics

    def test_aggregate_sum(self):
        """Test sum aggregation"""
        aggregator = DataAggregator()

        now = datetime.now()
        for i in range(5):
            aggregator.add_data("test", ReportDataPoint(
                timestamp=now + timedelta(seconds=i),
                value=10.0,
                tag_id="test"
            ))

        result = aggregator.aggregate(
            "test",
            now - timedelta(seconds=1),
            now + timedelta(seconds=10),
            TimeResolution.MINUTE,
            AggregationType.SUM
        )

        assert result.statistics['count'] == 5

    def test_aggregate_min_max(self):
        """Test min/max aggregation"""
        aggregator = DataAggregator()

        now = datetime.now()
        values = [5, 10, 15, 3, 20, 8]
        for i, v in enumerate(values):
            aggregator.add_data("test", ReportDataPoint(
                timestamp=now + timedelta(seconds=i),
                value=float(v),
                tag_id="test"
            ))

        result = aggregator.aggregate(
            "test",
            now - timedelta(seconds=1),
            now + timedelta(seconds=10),
            TimeResolution.MINUTE,
            AggregationType.AVERAGE
        )

        assert result.statistics['min'] == 3
        assert result.statistics['max'] == 20

    def test_time_bucketing(self):
        """Test time bucket calculation"""
        aggregator = DataAggregator()

        test_time = datetime(2024, 1, 15, 10, 37, 45)

        bucket = aggregator._get_bucket_time(test_time, TimeResolution.MINUTE)
        assert bucket.second == 0

        bucket = aggregator._get_bucket_time(test_time, TimeResolution.HOUR)
        assert bucket.minute == 0
        assert bucket.second == 0

        bucket = aggregator._get_bucket_time(test_time, TimeResolution.DAY)
        assert bucket.hour == 0


class TestChartGenerator:
    """Tests for chart generator"""

    def test_line_chart_generation(self):
        """Test line chart data generation"""
        generator = ChartGenerator()

        agg_data = AggregatedData(
            tag_id="water_level",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            resolution=TimeResolution.MINUTE,
            aggregation=AggregationType.AVERAGE,
            values=[
                {'timestamp': '2024-01-01T10:00:00', 'value': 5.0, 'count': 10},
                {'timestamp': '2024-01-01T10:01:00', 'value': 5.5, 'count': 10}
            ]
        )

        config = ChartConfig(
            chart_id="test_chart",
            title="Water Level",
            chart_type=ChartType.LINE,
            data_sources=["water_level"],
            show_legend=True
        )

        chart_data = generator.generate_line_chart([agg_data], config)

        assert chart_data['chart_type'] == 'line'
        assert chart_data['title'] == 'Water Level'
        assert len(chart_data['series']) == 1

    def test_bar_chart_generation(self):
        """Test bar chart data generation"""
        generator = ChartGenerator()

        agg_data = AggregatedData(
            tag_id="flow",
            start_time=datetime.now(),
            end_time=datetime.now(),
            resolution=TimeResolution.HOUR,
            aggregation=AggregationType.SUM,
            values=[
                {'timestamp': '10:00', 'value': 100, 'count': 1},
                {'timestamp': '11:00', 'value': 150, 'count': 1}
            ]
        )

        config = ChartConfig(
            chart_id="bar_test",
            title="Flow Rate",
            chart_type=ChartType.BAR,
            data_sources=["flow"]
        )

        chart_data = generator.generate_bar_chart([agg_data], config)

        assert chart_data['chart_type'] == 'bar'

    def test_gauge_generation(self):
        """Test gauge chart data generation"""
        generator = ChartGenerator()

        config = ChartConfig(
            chart_id="gauge_test",
            title="Pressure",
            chart_type=ChartType.GAUGE,
            data_sources=["pressure"],
            thresholds=[
                {'value': 80, 'color': 'green'},
                {'value': 90, 'color': 'yellow'},
                {'value': 100, 'color': 'red'}
            ]
        )

        gauge_data = generator.generate_gauge(75.0, config)

        assert gauge_data['chart_type'] == 'gauge'
        assert gauge_data['value'] == 75.0


class TestReportExporter:
    """Tests for report exporter"""

    def test_csv_export(self):
        """Test CSV export"""
        exporter = ReportExporter()

        data = [
            {'tag': 'level', 'value': 5.0, 'time': '10:00'},
            {'tag': 'level', 'value': 5.5, 'time': '11:00'}
        ]

        csv_output = exporter.export_csv(data, ['tag', 'value', 'time'])

        assert 'tag,value,time' in csv_output
        assert 'level,5.0,10:00' in csv_output

    def test_json_export(self):
        """Test JSON export"""
        exporter = ReportExporter()

        report = Report(
            report_id="test_001",
            template_id="template_001",
            name="Test Report",
            generated_at=datetime.now(),
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            sections=[],
            summary={'total': 100}
        )

        json_output = exporter.export_json(report)

        parsed = json.loads(json_output)
        assert parsed['report_id'] == "test_001"
        assert parsed['summary']['total'] == 100

    def test_html_export(self):
        """Test HTML export"""
        exporter = ReportExporter()

        report = Report(
            report_id="test_001",
            template_id="template_001",
            name="Daily Report",
            generated_at=datetime.now(),
            period_start=datetime.now() - timedelta(days=1),
            period_end=datetime.now(),
            sections=[
                {'title': 'Summary', 'section_type': 'text', 'content': 'Test content'}
            ],
            summary={'总流量': '1000 m³'}
        )

        html_output = exporter.export_html(report)

        assert '<!DOCTYPE html>' in html_output
        assert 'Daily Report' in html_output
        assert '总流量' in html_output


class TestReportGenerator:
    """Tests for report generator"""

    def test_generator_creation(self):
        """Test generator creation"""
        aggregator = DataAggregator()
        generator = ReportGenerator(aggregator)

        assert generator is not None

    def test_add_template(self):
        """Test adding template"""
        aggregator = DataAggregator()
        generator = ReportGenerator(aggregator)

        template = create_daily_operations_template()
        generator.add_template(template)

        assert "daily_ops" in generator.templates

    def test_generate_report(self):
        """Test report generation"""
        aggregator = DataAggregator()
        generator = ReportGenerator(aggregator)

        # Add some data
        now = datetime.now()
        for i in range(24):
            aggregator.add_data("water_level", ReportDataPoint(
                timestamp=now - timedelta(hours=24-i),
                value=5.0 + i * 0.1,
                tag_id="water_level"
            ))

        template = ReportTemplate(
            template_id="simple_test",
            name="Simple Test Report",
            description="Test",
            report_type=ReportType.DAILY,
            sections=[
                ReportSection(
                    section_id="intro",
                    title="Introduction",
                    section_type="text",
                    content="This is a test report."
                )
            ]
        )
        generator.add_template(template)

        report = generator.generate_report(
            "simple_test",
            now - timedelta(days=1),
            now
        )

        assert report.report_id is not None
        assert report.name == "Simple Test Report"
        assert len(report.sections) == 1


class TestDashboardManager:
    """Tests for dashboard manager"""

    def test_create_dashboard(self):
        """Test dashboard creation"""
        manager = DashboardManager()

        dashboard = Dashboard(
            dashboard_id="dash_001",
            name="Operations Dashboard",
            description="Main operations view"
        )

        dashboard_id = manager.create_dashboard(dashboard)

        assert dashboard_id == "dash_001"
        assert "dash_001" in manager.dashboards

    def test_add_widget(self):
        """Test adding widget to dashboard"""
        manager = DashboardManager()

        dashboard = Dashboard(
            dashboard_id="dash_001",
            name="Test Dashboard",
            description="Test"
        )
        manager.create_dashboard(dashboard)

        widget = DashboardWidget(
            widget_id="widget_001",
            title="Water Level",
            widget_type="gauge",
            config={'tag_id': 'water_level', 'min': 0, 'max': 10},
            position={'x': 0, 'y': 0, 'width': 4, 'height': 3}
        )

        result = manager.add_widget("dash_001", widget)

        assert result == True
        assert len(manager.dashboards["dash_001"].widgets) == 1

    def test_remove_widget(self):
        """Test removing widget"""
        manager = DashboardManager()

        dashboard = Dashboard(
            dashboard_id="dash_001",
            name="Test",
            description="Test"
        )
        manager.create_dashboard(dashboard)

        widget = DashboardWidget(
            widget_id="w1",
            title="Widget 1",
            widget_type="value",
            config={},
            position={'x': 0, 'y': 0, 'width': 2, 'height': 1}
        )
        manager.add_widget("dash_001", widget)

        result = manager.remove_widget("dash_001", "w1")

        assert result == True
        assert len(manager.dashboards["dash_001"].widgets) == 0

    def test_list_dashboards(self):
        """Test listing dashboards"""
        manager = DashboardManager()

        for i in range(3):
            manager.create_dashboard(Dashboard(
                dashboard_id=f"dash_{i}",
                name=f"Dashboard {i}",
                description=f"Description {i}"
            ))

        dashboard_list = manager.list_dashboards()

        assert len(dashboard_list) == 3


class TestReportingManager:
    """Tests for reporting manager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = ReportingManager()
        assert manager is not None

    def test_add_data(self):
        """Test adding data"""
        manager = ReportingManager()

        manager.add_data("water_level", 5.5)
        manager.add_data("water_level", 5.6)

        assert "water_level" in manager.aggregator._cache

    def test_add_batch_data(self):
        """Test adding batch data"""
        manager = ReportingManager()

        now = datetime.now()
        data = [
            (now - timedelta(minutes=i), 5.0 + i * 0.1)
            for i in range(10)
        ]

        manager.add_batch_data("flow", data)

        assert "flow" in manager.aggregator._cache

    def test_create_and_get_dashboard(self):
        """Test creating and getting dashboard"""
        manager = ReportingManager()

        dashboard = Dashboard(
            dashboard_id="main",
            name="Main Dashboard",
            description="Primary view"
        )

        dashboard_id = manager.create_dashboard(dashboard)
        retrieved = manager.get_dashboard(dashboard_id)

        assert retrieved is not None
        assert retrieved.name == "Main Dashboard"

    def test_get_status(self):
        """Test getting status"""
        manager = ReportingManager()

        manager.add_data("tag1", 10)
        manager.create_dashboard(Dashboard(
            dashboard_id="d1",
            name="D1",
            description="D1"
        ))

        status = manager.get_status()

        assert 'templates' in status
        assert 'dashboards' in status
        assert status['dashboards'] == 1


class TestIntegration:
    """Integration tests for Phase 1 modules"""

    def test_realtime_to_alarm_integration(self):
        """Test integration between real-time data and alarm system"""
        # Setup real-time data manager
        data_manager = RealtimeDataManager()
        config = OPCUAConfig(name="opcua", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)
        data_manager.add_adapter("opcua", adapter)

        tag = TagConfig(
            tag_id="water_level",
            name="Water Level",
            address="ns=2;s=WL"
        )
        data_manager.add_tag("opcua", tag)
        data_manager.connect_all()

        # Setup alarm manager
        alarm_manager = AlarmManager()
        alarm_manager.add_rule(create_high_limit_rule(
            "high_level",
            "High Water Level",
            "water_level",
            10.0,
            AlarmSeverity.ALERT
        ))

        # Read data and process alarms
        dp = data_manager.read_tag("water_level")
        alarm_manager.process_values({dp.tag_id: dp.value})

        # Check if alarm triggered based on value
        if dp.value > 10.0:
            assert len(alarm_manager.active_alarms) > 0

    def test_realtime_to_reporting_integration(self):
        """Test integration between real-time data and reporting"""
        # Setup real-time data
        data_manager = RealtimeDataManager()
        config = OPCUAConfig(name="opcua", host="localhost", port=4840)
        adapter = OPCUAAdapter(config)
        data_manager.add_adapter("opcua", adapter)
        data_manager.connect_all()

        # Setup reporting
        reporting_manager = ReportingManager()

        # Simulate data flow
        for i in range(10):
            data_manager.add_tag("opcua", TagConfig(
                tag_id=f"sensor_{i}",
                name=f"Sensor {i}",
                address=f"ns=2;s=S{i}"
            ))

        data_points = data_manager.read_all_tags()

        # Feed to reporting
        for dp in data_points:
            reporting_manager.add_data(dp.tag_id, dp.value, dp.timestamp)

        # Verify data in aggregator
        assert len(reporting_manager.aggregator._cache) == 10

    def test_alarm_to_reporting_integration(self):
        """Test integration between alarm and reporting systems"""
        alarm_manager = AlarmManager()
        reporting_manager = ReportingManager()

        # Setup alarm rule
        alarm_manager.add_rule(create_high_limit_rule(
            "high_temp",
            "High Temperature",
            "temperature",
            80.0,
            AlarmSeverity.CRITICAL
        ))

        # Register callback to send to reporting
        def on_alarm(alarm):
            reporting_manager.add_data(
                f"alarm_{alarm.rule_id}",
                1,
                alarm.timestamp
            )

        alarm_manager.register_callback(on_alarm)

        # Trigger alarm
        alarm_manager.process_values({"temperature": 90.0})

        # Verify alarm data in reporting
        assert "alarm_high_temp" in reporting_manager.aggregator._cache

    def test_full_pipeline_integration(self):
        """Test full data pipeline: collection -> alarm -> reporting"""
        # Real-time data collection
        data_manager = RealtimeDataManager()
        config = ModbusConfig(name="plc1", host="localhost", port=502, slave_id=1)
        adapter = ModbusAdapter(config)
        data_manager.add_adapter("plc1", adapter)
        data_manager.add_tag("plc1", TagConfig(
            tag_id="pressure",
            name="Pressure",
            address="HR:100"
        ))
        data_manager.connect_all()

        # Alarm management
        alarm_manager = AlarmManager()
        alarm_manager.add_rule(create_high_limit_rule(
            "high_pressure",
            "High Pressure",
            "pressure",
            1000.0,
            AlarmSeverity.ALERT
        ))

        # Reporting
        reporting_manager = ReportingManager()

        # Run pipeline
        for _ in range(5):
            dp = data_manager.read_tag("pressure")
            if dp and dp.value is not None:
                alarm_manager.process_values({dp.tag_id: dp.value})
                reporting_manager.add_data(dp.tag_id, dp.value, dp.timestamp)
            time.sleep(0.01)

        # Verify
        assert "pressure" in reporting_manager.aggregator._cache
        status = data_manager.get_status()
        assert status['adapters']['plc1']['connected'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
