"""
TAOS V3.10 Phase 3 Modules Comprehensive Tests
Phase 3 模块综合测试

Tests for:
- Digital Twin
- AIOps
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import Phase 3 modules
from digital_twin import (
    ObjectType, MaterialType, AnimationType, LayerType,
    Vector3, Quaternion, Transform, Material, Geometry, SceneObject,
    Animation, Camera, Light, Scene, StateSync, WaterSimulation,
    DigitalTwinManager, get_digital_twin_manager, create_sample_aqueduct_scene
)

from aiops import (
    AnomalyType, SeverityLevel, MaintenanceType, HealthStatus,
    Anomaly, DiagnosticResult, MaintenancePrediction, RemediationAction,
    AnomalyDetector, IntelligentDiagnostics, PredictiveMaintenance,
    AutoRemediation, AIOpsManager, get_aiops_manager
)


# ============================================================
# Digital Twin Tests
# ============================================================

class TestVector3:
    """Tests for Vector3"""

    def test_creation(self):
        """Test vector creation"""
        v = Vector3(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_to_dict(self):
        """Test serialization"""
        v = Vector3(1, 2, 3)
        d = v.to_dict()
        assert d == {'x': 1, 'y': 2, 'z': 3}

    def test_addition(self):
        """Test vector addition"""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_length(self):
        """Test vector length"""
        v = Vector3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalize(self):
        """Test vector normalization"""
        v = Vector3(3, 4, 0)
        n = v.normalize()
        assert abs(n.length() - 1.0) < 0.0001


class TestQuaternion:
    """Tests for Quaternion"""

    def test_default(self):
        """Test default quaternion (identity)"""
        q = Quaternion()
        assert q.w == 1.0
        assert q.x == 0.0

    def test_from_euler(self):
        """Test from Euler angles"""
        import math
        q = Quaternion.from_euler(0, math.pi/2, 0)
        assert q is not None
        # Should represent 90 degree rotation around Y


class TestTransform:
    """Tests for Transform"""

    def test_default_transform(self):
        """Test default transform"""
        t = Transform()
        assert t.position.x == 0
        assert t.scale.x == 1

    def test_to_dict(self):
        """Test transform serialization"""
        t = Transform(
            position=Vector3(1, 2, 3),
            scale=Vector3(2, 2, 2)
        )
        d = t.to_dict()
        assert d['position']['x'] == 1
        assert d['scale']['x'] == 2


class TestMaterial:
    """Tests for Material"""

    def test_default_material(self):
        """Test default material"""
        m = Material()
        assert m.material_type == MaterialType.DEFAULT

    def test_custom_material(self):
        """Test custom material"""
        m = Material(
            material_type=MaterialType.WATER,
            color=(0.2, 0.4, 0.8, 0.7),
            metallic=0.0
        )
        assert m.material_type == MaterialType.WATER
        assert m.color[3] == 0.7


class TestSceneObject:
    """Tests for SceneObject"""

    def test_object_creation(self):
        """Test object creation"""
        obj = SceneObject(
            object_id="test_001",
            name="Test Object",
            object_type=ObjectType.GATE
        )
        assert obj.object_id == "test_001"
        assert obj.object_type == ObjectType.GATE

    def test_object_serialization(self):
        """Test object serialization"""
        obj = SceneObject(
            object_id="test_001",
            name="Test Object",
            object_type=ObjectType.CHANNEL,
            visible=True
        )
        d = obj.to_dict()
        assert d['object_id'] == "test_001"
        assert d['object_type'] == "channel"


class TestScene:
    """Tests for Scene"""

    def test_scene_creation(self):
        """Test scene creation"""
        scene = Scene("scene_001", "Test Scene")
        assert scene.scene_id == "scene_001"
        assert len(scene.objects) == 0

    def test_add_object(self):
        """Test adding object to scene"""
        scene = Scene("scene_001", "Test Scene")
        obj = SceneObject(
            object_id="obj_001",
            name="Object 1",
            object_type=ObjectType.SENSOR
        )
        scene.add_object(obj)
        assert "obj_001" in scene.objects

    def test_remove_object(self):
        """Test removing object from scene"""
        scene = Scene("scene_001", "Test Scene")
        obj = SceneObject(
            object_id="obj_001",
            name="Object 1",
            object_type=ObjectType.SENSOR
        )
        scene.add_object(obj)
        result = scene.remove_object("obj_001")
        assert result == True
        assert "obj_001" not in scene.objects

    def test_parent_child_relationship(self):
        """Test parent-child object relationship"""
        scene = Scene("scene_001", "Test Scene")

        parent = SceneObject(
            object_id="parent",
            name="Parent",
            object_type=ObjectType.CHANNEL
        )
        scene.add_object(parent)

        child = SceneObject(
            object_id="child",
            name="Child",
            object_type=ObjectType.WATER,
            parent_id="parent"
        )
        scene.add_object(child)

        assert "child" in scene.objects["parent"].children

    def test_add_camera(self):
        """Test adding camera"""
        scene = Scene("scene_001", "Test Scene")
        camera = Camera(
            camera_id="cam_001",
            name="Main Camera",
            position=Vector3(10, 10, 10)
        )
        scene.add_camera(camera)
        assert "cam_001" in scene.cameras
        assert scene.active_camera_id == "cam_001"

    def test_add_light(self):
        """Test adding light"""
        scene = Scene("scene_001", "Test Scene")
        light = Light(
            light_id="light_001",
            name="Sun",
            light_type="directional"
        )
        scene.add_light(light)
        assert "light_001" in scene.lights

    def test_layer_visibility(self):
        """Test layer visibility control"""
        scene = Scene("scene_001", "Test Scene")

        obj = SceneObject(
            object_id="obj_001",
            name="Equipment",
            object_type=ObjectType.PUMP,
            layer=LayerType.EQUIPMENT
        )
        scene.add_object(obj)

        scene.set_layer_visibility(LayerType.EQUIPMENT, False)
        assert scene.objects["obj_001"].visible == False

    def test_get_objects_by_type(self):
        """Test getting objects by type"""
        scene = Scene("scene_001", "Test Scene")

        for i in range(3):
            scene.add_object(SceneObject(
                object_id=f"gate_{i}",
                name=f"Gate {i}",
                object_type=ObjectType.GATE
            ))

        scene.add_object(SceneObject(
            object_id="sensor_001",
            name="Sensor",
            object_type=ObjectType.SENSOR
        ))

        gates = scene.get_objects_by_type(ObjectType.GATE)
        assert len(gates) == 3


class TestStateSync:
    """Tests for StateSync"""

    def test_sync_creation(self):
        """Test state sync creation"""
        sync = StateSync()
        assert len(sync.state_cache) == 0

    def test_update_state(self):
        """Test state update"""
        sync = StateSync()
        sync.update_state("entity_001", {'value': 42})
        assert sync.get_state("entity_001")['value'] == 42

    def test_bind_property(self):
        """Test property binding"""
        sync = StateSync()
        sync.bind_property("entity_001", "object_001", "position")
        assert "entity_001" in sync.bindings

    def test_callback_notification(self):
        """Test callback notification"""
        sync = StateSync()

        received = []
        def callback(obj_id, state):
            received.append((obj_id, state))

        sync.register_callback(callback)
        sync.bind_property("entity_001", "object_001", "value", None)
        sync.update_state("entity_001", {'value': 100})

        assert len(received) == 1
        assert received[0][0] == "object_001"


class TestWaterSimulation:
    """Tests for WaterSimulation"""

    def test_simulation_creation(self):
        """Test simulation creation"""
        sim = WaterSimulation()
        assert len(sim.flow_rates) == 0

    def test_set_flow_rate(self):
        """Test setting flow rate"""
        sim = WaterSimulation()
        sim.set_flow_rate("channel_001", 5.0)
        assert sim.flow_rates["channel_001"] == 5.0

    def test_set_water_level(self):
        """Test setting water level"""
        sim = WaterSimulation()
        sim.set_water_level("channel_001", 2.5)
        assert sim.water_levels["channel_001"] == 2.5


class TestDigitalTwinManager:
    """Tests for DigitalTwinManager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = DigitalTwinManager()
        assert len(manager.scenes) == 0

    def test_create_scene(self):
        """Test scene creation"""
        manager = DigitalTwinManager()
        scene = manager.create_scene("Test Scene")
        assert scene is not None
        assert scene.name == "Test Scene"

    def test_create_channel_object(self):
        """Test channel object creation"""
        manager = DigitalTwinManager()
        scene = manager.create_scene("Test")

        obj = manager.create_channel_object(
            scene, "ch_001", "Main Channel",
            Vector3(0, 0, 0), Vector3(100, 0, 0),
            width=5, depth=3
        )

        assert obj.object_id == "channel_ch_001"
        assert "water_ch_001" in scene.objects  # Water surface added

    def test_create_gate_object(self):
        """Test gate object creation"""
        manager = DigitalTwinManager()
        scene = manager.create_scene("Test")

        obj = manager.create_gate_object(
            scene, "gate_001", "Main Gate",
            Vector3(50, 0, 0), width=5, height=3
        )

        assert obj.object_id == "gate_gate_001"
        assert obj.interactive == True

    def test_create_sensor_object(self):
        """Test sensor object creation"""
        manager = DigitalTwinManager()
        scene = manager.create_scene("Test")

        obj = manager.create_sensor_object(
            scene, "sensor_001", "Water Level",
            Vector3(25, 2, 0), "water_level"
        )

        assert obj.object_id == "sensor_sensor_001"

    def test_sample_scene_creation(self):
        """Test sample scene creation"""
        scene = create_sample_aqueduct_scene()

        assert scene is not None
        assert len(scene.objects) > 0
        assert len(scene.cameras) > 0
        assert len(scene.lights) > 0

    def test_export_scene(self):
        """Test scene export"""
        manager = DigitalTwinManager()
        scene = manager.create_scene("Export Test")
        scene.add_object(SceneObject(
            object_id="obj_001",
            name="Test",
            object_type=ObjectType.CHANNEL
        ))

        data = manager.export_scene(scene.scene_id)

        assert data is not None
        assert 'objects' in data
        assert 'obj_001' in data['objects']


# ============================================================
# AIOps Tests
# ============================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector"""

    def test_detector_creation(self):
        """Test detector creation"""
        detector = AnomalyDetector(window_size=100)
        assert detector.window_size == 100

    def test_add_data_point(self):
        """Test adding data points"""
        detector = AnomalyDetector()

        for i in range(20):
            detector.add_data_point("entity_001", "temperature", 25 + i * 0.1)

        key = "entity_001:temperature"
        assert key in detector.data_buffers
        assert len(detector.data_buffers[key]) == 20

    def test_baseline_update(self):
        """Test baseline statistics update"""
        detector = AnomalyDetector()

        for i in range(20):
            detector.add_data_point("entity_001", "temperature", 25 + i * 0.1)

        key = "entity_001:temperature"
        assert key in detector.baselines
        assert 'mean' in detector.baselines[key]
        assert 'std' in detector.baselines[key]

    def test_detect_spike(self):
        """Test spike detection"""
        detector = AnomalyDetector()

        # Build baseline
        for i in range(50):
            detector.add_data_point("entity_001", "value", 100 + (i % 5))

        # Add anomalous point
        anomaly = detector.detect("entity_001", "value", 200)

        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.SPIKE

    def test_detect_drop(self):
        """Test drop detection"""
        detector = AnomalyDetector()

        for i in range(50):
            detector.add_data_point("entity_001", "value", 100 + (i % 5))

        anomaly = detector.detect("entity_001", "value", 10)

        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.DROP

    def test_no_anomaly_normal(self):
        """Test no HIGH severity anomaly for normal values within expected range"""
        detector = AnomalyDetector()

        # Add data points with some natural variation
        for i in range(50):
            detector.add_data_point("entity_001", "value", 100.0)

        # Test with a value exactly at mean - should not trigger any anomaly
        anomaly = detector.detect("entity_001", "value", 100.0)

        # Constant data with constant test value should not trigger any anomaly
        assert anomaly is None

    def test_threshold_setting(self):
        """Test threshold setting"""
        detector = AnomalyDetector()
        detector.set_thresholds("entity_001", "value", sigma_factor=2.0,
                                 absolute_min=50, absolute_max=150)

        key = "entity_001:value"
        assert detector.thresholds[key]['sigma_factor'] == 2.0
        assert detector.thresholds[key]['absolute_min'] == 50


class TestIntelligentDiagnostics:
    """Tests for IntelligentDiagnostics"""

    def test_diagnostics_creation(self):
        """Test diagnostics creation"""
        diag = IntelligentDiagnostics()
        assert len(diag.diagnostic_rules) == 0

    def test_add_diagnostic_rule(self):
        """Test adding diagnostic rule"""
        diag = IntelligentDiagnostics()

        rule = {
            'rule_id': 'high_temp',
            'description': 'High temperature detected',
            'conditions': [
                {'type': 'metric_threshold', 'metric': 'temperature', 'operator': '>', 'threshold': 80}
            ],
            'recommendations': ['Check cooling system'],
            'root_cause': 'Cooling failure'
        }
        diag.add_diagnostic_rule(rule)

        assert len(diag.diagnostic_rules) == 1

    def test_diagnose_healthy(self):
        """Test diagnosis of healthy entity"""
        diag = IntelligentDiagnostics()

        metrics = {'temperature': 25, 'pressure': 100}
        result = diag.diagnose("entity_001", metrics, [])

        assert result.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def test_diagnose_with_anomalies(self):
        """Test diagnosis with anomalies"""
        diag = IntelligentDiagnostics()

        anomaly = Anomaly(
            anomaly_id="a1",
            entity_id="entity_001",
            metric_name="temperature",
            anomaly_type=AnomalyType.SPIKE,
            severity=SeverityLevel.HIGH,
            detected_at=datetime.now(),
            value=100,
            expected_value=25,
            deviation_score=5.0
        )

        metrics = {'temperature': 100}
        result = diag.diagnose("entity_001", metrics, [anomaly])

        assert len(result.issues) > 0
        assert result.health_status != HealthStatus.HEALTHY


class TestPredictiveMaintenance:
    """Tests for PredictiveMaintenance"""

    def test_maintenance_creation(self):
        """Test maintenance engine creation"""
        maint = PredictiveMaintenance()
        assert len(maint.equipment_data) == 0

    def test_register_equipment(self):
        """Test equipment registration"""
        maint = PredictiveMaintenance()
        maint.register_equipment(
            "pump_001",
            components=["motor", "impeller", "seal"],
            expected_lifetime=10000
        )

        assert "pump_001" in maint.equipment_data
        assert len(maint.equipment_data["pump_001"]['components']) == 3

    def test_update_operating_hours(self):
        """Test operating hours update"""
        maint = PredictiveMaintenance()
        maint.register_equipment("pump_001", ["motor"], 10000)
        maint.update_operating_hours("pump_001", 5000)

        assert maint.equipment_data["pump_001"]['operating_hours'] == 5000

    def test_predict_maintenance(self):
        """Test maintenance prediction"""
        maint = PredictiveMaintenance()
        maint.register_equipment("pump_001", ["motor", "seal"], 10000)
        maint.update_operating_hours("pump_001", 8000)

        predictions = maint.predict_maintenance("pump_001")

        assert len(predictions) == 2  # One per component
        for pred in predictions:
            assert pred.entity_id == "pump_001"
            assert pred.remaining_useful_life is not None


class TestAutoRemediation:
    """Tests for AutoRemediation"""

    def test_remediation_creation(self):
        """Test remediation engine creation"""
        remed = AutoRemediation()
        assert len(remed.remediation_rules) == 0

    def test_add_remediation_rule(self):
        """Test adding remediation rule"""
        remed = AutoRemediation()
        remed.add_remediation_rule(
            "rule_001",
            AnomalyType.SPIKE,
            "temperature",
            "reduce_load",
            {'reduction_percent': 20}
        )

        assert "rule_001" in remed.remediation_rules

    def test_get_remediation_actions(self):
        """Test getting remediation actions"""
        remed = AutoRemediation()
        remed.add_remediation_rule(
            "rule_001",
            AnomalyType.SPIKE,
            "temperature",
            "reduce_load",
            {'reduction_percent': 20}
        )

        anomaly = Anomaly(
            anomaly_id="a1",
            entity_id="entity_001",
            metric_name="temperature",
            anomaly_type=AnomalyType.SPIKE,
            severity=SeverityLevel.HIGH,
            detected_at=datetime.now(),
            value=100,
            expected_value=25,
            deviation_score=5.0
        )

        actions = remed.get_remediation_actions(anomaly)

        assert len(actions) == 1
        assert actions[0].action_type == "reduce_load"

    def test_execute_action(self):
        """Test action execution"""
        remed = AutoRemediation()

        executed = []
        def handler(params):
            executed.append(params)
            return True

        remed.register_action_handler("test_action", handler)

        action = RemediationAction(
            action_id="act_001",
            anomaly_id="a1",
            action_type="test_action",
            parameters={'value': 42}
        )

        result = remed.execute_action(action)

        assert result == True
        assert len(executed) == 1
        assert executed[0]['value'] == 42


class TestAIOpsManager:
    """Tests for AIOpsManager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = AIOpsManager()
        assert manager.anomaly_detector is not None
        assert manager.diagnostics is not None

    def test_process_metric(self):
        """Test metric processing"""
        manager = AIOpsManager()

        # Build baseline
        for i in range(50):
            manager.process_metric("entity_001", "value", 100 + (i % 5))

        # Normal value - no anomaly
        anomaly = manager.process_metric("entity_001", "value", 102)
        assert anomaly is None

    def test_process_metric_anomaly(self):
        """Test anomaly detection through manager"""
        manager = AIOpsManager()

        # Build baseline
        for i in range(50):
            manager.process_metric("entity_001", "value", 100 + (i % 5))

        # Anomalous value
        anomaly = manager.process_metric("entity_001", "value", 500)
        assert anomaly is not None

    def test_diagnose_entity(self):
        """Test entity diagnosis"""
        manager = AIOpsManager()

        metrics = {'temperature': 25, 'pressure': 100}
        result = manager.diagnose_entity("entity_001", metrics)

        assert result is not None
        assert result.entity_id == "entity_001"

    def test_predict_maintenance(self):
        """Test maintenance prediction through manager"""
        manager = AIOpsManager()
        manager.register_equipment("pump_001", ["motor"], 10000)

        predictions = manager.predict_maintenance("pump_001")

        assert len(predictions) > 0

    def test_resolve_anomaly(self):
        """Test anomaly resolution"""
        manager = AIOpsManager()

        # Build baseline and create anomaly
        for i in range(50):
            manager.process_metric("entity_001", "value", 100)

        anomaly = manager.process_metric("entity_001", "value", 500)
        assert anomaly is not None

        manager.resolve_anomaly(anomaly.anomaly_id)

        assert manager.active_anomalies[anomaly.anomaly_id].resolved == True

    def test_get_active_anomalies(self):
        """Test getting active anomalies"""
        manager = AIOpsManager()

        # Build baseline
        for i in range(50):
            manager.process_metric("entity_001", "value", 100)

        manager.process_metric("entity_001", "value", 500)

        active = manager.get_active_anomalies()
        assert len(active) > 0

    def test_statistics(self):
        """Test getting statistics"""
        manager = AIOpsManager()
        manager.register_equipment("pump_001", ["motor"], 10000)

        stats = manager.get_statistics()

        assert 'active_anomalies' in stats
        assert 'monitored_equipment' in stats
        assert stats['monitored_equipment'] == 1


class TestPhase3Integration:
    """Integration tests for Phase 3 modules"""

    def test_digital_twin_with_aiops(self):
        """Test digital twin integration with AIOps"""
        # Create digital twin
        dt_manager = DigitalTwinManager()
        scene = dt_manager.create_scene("Integration Test")

        dt_manager.create_channel_object(
            scene, "ch_001", "Main Channel",
            Vector3(0, 0, 0), Vector3(100, 0, 0),
            width=5, depth=3
        )

        # Create AIOps manager
        aiops = AIOpsManager()

        # Register callback to update twin
        def on_anomaly(anomaly):
            # Update digital twin state based on anomaly
            dt_manager.update_entity_state(
                anomaly.entity_id,
                {'alarm': True, 'severity': anomaly.severity.value}
            )

        aiops.register_anomaly_callback(on_anomaly)

        # Build baseline and trigger anomaly
        for i in range(50):
            aiops.process_metric("ch_001", "water_level", 2.5)

        anomaly = aiops.process_metric("ch_001", "water_level", 10.0)

        assert anomaly is not None

    def test_aiops_full_pipeline(self):
        """Test full AIOps pipeline"""
        manager = AIOpsManager()

        # Register equipment
        manager.register_equipment("gate_001", ["motor", "actuator"], 20000)

        # Add diagnostic rule
        manager.add_diagnostic_rule({
            'rule_id': 'high_current',
            'description': 'Motor overcurrent',
            'conditions': [
                {'type': 'metric_threshold', 'metric': 'current', 'operator': '>', 'threshold': 10}
            ],
            'recommendations': ['Check motor load'],
            'root_cause': 'Mechanical binding'
        })

        # Add remediation rule
        manager.add_remediation_rule(
            'reduce_speed',
            AnomalyType.SPIKE,
            'current',
            'adjust_speed',
            {'speed': 0.5},
            auto_execute=False
        )

        # Simulate operation
        for i in range(50):
            manager.process_metric("gate_001", "current", 5 + i * 0.05)
            manager.maintenance.update_operating_hours("gate_001", i * 10)

        # Check predictions
        predictions = manager.predict_maintenance("gate_001")
        assert len(predictions) > 0

        # Get statistics
        stats = manager.get_statistics()
        assert stats['monitored_equipment'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
