"""
TAOS V3.10 Phase 2 Modules Comprehensive Tests
Phase 2 模块综合测试

Tests for:
- MPC Controller
- Knowledge Graph
- Edge Computing
"""

import pytest
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import Phase 2 modules
from mpc_controller import (
    MPCType, SolverType, MPCConfig, MPCResult,
    SystemModel, LinearStateSpaceModel, NonlinearModel, AqueductModel,
    MPCController, EconomicMPC, DisturbanceObserver, MPCManager,
    create_aqueduct_mpc, get_mpc_manager
)

from knowledge_graph import (
    EntityType, RelationType, Entity, Relationship, InferenceRule,
    KnowledgeGraph, AqueductKnowledgeBuilder, KnowledgeGraphManager,
    get_knowledge_graph_manager, create_sample_aqueduct_graph
)

from edge_computing import (
    NodeStatus, TaskStatus, TaskPriority, SyncStrategy,
    EdgeNode, EdgeTask, SyncRecord, TaskHandler,
    DataProcessingHandler, ControlHandler, AlarmHandler,
    EdgeNodeExecutor, EdgeCloudSync, EdgeComputingManager,
    get_edge_computing_manager, create_edge_node, create_edge_task
)


# ============================================================
# MPC Controller Tests
# ============================================================

class TestMPCConfig:
    """Tests for MPC configuration"""

    def test_default_config(self):
        """Test default MPC configuration"""
        config = MPCConfig()
        assert config.prediction_horizon == 20
        assert config.control_horizon == 10
        assert config.Q is not None
        assert config.R is not None
        assert config.S is not None

    def test_custom_config(self):
        """Test custom MPC configuration"""
        config = MPCConfig(
            prediction_horizon=30,
            control_horizon=15,
            n_states=6,
            n_inputs=3,
            n_outputs=3
        )
        assert config.prediction_horizon == 30
        assert config.Q.shape == (3, 3)
        assert config.R.shape == (3, 3)


class TestLinearStateSpaceModel:
    """Tests for linear state space model"""

    def test_model_creation(self):
        """Test model creation"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.array([[1, 0]])
        
        model = LinearStateSpaceModel(A, B, C)
        
        assert model.n_states == 2
        assert model.n_inputs == 1
        assert model.n_outputs == 1

    def test_predict(self):
        """Test state prediction"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.array([[1, 0]])
        
        model = LinearStateSpaceModel(A, B, C)
        
        x = np.array([1.0, 0.5])
        u = np.array([0.5])
        
        x_next = model.predict(x, u)
        
        expected = A @ x + B @ u
        np.testing.assert_array_almost_equal(x_next, expected)

    def test_output(self):
        """Test output calculation"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.array([[1, 0], [0, 1]])
        
        model = LinearStateSpaceModel(A, B, C)
        
        x = np.array([1.0, 0.5])
        y = model.output(x)
        
        np.testing.assert_array_almost_equal(y, x)

    def test_linearization(self):
        """Test linearization (identity for linear model)"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.array([[1, 0]])
        
        model = LinearStateSpaceModel(A, B, C)
        
        A_lin, B_lin = model.get_linearization(np.zeros(2), np.zeros(1))
        
        np.testing.assert_array_almost_equal(A_lin, A)
        np.testing.assert_array_almost_equal(B_lin, B)


class TestAqueductModel:
    """Tests for aqueduct system model"""

    def test_model_creation(self):
        """Test aqueduct model creation"""
        model = AqueductModel()
        
        assert model.n_states == 4
        assert model.n_inputs == 2
        assert model.n_outputs == 2

    def test_predict(self):
        """Test state prediction"""
        model = AqueductModel()
        model.sample_time = 1.0
        
        x = np.array([2.0, 1.5, 5.0, 3.0])
        u = np.array([0.5, 0.5])
        
        x_next = model.predict(x, u)
        
        assert len(x_next) == 4
        assert all(x_next >= 0)

    def test_output(self):
        """Test output calculation"""
        model = AqueductModel()
        
        x = np.array([2.0, 1.5, 5.0, 3.0])
        y = model.output(x)
        
        assert len(y) == 2
        assert y[0] == x[0]
        assert y[1] == x[1]

    def test_linearization(self):
        """Test numerical linearization"""
        model = AqueductModel()
        model.sample_time = 1.0
        
        x = np.array([2.0, 1.5, 5.0, 3.0])
        u = np.array([0.5, 0.5])
        
        A, B = model.get_linearization(x, u)
        
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)


class TestMPCController:
    """Tests for MPC controller"""

    def test_controller_creation(self):
        """Test MPC controller creation"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.eye(2)
        
        model = LinearStateSpaceModel(A, B, C)
        config = MPCConfig(
            prediction_horizon=10,
            control_horizon=5,
            n_states=2,
            n_inputs=1,
            n_outputs=2
        )
        
        controller = MPCController(model, config)
        
        assert controller.N_p == 10
        assert controller.N_c == 5

    def test_set_reference(self):
        """Test setting reference trajectory"""
        A = np.eye(2) * 0.9
        B = np.array([[0.1], [0.1]])
        C = np.eye(2)
        
        model = LinearStateSpaceModel(A, B, C)
        config = MPCConfig(n_states=2, n_inputs=1, n_outputs=2)
        controller = MPCController(model, config)
        
        ref = np.array([1.0, 1.0])
        controller.set_reference(ref)
        
        assert controller.reference.shape == (controller.N_p, 2)

    def test_solve(self):
        """Test MPC solve"""
        A = np.array([[0.9, 0.1], [0, 0.8]])
        B = np.array([[0.1], [0.2]])
        C = np.eye(2)
        
        model = LinearStateSpaceModel(A, B, C)
        config = MPCConfig(
            prediction_horizon=10,
            control_horizon=5,
            n_states=2,
            n_inputs=1,
            n_outputs=2,
            u_min=np.array([-1.0]),
            u_max=np.array([1.0])
        )
        
        controller = MPCController(model, config)
        controller.set_reference(np.array([1.0, 0.5]))
        
        x0 = np.array([0.0, 0.0])
        result = controller.solve(x0)
        
        assert isinstance(result, MPCResult)
        assert result.optimal_control.shape[0] == config.control_horizon
        assert result.solve_time > 0

    def test_step(self):
        """Test MPC step execution"""
        A = np.eye(2) * 0.9
        B = np.array([[0.1], [0.1]])
        C = np.eye(2)
        
        model = LinearStateSpaceModel(A, B, C)
        config = MPCConfig(
            n_states=2, n_inputs=1, n_outputs=2,
            prediction_horizon=5, control_horizon=3
        )
        
        controller = MPCController(model, config)
        controller.set_reference(np.array([1.0, 1.0]))
        
        x0 = np.array([0.0, 0.0])
        u = controller.step(x0)
        
        assert len(u) == 1

    def test_statistics(self):
        """Test controller statistics"""
        A = np.eye(2) * 0.9
        B = np.array([[0.1], [0.1]])
        C = np.eye(2)
        
        model = LinearStateSpaceModel(A, B, C)
        config = MPCConfig(n_states=2, n_inputs=1, n_outputs=2)
        
        controller = MPCController(model, config)
        
        stats = controller.get_statistics()
        
        assert 'solve_count' in stats
        assert 'average_solve_time' in stats


class TestDisturbanceObserver:
    """Tests for disturbance observer"""

    def test_observer_creation(self):
        """Test observer creation"""
        model = AqueductModel()
        observer = DisturbanceObserver(model, observer_gain=0.5)
        
        assert observer.gain == 0.5
        assert len(observer.disturbance_estimate) == 4

    def test_update(self):
        """Test observer update"""
        model = AqueductModel()
        model.sample_time = 1.0
        observer = DisturbanceObserver(model)
        
        x = np.array([2.0, 1.5, 5.0, 3.0])
        u = np.array([0.5, 0.5])
        
        observer.update(x, u)
        observer.update(x * 1.1, u)
        
        estimate = observer.get_estimate()
        assert len(estimate) == 4

    def test_reset(self):
        """Test observer reset"""
        model = AqueductModel()
        observer = DisturbanceObserver(model)
        
        observer.disturbance_estimate = np.ones(4)
        observer.reset()
        
        assert np.allclose(observer.disturbance_estimate, np.zeros(4))


class TestMPCManager:
    """Tests for MPC manager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = MPCManager()
        assert len(manager.controllers) == 0

    def test_add_controller(self):
        """Test adding controller"""
        manager = MPCManager()
        
        controller, model = create_aqueduct_mpc()
        manager.add_controller("main", controller)
        
        assert "main" in manager.controllers

    def test_set_reference(self):
        """Test setting reference through manager"""
        manager = MPCManager()
        
        controller, model = create_aqueduct_mpc()
        manager.add_controller("main", controller)
        
        manager.set_reference("main", np.array([2.0, 1.5]))
        
        assert controller.reference[0, 0] == 2.0

    def test_step_through_manager(self):
        """Test MPC step through manager"""
        manager = MPCManager()
        
        controller, model = create_aqueduct_mpc()
        manager.add_controller("main", controller)
        manager.set_reference("main", np.array([2.0, 1.5]))
        
        x0 = np.array([1.5, 1.0, 3.0, 2.0])
        u = manager.step("main", x0)
        
        assert len(u) == 2

    def test_create_aqueduct_mpc(self):
        """Test aqueduct MPC factory function"""
        controller, model = create_aqueduct_mpc(sample_time=60.0)
        
        assert isinstance(controller, MPCController)
        assert isinstance(model, AqueductModel)
        assert model.sample_time == 60.0


# ============================================================
# Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraphBasics:
    """Tests for knowledge graph basics"""

    def test_entity_creation(self):
        """Test entity creation"""
        entity = Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Main Channel",
            properties={'length': 1000}
        )
        
        assert entity.entity_id == "ch_001"
        assert entity.entity_type == EntityType.CHANNEL

    def test_entity_serialization(self):
        """Test entity serialization"""
        entity = Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Main Channel"
        )
        
        data = entity.to_dict()
        
        assert data['entity_id'] == "ch_001"
        assert data['entity_type'] == "channel"

    def test_relationship_creation(self):
        """Test relationship creation"""
        rel = Relationship(
            relationship_id="rel_001",
            source_id="ch_001",
            target_id="ch_002",
            relation_type=RelationType.UPSTREAM_OF
        )
        
        assert rel.source_id == "ch_001"
        assert rel.target_id == "ch_002"


class TestKnowledgeGraph:
    """Tests for knowledge graph operations"""

    def test_graph_creation(self):
        """Test graph creation"""
        graph = KnowledgeGraph()
        
        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0

    def test_add_entity(self):
        """Test adding entity"""
        graph = KnowledgeGraph()
        
        entity = Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Main Channel"
        )
        
        entity_id = graph.add_entity(entity)
        
        assert entity_id == "ch_001"
        assert "ch_001" in graph.entities

    def test_get_entity(self):
        """Test getting entity"""
        graph = KnowledgeGraph()
        
        entity = Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Main Channel"
        )
        graph.add_entity(entity)
        
        retrieved = graph.get_entity("ch_001")
        
        assert retrieved is not None
        assert retrieved.name == "Main Channel"

    def test_update_entity(self):
        """Test updating entity"""
        graph = KnowledgeGraph()
        
        entity = Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Main Channel",
            properties={'length': 1000}
        )
        graph.add_entity(entity)
        
        result = graph.update_entity("ch_001", {'properties': {'width': 10}})
        
        assert result == True
        updated = graph.get_entity("ch_001")
        assert updated.properties['width'] == 10

    def test_add_relationship(self):
        """Test adding relationship"""
        graph = KnowledgeGraph()
        
        e1 = Entity(entity_id="ch_001", entity_type=EntityType.CHANNEL, name="C1")
        e2 = Entity(entity_id="ch_002", entity_type=EntityType.CHANNEL, name="C2")
        graph.add_entity(e1)
        graph.add_entity(e2)
        
        rel = Relationship(
            relationship_id="rel_001",
            source_id="ch_001",
            target_id="ch_002",
            relation_type=RelationType.UPSTREAM_OF
        )
        
        rel_id = graph.add_relationship(rel)
        
        assert rel_id == "rel_001"

    def test_get_neighbors(self):
        """Test getting neighbors"""
        graph = KnowledgeGraph()
        
        e1 = Entity(entity_id="ch_001", entity_type=EntityType.CHANNEL, name="C1")
        e2 = Entity(entity_id="ch_002", entity_type=EntityType.CHANNEL, name="C2")
        graph.add_entity(e1)
        graph.add_entity(e2)
        
        rel = Relationship(
            relationship_id="rel_001",
            source_id="ch_001",
            target_id="ch_002",
            relation_type=RelationType.UPSTREAM_OF
        )
        graph.add_relationship(rel)
        
        neighbors = graph.get_neighbors("ch_001", direction="out")
        
        assert len(neighbors) == 1
        assert neighbors[0].entity_id == "ch_002"

    def test_find_path(self):
        """Test path finding"""
        graph = KnowledgeGraph()
        
        for i in range(4):
            graph.add_entity(Entity(
                entity_id=f"ch_{i}",
                entity_type=EntityType.CHANNEL,
                name=f"Channel {i}"
            ))
        
        for i in range(3):
            graph.add_relationship(Relationship(
                relationship_id=f"rel_{i}",
                source_id=f"ch_{i}",
                target_id=f"ch_{i+1}",
                relation_type=RelationType.CONNECTS_TO
            ))
        
        path = graph.find_path("ch_0", "ch_3")
        
        assert path is not None
        assert path == ["ch_0", "ch_1", "ch_2", "ch_3"]

    def test_query_entities(self):
        """Test entity query"""
        graph = KnowledgeGraph()
        
        for i in range(5):
            graph.add_entity(Entity(
                entity_id=f"ch_{i}",
                entity_type=EntityType.CHANNEL,
                name=f"Channel {i}",
                properties={'section': i % 2}
            ))
        
        results = graph.query(entity_type=EntityType.CHANNEL)
        assert len(results) == 5
        
        results = graph.query(properties={'section': 0})
        assert len(results) == 3

    def test_get_entities_by_type(self):
        """Test getting entities by type"""
        graph = KnowledgeGraph()
        
        graph.add_entity(Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Channel 1"
        ))
        graph.add_entity(Entity(
            entity_id="gate_001",
            entity_type=EntityType.GATE,
            name="Gate 1"
        ))
        
        channels = graph.get_entities_by_type(EntityType.CHANNEL)
        gates = graph.get_entities_by_type(EntityType.GATE)
        
        assert len(channels) == 1
        assert len(gates) == 1

    def test_export_import(self):
        """Test graph export and import"""
        graph1 = KnowledgeGraph()
        
        graph1.add_entity(Entity(
            entity_id="ch_001",
            entity_type=EntityType.CHANNEL,
            name="Channel 1"
        ))
        
        data = graph1.export_to_dict()
        
        graph2 = KnowledgeGraph()
        graph2.import_from_dict(data)
        
        assert len(graph2.entities) == 1
        assert "ch_001" in graph2.entities


class TestAqueductKnowledgeBuilder:
    """Tests for aqueduct knowledge builder"""

    def test_add_channel(self):
        """Test adding channel"""
        graph = KnowledgeGraph()
        builder = AqueductKnowledgeBuilder(graph)
        
        channel_id = builder.add_channel(
            "ch_main", "Main Channel",
            length=1000, width=5, slope=0.0001
        )
        
        assert channel_id == "ch_main"
        channel = graph.get_entity("ch_main")
        assert channel.properties['length'] == 1000

    def test_add_gate(self):
        """Test adding gate"""
        graph = KnowledgeGraph()
        builder = AqueductKnowledgeBuilder(graph)
        
        gate_id = builder.add_gate(
            "gate_1", "Main Gate",
            max_opening=2.0, gate_type="sluice"
        )
        
        assert gate_id == "gate_1"
        gate = graph.get_entity("gate_1")
        assert gate.entity_type == EntityType.GATE

    def test_connect_upstream_downstream(self):
        """Test upstream-downstream connection"""
        graph = KnowledgeGraph()
        builder = AqueductKnowledgeBuilder(graph)
        
        builder.add_channel("ch_1", "C1", 1000, 5, 0.0001)
        builder.add_channel("ch_2", "C2", 1000, 5, 0.0001)
        builder.connect_upstream_downstream("ch_1", "ch_2")
        
        upstream = graph.get_upstream_entities("ch_2")
        assert len(upstream) == 1
        assert upstream[0].entity_id == "ch_1"

    def test_sample_graph_creation(self):
        """Test sample graph creation"""
        graph = create_sample_aqueduct_graph()
        
        stats = graph.get_statistics()
        
        assert stats['total_entities'] > 0
        assert stats['total_relationships'] > 0


class TestKnowledgeGraphManager:
    """Tests for knowledge graph manager"""

    def test_create_graph(self):
        """Test graph creation"""
        manager = KnowledgeGraphManager()
        
        graph = manager.create_graph("aqueduct")
        
        assert graph is not None
        assert "aqueduct" in manager.graphs

    def test_get_graph(self):
        """Test getting graph"""
        manager = KnowledgeGraphManager()
        manager.create_graph("test")
        
        graph = manager.get_graph("test")
        
        assert graph is not None

    def test_delete_graph(self):
        """Test deleting graph"""
        manager = KnowledgeGraphManager()
        manager.create_graph("test")
        
        result = manager.delete_graph("test")
        
        assert result == True
        assert manager.get_graph("test") is None


# ============================================================
# Edge Computing Tests
# ============================================================

class TestEdgeNode:
    """Tests for edge node"""

    def test_node_creation(self):
        """Test node creation"""
        node = EdgeNode(
            node_id="node_001",
            name="Gate Controller 1",
            location="Section A"
        )
        
        assert node.node_id == "node_001"
        assert node.status == NodeStatus.OFFLINE

    def test_node_serialization(self):
        """Test node serialization"""
        node = EdgeNode(
            node_id="node_001",
            name="Gate Controller 1",
            location="Section A",
            status=NodeStatus.ONLINE
        )
        
        data = node.to_dict()
        
        assert data['node_id'] == "node_001"
        assert data['status'] == "online"

    def test_node_availability(self):
        """Test node availability check"""
        node = EdgeNode(
            node_id="node_001",
            name="Node 1",
            location="A",
            status=NodeStatus.ONLINE,
            max_concurrent_tasks=10,
            current_tasks=5
        )
        
        assert node.is_available() == True
        
        node.current_tasks = 10
        assert node.is_available() == False


class TestEdgeTask:
    """Tests for edge task"""

    def test_task_creation(self):
        """Test task creation"""
        task = EdgeTask(
            task_id="task_001",
            task_type="data_aggregate",
            payload={'data': [1, 2, 3]}
        )
        
        assert task.task_id == "task_001"
        assert task.status == TaskStatus.PENDING

    def test_task_serialization(self):
        """Test task serialization"""
        task = EdgeTask(
            task_id="task_001",
            task_type="data_aggregate",
            payload={},
            priority=TaskPriority.HIGH
        )
        
        data = task.to_dict()
        
        assert data['task_id'] == "task_001"
        assert data['priority'] == 3


class TestTaskHandlers:
    """Tests for task handlers"""

    def test_data_processing_handler(self):
        """Test data processing handler"""
        handler = DataProcessingHandler()
        
        assert handler.can_handle("data_aggregate") == True
        assert handler.can_handle("unknown") == False
        
        task = EdgeTask(
            task_id="t1",
            task_type="data_aggregate",
            payload={'data': [10, 20, 30], 'operation': 'mean'}
        )
        
        result = handler.handle(task)
        assert result == 20.0

    def test_data_filter_handler(self):
        """Test data filter operation"""
        handler = DataProcessingHandler()
        
        task = EdgeTask(
            task_id="t1",
            task_type="data_filter",
            payload={'data': [1, 5, 10, 15, 20], 'threshold': 10, 'operator': '>'}
        )
        
        result = handler.handle(task)
        assert result == [15, 20]

    def test_control_handler(self):
        """Test control handler"""
        handler = ControlHandler()
        
        assert handler.can_handle("set_gate") == True
        
        task = EdgeTask(
            task_id="t1",
            task_type="set_gate",
            payload={'gate_id': 'gate_001', 'position': 0.5}
        )
        
        result = handler.handle(task)
        assert result['success'] == True
        assert result['new_position'] == 0.5

    def test_alarm_handler(self):
        """Test alarm handler"""
        handler = AlarmHandler()
        
        task = EdgeTask(
            task_id="t1",
            task_type="check_alarm",
            payload={'value': 15, 'threshold': 10, 'condition': '>'}
        )
        
        result = handler.handle(task)
        assert result['alarm_triggered'] == True


class TestEdgeNodeExecutor:
    """Tests for edge node executor"""

    def test_executor_creation(self):
        """Test executor creation"""
        executor = EdgeNodeExecutor("node_001")
        
        assert executor.node_id == "node_001"
        assert len(executor.handlers) > 0

    def test_submit_and_execute_task(self):
        """Test task submission and execution"""
        executor = EdgeNodeExecutor("node_001")
        executor.start(num_workers=1)
        
        task = EdgeTask(
            task_id="t1",
            task_type="data_aggregate",
            payload={'data': [10, 20, 30], 'operation': 'sum'}
        )
        
        executor.submit_task(task)
        time.sleep(0.5)
        
        assert task.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]
        
        executor.stop()


class TestEdgeCloudSync:
    """Tests for edge-cloud sync"""

    def test_sync_creation(self):
        """Test sync handler creation"""
        sync = EdgeCloudSync("node_001")
        
        assert sync.node_id == "node_001"
        assert sync.strategy == SyncStrategy.BATCH

    def test_add_to_sync(self):
        """Test adding data to sync queue"""
        sync = EdgeCloudSync("node_001")
        
        sync.add_to_sync("sensor_data", {'value': 42})
        
        assert len(sync.sync_queue) == 1

    def test_offline_buffer(self):
        """Test offline buffering"""
        sync = EdgeCloudSync("node_001")
        sync.is_connected = False
        
        sync.add_to_sync("sensor_data", {'value': 42})
        
        assert len(sync.offline_buffer) == 1
        assert len(sync.sync_queue) == 0


class TestEdgeComputingManager:
    """Tests for edge computing manager"""

    def test_manager_creation(self):
        """Test manager creation"""
        manager = EdgeComputingManager()
        
        assert len(manager.nodes) == 0

    def test_register_node(self):
        """Test node registration"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        node_id = manager.register_node(node)
        
        assert node_id in manager.nodes
        assert manager.nodes[node_id].status == NodeStatus.ONLINE

    def test_unregister_node(self):
        """Test node unregistration"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        node_id = manager.register_node(node)
        
        result = manager.unregister_node(node_id)
        
        assert result == True
        assert node_id not in manager.nodes

    def test_submit_task(self):
        """Test task submission"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        manager.register_node(node)
        
        task = create_edge_task(
            "data_aggregate",
            {'data': [1, 2, 3], 'operation': 'mean'}
        )
        
        task_id = manager.submit_task(task)
        
        assert task_id == task.task_id
        assert task.assigned_node is not None

    def test_get_task_status(self):
        """Test getting task status"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        manager.register_node(node)
        
        task = create_edge_task("data_aggregate", {'data': [1, 2, 3]})
        task_id = manager.submit_task(task)
        
        retrieved = manager.get_task_status(task_id)
        
        assert retrieved is not None
        assert retrieved.task_id == task_id

    def test_update_heartbeat(self):
        """Test heartbeat update"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        node_id = manager.register_node(node)
        
        metrics = {'cpu_usage': 50.0, 'memory_usage': 40.0}
        manager.update_heartbeat(node_id, metrics)
        
        updated_node = manager.nodes[node_id]
        assert updated_node.cpu_usage == 50.0
        assert updated_node.memory_usage == 40.0

    def test_sync_data(self):
        """Test data sync"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        node_id = manager.register_node(node)
        
        manager.sync_data(node_id, "sensor", {'value': 42})
        
        sync = manager.sync_handlers[node_id]
        assert len(sync.sync_queue) == 1

    def test_get_statistics(self):
        """Test getting statistics"""
        manager = EdgeComputingManager()
        
        node = create_edge_node("Controller 1", "Section A")
        manager.register_node(node)
        
        stats = manager.get_statistics()
        
        assert stats['total_nodes'] == 1
        assert stats['online_nodes'] == 1


class TestPhase2Integration:
    """Integration tests for Phase 2 modules"""

    def test_mpc_with_knowledge_graph(self):
        """Test MPC integration with knowledge graph"""
        # Create knowledge graph
        graph = create_sample_aqueduct_graph()
        
        # Get channel info from graph
        channels = graph.get_entities_by_type(EntityType.CHANNEL)
        assert len(channels) > 0
        
        # Create MPC for the system
        controller, model = create_aqueduct_mpc()
        controller.set_reference(np.array([2.0, 1.5]))
        
        x0 = np.array([1.5, 1.0, 3.0, 2.0])
        u = controller.step(x0)
        
        assert len(u) == 2

    def test_edge_with_mpc(self):
        """Test edge computing with MPC"""
        manager = EdgeComputingManager()
        
        node = create_edge_node(
            "MPC Controller Node",
            "Main Station",
            capabilities=["control", "mpc"]
        )
        manager.register_node(node)
        
        # Create control task
        task = create_edge_task(
            "set_gate",
            {'gate_id': 'gate_001', 'position': 0.7},
            priority=TaskPriority.HIGH
        )
        
        task_id = manager.submit_task(task)
        time.sleep(0.5)
        
        task = manager.get_task_status(task_id)
        assert task is not None

    def test_knowledge_graph_with_edge(self):
        """Test knowledge graph with edge computing"""
        # Create graph
        graph = create_sample_aqueduct_graph()
        
        # Create edge manager
        manager = EdgeComputingManager()
        
        # Register edge nodes for each gate in graph
        gates = graph.get_entities_by_type(EntityType.GATE)
        
        for gate in gates:
            node = create_edge_node(
                f"Controller for {gate.name}",
                gate.properties.get('location', 'Unknown'),
                capabilities=["control"]
            )
            manager.register_node(node)
        
        assert len(manager.nodes) == len(gates)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
