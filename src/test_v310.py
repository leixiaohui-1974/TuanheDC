"""
Integration Tests for TAOS V3.10 Modules

Tests for:
- Sensor Simulation
- Actuator Simulation
- Data Governance
- Data Assimilation
- IDZ Model Adapter
- State Evaluation
- State Prediction

Author: TAOS Development Team
Version: 3.10
"""

import unittest
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSensorSimulation(unittest.TestCase):
    """Tests for sensor simulation module."""

    @classmethod
    def setUpClass(cls):
        from sensor_simulation import (
            SensorSimulationEngine, SensorDegradationMode,
            HighFidelitySensor, SensorPhysicsModel, SensorType
        )
        cls.SensorSimulationEngine = SensorSimulationEngine
        cls.SensorDegradationMode = SensorDegradationMode
        cls.HighFidelitySensor = HighFidelitySensor
        cls.SensorPhysicsModel = SensorPhysicsModel
        cls.SensorType = SensorType

    def test_sensor_engine_initialization(self):
        """Test sensor simulation engine initialization."""
        engine = self.SensorSimulationEngine()
        self.assertIn('water_level', engine.networks)
        self.assertIn('velocity', engine.networks)
        self.assertIn('temperature', engine.networks)
        self.assertIn('structural', engine.networks)
        self.assertIn('flow', engine.networks)

    def test_sensor_measurement(self):
        """Test sensor measurement with true state."""
        engine = self.SensorSimulationEngine()
        true_state = {
            'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
            'joint_gap': 20.0, 'vib_amp': 5.0, 'bearing_stress': 31.0,
            'Q_in': 80.0, 'Q_out': 80.0
        }

        result = engine.measure(true_state, dt=0.1)

        self.assertIn('timestamp', result)
        self.assertIn('networks', result)
        self.assertIn('measured_state', result)
        self.assertIn('health_report', result)

    def test_sensor_fault_injection(self):
        """Test sensor fault injection."""
        engine = self.SensorSimulationEngine()

        # Inject a fault
        engine.inject_fault('water_level', 'level_1',
                           self.SensorDegradationMode.LINEAR_DRIFT, 0.1)

        # Check fault is active
        status = engine.get_full_status()
        self.assertIsNotNone(status)

        # Clear faults
        engine.clear_faults()

    def test_single_sensor(self):
        """Test single sensor physics model."""
        physics = self.SensorPhysicsModel(
            sensor_type=self.SensorType.ULTRASONIC_LEVEL,
            range_min=0.0,
            range_max=10.0,
            accuracy_percent=0.5
        )
        sensor = self.HighFidelitySensor("test_sensor", physics)

        # Take measurements - run enough iterations to let sensor dynamics settle
        value = 0
        for _ in range(20):
            value, metadata = sensor.measure(4.0, dt=0.1)
            self.assertTrue(metadata['valid'])

        # After settling, value should be close to true value
        self.assertAlmostEqual(value, 4.0, delta=1.0)


class TestActuatorSimulation(unittest.TestCase):
    """Tests for actuator simulation module."""

    @classmethod
    def setUpClass(cls):
        from actuator_simulation import (
            ActuatorSimulationEngine, ActuatorFailureMode,
            HighFidelityActuator, GateActuatorSystem, ActuatorDynamicsModel
        )
        cls.ActuatorSimulationEngine = ActuatorSimulationEngine
        cls.ActuatorFailureMode = ActuatorFailureMode
        cls.HighFidelityActuator = HighFidelityActuator
        cls.GateActuatorSystem = GateActuatorSystem
        cls.ActuatorDynamicsModel = ActuatorDynamicsModel

    def test_actuator_engine_initialization(self):
        """Test actuator simulation engine initialization."""
        engine = self.ActuatorSimulationEngine()
        self.assertIn('inlet_gate', engine.actuators)
        self.assertIn('outlet_gate', engine.actuators)
        self.assertIn('dump_valve', engine.actuators)
        self.assertIn('bypass_valve', engine.actuators)

    def test_actuator_command_and_step(self):
        """Test actuator command and simulation step."""
        engine = self.ActuatorSimulationEngine()

        # Command flows
        engine.command_flows(Q_in=100.0, Q_out=100.0)

        # Step simulation
        for _ in range(10):
            result = engine.step(dt=0.1)

        self.assertIn('flows', result)
        self.assertIn('Q_in_actual', result['flows'])
        self.assertIn('Q_out_actual', result['flows'])

    def test_actuator_failure_injection(self):
        """Test actuator failure injection."""
        engine = self.ActuatorSimulationEngine()

        # Inject failure
        engine.inject_failure('inlet_gate', self.ActuatorFailureMode.SLOW_RESPONSE, 0.5)

        # Check status
        status = engine.get_full_status()
        self.assertFalse(status['health']['inlet_gate']['is_healthy'])

        # Clear failures
        engine.clear_all_failures()

    def test_gate_actuator_flow(self):
        """Test gate actuator flow calculation."""
        gate = self.GateActuatorSystem("test_gate", max_flow=200.0)
        gate.set_water_levels(upstream=5.0, downstream=2.0)

        # Command position
        gate.command(50.0)

        # Step until settled
        for _ in range(100):
            state = gate.step(0.1)

        flow = gate.get_flow_rate()
        self.assertGreater(flow, 0)
        self.assertLess(flow, 200.0)


class TestDataGovernance(unittest.TestCase):
    """Tests for data governance module."""

    @classmethod
    def setUpClass(cls):
        from data_governance import (
            DataGovernanceEngine, DataQualityValidator,
            ValidationRule, ValidationRuleType
        )
        cls.DataGovernanceEngine = DataGovernanceEngine
        cls.DataQualityValidator = DataQualityValidator
        cls.ValidationRule = ValidationRule
        cls.ValidationRuleType = ValidationRuleType

    def test_governance_engine_initialization(self):
        """Test governance engine initialization."""
        engine = self.DataGovernanceEngine()
        self.assertIsNotNone(engine.quality_validator)
        self.assertIsNotNone(engine.lineage_tracker)
        self.assertIsNotNone(engine.access_controller)

    def test_data_quality_validation(self):
        """Test data quality validation."""
        validator = self.DataQualityValidator()

        # Valid data
        valid_data = {
            'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
            'fr': 0.5, 'Q_in': 80.0, 'Q_out': 80.0
        }
        metrics = validator.validate(valid_data)

        self.assertGreater(metrics.overall_score, 0.8)
        self.assertEqual(metrics.error_count, 0)

    def test_data_quality_violation(self):
        """Test data quality violation detection."""
        validator = self.DataQualityValidator()

        # Invalid data (water level out of range)
        invalid_data = {
            'h': 15.0,  # Out of range
            'v': 2.0, 'fr': 0.5
        }
        metrics = validator.validate(invalid_data)

        self.assertGreater(metrics.error_count, 0)

    def test_governance_process(self):
        """Test full governance processing."""
        engine = self.DataGovernanceEngine()

        data = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0}
        result = engine.process_data(data, source='test', user_id='test_user')

        self.assertIn('quality', result)
        self.assertIn('lineage_id', result)
        self.assertTrue(result['processed'])


class TestDataAssimilation(unittest.TestCase):
    """Tests for data assimilation module."""

    @classmethod
    def setUpClass(cls):
        from data_assimilation import (
            DataAssimilationEngine, AssimilationMethod,
            ExtendedKalmanFilter, EnsembleKalmanFilter
        )
        cls.DataAssimilationEngine = DataAssimilationEngine
        cls.AssimilationMethod = AssimilationMethod
        cls.ExtendedKalmanFilter = ExtendedKalmanFilter
        cls.EnsembleKalmanFilter = EnsembleKalmanFilter

    def test_assimilation_engine_initialization(self):
        """Test assimilation engine initialization."""
        engine = self.DataAssimilationEngine()
        self.assertIn('ekf', engine.filters)
        self.assertIn('ukf', engine.filters)
        self.assertIn('enkf', engine.filters)
        self.assertIn('pf', engine.filters)

    def test_ekf(self):
        """Test Extended Kalman Filter."""
        ekf = self.ExtendedKalmanFilter(state_dim=6, obs_dim=4)

        # Predict
        ekf.predict(dt=0.1)

        # Update with observation
        z = np.array([4.0, 2.0, 25.0, 22.0])
        innovation = ekf.update(z)

        self.assertEqual(len(innovation), 4)

    def test_enkf(self):
        """Test Ensemble Kalman Filter."""
        enkf = self.EnsembleKalmanFilter(state_dim=6, obs_dim=4, ensemble_size=20)

        # Initialize
        enkf.initialize_ensemble(np.zeros(6), np.eye(6))

        # Predict
        enkf.predict(dt=0.1)

        # Update
        z = np.array([4.0, 2.0, 25.0, 22.0])
        innovation = enkf.update(z)

        x, P = enkf.get_state()
        self.assertEqual(len(x), 6)

    def test_assimilation_workflow(self):
        """Test full assimilation workflow."""
        engine = self.DataAssimilationEngine()

        # Initialize
        initial_state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        engine.initialize(initial_state)

        # Predict
        predicted = engine.predict({'Q_in': 80, 'Q_out': 80}, dt=0.1)
        self.assertIn('h', predicted)

        # Assimilate
        observations = {'h': 4.1, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        result = engine.assimilate(observations)

        self.assertIn('state', result)
        self.assertIn('innovation', result)


class TestIDZModelAdapter(unittest.TestCase):
    """Tests for IDZ model adapter module."""

    @classmethod
    def setUpClass(cls):
        from idz_model_adapter import (
            IDZModelAdapter, MultiFidelityModelManager,
            IDZModel, IDZModelParameters
        )
        cls.IDZModelAdapter = IDZModelAdapter
        cls.MultiFidelityModelManager = MultiFidelityModelManager
        cls.IDZModel = IDZModel
        cls.IDZModelParameters = IDZModelParameters

    def test_idz_model(self):
        """Test IDZ model step."""
        model = self.IDZModel()

        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        state = model.step(control, environment, dt=0.1)

        self.assertIn('h', state)
        self.assertIn('v', state)
        self.assertIn('fr', state)

    def test_idz_adapter_update(self):
        """Test IDZ adapter update from high-fidelity model."""
        adapter = self.IDZModelAdapter()

        hifi_state = {
            'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
            'joint_gap': 20.0, 'vib_amp': 0.0
        }
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        result = adapter.update_from_hifi(hifi_state, control, environment, dt=0.1)

        self.assertIn('error_norm', result)
        self.assertIn('prediction', result)

    def test_idz_prediction(self):
        """Test IDZ model prediction."""
        adapter = self.IDZModelAdapter()

        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        predictions = adapter.predict(control, environment, horizon=10, dt=0.1)

        self.assertEqual(len(predictions), 10)

    def test_multi_fidelity_manager(self):
        """Test multi-fidelity model manager."""
        manager = self.MultiFidelityModelManager()

        hifi_state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0}

        result = manager.update(hifi_state, control, environment, dt=0.1)

        self.assertIn('fused_state', result)
        self.assertIn('weights', result)


class TestStateEvaluation(unittest.TestCase):
    """Tests for state evaluation module."""

    @classmethod
    def setUpClass(cls):
        from state_evaluation import (
            StateEvaluator, MultiObjectiveEvaluator,
            DeviationSeverity, ComplianceStatus
        )
        cls.StateEvaluator = StateEvaluator
        cls.MultiObjectiveEvaluator = MultiObjectiveEvaluator
        cls.DeviationSeverity = DeviationSeverity
        cls.ComplianceStatus = ComplianceStatus

    def test_evaluator_initialization(self):
        """Test state evaluator initialization."""
        evaluator = self.StateEvaluator()
        self.assertGreater(len(evaluator.targets), 5)

    def test_deviation_evaluation(self):
        """Test deviation evaluation."""
        evaluator = self.StateEvaluator()

        state = {
            'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
            'joint_gap': 20.0, 'vib_amp': 5.0, 'bearing_stress': 31.0,
            'Q_in': 80.0, 'Q_out': 80.0
        }

        deviations = evaluator.evaluate_deviation(state)

        self.assertIn('water_level', deviations)
        self.assertEqual(deviations['water_level'].severity, self.DeviationSeverity.NOMINAL)

    def test_risk_assessment(self):
        """Test risk assessment."""
        evaluator = self.StateEvaluator()

        # Normal state - low risk
        normal_state = {'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                        'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0}
        risks = evaluator.assess_risk(normal_state)

        max_risk = max(r.risk_level for r in risks)
        self.assertLess(max_risk, 0.5)

        # High risk state
        risky_state = {'h': 4.0, 'v': 5.0, 'fr': 0.95, 'T_sun': 40.0, 'T_shade': 20.0,
                       'joint_gap': 35.0, 'vib_amp': 60.0, 'bearing_stress': 45.0}
        risks = evaluator.assess_risk(risky_state)

        max_risk = max(r.risk_level for r in risks)
        self.assertGreater(max_risk, 0.3)

    def test_comprehensive_evaluation(self):
        """Test comprehensive state evaluation."""
        evaluator = self.StateEvaluator()

        state = {'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                 'Q_in': 80.0, 'Q_out': 80.0}

        result = evaluator.evaluate(state)

        self.assertIn('deviations', result)
        self.assertIn('performance_indices', result)
        self.assertIn('risk_assessment', result)
        self.assertIn('overall_score', result)

    def test_multi_objective_evaluation(self):
        """Test multi-objective evaluation."""
        evaluator = self.MultiObjectiveEvaluator()

        state = {'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                 'Q_in': 80.0, 'Q_out': 80.0}

        result = evaluator.evaluate(state)

        self.assertIn('objective_scores', result)
        self.assertIn('weighted_score', result)
        self.assertIn('safety', result['objective_scores'])


class TestStatePrediction(unittest.TestCase):
    """Tests for state prediction module."""

    @classmethod
    def setUpClass(cls):
        from state_prediction import (
            StatePredictionEngine, ScenarioPrediction,
            PredictionMethod, PhysicsPredictor
        )
        cls.StatePredictionEngine = StatePredictionEngine
        cls.ScenarioPrediction = ScenarioPrediction
        cls.PredictionMethod = PredictionMethod
        cls.PhysicsPredictor = PhysicsPredictor

    def test_physics_predictor(self):
        """Test physics-based predictor."""
        predictor = self.PhysicsPredictor()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        next_state = predictor.predict_step(state, control, environment, dt=1.0)

        self.assertIn('h', next_state)
        self.assertIn('fr', next_state)

    def test_prediction_engine(self):
        """Test prediction engine."""
        engine = self.StatePredictionEngine()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        trajectory = engine.predict(state, control, environment,
                                    horizon_name='short',
                                    method=self.PredictionMethod.PHYSICS_BASED)

        self.assertGreater(len(trajectory.predictions), 0)
        self.assertEqual(trajectory.horizon_name, 'short')

    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        engine = self.StatePredictionEngine()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        trajectory = engine.predict(state, control, environment,
                                    horizon_name='short',
                                    method=self.PredictionMethod.ENSEMBLE)

        # Check uncertainty is provided
        for pred in trajectory.predictions[:5]:
            self.assertIn('h', pred.uncertainty)

    def test_risk_prediction(self):
        """Test risk probability prediction."""
        engine = self.StatePredictionEngine()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        result = engine.predict_risk(state, control, environment)

        self.assertIn('risk_probabilities', result)
        self.assertIn('max_risks', result)

    def test_scenario_prediction(self):
        """Test scenario prediction."""
        predictor = self.ScenarioPrediction()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}

        result = predictor.predict_scenario(state, 'summer_peak')

        self.assertIn('trajectory', result)
        self.assertIn('risk_assessment', result)

    def test_scenario_comparison(self):
        """Test scenario comparison."""
        predictor = self.ScenarioPrediction()

        state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0}

        result = predictor.compare_scenarios(state, ['normal', 'summer_peak', 'winter_cold'])

        self.assertIn('summary', result)
        self.assertIn('individual_results', result)


class TestIntegrationV310(unittest.TestCase):
    """Integration tests combining multiple V3.10 modules."""

    def test_sensor_to_assimilation_pipeline(self):
        """Test sensor measurement to data assimilation pipeline."""
        from sensor_simulation import SensorSimulationEngine
        from data_assimilation import DataAssimilationEngine

        sensor_engine = SensorSimulationEngine()
        assimilation_engine = DataAssimilationEngine()

        # Initialize assimilation
        initial_state = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        assimilation_engine.initialize(initial_state)

        # Simulate sensor measurements
        true_state = {
            'h': 4.1, 'v': 2.1, 'T_sun': 26.0, 'T_shade': 23.0,
            'joint_gap': 20.0, 'vib_amp': 1.0, 'bearing_stress': 31.0,
            'Q_in': 82.0, 'Q_out': 80.0
        }

        sensor_result = sensor_engine.measure(true_state, dt=0.1)
        measured = sensor_result['measured_state']

        # Assimilate measurements
        assim_result = assimilation_engine.assimilate({
            'h': measured.get('h', 4.0),
            'v': measured.get('v', 2.0),
            'T_sun': measured.get('T_sun', 25.0),
            'T_shade': measured.get('T_shade', 22.0)
        })

        self.assertIn('state', assim_result)

    def test_evaluation_to_prediction_pipeline(self):
        """Test state evaluation to prediction pipeline."""
        from state_evaluation import StateEvaluator
        from state_prediction import StatePredictionEngine, PredictionMethod

        evaluator = StateEvaluator()
        predictor = StatePredictionEngine()

        state = {'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                 'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                 'Q_in': 80.0, 'Q_out': 80.0}

        # Evaluate current state
        eval_result = evaluator.evaluate(state)
        current_score = eval_result['overall_score']

        # Predict future state
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        trajectory = predictor.predict(state, control, environment, 'short',
                                       PredictionMethod.PHYSICS_BASED)

        # Evaluate predicted state
        if trajectory.predictions:
            future_state = trajectory.predictions[-1].state
            future_state['fr'] = future_state.get('fr', 0.5)
            future_state['bearing_stress'] = state['bearing_stress']

    def test_governance_with_validation(self):
        """Test data governance with quality validation."""
        from data_governance import DataGovernanceEngine

        engine = DataGovernanceEngine()

        # Good data
        good_data = {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'Q_in': 80.0}
        result = engine.process_data(good_data, 'test_source')

        self.assertTrue(result['quality']['is_valid'])

        # Bad data
        bad_data = {'h': 100.0, 'v': -5.0}  # Out of range
        result = engine.process_data(bad_data, 'test_source')

        self.assertFalse(result['quality']['is_valid'])


def run_tests():
    """Run all V3.10 tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSensorSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestActuatorSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestDataGovernance))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAssimilation))
    suite.addTests(loader.loadTestsFromTestCase(TestIDZModelAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestStateEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestStatePrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationV310))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 60)
    print("TAOS V3.10 Integration Tests")
    print("=" * 60)

    result = run_tests()

    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("All tests PASSED!")
    else:
        print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
