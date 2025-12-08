"""
V3.10 REST API Endpoint Tests for TAOS

Tests all V3.10 API endpoints including:
- Sensor simulation endpoints
- Actuator simulation endpoints
- Data governance endpoints
- Data assimilation endpoints
- IDZ model adapter endpoints
- State evaluation endpoints
- State prediction endpoints

Author: TAOS Development Team
Version: 3.10
"""

import unittest
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestV310APIEndpoints(unittest.TestCase):
    """Test V3.10 REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from server import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    # =========================================================================
    # Sensor Simulation API Tests
    # =========================================================================

    def test_sensor_measure(self):
        """Test sensor measurement endpoint."""
        data = {
            'true_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            },
            'dt': 0.1
        }
        response = self.client.post('/api/v310/sensor/measure',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('measured_state', result)

    def test_sensor_status(self):
        """Test sensor status endpoint."""
        response = self.client.get('/api/v310/sensor/status')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('overall_health', result)

    def test_sensor_inject_fault(self):
        """Test sensor fault injection endpoint."""
        data = {
            'network': 'water_level',
            'sensor_id': 'level_1',
            'degradation_mode': 'linear_drift',
            'factor': 0.01
        }
        response = self.client.post('/api/v310/sensor/inject_fault',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_sensor_clear_faults(self):
        """Test sensor clear faults endpoint."""
        response = self.client.post('/api/v310/sensor/clear_faults')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # Actuator Simulation API Tests
    # =========================================================================

    def test_actuator_command(self):
        """Test actuator command endpoint."""
        data = {'Q_in': 100.0, 'Q_out': 100.0}
        response = self.client.post('/api/v310/actuator/command',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_actuator_step(self):
        """Test actuator step endpoint."""
        data = {'dt': 0.1}
        response = self.client.post('/api/v310/actuator/step',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('flows', result)

    def test_actuator_status(self):
        """Test actuator status endpoint."""
        response = self.client.get('/api/v310/actuator/status')
        self.assertEqual(response.status_code, 200)

    def test_actuator_inject_failure(self):
        """Test actuator failure injection endpoint."""
        data = {
            'actuator_id': 'inlet_gate',
            'failure_mode': 'slow_response',
            'severity': 0.3
        }
        response = self.client.post('/api/v310/actuator/inject_failure',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_actuator_emergency_shutdown(self):
        """Test emergency shutdown endpoint."""
        response = self.client.post('/api/v310/actuator/emergency_shutdown')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # Data Governance API Tests
    # =========================================================================

    def test_governance_process(self):
        """Test data governance process endpoint."""
        data = {
            'data': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0},
            'source': 'test_sensor',
            'user_id': 'test_user'
        }
        response = self.client.post('/api/v310/governance/process',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('processed', result)

    def test_governance_dashboard(self):
        """Test governance dashboard endpoint."""
        response = self.client.get('/api/v310/governance/dashboard')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('quality_metrics', result)

    def test_governance_compliance_report(self):
        """Test compliance report endpoint."""
        response = self.client.get('/api/v310/governance/compliance_report')
        self.assertEqual(response.status_code, 200)

    def test_governance_quality_report(self):
        """Test quality report endpoint."""
        response = self.client.get('/api/v310/governance/quality_report')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # Data Assimilation API Tests
    # =========================================================================

    def test_assimilation_initialize(self):
        """Test assimilation initialize endpoint."""
        data = {
            'initial_state': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0},
            'initial_uncertainty': {'h': 0.1, 'v': 0.05}
        }
        response = self.client.post('/api/v310/assimilation/initialize',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_assimilation_predict(self):
        """Test assimilation predict endpoint."""
        data = {
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'dt': 0.1
        }
        response = self.client.post('/api/v310/assimilation/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_assimilation_assimilate(self):
        """Test assimilation assimilate endpoint."""
        data = {
            'observations': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        }
        response = self.client.post('/api/v310/assimilation/assimilate',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('state', result)

    def test_assimilation_state(self):
        """Test get assimilation state endpoint."""
        response = self.client.get('/api/v310/assimilation/state')
        self.assertEqual(response.status_code, 200)

    def test_assimilation_status(self):
        """Test assimilation status endpoint."""
        response = self.client.get('/api/v310/assimilation/status')
        self.assertEqual(response.status_code, 200)

    def test_assimilation_switch_method(self):
        """Test switch assimilation method endpoint."""
        data = {'method': 'ensemble_kalman'}
        response = self.client.post('/api/v310/assimilation/switch_method',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # IDZ Model Adapter API Tests
    # =========================================================================

    def test_idz_update(self):
        """Test IDZ model update endpoint."""
        data = {
            'hifi_state': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                          'joint_gap': 20.0, 'vib_amp': 0.0},
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0},
            'dt': 0.1
        }
        response = self.client.post('/api/v310/idz/update',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_idz_parameters(self):
        """Test IDZ get parameters endpoint."""
        response = self.client.get('/api/v310/idz/parameters')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('hydraulic_resistance', result)

    def test_idz_predict(self):
        """Test IDZ predict endpoint."""
        data = {
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0},
            'horizon': 10,
            'dt': 0.1
        }
        response = self.client.post('/api/v310/idz/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_idz_metrics(self):
        """Test IDZ metrics endpoint."""
        response = self.client.get('/api/v310/idz/metrics')
        self.assertEqual(response.status_code, 200)

    def test_idz_uncertainty(self):
        """Test IDZ uncertainty endpoint."""
        response = self.client.get('/api/v310/idz/uncertainty')
        self.assertEqual(response.status_code, 200)

    def test_multifidelity_update(self):
        """Test multi-fidelity update endpoint."""
        data = {
            'hifi_state': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                          'joint_gap': 20.0, 'vib_amp': 0.0},
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0},
            'dt': 0.1
        }
        response = self.client.post('/api/v310/multifidelity/update',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_multifidelity_status(self):
        """Test multi-fidelity status endpoint."""
        response = self.client.get('/api/v310/multifidelity/status')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # State Evaluation API Tests
    # =========================================================================

    def test_evaluation_evaluate(self):
        """Test comprehensive evaluation endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            },
            'control': {'Q_in': 80.0, 'Q_out': 80.0}
        }
        response = self.client.post('/api/v310/evaluation/evaluate',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('overall_score', result)

    def test_evaluation_deviation(self):
        """Test deviation evaluation endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            }
        }
        response = self.client.post('/api/v310/evaluation/deviation',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_performance(self):
        """Test performance indices endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            }
        }
        response = self.client.post('/api/v310/evaluation/performance',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_risk(self):
        """Test risk assessment endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0
            }
        }
        response = self.client.post('/api/v310/evaluation/risk',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_compliance(self):
        """Test compliance check endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            }
        }
        response = self.client.post('/api/v310/evaluation/compliance',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_trend(self):
        """Test evaluation trend endpoint."""
        response = self.client.get('/api/v310/evaluation/trend')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_status(self):
        """Test evaluation status endpoint."""
        response = self.client.get('/api/v310/evaluation/status')
        self.assertEqual(response.status_code, 200)

    def test_evaluation_multiobjective(self):
        """Test multi-objective evaluation endpoint."""
        data = {
            'state': {
                'h': 4.0, 'v': 2.0, 'fr': 0.5, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            },
            'control': {'Q_in': 80.0, 'Q_out': 80.0}
        }
        response = self.client.post('/api/v310/evaluation/multiobjective',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    # =========================================================================
    # State Prediction API Tests
    # =========================================================================

    def test_prediction_predict(self):
        """Test state prediction endpoint."""
        data = {
            'current_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0
            },
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0},
            'horizon': 'short',
            'method': 'physics_based'
        }
        response = self.client.post('/api/v310/prediction/predict',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('predictions', result)

    def test_prediction_risk(self):
        """Test risk prediction endpoint."""
        data = {
            'current_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0
            },
            'control': {'Q_in': 80.0, 'Q_out': 80.0},
            'environment': {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}
        }
        response = self.client.post('/api/v310/prediction/risk',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_prediction_scenario(self):
        """Test scenario prediction endpoint."""
        data = {
            'current_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0
            },
            'scenario': 'summer_peak'
        }
        response = self.client.post('/api/v310/prediction/scenario',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_prediction_scenarios_compare(self):
        """Test scenario comparison endpoint."""
        data = {
            'current_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0
            },
            'scenarios': ['normal', 'summer_peak', 'storm']
        }
        response = self.client.post('/api/v310/prediction/scenarios/compare',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_prediction_accuracy(self):
        """Test prediction accuracy endpoint."""
        response = self.client.get('/api/v310/prediction/accuracy')
        self.assertEqual(response.status_code, 200)

    def test_prediction_status(self):
        """Test prediction status endpoint."""
        response = self.client.get('/api/v310/prediction/status')
        self.assertEqual(response.status_code, 200)

    def test_prediction_update_history(self):
        """Test update prediction history endpoint."""
        data = {
            'state': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                     'joint_gap': 20.0, 'vib_amp': 0.0}
        }
        response = self.client.post('/api/v310/prediction/update_history',
                                     data=json.dumps(data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)


class TestV310APIIntegration(unittest.TestCase):
    """Integration tests for V3.10 API workflow."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client."""
        from server import app
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_sensor_to_assimilation_workflow(self):
        """Test sensor measurement to assimilation workflow."""
        # Initialize assimilation
        init_data = {
            'initial_state': {'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0}
        }
        response = self.client.post('/api/v310/assimilation/initialize',
                                     data=json.dumps(init_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Measure with sensors
        measure_data = {
            'true_state': {
                'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
                'joint_gap': 20.0, 'vib_amp': 0.0, 'bearing_stress': 31.0,
                'Q_in': 80.0, 'Q_out': 80.0
            },
            'dt': 0.1
        }
        response = self.client.post('/api/v310/sensor/measure',
                                     data=json.dumps(measure_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        measured = json.loads(response.data)

        # Assimilate measurements
        assim_data = {
            'observations': measured.get('measured_state', {})
        }
        response = self.client.post('/api/v310/assimilation/assimilate',
                                     data=json.dumps(assim_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

    def test_prediction_to_evaluation_workflow(self):
        """Test prediction to evaluation workflow."""
        state = {
            'h': 4.0, 'v': 2.0, 'T_sun': 25.0, 'T_shade': 22.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'fr': 0.5,
            'bearing_stress': 31.0, 'Q_in': 80.0, 'Q_out': 80.0
        }
        control = {'Q_in': 80.0, 'Q_out': 80.0}
        environment = {'T_ambient': 25.0, 'solar_rad': 0.5, 'wind_speed': 2.0}

        # Predict
        pred_data = {
            'current_state': state,
            'control': control,
            'environment': environment,
            'horizon': 'short',
            'method': 'physics_based'
        }
        response = self.client.post('/api/v310/prediction/predict',
                                     data=json.dumps(pred_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Evaluate current state
        eval_data = {'state': state, 'control': control}
        response = self.client.post('/api/v310/evaluation/evaluate',
                                     data=json.dumps(eval_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('overall_score', result)

    def test_full_simulation_cycle(self):
        """Test full simulation cycle."""
        # Command actuators
        cmd_data = {'Q_in': 100.0, 'Q_out': 100.0}
        response = self.client.post('/api/v310/actuator/command',
                                     data=json.dumps(cmd_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Step actuators
        step_data = {'dt': 0.1}
        response = self.client.post('/api/v310/actuator/step',
                                     data=json.dumps(step_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)
        actuator_result = json.loads(response.data)

        # Process through governance
        gov_data = {
            'data': actuator_result.get('flows', {}),
            'source': 'actuator_simulation'
        }
        response = self.client.post('/api/v310/governance/process',
                                     data=json.dumps(gov_data),
                                     content_type='application/json')
        self.assertEqual(response.status_code, 200)


def run_api_tests():
    """Run all API tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestV310APIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestV310APIIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("TAOS V3.10 REST API Endpoint Tests")
    print("=" * 70)
    print()

    result = run_api_tests()

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
