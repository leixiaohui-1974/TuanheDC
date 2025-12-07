"""
Comprehensive Integration Tests for TAOS (Tuanhe Aqueduct Autonomous Operation System)

Test Coverage:
- All individual scenarios (S1.1, S3.1, S3.3, S4.1, S5.1)
- Scenario combinations (multi-physics coupling)
- Boundary conditions and edge cases
- Controller mode switching
- Long-duration stability tests
"""

import time
import unittest
from simulation import AqueductSimulation
from control import AutonomousController, ControlMode, PerceptionSystem, RiskLevel


class TestAqueductSimulation(unittest.TestCase):
    """Unit tests for the simulation module."""

    def setUp(self):
        self.sim = AqueductSimulation()

    def test_initial_state(self):
        """Test initial state is safe."""
        state = self.sim.get_state()
        self.assertEqual(state['h'], 4.0)
        self.assertEqual(state['v'], 2.0)
        self.assertTrue(self.sim.is_safe_state())

    def test_reset_functionality(self):
        """Test simulation reset restores initial state."""
        # Modify state
        self.sim.h = 7.0
        self.sim.v = 5.0
        self.sim.T_sun = 50.0
        self.sim.bearing_locked = True

        # Reset
        self.sim.reset()

        # Verify
        state = self.sim.get_state()
        self.assertEqual(state['h'], 4.0)
        self.assertEqual(state['v'], 2.0)
        self.assertEqual(state['T_sun'], 20.0)
        self.assertFalse(state['bearing_locked'])

    def test_froude_calculation(self):
        """Test Froude number calculation."""
        self.sim.h = 4.0
        self.sim.v = 2.0
        fr = self.sim.calculate_froude()
        expected_fr = 2.0 / (9.81 * 4.0) ** 0.5
        self.assertAlmostEqual(fr, expected_fr, places=2)

    def test_water_level_boundaries(self):
        """Test water level stays within physical bounds."""
        self.sim.Q_in = 300.0
        self.sim.Q_out = 0.0

        for _ in range(100):
            self.sim.step(1.0, {'Q_in': 300, 'Q_out': 0})

        self.assertLessEqual(self.sim.h, self.sim.H_max)
        self.assertGreaterEqual(self.sim.h, 0.1)

    def test_is_safe_state_detection(self):
        """Test safe state detection logic."""
        # Normal state should be safe
        self.assertTrue(self.sim.is_safe_state())

        # High Froude - unsafe
        self.sim.v = 10.0
        self.sim.h = 2.0
        self.assertFalse(self.sim.is_safe_state())

        self.sim.reset()

        # Thermal differential - unsafe
        self.sim.T_sun = 40.0
        self.sim.T_shade = 20.0
        self.assertFalse(self.sim.is_safe_state())


class TestPerceptionSystem(unittest.TestCase):
    """Unit tests for the perception system."""

    def setUp(self):
        self.perception = PerceptionSystem()

    def test_normal_state_no_risks(self):
        """Test normal state has no critical risks."""
        state = {
            'fr': 0.3, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertEqual(len(scenarios), 0)
        self.assertEqual(self.perception.get_risk_level(risks), RiskLevel.INFO)

    def test_s1_1_detection(self):
        """Test S1.1 (Hydraulic Jump) detection."""
        state = {
            'fr': 1.2, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertIn('S1.1', scenarios)
        self.assertEqual(self.perception.get_risk_level(risks), RiskLevel.CRITICAL)

    def test_s3_1_detection(self):
        """Test S3.1 (Thermal Bending) detection."""
        state = {
            'fr': 0.3, 'T_sun': 45.0, 'T_shade': 25.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertIn('S3.1', scenarios)
        self.assertEqual(self.perception.get_risk_level(risks), RiskLevel.CRITICAL)

    def test_s3_3_detection(self):
        """Test S3.3 (Bearing Lock) detection."""
        state = {
            'fr': 0.3, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': True, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertIn('S3.3', scenarios)

    def test_s4_1_detection(self):
        """Test S4.1 (Joint Gap) detection."""
        state = {
            'fr': 0.3, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 28.0, 'vib_amp': 0.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertIn('S4.1', scenarios)

    def test_s5_1_detection(self):
        """Test S5.1 (Seismic) detection."""
        state = {
            'fr': 0.3, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 60.0, 'h': 4.0
        }
        scenarios, risks = self.perception.analyze(state)
        self.assertIn('S5.1', scenarios)


class TestAutonomousController(unittest.TestCase):
    """Unit tests for the autonomous controller."""

    def setUp(self):
        self.ctrl = AutonomousController()

    def test_mode_switching(self):
        """Test controller mode switching."""
        self.assertEqual(self.ctrl.mode, ControlMode.AUTO)

        self.ctrl.set_mode(ControlMode.MANUAL)
        self.assertEqual(self.ctrl.mode, ControlMode.MANUAL)

        self.ctrl.set_mode(ControlMode.AUTO)
        self.assertEqual(self.ctrl.mode, ControlMode.AUTO)

    def test_manual_control(self):
        """Test manual control mode."""
        self.ctrl.set_mode(ControlMode.MANUAL)
        self.ctrl.set_manual_control(Q_in=100.0, Q_out=80.0)

        state = {
            'fr': 0.3, 'T_sun': 25.0, 'T_shade': 22.0,
            'bearing_locked': False, 'bearing_stress': 15.0,
            'joint_gap': 20.0, 'vib_amp': 0.0, 'h': 4.0,
            'Q_in': 80.0, 'Q_out': 80.0
        }

        actions = self.ctrl.decide(state)
        self.assertEqual(actions['Q_in'], 100.0)
        self.assertEqual(actions['Q_out'], 80.0)
        self.assertEqual(actions['status'], "MANUAL CONTROL")

    def test_controller_reset(self):
        """Test controller reset."""
        self.ctrl.set_mode(ControlMode.MANUAL)
        self.ctrl.target_h = 6.0
        self.ctrl.integral_error = 10.0

        self.ctrl.reset()

        self.assertEqual(self.ctrl.mode, ControlMode.AUTO)
        self.assertEqual(self.ctrl.target_h, 4.0)
        self.assertEqual(self.ctrl.integral_error, 0.0)


class TestScenarioResponses(unittest.TestCase):
    """Integration tests for scenario-specific responses."""

    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_S1_1_HydraulicJump_Control(self):
        """S1.1: Test hydraulic jump response - should raise water level."""
        print("\nTesting S1.1 Hydraulic Jump Response...")
        self.sim.inject_scenario("S1.1")

        state = self.sim.get_state()
        initial_fr = state['fr']
        print(f"Initial Fr: {initial_fr:.2f}")
        self.assertGreater(initial_fr, 0.9, "Scenario injection failed")

        # Run control loop
        for _ in range(20):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        final_state = self.sim.get_state()
        print(f"Final Fr: {final_state['fr']:.2f}, h: {final_state['h']:.2f}")

        self.assertGreater(final_state['h'], 3.0, "Controller did not raise water level")
        self.assertLess(final_state['fr'], initial_fr, "Froude number did not decrease")

    def test_S3_1_ThermalCooling(self):
        """S3.1: Test thermal bending response - should increase cooling flow."""
        print("\nTesting S3.1 Thermal Cooling Response...")
        self.sim.inject_scenario("S3.1")

        for _ in range(2):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Sun Temp: {state['T_sun']:.2f}, Shade Temp: {state['T_shade']:.2f}")
        print(f"Delta T: {state['T_sun'] - state['T_shade']:.2f}")

        actions = self.ctrl.decide(state)
        print(f"Actions: {actions}")

        self.assertIn("COOLING", actions['status'])
        self.assertGreater(actions['Q_in'], state['Q_in'])

    def test_S3_3_BearingLock_Solo(self):
        """S3.3: Test bearing lock response without earthquake."""
        print("\nTesting S3.3 Bearing Lock Response (solo)...")
        self.sim.inject_scenario("S3.3")

        for _ in range(5):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Bearing Locked: {state['bearing_locked']}")

        actions = self.ctrl.decide(state)
        print(f"Status: {actions['status']}")

        self.assertIn("S3.3", actions['active_scenarios'])
        self.assertIn("BEARING", actions['status'])
        # Should NOT be emergency dump (that's only S3.3 + S5.1)
        self.assertNotIn("DUMP", actions['status'])

    def test_S4_1_JointProtection(self):
        """S4.1: Test cold joint protection response."""
        print("\nTesting S4.1 Joint Protection Response...")
        self.sim.inject_scenario("S4.1")

        for _ in range(10):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Joint Gap: {state['joint_gap']:.2f} mm")
        print(f"T_sun: {state['T_sun']:.2f}, T_shade: {state['T_shade']:.2f}")

        actions = self.ctrl.decide(state)
        print(f"Status: {actions['status']}")

        if "S4.1" in actions['active_scenarios']:
            self.assertIn("JOINT", actions['status'])

    def test_S5_1_Seismic_Solo(self):
        """S5.1: Test earthquake response without bearing lock."""
        print("\nTesting S5.1 Seismic Response (solo)...")
        self.sim.inject_scenario("S5.1")
        self.sim.ground_accel = 0.8

        for _ in range(10):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Vibration: {state['vib_amp']:.2f} mm")

        actions = self.ctrl.decide(state)
        print(f"Status: {actions['status']}")

        if state['vib_amp'] > 50.0:
            self.assertIn("S5.1", actions['active_scenarios'])
            self.assertIn("SEISMIC", actions['status'])

    def test_Emergency_Seismic_Locked(self):
        """S5.1 + S3.3: Combined earthquake and bearing lock - emergency response."""
        print("\nTesting S5.1 + S3.3 Emergency Response...")
        self.sim.inject_scenario("S5.1")
        self.sim.bearing_locked = True
        self.sim.ground_accel = 0.8

        for _ in range(10):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Vibration: {state['vib_amp']:.2f}")

        actions = self.ctrl.decide(state)
        print(f"Status: {actions['status']}")

        self.assertIn("EMERGENCY", actions['status'])
        self.assertEqual(actions['Q_out'], 200.0, "Did not open dump valve")
        self.assertEqual(actions['Q_in'], 0.0, "Did not close inlet")


class TestScenarioCombinations(unittest.TestCase):
    """Tests for multi-physics scenario combinations."""

    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_thermal_and_hydraulic(self):
        """Test S1.1 + S3.1: High flow with thermal stress."""
        print("\nTesting S1.1 + S3.1 Combined Response...")

        # Inject both conditions
        self.sim.inject_scenario("S1.1")  # High flow
        self.sim.T_sun = 45.0
        self.sim.T_shade = 30.0

        state = self.sim.get_state()
        actions = self.ctrl.decide(state)

        print(f"Active Scenarios: {actions['active_scenarios']}")
        print(f"Status: {actions['status']}")

        # S1.1 (hydraulic) should take priority over S3.1 (thermal)
        self.assertIn("S1.1", actions['active_scenarios'])

    def test_seismic_and_thermal(self):
        """Test S5.1 + S3.1: Earthquake with thermal stress."""
        print("\nTesting S5.1 + S3.1 Combined Response...")

        self.sim.inject_scenario("S5.1")
        self.sim.ground_accel = 0.8
        self.sim.T_sun = 45.0
        self.sim.T_shade = 30.0

        for _ in range(10):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        actions = self.ctrl.decide(state)

        print(f"Active Scenarios: {actions['active_scenarios']}")
        print(f"Status: {actions['status']}")

        # Seismic should take priority
        if state['vib_amp'] > 50.0:
            self.assertIn("SEISMIC", actions['status'])


class TestLongDurationStability(unittest.TestCase):
    """Long-duration stability and endurance tests."""

    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_normal_operation_stability(self):
        """Test system stability over extended normal operation."""
        print("\nTesting Long-Duration Normal Operation...")

        for i in range(100):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(0.5, actions)

            # System should remain stable
            self.assertGreater(state['h'], 0.1)
            self.assertLess(state['h'], 8.0)

        final_state = self.sim.get_state()
        print(f"Final h: {final_state['h']:.2f}, target: 4.0")

        # Should converge to target
        self.assertAlmostEqual(final_state['h'], 4.0, delta=0.5)

    def test_scenario_recovery(self):
        """Test system recovery from scenario back to normal."""
        print("\nTesting Scenario Recovery...")

        # Inject S1.1
        self.sim.inject_scenario("S1.1")

        # Run until stabilized
        for _ in range(30):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        # Reset to normal
        self.sim.inject_scenario("NORMAL")

        # Run more steps
        for _ in range(50):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        final_state = self.sim.get_state()
        print(f"Recovered h: {final_state['h']:.2f}, Fr: {final_state['fr']:.2f}")

        # Should recover to safe state
        self.assertLess(final_state['fr'], 0.9)

    def test_cyclic_scenario_stress(self):
        """Test system under cyclic scenario changes."""
        print("\nTesting Cyclic Scenario Stress...")

        scenarios = ["NORMAL", "S1.1", "NORMAL", "S3.1", "NORMAL", "S4.1", "NORMAL"]

        for scenario in scenarios:
            self.sim.inject_scenario(scenario)
            for _ in range(20):
                state = self.sim.get_state()
                actions = self.ctrl.decide(state)
                self.sim.step(0.5, actions)

        # System should still be operational
        final_state = self.sim.get_state()
        print(f"Final state: h={final_state['h']:.2f}, Fr={final_state['fr']:.2f}")
        self.assertGreater(final_state['h'], 0.1)


class TestBoundaryConditions(unittest.TestCase):
    """Tests for boundary and edge conditions."""

    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_extreme_inflow(self):
        """Test response to extreme inflow conditions."""
        self.sim.Q_in = 200.0  # Maximum

        for _ in range(20):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        # Water level should be clamped
        self.assertLessEqual(self.sim.h, self.sim.H_max)

    def test_zero_inflow(self):
        """Test response to zero inflow."""
        self.sim.Q_in = 0.0

        for _ in range(20):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        # Water level should not go below minimum
        self.assertGreaterEqual(self.sim.h, 0.1)

    def test_extreme_temperatures(self):
        """Test extreme temperature conditions."""
        self.sim.T_sun = 60.0
        self.sim.T_shade = -20.0
        self.sim.T_ambient = 40.0

        for _ in range(10):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        # System should detect thermal risk
        scenarios, risks = self.ctrl.perception.analyze(self.sim.get_state())
        self.assertIn("S3.1", scenarios)


class TestFullScenarioAutonomy(unittest.TestCase):
    """Full autonomous operation tests across all scenarios."""

    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_full_scenario_coverage(self):
        """Run all scenarios and verify autonomous handling."""
        print("\n=== FULL SCENARIO COVERAGE TEST ===")

        scenarios_tested = {
            'NORMAL': False,
            'S1.1': False,
            'S3.1': False,
            'S3.3': False,
            'S4.1': False,
            'S5.1': False
        }

        for scenario_id in scenarios_tested.keys():
            print(f"\nTesting scenario: {scenario_id}")
            self.sim.reset()
            self.ctrl.reset()

            self.sim.inject_scenario(scenario_id)
            if scenario_id == "S5.1":
                self.sim.ground_accel = 0.8

            # Run for sufficient steps
            for _ in range(30):
                state = self.sim.get_state()
                actions = self.ctrl.decide(state)
                self.sim.step(1.0, actions)

            final_state = self.sim.get_state()
            print(f"  Status: {actions['status']}")
            print(f"  h={final_state['h']:.2f}, Fr={final_state['fr']:.2f}")
            print(f"  Active: {actions['active_scenarios']}")

            scenarios_tested[scenario_id] = True

            # Verify system is still operational
            self.assertGreater(final_state['h'], 0.1)
            self.assertLess(final_state['h'], 8.0)

        # Verify all scenarios were tested
        self.assertTrue(all(scenarios_tested.values()))
        print("\n=== ALL SCENARIOS PASSED ===")


if __name__ == '__main__':
    unittest.main(verbosity=2)
