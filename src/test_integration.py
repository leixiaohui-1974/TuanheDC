import time
import unittest
from simulation import AqueductSimulation
from control import AutonomousController

class TestAqueductSystem(unittest.TestCase):
    def setUp(self):
        self.sim = AqueductSimulation()
        self.ctrl = AutonomousController()

    def test_S1_1_HydraulicJump_Control(self):
        print("\nTesting S1.1 Hydraulic Jump Response...")
        # Inject S1.1
        self.sim.inject_scenario("S1.1")

        # Verify initial bad state
        state = self.sim.get_state()
        print(f"Initial Fr: {state['fr']:.2f}")
        self.assertTrue(state['fr'] > 0.9, "Scenario injection failed to create high Fr")

        # Run Control Loop for some steps
        for i in range(20):
            state = self.sim.get_state()
            actions = self.ctrl.decide(state)
            self.sim.step(1.0, actions)

        final_state = self.sim.get_state()
        print(f"Final Fr: {final_state['fr']:.2f}, h: {final_state['h']:.2f}")

        # Expectation: Controller should have reduced Fr, likely by raising water level (h) or reducing flow?
        # My logic in control.py was: Target higher water level (h=7.0).
        self.assertTrue(final_state['h'] > 3.0, "Controller did not raise water level")
        self.assertTrue(final_state['fr'] < state['fr'], "Froude number did not decrease")

    def test_S3_1_ThermalCooling(self):
        print("\nTesting S3.1 Thermal Cooling Response...")
        self.sim.inject_scenario("S3.1") # Sets T_sun=45, T_shade=28

        # Let temp build up
        for i in range(2):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Sun Temp: {state['T_sun']:.2f}, Shade Temp: {state['T_shade']:.2f}")
        print(f"Delta T: {state['T_sun'] - state['T_shade']:.2f}")

        # Check Controller Action
        actions = self.ctrl.decide(state)
        print(f"Actions: {actions}")

        # Expectation: 'status' should be COOLING MODE, Q_in should be high
        self.assertIn("COOLING", actions['status'])
        self.assertTrue(actions['Q_in'] > state['Q_in'], "Controller did not increase flow for cooling")

    def test_Emergency_Seismic_Locked(self):
        print("\nTesting S5.1 + S3.3 Emergency Response...")
        self.sim.inject_scenario("S5.1") # Earthquake
        self.sim.bearing_locked = True
        self.sim.ground_accel = 0.8 # Increased acceleration to ensure threshold crossing

        # Step simulation to let vibration build up
        for i in range(10):
            self.sim.step(1.0, {})

        state = self.sim.get_state()
        print(f"Vibration: {state['vib_amp']:.2f}")

        actions = self.ctrl.decide(state)

        print(f"Status: {actions['status']}")
        self.assertIn("EMERGENCY", actions['status'])
        self.assertEqual(actions['Q_out'], 200.0, "Did not open dump valve")
        self.assertEqual(actions['Q_in'], 0.0, "Did not close inlet")

if __name__ == '__main__':
    unittest.main()
