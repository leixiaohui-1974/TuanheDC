from flask import Flask, jsonify, request, render_template
import threading
import time
from simulation import AqueductSimulation
from control import AutonomousController

app = Flask(__name__)

# Global instances
sim = AqueductSimulation()
controller = AutonomousController()
simulation_running = True
last_control_actions = {}
sim_lock = threading.Lock()

def simulation_loop():
    global last_control_actions
    dt = 0.5 # Simulation step size (seconds)
    while simulation_running:
        with sim_lock:
            state = sim.get_state()

            # Determine control actions
            actions = controller.decide(state)
            last_control_actions = actions

            # Evolve simulation
            sim.step(dt, actions)

        time.sleep(0.1) # Run faster than real-time (10x speed roughly if dt=0.5 and sleep=0.1)

# Start background thread
bg_thread = threading.Thread(target=simulation_loop)
bg_thread.daemon = True
bg_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    with sim_lock:
        state = sim.get_state()
        # Merge control status
        state.update(last_control_actions)
    return jsonify(state)

@app.route('/api/scenario', methods=['POST'])
def set_scenario():
    data = request.json
    scenario_id = data.get('scenario_id')
    with sim_lock:
        sim.inject_scenario(scenario_id)
    return jsonify({"status": "ok", "scenario": scenario_id})

@app.route('/api/control', methods=['POST'])
def set_control():
    # Manual override or set target (not fully impl in controller yet, but hooks are here)
    data = request.json
    if 'target_h' in data:
        controller.target_h = float(data['target_h'])
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
