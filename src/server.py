from flask import Flask, jsonify, request, render_template
import threading
import time
from collections import deque
from datetime import datetime
from simulation import AqueductSimulation
from control import AutonomousController, ControlMode

app = Flask(__name__)

# Configuration
HISTORY_MAX_SIZE = 1000  # Maximum number of state history records
SIMULATION_DT = 0.5      # Simulation step size (seconds)
SIMULATION_SLEEP = 0.1   # Sleep between simulation steps

# Global instances
sim = AqueductSimulation()
controller = AutonomousController()
simulation_running = True
simulation_paused = False
last_control_actions = {}
sim_lock = threading.Lock()
state_history = deque(maxlen=HISTORY_MAX_SIZE)
start_time = datetime.now()


def simulation_loop():
    """Background simulation loop."""
    global last_control_actions
    dt = SIMULATION_DT

    while simulation_running:
        if not simulation_paused:
            with sim_lock:
                state = sim.get_state()

                # Determine control actions
                actions = controller.decide(state)
                last_control_actions = actions

                # Evolve simulation
                sim.step(dt, actions)

                # Record history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'sim_time': state['time'],
                    **state,
                    'status': actions.get('status', 'UNKNOWN'),
                    'active_scenarios': actions.get('active_scenarios', [])
                }
                state_history.append(history_entry)

        time.sleep(SIMULATION_SLEEP)


# Start background thread
bg_thread = threading.Thread(target=simulation_loop)
bg_thread.daemon = True
bg_thread.start()


@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    uptime = (datetime.now() - start_time).total_seconds()
    with sim_lock:
        state = sim.get_state()
        is_safe = sim.is_safe_state()

    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'simulation_running': simulation_running,
        'simulation_paused': simulation_paused,
        'system_safe': is_safe,
        'sim_time': state['time'],
        'history_size': len(state_history)
    })


@app.route('/api/state')
def get_state():
    """Get current simulation state with control actions."""
    try:
        with sim_lock:
            state = sim.get_state()
            # Merge control status
            state.update(last_control_actions)
        return jsonify(state)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get state history with optional limit."""
    try:
        limit = request.args.get('limit', default=100, type=int)
        limit = min(limit, HISTORY_MAX_SIZE)

        with sim_lock:
            history = list(state_history)[-limit:]

        return jsonify({
            'count': len(history),
            'total': len(state_history),
            'data': history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scenario', methods=['POST'])
def set_scenario():
    """Inject a scenario into the simulation."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        scenario_id = data.get('scenario_id')
        if not scenario_id:
            return jsonify({'error': 'scenario_id is required'}), 400

        valid_scenarios = ['NORMAL', 'S1.1', 'S3.1', 'S3.3', 'S4.1', 'S5.1']
        if scenario_id not in valid_scenarios:
            return jsonify({
                'error': f'Invalid scenario_id. Valid options: {valid_scenarios}'
            }), 400

        with sim_lock:
            sim.inject_scenario(scenario_id)

        return jsonify({'status': 'ok', 'scenario': scenario_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/control', methods=['POST'])
def set_control():
    """Set control parameters or mode."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        with sim_lock:
            if 'target_h' in data:
                controller.target_h = float(data['target_h'])

            if 'mode' in data:
                mode_str = data['mode'].upper()
                if mode_str == 'AUTO':
                    controller.set_mode(ControlMode.AUTO)
                elif mode_str == 'MANUAL':
                    controller.set_mode(ControlMode.MANUAL)
                elif mode_str == 'EMERGENCY':
                    controller.set_mode(ControlMode.EMERGENCY)
                else:
                    return jsonify({'error': 'Invalid mode'}), 400

            if 'Q_in' in data or 'Q_out' in data:
                controller.set_manual_control(
                    Q_in=data.get('Q_in'),
                    Q_out=data.get('Q_out')
                )

        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulation/pause', methods=['POST'])
def pause_simulation():
    """Pause the simulation."""
    global simulation_paused
    simulation_paused = True
    return jsonify({'status': 'ok', 'paused': True})


@app.route('/api/simulation/resume', methods=['POST'])
def resume_simulation():
    """Resume the simulation."""
    global simulation_paused
    simulation_paused = False
    return jsonify({'status': 'ok', 'paused': False})


@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation and controller to initial state."""
    try:
        with sim_lock:
            sim.reset()
            controller.reset()
            state_history.clear()

        return jsonify({'status': 'ok', 'message': 'Simulation reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get statistical summary of the simulation."""
    try:
        with sim_lock:
            history = list(state_history)

        if not history:
            return jsonify({'error': 'No history data available'}), 404

        # Calculate statistics
        h_values = [h['h'] for h in history]
        fr_values = [h['fr'] for h in history]
        T_sun_values = [h['T_sun'] for h in history]
        T_shade_values = [h['T_shade'] for h in history]

        stats = {
            'sample_count': len(history),
            'water_level': {
                'min': min(h_values),
                'max': max(h_values),
                'avg': sum(h_values) / len(h_values)
            },
            'froude_number': {
                'min': min(fr_values),
                'max': max(fr_values),
                'avg': sum(fr_values) / len(fr_values)
            },
            'temperature': {
                'T_sun_avg': sum(T_sun_values) / len(T_sun_values),
                'T_shade_avg': sum(T_shade_values) / len(T_shade_values),
                'delta_T_avg': sum(T_sun_values) / len(T_sun_values) - sum(T_shade_values) / len(T_shade_values)
            }
        }

        # Count scenarios
        scenario_counts = {}
        for h in history:
            for s in h.get('active_scenarios', []):
                scenario_counts[s] = scenario_counts.get(s, 0) + 1

        stats['scenario_occurrences'] = scenario_counts

        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
