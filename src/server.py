from flask import Flask, jsonify, request, render_template
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from simulation import AqueductSimulation
from control import AutonomousController, ControlMode, PerceptionSystem
from mpc_controller import HybridController, AdaptiveMPC
from scenario_generator import ScenarioGenerator
from prediction_planning import (
    PredictiveScenarioManager, WeatherForecast, FlowForecast,
    SeismicAlert, MaintenancePlan, OperationSchedule,
    PlanType, AlertLevel, ForecastSource
)

app = Flask(__name__)

# Configuration
HISTORY_MAX_SIZE = 1000  # Maximum number of state history records
SIMULATION_DT = 0.5      # Simulation step size (seconds)
SIMULATION_SLEEP = 0.1   # Sleep between simulation steps

# Global instances - 分层智能体架构
sim = AqueductSimulation()
controller = AutonomousController()
perception = PerceptionSystem()      # 感知层
mpc_controller = AdaptiveMPC()       # MPC控制层
hybrid_controller = HybridController()  # 混合控制器
scenario_gen = ScenarioGenerator()   # 场景生成器
prediction_manager = PredictiveScenarioManager()  # 预测管理器

simulation_running = True
simulation_paused = False
last_control_actions = {}
last_perception_result = {}     # 场景识别结果
last_mpc_state = {}             # MPC控制器状态
last_prediction_state = {}      # 预测状态
sim_lock = threading.Lock()
state_history = deque(maxlen=HISTORY_MAX_SIZE)
start_time = datetime.now()


def simulation_loop():
    """
    Background simulation loop with full agent hierarchy.
    分层智能体架构：感知层 -> 决策层 -> 执行层
    """
    global last_control_actions, last_perception_result, last_mpc_state
    dt = SIMULATION_DT

    while simulation_running:
        if not simulation_paused:
            with sim_lock:
                state = sim.get_state()
                state['time'] = sim.time

                # === 感知层：场景识别 ===
                detected_scenarios, risks = perception.analyze(state)
                risk_level = perception.get_risk_level(risks)

                last_perception_result = {
                    'detected_scenarios': detected_scenarios,
                    'risks': risks,
                    'risk_level': risk_level.value if hasattr(risk_level, 'value') else str(risk_level),
                    'multi_physics_active': 'MULTI_PHYSICS' in detected_scenarios
                }

                # === 决策层：MPC增益自适应 ===
                mpc_controller._update_gains(detected_scenarios)
                mpc_result = mpc_controller.compute(state, detected_scenarios)

                last_mpc_state = {
                    'method': mpc_result.get('method', 'UNKNOWN'),
                    'gains': mpc_result.get('gains', {}),
                    'target_h': mpc_controller.config.h_target,
                    'solve_count': mpc_controller.solve_count,
                    'fallback_count': mpc_controller.fallback_count,
                    'fallback_rate': mpc_controller.fallback_count / max(1, mpc_controller.solve_count)
                }

                # === 执行层：控制决策 ===
                actions = controller.decide(state)
                last_control_actions = actions

                # 合并MPC建议（如果控制器处于AUTO模式）
                if actions.get('status') == 'NORMAL':
                    actions['Q_in'] = mpc_result.get('Q_in', actions.get('Q_in', 80.0))
                    actions['Q_out'] = mpc_result.get('Q_out', actions.get('Q_out', 80.0))

                # Evolve simulation
                sim.step(dt, actions)

                # Record comprehensive history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'sim_time': state['time'],
                    **state,
                    'status': actions.get('status', 'UNKNOWN'),
                    'active_scenarios': detected_scenarios,
                    'risk_level': last_perception_result['risk_level'],
                    'mpc_method': last_mpc_state['method'],
                    'mpc_target_h': last_mpc_state['target_h'],
                    'Q_in_cmd': actions.get('Q_in', 80.0),
                    'Q_out_cmd': actions.get('Q_out', 80.0)
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


@app.route('/api/perception')
def get_perception():
    """获取感知层状态 - 场景识别结果和风险评估"""
    try:
        with sim_lock:
            result = {
                **last_perception_result,
                'thresholds': {
                    'froude_critical': perception.FROUDE_CRITICAL,
                    'froude_warning': perception.FROUDE_WARNING,
                    'thermal_delta_critical': perception.THERMAL_DELTA_CRITICAL,
                    'vibration_critical': perception.VIBRATION_CRITICAL,
                    'joint_gap_max': perception.JOINT_GAP_MAX_CRITICAL,
                    'joint_gap_min': perception.JOINT_GAP_MIN_CRITICAL
                },
                'supported_scenarios': [
                    'S1.1', 'S1.2', 'S2.1', 'S3.1', 'S3.2', 'S3.3',
                    'S4.1', 'S4.2', 'S5.1', 'S5.2', 'S6.1', 'S6.2',
                    'MULTI_PHYSICS'
                ]
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mpc')
def get_mpc_state():
    """获取MPC控制器状态 - 增益调度和优化结果"""
    try:
        with sim_lock:
            # 获取当前MPC配置
            result = {
                **last_mpc_state,
                'config': {
                    'prediction_horizon': mpc_controller.config.prediction_horizon,
                    'control_horizon': mpc_controller.config.control_horizon,
                    'dt': mpc_controller.config.dt,
                    'Q_min': mpc_controller.config.Q_min,
                    'Q_max': mpc_controller.config.Q_max,
                    'dQ_max': mpc_controller.config.dQ_max
                },
                'current_weights': {
                    'w_h': mpc_controller.config.w_h,
                    'w_fr': mpc_controller.config.w_fr,
                    'w_T_delta': mpc_controller.config.w_T_delta,
                    'h_target': mpc_controller.config.h_target
                },
                'scenario_gain_table': {
                    k: {
                        'w_h': v['w_h'],
                        'w_fr': v['w_fr'],
                        'w_T': v['w_T'],
                        'target_h': v.get('target_h', 4.0)
                    }
                    for k, v in mpc_controller.scenario_gains.items()
                },
                'last_control': list(mpc_controller.last_u)
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scenarios')
def get_scenarios():
    """获取所有可用场景及其配置"""
    try:
        scenarios_info = {}
        for scenario_id, profile in scenario_gen.scenarios.items():
            scenarios_info[scenario_id] = {
                'name': profile.name,
                'description': profile.description,
                'severity': profile.severity.name,
                'duration_range': profile.duration_range,
                'parameters': profile.parameters
            }

        result = {
            'available_scenarios': scenarios_info,
            'active_scenarios': scenario_gen.active_scenarios,
            'environment_profile': scenario_gen.environment.profile_type,
            'transition_matrix': scenario_gen.transition_matrix
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent_hierarchy')
def get_agent_hierarchy():
    """获取分层智能体架构状态"""
    try:
        with sim_lock:
            state = sim.get_state()
            is_safe = sim.is_safe_state()

            result = {
                'architecture': {
                    'perception_layer': {
                        'name': 'PerceptionSystem',
                        'function': '场景识别与风险评估',
                        'detected_scenarios': last_perception_result.get('detected_scenarios', []),
                        'risk_level': last_perception_result.get('risk_level', 'INFO')
                    },
                    'decision_layer': {
                        'name': 'AdaptiveMPC + HybridController',
                        'function': '自适应MPC + 场景优先级控制',
                        'mpc_method': last_mpc_state.get('method', 'UNKNOWN'),
                        'current_gains': last_mpc_state.get('gains', {}),
                        'target_h': last_mpc_state.get('target_h', 4.0)
                    },
                    'execution_layer': {
                        'name': 'AutonomousController',
                        'function': '执行控制决策',
                        'mode': str(controller.mode),
                        'status': last_control_actions.get('status', 'UNKNOWN'),
                        'Q_in': last_control_actions.get('Q_in', 80.0),
                        'Q_out': last_control_actions.get('Q_out', 80.0)
                    }
                },
                'system_state': {
                    'is_safe': is_safe,
                    'water_level': state['h'],
                    'froude_number': state['fr'],
                    'thermal_delta': state['T_sun'] - state['T_shade'],
                    'vibration': state['vib_amp']
                },
                'adaptation_info': {
                    'scenario_to_gains_mapping': True,
                    'dynamic_target_update': True,
                    'emergency_override': True,
                    'fallback_rate': last_mpc_state.get('fallback_rate', 0.0)
                }
            }
        return jsonify(result)
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

        # 扩展支持的场景列表
        valid_scenarios = list(scenario_gen.scenarios.keys())
        if scenario_id not in valid_scenarios:
            return jsonify({
                'error': f'Invalid scenario_id. Valid options: {valid_scenarios}'
            }), 400

        with sim_lock:
            scenario_gen.inject_scenario(scenario_id, sim)
            if scenario_id == 'S5.1':
                sim.ground_accel = 0.5

        return jsonify({
            'status': 'ok',
            'scenario': scenario_id,
            'description': scenario_gen.scenarios[scenario_id].description
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction')
def get_prediction():
    """获取预测信息系统状态"""
    try:
        forecast = prediction_manager.get_scenario_forecast(24)
        return jsonify(forecast)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/weather', methods=['POST'])
def update_weather_forecast():
    """更新天气预报"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        forecast = WeatherForecast(
            timestamp=datetime.now(),
            valid_hours=data.get('valid_hours', 24),
            T_max=data.get('T_max', 30.0),
            T_min=data.get('T_min', 20.0),
            T_trend=data.get('T_trend', 'stable'),
            cooling_rate=data.get('cooling_rate', 0.0),
            wind_speed_max=data.get('wind_speed_max', 5.0),
            wind_direction=data.get('wind_direction', 'N'),
            gust_speed=data.get('gust_speed', 10.0),
            precipitation=data.get('precipitation', 0.0),
            precipitation_prob=data.get('precipitation_prob', 0.0),
            storm_warning=data.get('storm_warning', False),
            solar_hours=data.get('solar_hours', 8.0),
            confidence=data.get('confidence', 0.8)
        )

        prediction_manager.update_predictions(weather=forecast)

        # Update perception system with predictions
        with sim_lock:
            perception.update_predictions(
                prediction_manager.prediction.scenario_probabilities,
                prediction_manager.planning.get_plan_impacts()
            )

        return jsonify({
            'status': 'ok',
            'forecast_summary': prediction_manager.prediction.get_forecast_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/flow', methods=['POST'])
def update_flow_forecast():
    """更新流量预报"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        forecast = FlowForecast(
            timestamp=datetime.now(),
            valid_hours=data.get('valid_hours', 24),
            Q_upstream=data.get('Q_upstream', 80.0),
            Q_upstream_max=data.get('Q_upstream_max', 100.0),
            Q_upstream_min=data.get('Q_upstream_min', 60.0),
            Q_downstream_demand=data.get('Q_downstream_demand', 80.0),
            reservoir_release=data.get('reservoir_release', 0.0),
            surge_probability=data.get('surge_probability', 0.0),
            low_flow_risk=data.get('low_flow_risk', False),
            confidence=data.get('confidence', 0.85)
        )

        prediction_manager.update_predictions(flow=forecast)

        with sim_lock:
            perception.update_predictions(
                prediction_manager.prediction.scenario_probabilities,
                prediction_manager.planning.get_plan_impacts()
            )

        return jsonify({
            'status': 'ok',
            'forecast_summary': prediction_manager.prediction.get_forecast_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/seismic', methods=['POST'])
def update_seismic_alert():
    """更新地震预警"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        alert = SeismicAlert(
            timestamp=datetime.now(),
            alert_level=AlertLevel(data.get('alert_level', 1)),
            magnitude=data.get('magnitude', 0.0),
            epicenter_distance=data.get('epicenter_distance', 0.0),
            estimated_arrival=data.get('estimated_arrival', 0.0),
            expected_pga=data.get('expected_pga', 0.0),
            expected_duration=data.get('expected_duration', 0.0),
            aftershock_probability=data.get('aftershock_probability', 0.0),
            recommended_action=data.get('recommended_action', 'MONITOR')
        )

        prediction_manager.update_predictions(seismic=alert)

        with sim_lock:
            perception.update_predictions(
                prediction_manager.prediction.scenario_probabilities,
                prediction_manager.planning.get_plan_impacts()
            )

        return jsonify({
            'status': 'ok',
            'alert': prediction_manager.prediction._alert_to_dict(alert),
            'scenario_probabilities': prediction_manager.prediction.scenario_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/planning/maintenance', methods=['POST'])
def add_maintenance_plan():
    """添加维护计划"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        now = datetime.now()
        plan = MaintenancePlan(
            plan_id=data.get('plan_id', f'M{int(now.timestamp())}'),
            plan_type=PlanType(data.get('plan_type', 'maintenance')),
            start_time=datetime.fromisoformat(data['start_time']) if 'start_time' in data else now,
            end_time=datetime.fromisoformat(data['end_time']) if 'end_time' in data else now + timedelta(hours=2),
            affected_systems=data.get('affected_systems', []),
            affected_sensors=data.get('affected_sensors', []),
            affected_actuators=data.get('affected_actuators', []),
            sensor_availability=data.get('sensor_availability', 1.0),
            actuator_availability=data.get('actuator_availability', 1.0),
            control_capacity=data.get('control_capacity', 1.0),
            description=data.get('description', '')
        )

        prediction_manager.planning.add_maintenance_plan(plan)

        with sim_lock:
            perception.update_predictions(
                prediction_manager.prediction.scenario_probabilities,
                prediction_manager.planning.get_plan_impacts()
            )

        return jsonify({
            'status': 'ok',
            'plan_id': plan.plan_id,
            'plan_impacts': prediction_manager.planning.get_plan_impacts()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/planning')
def get_planning():
    """获取计划信息"""
    try:
        return jsonify({
            'active_plans': prediction_manager.planning.get_active_plans(),
            'upcoming_plans': prediction_manager.planning.get_upcoming_plans(48),
            'plan_impacts': prediction_manager.planning.get_plan_impacts()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations')
def get_recommendations():
    """获取建议措施"""
    try:
        recommendations = prediction_manager.get_recommended_preparations()
        return jsonify({
            'recommendations': recommendations,
            'scenario_probabilities': prediction_manager.prediction.scenario_probabilities,
            'alerts': prediction_manager.get_scenario_forecast(24).get('alerts', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
