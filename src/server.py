from flask import Flask, jsonify, request, render_template, Response, make_response
import json
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
# 新增模块导入
from data_persistence import get_persistence, DataAnalytics, AggregationType, ExportFormat
from intelligence import get_intelligence
from visualization import get_dashboard, DashboardLayout, SVGGenerator
from safety import get_safety_manager, SafetyLevel
from scada_interface import get_integration
# V3.5 新增模块
from config_manager import get_config
from logging_system import get_logger, log_info, log_warning, log_error, audit
from api_docs import get_generator, get_swagger_ui_html, get_redoc_html
# V3.8 新增模块 - 分布式与智能化
from cluster import get_cluster, ClusterManager, NodeRole, NodeStatus, ServiceType
from edge_computing import get_edge_manager, EdgeDeviceManager, EdgeDeviceType, EdgeDeviceStatus
from digital_twin import get_twin_manager, DigitalTwinManager, ModelResolution
from ml_control import get_ml_manager, MLControlManager, ModelType
# V3.9 新增模块 - 移动端与可视化增强
from mobile_api import get_mobile_manager, MobileDeviceManager, NotificationType, DevicePlatform
from advanced_visualization import get_viz_manager, AdvancedVisualizationManager, ChartType
from i18n import get_i18n_manager, I18nManager, t as translate, SupportedLocale
from blockchain_audit import get_audit_manager, BlockchainAuditManager, AuditEventType, AuditSeverity

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

# 新增功能模块
persistence = get_persistence()       # 数据持久化
intelligence = get_intelligence()     # 智能化模块
dashboard = get_dashboard()           # 可视化仪表盘
safety_manager = get_safety_manager() # 安全管理
integration = get_integration()       # 工程集成
# V3.5 新增模块
config = get_config()                 # 配置管理
logger = get_logger()                 # 日志系统
api_generator = get_generator()       # API文档生成
# V3.8 新增模块
cluster_manager = get_cluster()       # 集群管理
edge_manager = get_edge_manager()     # 边缘计算
twin_manager = get_twin_manager()     # 数字孪生
ml_manager = get_ml_manager()         # ML控制
# V3.9 新增模块
mobile_manager = get_mobile_manager()   # 移动端API
viz_manager = get_viz_manager()         # 高级可视化
i18n_manager = get_i18n_manager()       # 国际化
audit_manager = get_audit_manager()     # 区块链审计

# 状态记录
last_intelligence_result = {}         # 智能分析结果
last_safety_result = {}               # 安全评估结果

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
    集成：数据持久化、智能分析、安全监控、工程接口
    """
    global last_control_actions, last_perception_result, last_mpc_state
    global last_intelligence_result, last_safety_result
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

                # === 新增：数据持久化 ===
                persistence.record_state(history_entry)

                # === 新增：智能分析 ===
                last_intelligence_result = intelligence.process_state(state)

                # === 新增：安全监控 ===
                last_safety_result = safety_manager.process_state(state)

                # === 新增：可视化更新 ===
                dashboard.update_history(state)

                # === 新增：工程接口更新 ===
                integration.update_all(state)

                # === V3.8: 数字孪生同步 ===
                twin_manager.update_from_real_time(state)

                # === V3.8: ML控制处理 ===
                ml_manager.process_state(state)

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


# ============================================================
# 新增API端点 - 数据持久化
# ============================================================

@app.route('/api/persistence/stats')
def get_persistence_stats():
    """获取数据库统计信息"""
    try:
        return jsonify(persistence.get_database_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persistence/history')
def get_persistence_history():
    """获取历史数据"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        aggregation = request.args.get('aggregation', default='raw', type=str)

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        agg_type = AggregationType(aggregation) if aggregation != 'raw' else AggregationType.RAW
        data = persistence.query_state_history(start_time, end_time, agg_type, limit=1000)

        return jsonify({
            'count': len(data),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'aggregation': aggregation,
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persistence/statistics')
def get_persistence_statistics():
    """获取统计数据"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        stats = persistence.get_statistics(start_time, end_time)

        return jsonify({
            'period_hours': hours,
            'total_samples': stats.total_samples,
            'water_level': {
                'mean': stats.h_mean,
                'std': stats.h_std,
                'min': stats.h_min,
                'max': stats.h_max
            },
            'froude': {
                'mean': stats.fr_mean,
                'max': stats.fr_max
            },
            'thermal': {
                'delta_mean': stats.T_delta_mean,
                'delta_max': stats.T_delta_max
            },
            'vibration': {
                'mean': stats.vib_mean,
                'max': stats.vib_max
            },
            'scenario_counts': stats.scenario_counts,
            'risk_distribution': stats.risk_level_distribution,
            'safe_operation_ratio': stats.safe_operation_ratio
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persistence/alerts')
def get_persistence_alerts():
    """获取告警历史"""
    try:
        limit = request.args.get('limit', default=50, type=int)
        unack = request.args.get('unacknowledged', default='false', type=str).lower() == 'true'
        return jsonify(persistence.get_recent_alerts(limit, unack))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 新增API端点 - 智能分析
# ============================================================

@app.route('/api/intelligence')
def get_intelligence_status():
    """获取智能分析状态"""
    try:
        with sim_lock:
            return jsonify({
                'last_result': last_intelligence_result,
                'summary': intelligence.get_summary()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/intelligence/predictions')
def get_intelligence_predictions():
    """获取预测结果"""
    try:
        variables = request.args.getlist('var') or ['h', 'fr', 'Q_in', 'Q_out']
        horizon = request.args.get('horizon', default=5, type=int)

        predictions = {}
        for var in variables:
            pred = intelligence.predictor.predict(var, horizon)
            if pred:
                predictions[var] = {
                    'value': pred.value,
                    'confidence': pred.confidence,
                    'lower_bound': pred.lower_bound,
                    'upper_bound': pred.upper_bound
                }

        return jsonify({
            'horizon_minutes': horizon,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/intelligence/anomalies')
def get_intelligence_anomalies():
    """获取异常检测结果"""
    try:
        with sim_lock:
            anomalies = last_intelligence_result.get('anomalies', [])
        return jsonify({
            'anomalies': anomalies,
            'detector_summary': intelligence.anomaly_detector.get_anomaly_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/intelligence/patterns')
def get_intelligence_patterns():
    """获取模式识别结果"""
    try:
        with sim_lock:
            state = sim.get_state()
        patterns = intelligence.pattern_recognizer.recognize(state)
        probabilities = intelligence.pattern_recognizer.get_pattern_probabilities(state)

        return jsonify({
            'matched_patterns': [p.pattern_id for p in patterns],
            'pattern_probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/intelligence/rl')
def get_intelligence_rl():
    """获取强化学习状态"""
    try:
        return jsonify({
            'policy_stats': intelligence.rl_agent.get_policy_stats(),
            'last_action': last_intelligence_result.get('rl_action')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 新增API端点 - 可视化
# ============================================================

@app.route('/api/dashboard/gauges')
def get_dashboard_gauges():
    """获取仪表盘数据"""
    try:
        with sim_lock:
            state = sim.get_state()

        gauges = {}
        for var in ['h', 'fr', 'Q_in', 'Q_out', 'T_sun', 'T_shade', 'vib_amp', 'joint_gap']:
            if var in state:
                gauges[var] = dashboard.get_gauge_data(var, state[var])

        # 添加温差
        T_delta = abs(state.get('T_sun', 25) - state.get('T_shade', 25))
        gauges['T_delta'] = dashboard.get_gauge_data('T_delta', T_delta)

        return jsonify(gauges)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/charts')
def get_dashboard_charts():
    """获取图表数据"""
    try:
        variables = request.args.getlist('var') or ['h', 'Q_in', 'Q_out']
        minutes = request.args.get('minutes', default=10, type=int)
        return jsonify(dashboard.get_chart_data(variables, minutes))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/topology')
def get_dashboard_topology():
    """获取系统拓扑"""
    try:
        with sim_lock:
            state = sim.get_state()
        return jsonify(dashboard.get_system_topology(state))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/layout')
def get_dashboard_layout():
    """获取仪表盘布局配置"""
    try:
        layout_type = request.args.get('type', default='default', type=str)
        if layout_type == 'monitoring':
            return jsonify(DashboardLayout.get_monitoring_layout())
        return jsonify(DashboardLayout.get_default_layout())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/alerts')
def get_dashboard_alerts():
    """获取可视化告警"""
    try:
        limit = request.args.get('limit', default=20, type=int)
        return jsonify(dashboard.get_alerts(limit))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/scenario-timeline')
def get_scenario_timeline():
    """获取场景时间线"""
    try:
        hours = request.args.get('hours', default=1, type=int)
        with sim_lock:
            history = list(state_history)
        return jsonify(dashboard.get_scenario_timeline(history, hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 新增API端点 - 安全管理
# ============================================================

@app.route('/api/safety')
def get_safety_status():
    """获取安全状态"""
    try:
        return jsonify(safety_manager.get_safety_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/faults')
def get_safety_faults():
    """获取故障信息"""
    try:
        active = safety_manager.fault_diagnosis.get_active_faults()
        stats = safety_manager.fault_diagnosis.get_fault_statistics()

        return jsonify({
            'active_faults': [
                {
                    'id': f.fault_id,
                    'type': f.fault_type.value,
                    'severity': f.severity.name,
                    'location': f.location,
                    'description': f.description,
                    'timestamp': f.timestamp.isoformat()
                }
                for f in active
            ],
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/interlocks')
def get_safety_interlocks():
    """获取联锁状态"""
    try:
        return jsonify(safety_manager.interlocks.get_interlock_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/channels')
def get_safety_channels():
    """获取冗余通道状态"""
    try:
        return jsonify(safety_manager.redundant_control.get_channel_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/emergency')
def get_safety_emergency():
    """获取应急状态"""
    try:
        return jsonify(safety_manager.emergency.get_emergency_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/emergency/activate', methods=['POST'])
def activate_emergency():
    """激活应急响应"""
    try:
        data = request.json
        if not data or 'type' not in data:
            return jsonify({'error': 'Emergency type required'}), 400

        result = safety_manager.emergency.activate_emergency(data['type'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/safety/emergency/deactivate', methods=['POST'])
def deactivate_emergency():
    """解除应急状态"""
    try:
        data = request.json or {}
        reason = data.get('reason', 'resolved')
        success = safety_manager.emergency.deactivate_emergency(reason)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 新增API端点 - 工程集成
# ============================================================

@app.route('/api/scada')
def get_scada_status():
    """获取SCADA状态"""
    try:
        return jsonify(integration.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scada/values')
def get_scada_values():
    """获取SCADA数据点值"""
    try:
        return jsonify(integration.get_scada_values())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scada/alarms')
def get_scada_alarms():
    """获取SCADA告警"""
    try:
        return jsonify(integration.scada.get_active_alarms())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scada/write', methods=['POST'])
def write_scada_value():
    """写入SCADA设定值"""
    try:
        data = request.json
        if not data or 'parameter' not in data or 'value' not in data:
            return jsonify({'error': 'Parameter and value required'}), 400

        success = integration.write_setpoint(data['parameter'], float(data['value']))
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/opcua/nodes')
def get_opcua_nodes():
    """获取OPC-UA节点"""
    try:
        return jsonify(integration.get_opcua_nodes())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/opcua/read')
def read_opcua_node():
    """读取OPC-UA节点值"""
    try:
        node_id = request.args.get('node_id')
        if not node_id:
            return jsonify({'error': 'node_id required'}), 400

        value, status = integration.opcua.read_value(node_id)
        return jsonify({
            'node_id': node_id,
            'value': value,
            'status_code': status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/opcua/write', methods=['POST'])
def write_opcua_node():
    """写入OPC-UA节点值"""
    try:
        data = request.json
        if not data or 'node_id' not in data or 'value' not in data:
            return jsonify({'error': 'node_id and value required'}), 400

        status = integration.opcua.write_value(data['node_id'], data['value'])
        return jsonify({
            'node_id': data['node_id'],
            'status_code': status,
            'success': status == 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 系统综合API
# ============================================================

@app.route('/api/system/full-status')
def get_full_system_status():
    """获取系统完整状态"""
    try:
        with sim_lock:
            state = sim.get_state()
            is_safe = sim.is_safe_state()

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - start_time).total_seconds(),
            'simulation': {
                'running': simulation_running,
                'paused': simulation_paused,
                'time': state.get('time', 0)
            },
            'state': state,
            'perception': last_perception_result,
            'mpc': last_mpc_state,
            'intelligence': {
                'anomalies': last_intelligence_result.get('anomalies', []),
                'patterns': last_intelligence_result.get('patterns', [])
            },
            'safety': {
                'level': last_safety_result.get('safety_level', SafetyLevel.NORMAL).value
                    if hasattr(last_safety_result.get('safety_level'), 'value')
                    else str(last_safety_result.get('safety_level', 'NORMAL')),
                'faults': last_safety_result.get('faults', []),
                'interlocks': last_safety_result.get('interlocks_triggered', [])
            },
            'is_safe': is_safe
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.5 新增API端点 - 配置管理
# ============================================================

@app.route('/api/config')
def get_all_config():
    """获取所有配置"""
    try:
        return jsonify(config.get_all())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/<section>')
def get_config_section(section):
    """获取配置区段"""
    try:
        data = config.get_section(section)
        if not data:
            return jsonify({'error': f'Section {section} not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/<section>', methods=['PUT'])
def update_config_section(section):
    """更新配置区段"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        success = config.update_section(section, data, author=request.remote_addr)
        if success:
            log_info(f"Configuration updated: {section}", category="config",
                    data=data, user=request.remote_addr)
            audit("config_update", request.remote_addr, target=section,
                 details=data, result="success")
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'error': 'Update failed, validation error'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/<section>/<key>', methods=['PUT'])
def update_config_value(section, key):
    """更新单个配置值"""
    try:
        data = request.json
        if 'value' not in data:
            return jsonify({'error': 'Value required'}), 400

        old_value = config.get(section, key)
        success = config.set(section, key, data['value'], author=request.remote_addr)
        if success:
            log_info(f"Config value updated: {section}.{key}", category="config",
                    data={'old': old_value, 'new': data['value']}, user=request.remote_addr)
            return jsonify({'status': 'ok', 'old_value': old_value, 'new_value': data['value']})
        else:
            return jsonify({'error': 'Update failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/history')
def get_config_history():
    """获取配置变更历史"""
    try:
        limit = request.args.get('limit', default=20, type=int)
        return jsonify(config.get_version_history(limit))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/export')
def export_config():
    """导出配置"""
    try:
        format_type = request.args.get('format', default='yaml', type=str)
        config_str = config.export_config(format_type)

        response = make_response(config_str)
        if format_type == 'json':
            response.headers['Content-Type'] = 'application/json'
            response.headers['Content-Disposition'] = 'attachment; filename=taos_config.json'
        else:
            response.headers['Content-Type'] = 'application/x-yaml'
            response.headers['Content-Disposition'] = 'attachment; filename=taos_config.yaml'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/import', methods=['POST'])
def import_config():
    """导入配置"""
    try:
        data = request.json
        if not data or 'config' not in data:
            return jsonify({'error': 'Config data required'}), 400

        format_type = data.get('format', 'yaml')
        success = config.import_config(data['config'], format_type,
                                       author=request.remote_addr)
        if success:
            audit("config_import", request.remote_addr, target="all",
                 details={'format': format_type}, result="success")
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'error': 'Import failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/reset', methods=['POST'])
def reset_config():
    """重置配置到默认值"""
    try:
        data = request.json or {}
        section = data.get('section')
        config.reset_to_defaults(section)
        audit("config_reset", request.remote_addr, target=section or "all",
             result="success")
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.5 新增API端点 - 日志系统
# ============================================================

@app.route('/api/logs')
def get_logs():
    """获取日志"""
    try:
        level = request.args.get('level')
        category = request.args.get('category')
        limit = request.args.get('limit', default=100, type=int)
        search = request.args.get('search')

        logs = logger.get_recent_logs(count=limit, level=level, category=category)
        if search:
            logs = [l for l in logs if search.lower() in l.get('message', '').lower()]

        return jsonify({
            'count': len(logs),
            'logs': logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/query')
def query_logs():
    """查询日志"""
    try:
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        level = request.args.get('level')
        category = request.args.get('category')
        search = request.args.get('search')
        limit = request.args.get('limit', default=100, type=int)

        start = datetime.fromisoformat(start_time) if start_time else None
        end = datetime.fromisoformat(end_time) if end_time else None

        logs = logger.query_logs(start_time=start, end_time=end,
                                level=level, category=category,
                                search=search, limit=limit)

        return jsonify({
            'count': len(logs),
            'logs': logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/statistics')
def get_log_statistics():
    """获取日志统计"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(logger.get_log_statistics(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audit')
def get_audit_trail():
    """获取审计记录"""
    try:
        category = request.args.get('category')
        actor = request.args.get('actor')
        action = request.args.get('action')
        limit = request.args.get('limit', default=100, type=int)

        entries = logger.get_audit_trail(category=category, actor=actor,
                                        action=action, limit=limit)
        return jsonify({
            'count': len(entries),
            'entries': entries
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audit/statistics')
def get_audit_statistics():
    """获取审计统计"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        start = datetime.now() - timedelta(hours=hours)
        return jsonify(logger.get_audit_statistics(start_time=start))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.5 新增API端点 - API文档
# ============================================================

@app.route('/api/docs')
@app.route('/api/docs/')
def api_docs():
    """Swagger UI 文档页面"""
    return get_swagger_ui_html('/api/openapi.json')


@app.route('/api/redoc')
def api_redoc():
    """ReDoc 文档页面"""
    return get_redoc_html('/api/openapi.json')


@app.route('/api/openapi.json')
def openapi_json():
    """OpenAPI JSON规范"""
    return Response(api_generator.get_openapi_json(),
                   mimetype='application/json')


@app.route('/api/openapi.yaml')
def openapi_yaml():
    """OpenAPI YAML规范"""
    return Response(api_generator.get_openapi_yaml(),
                   mimetype='application/x-yaml')


# ============================================================
# V3.5 新增路由 - 增强版仪表盘
# ============================================================

@app.route('/dashboard')
def enhanced_dashboard():
    """增强版仪表盘页面"""
    return render_template('dashboard.html')


@app.route('/api/metrics')
def get_system_metrics():
    """获取系统性能指标"""
    try:
        import os
        import psutil

        process = psutil.Process(os.getpid())

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - start_time).total_seconds(),
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'thread_count': threading.active_count(),
            'simulation': {
                'running': simulation_running,
                'paused': simulation_paused,
                'history_size': len(state_history)
            },
            'modules': {
                'persistence': persistence.get_database_stats() if hasattr(persistence, 'get_database_stats') else {},
                'safety': safety_manager.get_safety_summary() if hasattr(safety_manager, 'get_safety_summary') else {},
                'config': {'version': config.current_version}
            }
        })
    except ImportError:
        # psutil not available
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - start_time).total_seconds(),
            'thread_count': threading.active_count(),
            'simulation': {
                'running': simulation_running,
                'paused': simulation_paused,
                'history_size': len(state_history)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.8 新增API端点 - 集群管理
# ============================================================

@app.route('/api/cluster')
def get_cluster_status():
    """获取集群状态"""
    try:
        return jsonify(cluster_manager.get_cluster_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/nodes')
def get_cluster_nodes():
    """获取集群节点列表"""
    try:
        nodes = [n.to_dict() for n in cluster_manager.nodes.values()]
        return jsonify({
            'count': len(nodes),
            'nodes': nodes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/nodes/<node_id>')
def get_cluster_node(node_id):
    """获取指定节点信息"""
    try:
        node = cluster_manager.nodes.get(node_id)
        if not node:
            return jsonify({'error': 'Node not found'}), 404
        return jsonify(node.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/nodes/<node_id>/metrics')
def get_cluster_node_metrics(node_id):
    """获取节点指标"""
    try:
        return jsonify(cluster_manager.get_node_metrics(node_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/register', methods=['POST'])
def register_cluster_node():
    """注册新节点"""
    try:
        data = request.json
        if not data or 'node_id' not in data:
            return jsonify({'error': 'node_id required'}), 400

        success = cluster_manager.register_node(data)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/load-balancer')
def get_load_balancer_status():
    """获取负载均衡状态"""
    try:
        return jsonify({
            'strategy': cluster_manager.load_balancer.current_strategy,
            'available_strategies': list(cluster_manager.load_balancer.strategies.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/load-balancer/strategy', methods=['POST'])
def set_load_balancer_strategy():
    """设置负载均衡策略"""
    try:
        data = request.json
        if not data or 'strategy' not in data:
            return jsonify({'error': 'strategy required'}), 400

        cluster_manager.load_balancer.set_strategy(data['strategy'])
        return jsonify({'status': 'ok', 'strategy': data['strategy']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.8 新增API端点 - 边缘计算
# ============================================================

@app.route('/api/edge')
def get_edge_status():
    """获取边缘计算状态"""
    try:
        return jsonify(edge_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/devices')
def get_edge_devices():
    """获取边缘设备列表"""
    try:
        devices = [d.to_dict() for d in edge_manager.get_all_devices()]
        return jsonify({
            'count': len(devices),
            'devices': devices
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/devices/<device_id>')
def get_edge_device(device_id):
    """获取指定边缘设备"""
    try:
        device = edge_manager.get_device(device_id)
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        return jsonify(device.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/devices/<device_id>/data', methods=['POST'])
def process_edge_device_data(device_id):
    """处理边缘设备数据"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        result = edge_manager.process_device_data(device_id, data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/devices/<device_id>/aggregated')
def get_edge_device_aggregated(device_id):
    """获取边缘设备聚合数据"""
    try:
        return jsonify(edge_manager.get_aggregated_data(device_id))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/devices/<device_id>/command', methods=['POST'])
def send_edge_device_command(device_id):
    """发送命令到边缘设备"""
    try:
        data = request.json
        if not data or 'command' not in data:
            return jsonify({'error': 'command required'}), 400

        result = edge_manager.send_command_to_device(
            device_id, data['command'], data.get('params', {})
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/register', methods=['POST'])
def register_edge_device():
    """注册新边缘设备"""
    try:
        data = request.json
        if not data or 'device_id' not in data:
            return jsonify({'error': 'device_id required'}), 400

        device = edge_manager.register_device(data)
        return jsonify(device.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge/sync')
def get_edge_sync_status():
    """获取边缘同步状态"""
    try:
        return jsonify(edge_manager.sync_manager.get_sync_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.8 新增API端点 - 数字孪生
# ============================================================

@app.route('/api/twin')
def get_twin_status():
    """获取数字孪生状态"""
    try:
        return jsonify(twin_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/model')
def get_twin_model():
    """获取数字孪生3D模型数据"""
    try:
        return jsonify(twin_manager.model.get_3d_model_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/components')
def get_twin_components():
    """获取数字孪生组件列表"""
    try:
        components = {cid: c.to_dict()
                     for cid, c in twin_manager.model.components.items()}
        return jsonify({
            'count': len(components),
            'components': components
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/components/<component_id>')
def get_twin_component(component_id):
    """获取指定组件"""
    try:
        comp = twin_manager.model.get_component(component_id)
        if not comp:
            return jsonify({'error': 'Component not found'}), 404
        return jsonify(comp.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/virtual-sensors')
def get_twin_virtual_sensors():
    """获取虚拟传感器"""
    try:
        sensors = {vid: v.to_dict()
                  for vid, v in twin_manager.model.virtual_sensors.items()}
        return jsonify({
            'count': len(sensors),
            'sensors': sensors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/virtual-sensors/readings')
def get_twin_virtual_sensor_readings():
    """获取虚拟传感器读数"""
    try:
        return jsonify(twin_manager.model.get_virtual_sensor_readings())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/cross-section')
def get_twin_cross_section():
    """获取指定位置的断面数据"""
    try:
        chainage = request.args.get('chainage', default=500, type=float)
        return jsonify(twin_manager.model.get_cross_section_at(chainage))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/state-history')
def get_twin_state_history():
    """获取数字孪生状态历史"""
    try:
        minutes = request.args.get('minutes', default=60, type=int)
        return jsonify(twin_manager.model.get_state_history(minutes))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/scenario', methods=['POST'])
def run_twin_scenario():
    """运行数字孪生场景分析"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Scenario data required'}), 400

        name = data.get('name', f'scenario_{int(time.time())}')
        scenario = data.get('scenario', {})
        duration = data.get('duration', 300.0)

        result = twin_manager.run_scenario_analysis(name, scenario, duration)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/scenarios')
def get_twin_scenarios():
    """获取保存的场景分析结果"""
    try:
        return jsonify(twin_manager.get_saved_scenarios())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/twin/export')
def export_twin_model():
    """导出数字孪生模型"""
    try:
        model_json = twin_manager.model.export_to_json()
        response = make_response(model_json)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=digital_twin.json'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.8 新增API端点 - ML控制
# ============================================================

@app.route('/api/ml')
def get_ml_status():
    """获取ML控制状态"""
    try:
        return jsonify(ml_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/models')
def get_ml_models():
    """获取ML模型列表"""
    try:
        return jsonify(ml_manager.get_model_list())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/models/<model_id>')
def get_ml_model(model_id):
    """获取指定ML模型信息"""
    try:
        info = ml_manager.predictor.get_model_info(model_id)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/predict/<model_id>')
def get_ml_prediction(model_id):
    """获取ML预测"""
    try:
        horizon = request.args.get('horizon', default=30, type=int)

        with sim_lock:
            state = sim.get_state()

        pred = ml_manager.get_prediction(model_id, state, horizon)
        if pred:
            return jsonify(pred)
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/anomaly')
def get_ml_anomaly():
    """获取异常检测结果"""
    try:
        with sim_lock:
            state = sim.get_state()

        anomaly = ml_manager.predictor.detect_anomaly(state)
        return jsonify(anomaly)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/train', methods=['POST'])
def train_ml_models():
    """训练ML模型"""
    try:
        data = request.json or {}
        epochs = data.get('epochs', 100)

        results = ml_manager.train_models(epochs)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/rl')
def get_ml_rl_status():
    """获取强化学习状态"""
    try:
        return jsonify(ml_manager.rl_controller.get_policy_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/rl/action')
def get_ml_rl_action():
    """获取RL建议的动作"""
    try:
        with sim_lock:
            state = sim.get_state()

        action = ml_manager.rl_controller.get_action(state, explore=False)
        return jsonify(action)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/rl/enable', methods=['POST'])
def enable_ml_rl():
    """启用/禁用RL控制"""
    try:
        data = request.json or {}
        enable = data.get('enable', True)
        ml_manager.enable_rl_control(enable)
        return jsonify({'status': 'ok', 'rl_enabled': enable})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.9 新增API端点 - 移动端API
# ============================================================

@app.route('/api/mobile')
def get_mobile_status():
    """获取移动端API状态"""
    try:
        return jsonify(mobile_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/devices')
def get_mobile_devices():
    """获取已注册的移动设备列表"""
    try:
        devices = mobile_manager.get_all_devices()
        return jsonify({
            'count': len(devices),
            'devices': [d.to_dict() for d in devices]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/devices/<device_id>')
def get_mobile_device(device_id):
    """获取指定移动设备信息"""
    try:
        device = mobile_manager.get_device(device_id)
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        return jsonify(device.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/register', methods=['POST'])
def register_mobile_device():
    """注册移动设备"""
    try:
        data = request.json
        if not data or 'device_id' not in data:
            return jsonify({'error': 'device_id required'}), 400

        device = mobile_manager.register_device(data)
        audit_manager.log_data_create(
            actor=data.get('user_id', 'anonymous'),
            target=f"mobile_device:{device.device_id}",
            data={'platform': data.get('platform', 'unknown')}
        )
        return jsonify(device.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/devices/<device_id>/push-token', methods=['PUT'])
def update_push_token(device_id):
    """更新推送令牌"""
    try:
        data = request.json
        if not data or 'push_token' not in data:
            return jsonify({'error': 'push_token required'}), 400

        success = mobile_manager.update_push_token(device_id, data['push_token'])
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/notifications', methods=['POST'])
def send_mobile_notification():
    """发送推送通知"""
    try:
        data = request.json
        if not data or 'title' not in data:
            return jsonify({'error': 'title required'}), 400

        result = mobile_manager.notification_service.send_notification(
            device_id=data.get('device_id'),
            title=data['title'],
            body=data.get('body', ''),
            notification_type=data.get('type', 'info'),
            data=data.get('data', {})
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/notifications/broadcast', methods=['POST'])
def broadcast_mobile_notification():
    """广播推送通知"""
    try:
        data = request.json
        if not data or 'title' not in data:
            return jsonify({'error': 'title required'}), 400

        result = mobile_manager.notification_service.broadcast(
            title=data['title'],
            body=data.get('body', ''),
            notification_type=data.get('type', 'info')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/qr-code')
def get_mobile_qr_code():
    """生成移动端绑定二维码"""
    try:
        user_id = request.args.get('user_id', 'default')
        qr_data = mobile_manager.qr_generator.generate_device_binding_qr(user_id)
        return jsonify(qr_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/sync/status')
def get_mobile_sync_status():
    """获取移动端同步状态"""
    try:
        return jsonify(mobile_manager.sync_manager.get_sync_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/sync/data', methods=['POST'])
def sync_mobile_data():
    """同步移动端数据"""
    try:
        data = request.json
        if not data or 'device_id' not in data:
            return jsonify({'error': 'device_id required'}), 400

        result = mobile_manager.sync_manager.sync_device(
            data['device_id'],
            data.get('last_sync_time')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mobile/dashboard')
def get_mobile_dashboard():
    """获取移动端仪表盘数据"""
    try:
        with sim_lock:
            state = sim.get_state()

        dashboard_data = mobile_manager.get_mobile_dashboard(state)
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.9 新增API端点 - 高级可视化
# ============================================================

@app.route('/api/viz')
def get_viz_status():
    """获取可视化系统状态"""
    try:
        return jsonify(viz_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/charts/<chart_type>')
def get_viz_chart(chart_type):
    """获取指定类型的图表数据"""
    try:
        with sim_lock:
            state = sim.get_state()
            history = list(state_history)

        chart_data = viz_manager.chart_generator.generate_chart(
            chart_type, state, history
        )
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/gis')
def get_viz_gis_data():
    """获取GIS可视化数据"""
    try:
        with sim_lock:
            state = sim.get_state()

        gis_data = viz_manager.gis_viz.get_gis_data(state)
        return jsonify(gis_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/gis/layers')
def get_viz_gis_layers():
    """获取GIS图层列表"""
    try:
        return jsonify(viz_manager.gis_viz.get_available_layers())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/gis/layer/<layer_id>')
def get_viz_gis_layer(layer_id):
    """获取指定GIS图层数据"""
    try:
        layer_data = viz_manager.gis_viz.get_layer_data(layer_id)
        if not layer_data:
            return jsonify({'error': 'Layer not found'}), 404
        return jsonify(layer_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder')
def get_viz_dashboard_builder():
    """获取仪表盘构建器状态"""
    try:
        return jsonify(viz_manager.dashboard_builder.get_builder_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/templates')
def get_viz_dashboard_templates():
    """获取仪表盘模板列表"""
    try:
        return jsonify(viz_manager.dashboard_builder.get_templates())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/dashboards')
def get_viz_custom_dashboards():
    """获取自定义仪表盘列表"""
    try:
        return jsonify(viz_manager.dashboard_builder.get_all_dashboards())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/dashboards', methods=['POST'])
def create_viz_dashboard():
    """创建自定义仪表盘"""
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({'error': 'name required'}), 400

        dashboard = viz_manager.dashboard_builder.create_dashboard(data)
        audit_manager.log_data_create(
            actor=request.remote_addr,
            target=f"dashboard:{dashboard['id']}",
            data={'name': data['name']}
        )
        return jsonify(dashboard)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/dashboards/<dashboard_id>')
def get_viz_dashboard_by_id(dashboard_id):
    """获取指定仪表盘"""
    try:
        dashboard = viz_manager.dashboard_builder.get_dashboard(dashboard_id)
        if not dashboard:
            return jsonify({'error': 'Dashboard not found'}), 404
        return jsonify(dashboard)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/dashboards/<dashboard_id>', methods=['PUT'])
def update_viz_dashboard(dashboard_id):
    """更新仪表盘"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        dashboard = viz_manager.dashboard_builder.update_dashboard(dashboard_id, data)
        if not dashboard:
            return jsonify({'error': 'Dashboard not found'}), 404

        audit_manager.log_data_update(
            actor=request.remote_addr,
            target=f"dashboard:{dashboard_id}",
            data={'changes': list(data.keys())}
        )
        return jsonify(dashboard)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/dashboard-builder/dashboards/<dashboard_id>', methods=['DELETE'])
def delete_viz_dashboard(dashboard_id):
    """删除仪表盘"""
    try:
        success = viz_manager.dashboard_builder.delete_dashboard(dashboard_id)
        if not success:
            return jsonify({'error': 'Dashboard not found'}), 404

        audit_manager.log_data_delete(
            actor=request.remote_addr,
            target=f"dashboard:{dashboard_id}"
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/3d-model')
def get_viz_3d_model():
    """获取3D模型数据"""
    try:
        with sim_lock:
            state = sim.get_state()

        model_data = viz_manager.get_3d_visualization(state)
        return jsonify(model_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/heatmap')
def get_viz_heatmap():
    """获取热力图数据"""
    try:
        variable = request.args.get('variable', 'h')
        with sim_lock:
            state = sim.get_state()
            history = list(state_history)

        heatmap_data = viz_manager.chart_generator.generate_heatmap(
            variable, state, history
        )
        return jsonify(heatmap_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/viz/export/<format_type>')
def export_viz_data(format_type):
    """导出可视化数据"""
    try:
        with sim_lock:
            state = sim.get_state()
            history = list(state_history)

        if format_type == 'svg':
            svg_data = viz_manager.export_to_svg(state)
            response = make_response(svg_data)
            response.headers['Content-Type'] = 'image/svg+xml'
            response.headers['Content-Disposition'] = 'attachment; filename=visualization.svg'
            return response
        elif format_type == 'json':
            json_data = viz_manager.export_to_json(state, history)
            response = make_response(json_data)
            response.headers['Content-Type'] = 'application/json'
            response.headers['Content-Disposition'] = 'attachment; filename=visualization.json'
            return response
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.9 新增API端点 - 国际化
# ============================================================

@app.route('/api/i18n')
def get_i18n_status():
    """获取国际化状态"""
    try:
        return jsonify(i18n_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/locales')
def get_i18n_locales():
    """获取支持的语言列表"""
    try:
        return jsonify({
            'locales': i18n_manager.get_supported_locales(),
            'current': i18n_manager.get_locale()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/locale', methods=['PUT'])
def set_i18n_locale():
    """设置当前语言"""
    try:
        data = request.json
        if not data or 'locale' not in data:
            return jsonify({'error': 'locale required'}), 400

        success = i18n_manager.set_locale(data['locale'])
        if not success:
            return jsonify({'error': 'Unsupported locale'}), 400
        return jsonify({'status': 'ok', 'locale': data['locale']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/translate')
def translate_text():
    """翻译文本"""
    try:
        key = request.args.get('key')
        locale = request.args.get('locale')

        if not key:
            return jsonify({'error': 'key required'}), 400

        translation = i18n_manager.t(key, locale)
        return jsonify({
            'key': key,
            'locale': locale or i18n_manager.get_locale(),
            'translation': translation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/translations/<locale>')
def get_i18n_translations(locale):
    """获取指定语言的所有翻译"""
    try:
        translations = i18n_manager.translation_manager.get_translations_for_locale(locale)
        return jsonify({
            'locale': locale,
            'count': len(translations),
            'translations': translations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/translations', methods=['POST'])
def add_i18n_translation():
    """添加翻译"""
    try:
        data = request.json
        if not data or 'key' not in data or 'locale' not in data or 'text' not in data:
            return jsonify({'error': 'key, locale, and text required'}), 400

        i18n_manager.add_translation(data['key'], data['locale'], data['text'])
        audit_manager.log_data_create(
            actor=request.remote_addr,
            target=f"translation:{data['key']}:{data['locale']}"
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/export')
def export_i18n():
    """导出翻译"""
    try:
        locale = request.args.get('locale')
        data = i18n_manager.export_all_translations()

        response = make_response(json.dumps(data, ensure_ascii=False, indent=2))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = 'attachment; filename=translations.json'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/import', methods=['POST'])
def import_i18n():
    """导入翻译"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        i18n_manager.import_translations(data)
        audit_manager.log_data_create(
            actor=request.remote_addr,
            target="translations",
            data={'imported_keys': len(data.get('translations', {}))}
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/format/date')
def format_i18n_date():
    """格式化日期"""
    try:
        timestamp = request.args.get('timestamp')
        locale = request.args.get('locale')
        format_type = request.args.get('format', 'datetime')

        if timestamp:
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = datetime.now()

        formatted = i18n_manager.format_date(dt, locale, format_type)
        return jsonify({
            'timestamp': dt.isoformat(),
            'locale': locale or i18n_manager.get_locale(),
            'format_type': format_type,
            'formatted': formatted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/i18n/format/number')
def format_i18n_number():
    """格式化数字"""
    try:
        number = request.args.get('number', type=float)
        locale = request.args.get('locale')
        decimals = request.args.get('decimals', default=2, type=int)

        if number is None:
            return jsonify({'error': 'number required'}), 400

        formatted = i18n_manager.format_number(number, locale, decimals)
        return jsonify({
            'number': number,
            'locale': locale or i18n_manager.get_locale(),
            'formatted': formatted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# V3.9 新增API端点 - 区块链审计
# ============================================================

@app.route('/api/blockchain')
def get_blockchain_status():
    """获取区块链审计状态"""
    try:
        return jsonify(audit_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/verify')
def verify_blockchain():
    """验证区块链完整性"""
    try:
        return jsonify(audit_manager.verify_chain())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/stats')
def get_blockchain_stats():
    """获取区块链统计"""
    try:
        return jsonify(audit_manager.get_chain_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/events')
def get_blockchain_events():
    """查询审计事件"""
    try:
        actor = request.args.get('actor')
        target = request.args.get('target')
        event_type = request.args.get('event_type')
        severity = request.args.get('severity')
        limit = request.args.get('limit', default=100, type=int)

        events = audit_manager.query_events(
            actor=actor, target=target,
            event_type=event_type, severity=severity, limit=limit
        )
        return jsonify({
            'count': len(events),
            'events': events
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/versions/<entity_type>/<entity_id>')
def get_blockchain_version_history(entity_type, entity_id):
    """获取实体版本历史"""
    try:
        history = audit_manager.get_version_history(entity_type, entity_id)
        return jsonify({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'versions': history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/versions/<entity_type>/<entity_id>/compare')
def compare_blockchain_versions(entity_type, entity_id):
    """比较实体版本"""
    try:
        v1 = request.args.get('v1', type=int)
        v2 = request.args.get('v2', type=int)

        if v1 is None or v2 is None:
            return jsonify({'error': 'v1 and v2 required'}), 400

        comparison = audit_manager.compare_versions(entity_type, entity_id, v1, v2)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/reports/access')
def get_blockchain_access_report():
    """获取访问审计报告"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(audit_manager.generate_access_report(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/reports/change')
def get_blockchain_change_report():
    """获取变更审计报告"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(audit_manager.generate_change_report(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/reports/security')
def get_blockchain_security_report():
    """获取安全审计报告"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(audit_manager.generate_security_report(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/reports/control')
def get_blockchain_control_report():
    """获取控制审计报告"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(audit_manager.generate_control_report(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/reports/compliance')
def get_blockchain_compliance_report():
    """获取合规性报告"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        return jsonify(audit_manager.generate_compliance_summary(hours))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/blockchain/export')
def export_blockchain():
    """导出区块链数据"""
    try:
        chain_json = audit_manager.export_chain()
        response = make_response(chain_json)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=audit_chain.json'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 更新版本信息
# ============================================================

@app.route('/api/version')
def get_version():
    """获取系统版本信息"""
    return jsonify({
        'name': 'TAOS - Tuanhe Aqueduct Autonomous Operation System',
        'name_cn': '团河渡槽自主运行系统',
        'version': '3.10.0',
        'build_date': '2025-12-07',
        'features': [
            '全场景自主运行',
            '预测与规划',
            '数据持久化',
            '智能分析',
            '可视化仪表盘',
            '安全管理',
            '工程集成',
            '配置管理',
            '日志审计',
            'API文档',
            # V3.6-V3.7
            'WebSocket实时通信',
            '多渠道告警',
            '自动报告生成',
            '用户认证授权',
            'Docker容器化',
            '性能监控',
            '备份恢复',
            # V3.8
            '分布式集群',
            '边缘计算',
            '数字孪生',
            'AI/ML控制',
            # V3.9
            '移动端API',
            '高级可视化',
            '国际化支持',
            '区块链审计',
            # V3.10
            '高保真传感器仿真',
            '高保真执行器仿真',
            '数据治理',
            '数据同化',
            'IDZ模型参数动态更新',
            '系统状态实时评价',
            '系统状态实时预测',
            '多保真度模型融合'
        ],
        'modules': {
            'cluster': cluster_manager.get_cluster_status()['node_count'],
            'edge_devices': len(edge_manager.get_all_devices()),
            'twin_components': len(twin_manager.model.components),
            'ml_models': len(ml_manager.predictor.models),
            'mobile_devices': len(mobile_manager.get_all_devices()),
            'supported_locales': len(i18n_manager.get_supported_locales()),
            'audit_blocks': len(audit_manager.audit_chain.chain)
        }
    })


# ============================================================
# V3.10 新功能API - 传感器仿真、执行器仿真、数据治理、数据同化、
# IDZ模型参数动态更新、系统状态评价、系统状态预测
# ============================================================

# 导入V3.10模块
try:
    from sensor_simulation import SensorSimulationEngine, SensorDegradationMode
    from actuator_simulation import ActuatorSimulationEngine, ActuatorFailureMode
    from data_governance import DataGovernanceEngine
    from data_assimilation import DataAssimilationEngine, AssimilationMethod
    from idz_model_adapter import IDZModelAdapter, MultiFidelityModelManager
    from state_evaluation import StateEvaluator, MultiObjectiveEvaluator
    from state_prediction import StatePredictionEngine, ScenarioPrediction, PredictionMethod

    # 初始化V3.10模块
    sensor_sim_engine = SensorSimulationEngine()
    actuator_sim_engine = ActuatorSimulationEngine()
    data_governance_engine = DataGovernanceEngine()
    data_assimilation_engine = DataAssimilationEngine()
    idz_adapter = IDZModelAdapter()
    multi_fidelity_manager = MultiFidelityModelManager()
    state_evaluator = StateEvaluator()
    multi_objective_evaluator = MultiObjectiveEvaluator()
    state_prediction_engine = StatePredictionEngine()
    scenario_predictor = ScenarioPrediction()

    V310_ENABLED = True
    log_info("V3.10 modules loaded successfully", category="system")
except ImportError as e:
    V310_ENABLED = False
    log_info(f"V3.10 modules not available: {e}", category="system")


# ---------- 传感器仿真 API ----------

@app.route('/api/v310/sensor/measure', methods=['POST'])
def sensor_simulation_measure():
    """执行传感器仿真测量"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        true_state = data.get('true_state', {})
        dt = data.get('dt', 0.1)

        # 更新环境条件
        if 'environment' in data:
            sensor_sim_engine.update_environment(**data['environment'])

        result = sensor_sim_engine.measure(true_state, dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/sensor/status')
def get_sensor_simulation_status():
    """获取传感器仿真状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(sensor_sim_engine.get_full_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/sensor/inject_fault', methods=['POST'])
def inject_sensor_fault():
    """注入传感器故障"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        network_name = data.get('network')
        sensor_name = data.get('sensor')
        degradation_mode = data.get('mode', 'linear_drift')
        factor = data.get('factor', 0.1)

        mode = SensorDegradationMode[degradation_mode.upper()]
        sensor_sim_engine.inject_fault(network_name, sensor_name, mode, factor)

        return jsonify({'status': 'fault_injected', 'sensor': sensor_name, 'mode': degradation_mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/sensor/clear_faults', methods=['POST'])
def clear_sensor_faults():
    """清除传感器故障"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        sensor_sim_engine.clear_faults()
        return jsonify({'status': 'faults_cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 执行器仿真 API ----------

@app.route('/api/v310/actuator/command', methods=['POST'])
def actuator_simulation_command():
    """发送执行器命令"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        Q_in = data.get('Q_in', 80.0)
        Q_out = data.get('Q_out', 80.0)
        emergency_dump = data.get('emergency_dump', False)

        actuator_sim_engine.command_flows(Q_in, Q_out, emergency_dump)
        return jsonify({'status': 'command_sent', 'Q_in': Q_in, 'Q_out': Q_out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/actuator/step', methods=['POST'])
def actuator_simulation_step():
    """执行执行器仿真步进"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        dt = data.get('dt', 0.1)

        result = actuator_sim_engine.step(dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/actuator/status')
def get_actuator_simulation_status():
    """获取执行器仿真状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(actuator_sim_engine.get_full_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/actuator/inject_failure', methods=['POST'])
def inject_actuator_failure():
    """注入执行器故障"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        actuator_name = data.get('actuator')
        mode = data.get('mode', 'slow_response')
        severity = data.get('severity', 0.5)

        failure_mode = ActuatorFailureMode[mode.upper()]
        actuator_sim_engine.inject_failure(actuator_name, failure_mode, severity)

        return jsonify({'status': 'failure_injected', 'actuator': actuator_name, 'mode': mode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/actuator/emergency_shutdown', methods=['POST'])
def actuator_emergency_shutdown():
    """执行器紧急停机"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        actuator_sim_engine.emergency_shutdown()
        return jsonify({'status': 'emergency_shutdown_activated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 数据治理 API ----------

@app.route('/api/v310/governance/process', methods=['POST'])
def data_governance_process():
    """数据治理处理"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        source = data.get('source', 'sensor_data')
        user_id = data.get('user_id', 'system')
        payload = data.get('data', {})

        result = data_governance_engine.process_data(payload, source, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/governance/dashboard')
def get_governance_dashboard():
    """获取数据治理仪表盘"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(data_governance_engine.get_governance_dashboard())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/governance/compliance_report')
def get_governance_compliance_report():
    """获取合规报告"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)

        report = data_governance_engine.export_compliance_report(start_time, end_time)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/governance/quality_report')
def get_data_quality_report():
    """获取数据质量报告"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        lookback = request.args.get('lookback_seconds', default=3600, type=float)
        report = data_governance_engine.quality_validator.get_validation_report(lookback)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 数据同化 API ----------

@app.route('/api/v310/assimilation/initialize', methods=['POST'])
def initialize_data_assimilation():
    """初始化数据同化"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        initial_state = data.get('initial_state', {})
        uncertainty = data.get('uncertainty', {})

        data_assimilation_engine.initialize(initial_state, uncertainty)
        return jsonify({'status': 'initialized', 'state': initial_state})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/assimilation/predict', methods=['POST'])
def data_assimilation_predict():
    """数据同化预测步"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        control = data.get('control', {})
        dt = data.get('dt', 0.1)

        result = data_assimilation_engine.predict(control, dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/assimilation/assimilate', methods=['POST'])
def data_assimilation_update():
    """数据同化更新步"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        observations = data.get('observations', {})
        timestamp = data.get('timestamp')

        result = data_assimilation_engine.assimilate(observations, timestamp)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/assimilation/state')
def get_assimilation_state():
    """获取同化状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(data_assimilation_engine.get_current_state())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/assimilation/status')
def get_assimilation_status():
    """获取数据同化系统状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(data_assimilation_engine.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/assimilation/switch_method', methods=['POST'])
def switch_assimilation_method():
    """切换同化方法"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        method_name = data.get('method', 'ensemble_kalman')

        method = AssimilationMethod[method_name.upper()]
        data_assimilation_engine.switch_method(method)

        return jsonify({'status': 'method_switched', 'method': method_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- IDZ模型参数动态更新 API ----------

@app.route('/api/v310/idz/update', methods=['POST'])
def idz_model_update():
    """基于高保真模型更新IDZ参数"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        hifi_state = data.get('hifi_state', {})
        control = data.get('control', {})
        environment = data.get('environment', {})
        dt = data.get('dt', 0.1)

        result = idz_adapter.update_from_hifi(hifi_state, control, environment, dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/idz/parameters')
def get_idz_parameters():
    """获取当前IDZ模型参数"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(idz_adapter.get_current_parameters())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/idz/predict', methods=['POST'])
def idz_model_predict():
    """使用IDZ模型预测"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        control = data.get('control', {})
        environment = data.get('environment', {})
        horizon = data.get('horizon', 10)
        dt = data.get('dt', 0.1)

        predictions = idz_adapter.predict(control, environment, horizon, dt)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/idz/metrics')
def get_idz_metrics():
    """获取IDZ模型适配指标"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(idz_adapter.get_adaptation_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/idz/uncertainty')
def get_idz_uncertainty():
    """获取IDZ模型不确定性"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(idz_adapter.get_model_uncertainty())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/multifidelity/update', methods=['POST'])
def multifidelity_update():
    """更新多保真度模型"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        hifi_state = data.get('hifi_state', {})
        control = data.get('control', {})
        environment = data.get('environment', {})
        dt = data.get('dt', 0.1)

        result = multi_fidelity_manager.update(hifi_state, control, environment, dt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/multifidelity/status')
def get_multifidelity_status():
    """获取多保真度模型状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(multi_fidelity_manager.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 系统状态评价 API ----------

@app.route('/api/v310/evaluation/evaluate', methods=['POST'])
def evaluate_system_state():
    """评价系统状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})
        control = data.get('control', {})
        timestamp = data.get('timestamp')

        result = state_evaluator.evaluate(state, control, timestamp)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/deviation', methods=['POST'])
def evaluate_deviation():
    """评价与控制目标的偏差"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})
        timestamp = data.get('timestamp')

        result = state_evaluator.evaluate_deviation(state, timestamp)
        return jsonify({
            name: {
                'target': r.target,
                'actual': r.actual,
                'deviation': r.deviation,
                'severity': r.severity.value,
                'within_tolerance': r.within_tolerance
            }
            for name, r in result.items()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/performance', methods=['POST'])
def evaluate_performance():
    """计算性能指标"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})
        control = data.get('control', {})

        indices = state_evaluator.calculate_performance_indices(state, control)
        return jsonify([
            {
                'name': p.name,
                'category': p.category.value,
                'value': p.value,
                'target': p.target,
                'unit': p.unit
            }
            for p in indices
        ])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/risk', methods=['POST'])
def evaluate_risk():
    """评估系统风险"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})

        risks = state_evaluator.assess_risk(state)
        return jsonify([
            {
                'category': r.category,
                'risk_level': r.risk_level,
                'probability': r.probability,
                'consequence': r.consequence,
                'description': r.description
            }
            for r in risks
        ])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/compliance', methods=['POST'])
def evaluate_compliance():
    """检查合规性"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})

        compliance = state_evaluator.check_compliance(state)
        return jsonify({name: status.value for name, status in compliance.items()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/trend')
def get_evaluation_trend():
    """获取评价趋势"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        lookback = request.args.get('lookback_seconds', default=3600, type=float)
        return jsonify(state_evaluator.get_evaluation_trend(lookback))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/status')
def get_evaluation_status():
    """获取评价系统状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(state_evaluator.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/evaluation/multiobjective', methods=['POST'])
def evaluate_multiobjective():
    """多目标评价"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})
        control = data.get('control', {})

        result = multi_objective_evaluator.evaluate(state, control)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 系统状态预测 API ----------

@app.route('/api/v310/prediction/predict', methods=['POST'])
def predict_system_state():
    """预测系统状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        current_state = data.get('current_state', {})
        control = data.get('control', {})
        environment = data.get('environment', {})
        horizon = data.get('horizon', 'short')
        method = data.get('method', 'ensemble')

        pred_method = PredictionMethod[method.upper()]
        trajectory = state_prediction_engine.predict(
            current_state, control, environment, horizon, pred_method
        )

        return jsonify({
            'start_time': trajectory.start_time,
            'horizon_name': trajectory.horizon_name,
            'method': trajectory.method.value,
            'predictions': [
                {
                    'timestamp': p.timestamp,
                    'horizon_seconds': p.horizon_seconds,
                    'state': p.state,
                    'uncertainty': p.uncertainty,
                    'confidence': p.confidence
                }
                for p in trajectory.predictions
            ],
            'metadata': trajectory.metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/risk', methods=['POST'])
def predict_risk():
    """预测风险概率"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        current_state = data.get('current_state', {})
        control = data.get('control', {})
        environment = data.get('environment', {})
        thresholds = data.get('thresholds', {})

        result = state_prediction_engine.predict_risk(
            current_state, control, environment, thresholds
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/scenario', methods=['POST'])
def predict_scenario():
    """场景预测"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        current_state = data.get('current_state', {})
        scenario_name = data.get('scenario', 'normal')
        base_control = data.get('control', {})
        base_environment = data.get('environment', {})

        result = scenario_predictor.predict_scenario(
            current_state, scenario_name, base_control, base_environment
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/scenarios/compare', methods=['POST'])
def compare_scenarios():
    """比较多个场景"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        current_state = data.get('current_state', {})
        scenarios = data.get('scenarios', ['normal', 'summer_peak'])
        base_control = data.get('control', {})
        base_environment = data.get('environment', {})

        result = scenario_predictor.compare_scenarios(
            current_state, scenarios, base_control, base_environment
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/accuracy')
def get_prediction_accuracy():
    """获取预测精度"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(state_prediction_engine.get_prediction_accuracy())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/status')
def get_prediction_status():
    """获取预测系统状态"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        return jsonify(state_prediction_engine.get_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v310/prediction/update_history', methods=['POST'])
def update_prediction_history():
    """更新预测历史"""
    if not V310_ENABLED:
        return jsonify({'error': 'V3.10 modules not available'}), 503

    try:
        data = request.get_json()
        state = data.get('state', {})

        state_prediction_engine.update_history(state)
        return jsonify({'status': 'history_updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 启动时记录日志
log_info("TAOS V3.10 Server starting", category="system")
audit_manager.log_system_event("start", {"version": "3.10.0"})


if __name__ == '__main__':
    # 启动V3.8模块
    cluster_manager.start()
    edge_manager.start()
    twin_manager.start()
    ml_manager.start()

    log_info("TAOS V3.9 Server started on port 5000", category="system")
    app.run(host='0.0.0.0', port=5000)
