# Tuanhe Aqueduct Autonomous Operation System (TAOS) V3.10

## Overview
This project implements a comprehensive **Autonomous Operation & Health Management System** for the Tuanhe Aqueduct (South-to-North Water Diversion Project). The system addresses "Wind-Water-Thermal-Structure" multi-physics coupling risks through intelligent perception, adaptive control, and comprehensive simulation.

## Key Features

### Core Capabilities (V3.0-V3.5)
- **High-Fidelity Multi-Physics Simulation**: Models fluid dynamics, thermal effects, structural behavior, and environmental conditions
- **Comprehensive Scenario Coverage**: 1.78M+ scenarios including hydraulic jumps, thermal bending, bearing lock, joint tearing, and seismic events
- **Adaptive MPC Control**: Model Predictive Control with scenario-based gain scheduling and PID fallback
- **Full Hardware-in-the-Loop Testing**: Sensor and actuator models with fault injection capability
- **Real-time Web Dashboard**: Live visualization and scenario injection interface
- **Prediction & Planning**: Weather, flow, and seismic forecasting integration
- **Data Persistence**: SQLite-based time series storage with multi-level aggregation
- **Safety Management**: Fault diagnosis, redundant control, safety interlocks

### V3.6-V3.7 Enterprise Features
- **WebSocket Communication**: Real-time bidirectional messaging
- **Multi-Channel Alerts**: Email, SMS, webhook notifications with escalation
- **Report Generation**: Automated daily, weekly, monthly reports
- **Authentication & RBAC**: User management with role-based access control
- **Docker Containerization**: Production-ready deployment
- **Performance Monitoring**: Prometheus-compatible metrics
- **Backup & Restore**: Automated backup scheduling with verification

### V3.8 Distributed & Intelligent Features
- **Distributed Cluster**: Multi-node deployment with leader election and failover
- **Edge Computing**: Edge device management, local processing, cloud sync
- **Digital Twin**: High-fidelity 3D model with 75 components, virtual sensors
- **AI/ML Control**: Deep learning prediction, reinforcement learning control

### V3.9 Mobile & Visualization Enhancement
- **Mobile API**: Device registration, push notifications, QR codes, offline sync
- **Advanced Visualization**: ECharts-compatible charts, GIS mapping, dashboard builder
- **Internationalization (i18n)**: 10 languages, locale-aware formatting
- **Blockchain Audit**: Immutable audit trail, data versioning, compliance reports

### V3.10 Simulation, Data Assimilation & State Prediction (NEW)
- **High-Fidelity Sensor Simulation**: 12 sensor types with physics-based modeling, environmental interference, degradation
- **High-Fidelity Actuator Simulation**: 2nd-order dynamics, friction/wear models, failure mode injection
- **Data Governance**: Data quality validation (ISO 8000), lineage tracking, access control, lifecycle management
- **Data Assimilation**: EKF, UKF, EnKF, Particle Filter for state estimation from observations
- **IDZ Model Parameter Adaptation**: Dynamic parameter updating based on high-fidelity simulation, multi-fidelity fusion
- **Real-time State Evaluation**: Deviation analysis, performance indices, risk assessment, compliance monitoring
- **Real-time State Prediction**: Multi-horizon forecasting, ensemble prediction, scenario analysis, uncertainty quantification

## System Architecture

```
src/
├── Core Simulation
│   ├── simulation.py          # Multi-physics plant model (16 state variables)
│   ├── control.py             # Perception system and autonomous controller
│   ├── mpc_controller.py      # Adaptive MPC with hybrid control
│   ├── sensors.py             # Sensor models with noise and faults
│   ├── actuators.py           # Actuator dynamics and faults
│   └── scenario_generator.py  # Multi-physics scenario generation
│
├── Prediction & Intelligence
│   ├── prediction_planning.py # Weather/flow/seismic forecasting
│   ├── intelligence.py        # ML prediction and anomaly detection
│   ├── ml_control.py          # Deep learning & RL control (V3.8)
│   └── scenario_space.py      # Full scenario space (1.78M+)
│
├── Distributed System (V3.8)
│   ├── cluster.py             # Cluster management, leader election
│   ├── edge_computing.py      # Edge device management
│   └── digital_twin.py        # Digital twin 3D model
│
├── Mobile & Visualization (V3.9)
│   ├── mobile_api.py          # Mobile device API
│   ├── advanced_visualization.py  # Advanced charts & GIS
│   ├── i18n.py                # Internationalization
│   └── blockchain_audit.py    # Blockchain audit trail
│
├── Simulation & State Estimation (V3.10)
│   ├── sensor_simulation.py   # High-fidelity sensor models
│   ├── actuator_simulation.py # High-fidelity actuator models
│   ├── data_governance.py     # Data quality & governance
│   ├── data_assimilation.py   # EKF/UKF/EnKF/PF filters
│   ├── idz_model_adapter.py   # IDZ parameter adaptation
│   ├── state_evaluation.py    # Real-time state evaluation
│   └── state_prediction.py    # Multi-horizon prediction
│
├── Enterprise Features
│   ├── data_persistence.py    # SQLite time series storage
│   ├── safety.py              # Fault diagnosis, interlocks
│   ├── scada_interface.py     # SCADA/OPC-UA integration
│   ├── auth_system.py         # Authentication & RBAC
│   ├── alert_system.py        # Multi-channel notifications
│   ├── report_generator.py    # Automated reports
│   ├── monitoring.py          # Performance metrics
│   └── backup_restore.py      # Backup management
│
├── Web Interface
│   ├── server.py              # Flask REST API (100+ endpoints)
│   ├── websocket_hub.py       # Real-time WebSocket
│   ├── visualization.py       # Dashboard data
│   └── api_docs.py            # OpenAPI/Swagger
│
└── Testing
    ├── test_integration.py    # Integration tests (29 tests)
    ├── test_hil.py            # HIL test framework
    └── hifi_simulation.py     # High-fidelity simulation
```

## V3.8 New Modules

### Cluster Management (`cluster.py`)
| Component | Description |
|-----------|-------------|
| `ClusterManager` | Node registration, leader election (Raft-like), state replication |
| `LoadBalancer` | Round-robin, least-connections, weighted, IP-hash strategies |
| `HealthChecker` | Node health monitoring with history tracking |
| `StateReplicator` | Log-based state replication with snapshots |
| `FailoverManager` | Automatic failover on node/leader failure |

### Edge Computing (`edge_computing.py`)
| Component | Description |
|-----------|-------------|
| `EdgeDeviceManager` | Register, monitor, and command edge devices |
| `EdgeDataProcessor` | Local data processing with alert rules |
| `EdgeSyncManager` | Cloud synchronization with offline storage |
| Device Types | Gateway, Sensor Hub, PLC, RTU, IPC, Smart Sensor |

### Digital Twin (`digital_twin.py`)
| Component | Description |
|-----------|-------------|
| `DigitalTwinModel` | 75 physical components (sections, gates, bearings, joints) |
| `PhysicsEngine` | Hydraulic, thermal, structural calculations |
| `VirtualSensor` | 20+ interpolated sensor points |
| What-If Analysis | Run scenario simulations on digital model |

### AI/ML Control (`ml_control.py`)
| Component | Description |
|-----------|-------------|
| `DeepLearningPredictor` | 5 neural network models (LSTM, GRU, MLP, Autoencoder) |
| `ReinforcementLearningController` | Q-learning based control optimization |
| `AnomalyDetector` | Autoencoder-based anomaly detection |
| `ScenarioClassifier` | Multi-class scenario probability prediction |

## V3.9 New Modules

### Mobile API (`mobile_api.py`)
| Component | Description |
|-----------|-------------|
| `MobileDeviceManager` | Device registration, authentication, management |
| `PushNotificationService` | APNs/FCM push notification support |
| `QRCodeGenerator` | Device binding and data sharing QR codes |
| `OfflineSyncManager` | Offline data storage with background sync |

### Advanced Visualization (`advanced_visualization.py`)
| Component | Description |
|-----------|-------------|
| `ChartDataGenerator` | ECharts-compatible line, gauge, heatmap, radar charts |
| `GISVisualization` | Geographic mapping with aqueduct overlay |
| `DashboardBuilder` | Custom dashboard creation with templates |
| `3DVisualization` | ThreeJS-compatible 3D model data |

### Internationalization (`i18n.py`)
| Component | Description |
|-----------|-------------|
| `TranslationManager` | Key-value translation with fallback |
| `LocaleManager` | 10 locales (zh-CN, en-US, ja-JP, ko-KR, etc.) |
| `DateFormatter` | Locale-aware date/time formatting |
| `NumberFormatter` | Locale-aware number/currency formatting |

### Blockchain Audit (`blockchain_audit.py`)
| Component | Description |
|-----------|-------------|
| `AuditChain` | Immutable blockchain with proof-of-work |
| `DataVersionManager` | Entity version tracking with change history |
| `MerkleTree` | Efficient verification of audit data |
| `ComplianceReporter` | Access, change, security, control reports |

## V3.10 New Modules

### Sensor Simulation (`sensor_simulation.py`)
| Component | Description |
|-----------|-------------|
| `HighFidelitySensor` | Physics-based sensor model with 12 sensor types |
| `SensorPhysicsModel` | Measurement dynamics, noise, environmental effects |
| `SensorNetwork` | Cross-correlation, fusion, redundancy management |
| `SensorDegradation` | Linear drift, exponential decay, fouling, corrosion |
| `SensorSimulationEngine` | Complete sensor network simulation |

### Actuator Simulation (`actuator_simulation.py`)
| Component | Description |
|-----------|-------------|
| `HighFidelityActuator` | 2nd-order dynamics with friction modeling |
| `GateActuatorSystem` | Specialized gate with hydraulic force calculation |
| `WearModel` | Linear, bathtub curve, fatigue-based degradation |
| `ActuatorFailureMode` | Stuck, slow, partial, oscillation, runaway modes |
| `ActuatorSimulationEngine` | Complete actuator system simulation |

### Data Governance (`data_governance.py`)
| Component | Description |
|-----------|-------------|
| `DataQualityValidator` | ISO 8000 data quality dimensions |
| `ValidationRule` | Range, null, type, temporal, statistical checks |
| `DataLineageTracker` | Data provenance and transformation tracking |
| `DataAccessController` | RBAC access control with audit logging |
| `DataLifecycleManager` | Retention policies and data archival |

### Data Assimilation (`data_assimilation.py`)
| Component | Description |
|-----------|-------------|
| `ExtendedKalmanFilter` | EKF for nonlinear state estimation |
| `UnscentedKalmanFilter` | UKF with sigma point propagation |
| `EnsembleKalmanFilter` | EnKF for high-dimensional systems |
| `ParticleFilter` | Sequential Monte Carlo for non-Gaussian systems |
| `DataAssimilationEngine` | Unified interface for all assimilation methods |

### IDZ Model Adapter (`idz_model_adapter.py`)
| Component | Description |
|-----------|-------------|
| `IDZModel` | Simplified control-oriented aqueduct model |
| `RecursiveLeastSquares` | Online parameter identification |
| `IDZModelAdapter` | Dynamic parameter updating from hi-fi model |
| `MultiFidelityModelManager` | Model fusion and weight adaptation |

### State Evaluation (`state_evaluation.py`)
| Component | Description |
|-----------|-------------|
| `StateEvaluator` | Control target deviation analysis |
| `PerformanceIndex` | Flow efficiency, Froude stability, structural health |
| `RiskAssessment` | Hydraulic jump, thermal cracking, joint failure risks |
| `ComplianceChecker` | Operational constraint monitoring |
| `MultiObjectiveEvaluator` | Pareto-based multi-objective evaluation |

### State Prediction (`state_prediction.py`)
| Component | Description |
|-----------|-------------|
| `PhysicsPredictor` | Physics-based state prediction |
| `StatisticalPredictor` | Exponential smoothing, ARMA forecasting |
| `EnsemblePredictor` | Monte Carlo ensemble prediction |
| `StatePredictionEngine` | Multi-horizon, multi-method prediction |
| `ScenarioPrediction` | What-if scenario analysis |

### V3.10 Enhanced Features (Phase 1-3)

#### Real-time Data Interface (`realtime_data_interface.py`)
| Component | Description |
|-----------|-------------|
| `RealtimeDataManager` | Multi-protocol real-time data management |
| `OPCUAAdapter` | OPC-UA protocol adapter with browsing and subscription |
| `ModbusAdapter` | Modbus TCP/RTU protocol with batch read/write |
| `MQTTAdapter` | MQTT protocol with topic subscription |
| `DataBuffer` | Circular buffer with statistics and quality tracking |

#### Alarm Event Management (`alarm_event_management.py`)
| Component | Description |
|-----------|-------------|
| `AlarmManager` | Multi-level alarm management system |
| `AlarmRule` | Configurable alarm rules with conditions |
| `EventCorrelator` | Event correlation and pattern detection |
| `NotificationDispatcher` | Email, SMS, Webhook notification support |

#### Reporting Visualization (`reporting_visualization.py`)
| Component | Description |
|-----------|-------------|
| `ReportingManager` | Comprehensive reporting system |
| `DataAggregator` | Time-series data aggregation (hour, day, week, month) |
| `ChartGenerator` | ECharts-compatible chart generation |
| `ReportExporter` | Export to PDF, Excel, CSV, HTML |
| `DashboardManager` | Customizable dashboard management |

#### Knowledge Graph (`knowledge_graph.py`)
| Component | Description |
|-----------|-------------|
| `KnowledgeGraphManager` | Graph-based knowledge management |
| `KnowledgeGraph` | Entity-relationship graph operations |
| `InferenceRule` | Rule-based inference engine |
| `AqueductKnowledgeBuilder` | Domain-specific knowledge builder |

#### AIOps (`aiops.py`)
| Component | Description |
|-----------|-------------|
| `AIOpsManager` | Intelligent operations management |
| `AnomalyDetector` | Statistical anomaly detection (spike, drop, trend, oscillation) |
| `IntelligentDiagnostics` | Root cause analysis and diagnosis |
| `PredictiveMaintenance` | Equipment health and maintenance prediction |
| `AutoRemediation` | Automated remediation actions |

## API Endpoints (230+)

### Core APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current simulation state |
| `/api/health` | GET | System health check |
| `/api/history` | GET | Historical state data |
| `/api/scenario` | POST | Inject scenario |
| `/api/control` | POST | Set control mode/parameters |
| `/api/version` | GET | System version info |

### V3.8 Cluster APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cluster` | GET | Cluster status |
| `/api/cluster/nodes` | GET | List all nodes |
| `/api/cluster/nodes/<id>` | GET | Node details |
| `/api/cluster/register` | POST | Register new node |
| `/api/cluster/load-balancer/strategy` | POST | Set LB strategy |

### V3.8 Edge APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/edge` | GET | Edge system status |
| `/api/edge/devices` | GET | List edge devices |
| `/api/edge/devices/<id>/data` | POST | Process device data |
| `/api/edge/devices/<id>/command` | POST | Send command |
| `/api/edge/sync` | GET | Sync status |

### V3.8 Digital Twin APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/twin` | GET | Twin status |
| `/api/twin/model` | GET | 3D model data |
| `/api/twin/components` | GET | List components |
| `/api/twin/virtual-sensors/readings` | GET | Virtual sensor data |
| `/api/twin/scenario` | POST | Run what-if analysis |
| `/api/twin/export` | GET | Export model JSON |

### V3.8 ML APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml` | GET | ML system status |
| `/api/ml/models` | GET | List ML models |
| `/api/ml/predict/<model>` | GET | Get prediction |
| `/api/ml/anomaly` | GET | Anomaly detection |
| `/api/ml/train` | POST | Train models |
| `/api/ml/rl` | GET | RL controller status |
| `/api/ml/rl/action` | GET | Get RL action |

### V3.9 Mobile APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mobile` | GET | Mobile API status |
| `/api/mobile/devices` | GET | List mobile devices |
| `/api/mobile/register` | POST | Register device |
| `/api/mobile/notifications` | POST | Send notification |
| `/api/mobile/qr-code` | GET | Generate QR code |
| `/api/mobile/sync/data` | POST | Sync device data |
| `/api/mobile/dashboard` | GET | Mobile dashboard data |

### V3.9 Visualization APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/viz` | GET | Visualization status |
| `/api/viz/charts/<type>` | GET | Chart data by type |
| `/api/viz/gis` | GET | GIS map data |
| `/api/viz/gis/layers` | GET | Available map layers |
| `/api/viz/dashboard-builder/dashboards` | GET/POST | Custom dashboards |
| `/api/viz/3d-model` | GET | 3D model data |
| `/api/viz/heatmap` | GET | Heatmap data |

### V3.9 i18n APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/i18n` | GET | i18n status |
| `/api/i18n/locales` | GET | Supported locales |
| `/api/i18n/locale` | PUT | Set locale |
| `/api/i18n/translate` | GET | Translate text |
| `/api/i18n/translations/<locale>` | GET | All translations |
| `/api/i18n/format/date` | GET | Format date |
| `/api/i18n/format/number` | GET | Format number |

### V3.9 Blockchain Audit APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/blockchain` | GET | Audit status |
| `/api/blockchain/verify` | GET | Verify chain integrity |
| `/api/blockchain/stats` | GET | Chain statistics |
| `/api/blockchain/events` | GET | Query audit events |
| `/api/blockchain/versions/<type>/<id>` | GET | Version history |
| `/api/blockchain/reports/compliance` | GET | Compliance report |
| `/api/blockchain/export` | GET | Export audit chain |

### V3.10 Sensor Simulation APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/sensor/measure` | POST | Execute sensor measurement |
| `/api/v310/sensor/status` | GET | Sensor simulation status |
| `/api/v310/sensor/inject_fault` | POST | Inject sensor fault |
| `/api/v310/sensor/clear_faults` | POST | Clear all faults |

### V3.10 Actuator Simulation APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/actuator/command` | POST | Send actuator command |
| `/api/v310/actuator/step` | POST | Execute simulation step |
| `/api/v310/actuator/status` | GET | Actuator system status |
| `/api/v310/actuator/inject_failure` | POST | Inject actuator failure |
| `/api/v310/actuator/emergency_shutdown` | POST | Emergency shutdown |

### V3.10 Data Governance APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/governance/process` | POST | Process data through governance |
| `/api/v310/governance/dashboard` | GET | Governance dashboard |
| `/api/v310/governance/compliance_report` | GET | Compliance report |
| `/api/v310/governance/quality_report` | GET | Data quality report |

### V3.10 Data Assimilation APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/assimilation/initialize` | POST | Initialize assimilation |
| `/api/v310/assimilation/predict` | POST | Prediction step |
| `/api/v310/assimilation/assimilate` | POST | Assimilate observations |
| `/api/v310/assimilation/state` | GET | Current state estimate |
| `/api/v310/assimilation/status` | GET | Assimilation status |
| `/api/v310/assimilation/switch_method` | POST | Switch assimilation method |

### V3.10 IDZ Model APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/idz/update` | POST | Update IDZ from hi-fi model |
| `/api/v310/idz/parameters` | GET | Current IDZ parameters |
| `/api/v310/idz/predict` | POST | IDZ model prediction |
| `/api/v310/idz/metrics` | GET | Adaptation metrics |
| `/api/v310/idz/uncertainty` | GET | Model uncertainty |
| `/api/v310/multifidelity/update` | POST | Multi-fidelity update |
| `/api/v310/multifidelity/status` | GET | Multi-fidelity status |

### V3.10 State Evaluation APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/evaluation/evaluate` | POST | Comprehensive evaluation |
| `/api/v310/evaluation/deviation` | POST | Deviation from targets |
| `/api/v310/evaluation/performance` | POST | Performance indices |
| `/api/v310/evaluation/risk` | POST | Risk assessment |
| `/api/v310/evaluation/compliance` | POST | Compliance check |
| `/api/v310/evaluation/trend` | GET | Evaluation trend |
| `/api/v310/evaluation/multiobjective` | POST | Multi-objective evaluation |

### V3.10 State Prediction APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/prediction/predict` | POST | State prediction |
| `/api/v310/prediction/risk` | POST | Risk probability prediction |
| `/api/v310/prediction/scenario` | POST | Scenario prediction |
| `/api/v310/prediction/scenarios/compare` | POST | Compare scenarios |
| `/api/v310/prediction/accuracy` | GET | Prediction accuracy |
| `/api/v310/prediction/status` | GET | Prediction engine status |

### V3.10 Real-time Data APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/realtime/status` | GET | Real-time data system status |
| `/api/v310/realtime/connections` | GET | List all connections |
| `/api/v310/realtime/data/<name>` | GET | Get data from connection |
| `/api/v310/realtime/subscribe` | POST | Subscribe to data tags |

### V3.10 Alarm APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/alarm/status` | GET | Alarm system status |
| `/api/v310/alarm/active` | GET | Active alarms |
| `/api/v310/alarm/trigger` | POST | Trigger alarm |
| `/api/v310/alarm/acknowledge` | POST | Acknowledge alarm |
| `/api/v310/alarm/history` | GET | Alarm history |
| `/api/v310/event/log` | POST | Log event |
| `/api/v310/event/query` | GET | Query events |

### V3.10 Report APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/report/status` | GET | Report system status |
| `/api/v310/report/generate` | POST | Generate report |
| `/api/v310/report/templates` | GET | Report templates |
| `/api/v310/report/chart` | POST | Generate chart |
| `/api/v310/report/dashboard` | GET | Get dashboard |
| `/api/v310/report/export` | POST | Export report |

### V3.10 Knowledge Graph APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/kg/status` | GET | Knowledge graph status |
| `/api/v310/kg/entities` | GET | List entities |
| `/api/v310/kg/entity/<id>` | GET | Get entity details |
| `/api/v310/kg/entity` | POST | Create entity |
| `/api/v310/kg/relation` | POST | Create relation |
| `/api/v310/kg/query` | POST | Query knowledge graph |
| `/api/v310/kg/path` | POST | Find path between entities |
| `/api/v310/kg/inference` | POST | Run inference |

### V3.10 AIOps APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v310/aiops/status` | GET | AIOps system status |
| `/api/v310/aiops/anomalies` | GET | Active anomalies |
| `/api/v310/aiops/process_metric` | POST | Process metric data |
| `/api/v310/aiops/diagnose` | POST | Run diagnosis |
| `/api/v310/aiops/maintenance/predict` | POST | Predict maintenance |
| `/api/v310/aiops/remediate` | POST | Get remediation actions |
| `/api/v310/aiops/resolve` | POST | Resolve anomaly |
| `/api/v310/aiops/health/<id>` | GET | Entity health status |

## Installation & Usage

### Requirements
```bash
pip install flask numpy pyyaml
```

### Run Server
```bash
cd src && python server.py
# Access: http://localhost:5000
# API Docs: http://localhost:5000/api/docs
```

### Docker Deployment
```bash
docker-compose up -d
```

### Run Tests
```bash
# Integration tests (29 tests)
cd src && python -m unittest test_integration -v

# HIL tests (11 tests)
cd src && python test_hil.py

# Test V3.8 modules
cd src && python cluster.py
cd src && python edge_computing.py
cd src && python digital_twin.py
cd src && python ml_control.py

# Test V3.9 modules
cd src && python mobile_api.py
cd src && python advanced_visualization.py
cd src && python i18n.py
cd src && python blockchain_audit.py

# Test V3.10 modules
cd src && python sensor_simulation.py
cd src && python actuator_simulation.py
cd src && python data_governance.py
cd src && python data_assimilation.py
cd src && python idz_model_adapter.py
cd src && python state_evaluation.py
cd src && python state_prediction.py
```

## Module Statistics

| Category | Count |
|----------|-------|
| Total Python Modules | 45 |
| API Endpoints | 230+ |
| Scenario Coverage | 1,778,880 |
| Digital Twin Components | 75 |
| Virtual Sensors | 20+ |
| ML Models | 5 |
| Edge Device Types | 6 |
| Supported Languages | 10 |
| Integration Tests | 223+ |

## Version History

| Version | Features |
|---------|----------|
| V3.0 | HIL simulation, sensor/actuator models, adaptive MPC |
| V3.1 | Full-scenario autonomous operation |
| V3.2 | Full scenario space (1.78M+ scenarios) |
| V3.3 | Prediction & planning integration |
| V3.4 | Data persistence, intelligence, safety, SCADA |
| V3.5 | Configuration management, logging, API docs |
| V3.6 | WebSocket, alerts, reports, authentication |
| V3.7 | Docker, monitoring, backup/restore |
| V3.8 | Distributed cluster, edge computing, digital twin, AI/ML |
| V3.9 | Mobile API, advanced visualization, i18n, blockchain audit |
| V3.10 | Sensor/actuator simulation, data governance, data assimilation, state evaluation/prediction, real-time data interface, alarm management, knowledge graph, AIOps |

## License
This project is part of the South-to-North Water Diversion Project infrastructure management system.
