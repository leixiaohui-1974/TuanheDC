# Tuanhe Aqueduct Autonomous Operation System (TAOS) V3.9

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

### V3.9 Mobile & Visualization Enhancement (NEW)
- **Mobile API**: Device registration, push notifications, QR codes, offline sync
- **Advanced Visualization**: ECharts-compatible charts, GIS mapping, dashboard builder
- **Internationalization (i18n)**: 10 languages, locale-aware formatting
- **Blockchain Audit**: Immutable audit trail, data versioning, compliance reports

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

## API Endpoints (150+)

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
```

## Module Statistics

| Category | Count |
|----------|-------|
| Total Python Modules | 32 |
| API Endpoints | 150+ |
| Scenario Coverage | 1,778,880 |
| Digital Twin Components | 75 |
| Virtual Sensors | 20+ |
| ML Models | 5 |
| Edge Device Types | 6 |
| Supported Languages | 10 |
| Integration Tests | 29 |

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

## License
This project is part of the South-to-North Water Diversion Project infrastructure management system.
