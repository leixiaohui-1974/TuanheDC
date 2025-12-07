# Tuanhe Aqueduct Autonomous Operation System (TAOS) V3.8

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

### V3.8 Distributed & Intelligent Features (NEW)
- **Distributed Cluster**: Multi-node deployment with leader election and failover
- **Edge Computing**: Edge device management, local processing, cloud sync
- **Digital Twin**: High-fidelity 3D model with 75 components, virtual sensors
- **AI/ML Control**: Deep learning prediction, reinforcement learning control

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

## API Endpoints (100+)

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
```

## Module Statistics

| Category | Count |
|----------|-------|
| Total Python Modules | 28 |
| API Endpoints | 100+ |
| Scenario Coverage | 1,778,880 |
| Digital Twin Components | 75 |
| Virtual Sensors | 20+ |
| ML Models | 5 |
| Edge Device Types | 6 |
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

## License
This project is part of the South-to-North Water Diversion Project infrastructure management system.
