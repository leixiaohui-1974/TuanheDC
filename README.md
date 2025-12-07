# Tuanhe Aqueduct Autonomous Operation System (TAOS) V2.1

## Overview
This project implements a comprehensive Autonomous Operation & Health Management System for the Tuanhe Aqueduct (South-to-North Water Diversion Project). It addresses the "Wind-Water-Thermal-Structure" multi-physics coupling risks through intelligent perception, autonomous control, and real-time monitoring.

## Key Features

### Core Capabilities
- **High-Fidelity Simulation**: Models fluid dynamics (surges, sloshing), aerodynamics (VIV), thermal effects (sunlight, water cooling), and structural behavior (joints, bearings).
- **Scenario Atlas**: Covers all critical scenarios including:
  - S1.1: Hydraulic Jump / Flow Instability
  - S3.1: Thermal Bending (Sun/Shade differential)
  - S3.3: Bearing Lock
  - S4.1: Joint Tearing (Cold expansion)
  - S5.1: Seismic Activity
- **Autonomous Control**: PID-based controller with scenario-specific overrides for thermal cooling, Froude number control, and emergency response.
- **Web Interface**: Real-time visualization of the aqueduct state, sensor data, and control actions.

### Enhanced Features (V2.1)
- **Multi-Mode Control**: AUTO, MANUAL, and EMERGENCY operation modes
- **Comprehensive Risk Detection**: Configurable thresholds for all risk types
- **State History Recording**: Track up to 1000 historical states for analysis
- **Health Monitoring API**: Real-time system health and statistics endpoints
- **Full Scenario Coverage**: Autonomous handling of all 6 scenario types

## Architecture

```
src/
├── simulation.py      # Multi-physics simulation engine
├── control.py         # Perception system and autonomous controller
├── server.py          # Flask REST API backend
├── test_integration.py # Comprehensive integration tests (29 tests)
├── test_hil.py        # Hardware-in-the-Loop test framework (11 tests)
└── templates/
    └── index.html     # Real-time dashboard frontend
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| `AqueductSimulation` | 16-variable physics model with thermal, hydraulic, and structural dynamics |
| `PerceptionSystem` | Multi-physics risk detection with configurable thresholds |
| `AutonomousController` | PID control with scenario-specific response strategies |
| `Flask Server` | REST API with health checks, history, and control endpoints |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current simulation state with control status |
| `/api/health` | GET | System health check and metrics |
| `/api/history` | GET | Historical state data (limit via query param) |
| `/api/stats` | GET | Statistical summary of simulation |
| `/api/scenario` | POST | Inject a scenario for testing |
| `/api/control` | POST | Set control parameters or mode |
| `/api/simulation/pause` | POST | Pause simulation |
| `/api/simulation/resume` | POST | Resume simulation |
| `/api/simulation/reset` | POST | Reset to initial state |

## Running the System

### Installation
```bash
pip install -r requirements.txt
```

### Start the Server
```bash
python src/server.py
```
Access via browser at `http://localhost:5000`

### Run Tests
```bash
# Integration tests (29 tests)
cd src && python -m unittest test_integration -v

# Hardware-in-the-Loop tests (11 tests)
cd src && python test_hil.py
```

## Test Coverage

### Integration Tests (29 tests)
- Unit tests for simulation, perception, and controller
- Scenario-specific response tests (S1.1, S3.1, S3.3, S4.1, S5.1)
- Multi-physics coupling tests
- Long-duration stability tests
- Boundary condition tests
- Full scenario autonomy validation

### HIL Tests (11 tests)
- Full scenario simulation with metrics
- Stress tests (rapid switching, extreme conditions)
- Autonomy validation (detection, emergency, recovery)

## Scenario Response Matrix

| Scenario | Trigger Condition | Controller Response |
|----------|------------------|---------------------|
| S1.1 | Fr > 0.9 | Raise water level to 7.0m |
| S3.1 | ΔT > 10°C | Increase flow for cooling |
| S3.3 | Bearing locked | Reduce water level to 3.0m |
| S4.1 | Joint gap > 25mm | Increase water level to 5.0m |
| S5.1 | Vibration > 50mm | Reduce flow, prepare emergency |
| S5.1 + S3.3 | Combined | EMERGENCY: Dump water |

## Control Modes

1. **AUTO**: Autonomous operation with PID level control and scenario responses
2. **MANUAL**: Direct control of Q_in and Q_out via API
3. **EMERGENCY**: Automatic activation during critical combined scenarios

## License
This project is part of the South-to-North Water Diversion Project infrastructure management system.
