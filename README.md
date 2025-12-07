# Tuanhe Aqueduct Autonomous Operation System (TAOS) V3.0

## Overview
This project implements a comprehensive **Autonomous Operation & Health Management System** for the Tuanhe Aqueduct (South-to-North Water Diversion Project). The system addresses "Wind-Water-Thermal-Structure" multi-physics coupling risks through intelligent perception, adaptive control, and comprehensive simulation.

## Key Features

### Core Capabilities
- **High-Fidelity Multi-Physics Simulation**: Models fluid dynamics, thermal effects, structural behavior, and environmental conditions
- **Comprehensive Scenario Coverage**: 14+ scenarios including hydraulic jumps, thermal bending, bearing lock, joint tearing, and seismic events
- **Adaptive MPC Control**: Model Predictive Control with scenario-based gain scheduling and PID fallback
- **Full Hardware-in-the-Loop Testing**: Sensor and actuator models with fault injection capability
- **Real-time Web Dashboard**: Live visualization and scenario injection interface

### V3.0 Enhanced Features
- **Sensor Models**: Noise, delay, drift, redundancy, Kalman filtering
- **Actuator Models**: Response dynamics, rate limits, saturation, fault simulation
- **Adaptive MPC**: Prediction horizon optimization with scenario-specific gains
- **Scenario Generator**: Multi-physics coupling, time-varying environments, probabilistic sequences
- **Closed-Loop Simulation**: Complete plant model integration for HIL testing

## System Architecture

```
src/
├── simulation.py          # Multi-physics plant model (16 state variables)
├── control.py             # Perception system and autonomous controller
├── mpc_controller.py      # Adaptive MPC with hybrid control
├── sensors.py             # Sensor models with noise and faults
├── actuators.py           # Actuator dynamics and faults
├── scenario_generator.py  # Multi-physics scenario generation
├── hifi_simulation.py     # High-fidelity closed-loop simulation
├── server.py              # Flask REST API backend
├── test_integration.py    # Integration tests (29 tests)
├── test_hil.py            # HIL test framework (11 tests)
└── templates/
    └── index.html         # Real-time dashboard
```

## Component Overview

| Module | Purpose |
|--------|---------|
| `AqueductSimulation` | 16-variable physics model with thermal, hydraulic, structural dynamics |
| `PerceptionSystem` | Multi-physics risk detection with configurable thresholds |
| `AutonomousController` | PID control with scenario-specific response strategies |
| `AdaptiveMPC` | Model Predictive Control with adaptive gain scheduling |
| `SensorSuite` | Redundant sensors with Kalman filtering and fault injection |
| `ActuatorSuite` | Gate/valve models with dynamics and safety interlocks |
| `ScenarioGenerator` | Time-varying environment and multi-physics scenario generation |
| `ClosedLoopSimulation` | Complete plant-controller integration for HIL testing |

## Scenario Coverage Matrix

| ID | Scenario | Trigger | Controller Response |
|----|----------|---------|---------------------|
| S1.1 | Hydraulic Jump | Fr > 0.9 | Raise water level to 7.0m |
| S1.2 | Surge Wave | Transient flow | Adaptive flow control |
| S2.1 | Vortex-Induced Vibration | Wind > 12 m/s | Monitor and alert |
| S3.1 | Thermal Bending | ΔT > 10°C | Increase flow for cooling |
| S3.2 | Rapid Cooling | Temp drop | Thermal management |
| S3.3 | Bearing Lock | Locked bearing | Reduce water level to 3.0m |
| S4.1 | Joint Gap Expansion | Cold weather | Increase water level to 5.0m |
| S4.2 | Joint Compression | Hot weather | Temperature monitoring |
| S5.1 | Seismic Event | Ground accel | Emergency response |
| S5.2 | Aftershock Sequence | Multiple events | Sequential response |
| S6.1 | Sensor Degradation | Fault injection | Redundancy switching |
| S6.2 | Actuator Fault | Fault injection | Graceful degradation |
| COMBINED | Thermal + Seismic | Multi-physics | EMERGENCY: Dump water |
| MULTI_PHYSICS | Wind + Thermal + Hydraulic | Coupling | Adaptive priority |

## Control Modes

1. **AUTO**: Autonomous MPC/PID with scenario-specific overrides
2. **MANUAL**: Direct control via API
3. **EMERGENCY**: Automatic activation for critical scenarios (Q_out=200, Q_in=0)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current simulation state |
| `/api/health` | GET | System health check |
| `/api/history` | GET | Historical state data |
| `/api/stats` | GET | Statistical summary |
| `/api/scenario` | POST | Inject scenario |
| `/api/control` | POST | Set control mode/parameters |
| `/api/simulation/pause` | POST | Pause simulation |
| `/api/simulation/resume` | POST | Resume simulation |
| `/api/simulation/reset` | POST | Reset to initial state |

## Installation & Usage

### Requirements
```bash
pip install flask numpy
```

### Run Server
```bash
python src/server.py
# Access: http://localhost:5000
```

### Run Tests
```bash
# Integration tests (29 tests)
cd src && python -m unittest test_integration -v

# HIL tests (11 tests)
cd src && python test_hil.py

# Full system HIL tests
cd src && python hifi_simulation.py
```

## Test Coverage

### Integration Tests (29 tests)
- Unit tests for simulation, perception, and controller
- All scenario responses (S1.1, S3.1, S3.3, S4.1, S5.1)
- Multi-physics coupling scenarios
- Long-duration stability tests
- Boundary condition tests
- Full scenario autonomy validation

### HIL Tests (11+ tests)
- Full scenario simulation with metrics
- Stress tests (rapid switching, extreme conditions)
- Autonomy validation (detection, emergency, recovery)
- Sensor/actuator fault injection
- Closed-loop performance verification

## Sensor Model Features

- **Noise Models**: Gaussian noise, bias, drift
- **Time Delays**: Configurable measurement delays
- **Redundancy**: Triple-redundant critical sensors with voting
- **Kalman Filtering**: State estimation and smoothing
- **Fault Injection**: Stuck, drift, noise increase, intermittent, complete failure

## Actuator Model Features

- **Response Dynamics**: First-order response with time constants
- **Rate Limiting**: Maximum rate of change constraints
- **Saturation**: Physical limits on positions and flows
- **Gate Characteristics**: Non-linear flow vs. position
- **Fault Injection**: Stuck, slow, partial failure, oscillation

## MPC Controller Features

- **Prediction Horizon**: Configurable Np steps
- **Control Horizon**: Configurable Nc steps
- **Adaptive Gains**: Scenario-based weight scheduling
- **Constraint Handling**: Flow limits, rate limits
- **PID Fallback**: Automatic fallback when MPC fails
- **Performance Tracking**: Solve count, fallback rate

## License
This project is part of the South-to-North Water Diversion Project infrastructure management system.
