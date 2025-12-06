# Tuanhe Aqueduct Autonomous Operation System (TAOS)

## Overview
This project implements a comprehensive Autonomous Operation & Health Management System for the Tuanhe Aqueduct (South-to-North Water Diversion Project). It addresses the "Wind-Water-Thermal-Structure" multi-physics coupling risks.

## Key Features
- **High-Fidelity Simulation**: Models fluid dynamics (surges, sloshing), aerodynamics (VIV), thermal effects (sunlight, water cooling), and structural behavior (joints, bearings).
- **Scenario Atlas**: Covers 14 scenarios including Hydraulic Jumps (S1.1), Thermal Bending (S3.1), and Joint Tearing (S4.1).
- **Autonomous Control**: MPC-based controller with thermal and joint constraints, capable of "Water Cooling" scheduling and Froude number control.
- **Web Interface**: Real-time visualization of the aqueduct state, sensor data, and control actions.

## Architecture
- `src/simulation.py`: The physics engine.
- `src/control.py`: Perception and Control algorithms (MPC/PID).
- `src/server.py`: Flask backend.
- `src/templates/`: Frontend dashboard.

## Running the System
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python src/server.py`
3. Access via browser at `http://localhost:5000`
