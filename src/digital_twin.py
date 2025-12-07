#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.8 - Digital Twin Enhancement
团河渡槽自主运行系统 - 数字孪生增强模块

Features:
- High-fidelity 3D model representation
- Real-time state synchronization
- Physics-based simulation
- Predictive modeling
- What-if scenario analysis
- Virtual sensor placement
- Historical playback
- Multi-resolution modeling
"""

import time
import json
import math
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from collections import deque
from pathlib import Path
import sqlite3


class TwinSyncState(Enum):
    """Digital twin synchronization state"""
    SYNCHRONIZED = "synchronized"
    SYNCING = "syncing"
    DELAYED = "delayed"
    DISCONNECTED = "disconnected"
    SIMULATION = "simulation"


class ModelResolution(Enum):
    """Model fidelity levels"""
    LOW = "low"           # Simplified lumped model
    MEDIUM = "medium"     # Section-averaged model
    HIGH = "high"         # Full 3D discretized model
    ULTRA = "ultra"       # CFD-level detail


class ComponentType(Enum):
    """Physical component types"""
    AQUEDUCT_SECTION = "aqueduct_section"
    GATE = "gate"
    SENSOR = "sensor"
    BEARING = "bearing"
    JOINT = "joint"
    PIER = "pier"
    FOUNDATION = "foundation"
    PUMP_STATION = "pump_station"


@dataclass
class Vector3D:
    """3D vector representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class BoundingBox:
    """3D bounding box"""
    min_point: Vector3D
    max_point: Vector3D

    def to_dict(self) -> Dict[str, Any]:
        return {
            'min': self.min_point.to_dict(),
            'max': self.max_point.to_dict()
        }

    def center(self) -> Vector3D:
        return Vector3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )

    def dimensions(self) -> Vector3D:
        return Vector3D(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )


@dataclass
class PhysicalComponent:
    """Physical component in the digital twin"""
    component_id: str
    component_type: ComponentType
    name: str
    position: Vector3D
    rotation: Vector3D
    scale: Vector3D
    bounding_box: BoundingBox
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    sensors: List[str] = field(default_factory=list)
    material: str = "concrete"
    health_score: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'name': self.name,
            'position': self.position.to_dict(),
            'rotation': self.rotation.to_dict(),
            'scale': self.scale.to_dict(),
            'bounding_box': self.bounding_box.to_dict(),
            'parent_id': self.parent_id,
            'children': self.children,
            'properties': self.properties,
            'state': self.state,
            'sensors': self.sensors,
            'material': self.material,
            'health_score': self.health_score,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class VirtualSensor:
    """Virtual sensor placement"""
    sensor_id: str
    name: str
    sensor_type: str
    position: Vector3D
    attached_to: str  # Component ID
    measurement_type: str
    unit: str
    current_value: float = 0.0
    confidence: float = 1.0
    interpolation_method: str = "linear"
    source_sensors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sensor_id': self.sensor_id,
            'name': self.name,
            'sensor_type': self.sensor_type,
            'position': self.position.to_dict(),
            'attached_to': self.attached_to,
            'measurement_type': self.measurement_type,
            'unit': self.unit,
            'current_value': self.current_value,
            'confidence': self.confidence,
            'interpolation_method': self.interpolation_method,
            'source_sensors': self.source_sensors
        }


@dataclass
class SimulationState:
    """Complete simulation state snapshot"""
    timestamp: datetime
    time_step: float
    water_level: float
    flow_velocity: float
    flow_rate_in: float
    flow_rate_out: float
    temperature_sun: float
    temperature_shade: float
    thermal_gradient: float
    thermal_bending: float
    vibration_amplitude: float
    joint_gap: float
    bearing_stress: float
    bearing_locked: bool
    ground_acceleration: float
    froude_number: float
    reynolds_number: float
    water_volume: float
    energy_dissipation: float
    active_scenarios: List[str] = field(default_factory=list)
    control_mode: str = "AUTO"
    risk_level: str = "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhysicsEngine:
    """
    Physics simulation engine for digital twin
    """

    def __init__(self):
        # Physical constants
        self.g = 9.81  # Gravity (m/s^2)
        self.rho = 1000  # Water density (kg/m^3)
        self.nu = 1e-6  # Kinematic viscosity (m^2/s)

        # Aqueduct parameters (Tuanhe)
        self.length = 1292.0  # Total length (m)
        self.width = 7.0  # Channel width (m)
        self.depth = 7.45  # Design depth (m)
        self.slope = 0.00015  # Bed slope

        # Thermal properties
        self.alpha_concrete = 1e-5  # Thermal expansion coefficient
        self.conductivity = 1.7  # Thermal conductivity (W/mK)

        # Structural properties
        self.E_concrete = 30e9  # Elastic modulus (Pa)
        self.density_concrete = 2400  # Concrete density (kg/m^3)

    def compute_hydraulics(self, h: float, Q_in: float, Q_out: float,
                          dt: float) -> Dict[str, float]:
        """Compute hydraulic state"""
        # Cross-sectional area
        A = self.width * h

        # Flow velocity
        v = (Q_in + Q_out) / 2 / A if A > 0 else 0

        # Froude number
        Fr = v / math.sqrt(self.g * h) if h > 0 else 0

        # Reynolds number
        R_h = A / (self.width + 2 * h)  # Hydraulic radius
        Re = v * 4 * R_h / self.nu if R_h > 0 else 0

        # Manning's equation for friction
        n = 0.013  # Manning's coefficient for concrete
        S_f = (n * v / R_h**(2/3))**2 if R_h > 0 else 0

        # Water volume
        volume = A * self.length

        # Energy dissipation rate
        E_diss = self.rho * self.g * (Q_in - Q_out) * h if h > 0 else 0

        return {
            'velocity': v,
            'froude_number': Fr,
            'reynolds_number': Re,
            'hydraulic_radius': R_h,
            'friction_slope': S_f,
            'volume': volume,
            'energy_dissipation': E_diss,
            'cross_area': A
        }

    def compute_thermal(self, T_sun: float, T_shade: float,
                       current_bending: float, dt: float) -> Dict[str, float]:
        """Compute thermal state"""
        # Temperature gradient
        delta_T = T_sun - T_shade

        # Thermal bending (simplified beam theory)
        # curvature = alpha * delta_T / thickness
        thickness = 0.5  # Wall thickness (m)
        curvature = self.alpha_concrete * delta_T / thickness

        # Thermal stress
        sigma_thermal = self.E_concrete * self.alpha_concrete * delta_T

        # Thermal moment
        I = self.width * thickness**3 / 12  # Second moment of area
        M_thermal = sigma_thermal * I / thickness

        # Update bending (first-order response)
        tau_thermal = 3600  # Time constant (1 hour)
        target_bending = curvature * 1000  # mm
        new_bending = current_bending + (target_bending - current_bending) * dt / tau_thermal

        return {
            'gradient': delta_T,
            'curvature': curvature,
            'thermal_stress': sigma_thermal,
            'thermal_moment': M_thermal,
            'bending': new_bending
        }

    def compute_structural(self, h: float, vibration: float, bearing_stress: float,
                          joint_gap: float, ground_accel: float) -> Dict[str, float]:
        """Compute structural state"""
        # Hydrostatic pressure
        p_hydro = self.rho * self.g * h

        # Wall loading (simplified)
        F_water = 0.5 * p_hydro * h * self.width

        # Natural frequency (simplified beam)
        m_per_length = self.density_concrete * self.width * 0.5 + self.rho * self.width * h
        k_effective = self.E_concrete * self.width * 0.5**3 / (12 * 10**3)
        f_natural = 1 / (2 * math.pi) * math.sqrt(k_effective / m_per_length) if m_per_length > 0 else 1

        # Seismic response
        Sa = ground_accel * 2.5  # Spectral acceleration (simplified)
        F_seismic = Sa * m_per_length * 10  # Seismic force per unit length

        # Safety factors
        SF_bearing = 50.0 / bearing_stress if bearing_stress > 0 else 10
        SF_joint = 30.0 / abs(joint_gap - 20) if abs(joint_gap - 20) > 1 else 10

        return {
            'hydrostatic_pressure': p_hydro,
            'water_force': F_water,
            'natural_frequency': f_natural,
            'spectral_acceleration': Sa,
            'seismic_force': F_seismic,
            'bearing_safety_factor': SF_bearing,
            'joint_safety_factor': SF_joint
        }


class DigitalTwinModel:
    """
    Digital Twin model of the Tuanhe Aqueduct
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "twin"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Model parameters
        self.model_id = "tuanhe-aqueduct-twin"
        self.version = "3.8.0"
        self.resolution = ModelResolution.HIGH
        self.sync_state = TwinSyncState.SYNCHRONIZED

        # Components
        self.components: Dict[str, PhysicalComponent] = {}
        self.virtual_sensors: Dict[str, VirtualSensor] = {}

        # State history
        self.state_history: deque = deque(maxlen=10000)
        self.current_state: Optional[SimulationState] = None

        # Physics engine
        self.physics = PhysicsEngine()

        # Real-time data
        self.real_time_data: Dict[str, Any] = {}
        self.last_sync_time = datetime.now()
        self.sync_delay_ms = 0

        # Threading
        self.lock = threading.Lock()

        # Initialize model structure
        self._init_model_structure()

    def _init_model_structure(self):
        """Initialize the digital twin model structure"""
        # Define aqueduct sections
        num_sections = 20
        section_length = self.physics.length / num_sections

        for i in range(num_sections):
            section_id = f"section_{i+1:03d}"
            start_x = i * section_length
            end_x = (i + 1) * section_length

            self.components[section_id] = PhysicalComponent(
                component_id=section_id,
                component_type=ComponentType.AQUEDUCT_SECTION,
                name=f"Section {i+1}",
                position=Vector3D(x=start_x + section_length/2, y=0, z=0),
                rotation=Vector3D(0, 0, 0),
                scale=Vector3D(section_length, self.physics.width, self.physics.depth),
                bounding_box=BoundingBox(
                    Vector3D(start_x, -self.physics.width/2, 0),
                    Vector3D(end_x, self.physics.width/2, self.physics.depth)
                ),
                properties={
                    'section_number': i + 1,
                    'chainage_start': start_x,
                    'chainage_end': end_x,
                    'design_flow': 50.0,
                    'design_level': 4.5
                }
            )

            # Add joints between sections
            if i > 0:
                joint_id = f"joint_{i:03d}"
                self.components[joint_id] = PhysicalComponent(
                    component_id=joint_id,
                    component_type=ComponentType.JOINT,
                    name=f"Joint {i}",
                    position=Vector3D(x=start_x, y=0, z=self.physics.depth/2),
                    rotation=Vector3D(0, 0, 0),
                    scale=Vector3D(0.1, self.physics.width, self.physics.depth),
                    bounding_box=BoundingBox(
                        Vector3D(start_x-0.05, -self.physics.width/2, 0),
                        Vector3D(start_x+0.05, self.physics.width/2, self.physics.depth)
                    ),
                    properties={
                        'joint_type': 'expansion',
                        'design_gap': 20.0,  # mm
                        'max_gap': 40.0,
                        'min_gap': 5.0
                    },
                    parent_id=section_id
                )

        # Add gates
        gate_positions = [0, 430, 860, 1292]  # Approximate positions
        for i, pos in enumerate(gate_positions):
            gate_id = f"gate_{i+1:02d}"
            self.components[gate_id] = PhysicalComponent(
                component_id=gate_id,
                component_type=ComponentType.GATE,
                name=f"Control Gate {i+1}",
                position=Vector3D(x=pos, y=0, z=self.physics.depth/2),
                rotation=Vector3D(0, 0, 0),
                scale=Vector3D(1.0, self.physics.width, self.physics.depth),
                bounding_box=BoundingBox(
                    Vector3D(pos-0.5, -self.physics.width/2, 0),
                    Vector3D(pos+0.5, self.physics.width/2, self.physics.depth)
                ),
                properties={
                    'gate_type': 'radial',
                    'width': 6.0,
                    'height': 4.0,
                    'max_opening': 3.5,
                    'actuator_type': 'hydraulic'
                },
                state={
                    'opening': 0.5,
                    'position_mm': 1750
                }
            )

        # Add bearings
        bearing_spacing = 40.0  # meters
        num_bearings = int(self.physics.length / bearing_spacing)
        for i in range(num_bearings):
            bearing_id = f"bearing_{i+1:03d}"
            pos = i * bearing_spacing
            self.components[bearing_id] = PhysicalComponent(
                component_id=bearing_id,
                component_type=ComponentType.BEARING,
                name=f"Bearing {i+1}",
                position=Vector3D(x=pos, y=0, z=-0.5),
                rotation=Vector3D(0, 0, 0),
                scale=Vector3D(1.0, 0.5, 0.5),
                bounding_box=BoundingBox(
                    Vector3D(pos-0.5, -0.25, -0.75),
                    Vector3D(pos+0.5, 0.25, -0.25)
                ),
                properties={
                    'bearing_type': 'pot',
                    'capacity': 5000,  # kN
                    'friction_coefficient': 0.03
                },
                state={
                    'stress': 20.0,
                    'locked': False
                }
            )

        # Add virtual sensors
        self._init_virtual_sensors()

    def _init_virtual_sensors(self):
        """Initialize virtual sensors"""
        # Virtual water level sensors along the aqueduct
        for i in range(0, int(self.physics.length), 100):
            vs_id = f"vs_level_{i:04d}"
            self.virtual_sensors[vs_id] = VirtualSensor(
                sensor_id=vs_id,
                name=f"Virtual Level at {i}m",
                sensor_type='virtual',
                position=Vector3D(x=i, y=0, z=2.0),
                attached_to=f"section_{int(i/64.6)+1:03d}",
                measurement_type='water_level',
                unit='m',
                interpolation_method='linear'
            )

        # Virtual temperature sensors
        for i in range(0, int(self.physics.length), 200):
            vs_id = f"vs_temp_{i:04d}"
            self.virtual_sensors[vs_id] = VirtualSensor(
                sensor_id=vs_id,
                name=f"Virtual Temp at {i}m",
                sensor_type='virtual',
                position=Vector3D(x=i, y=3.5, z=4.0),
                attached_to=f"section_{int(i/64.6)+1:03d}",
                measurement_type='temperature',
                unit='C',
                interpolation_method='spline'
            )

    def update_from_simulation(self, sim_state: Dict[str, Any]):
        """Update twin from simulation state"""
        with self.lock:
            # Create state snapshot
            state = SimulationState(
                timestamp=datetime.now(),
                time_step=sim_state.get('time', 0),
                water_level=sim_state.get('h', 4.5),
                flow_velocity=sim_state.get('v', 2.0),
                flow_rate_in=sim_state.get('Q_in', 85.0),
                flow_rate_out=sim_state.get('Q_out', 85.0),
                temperature_sun=sim_state.get('T_sun', 25.0),
                temperature_shade=sim_state.get('T_shade', 20.0),
                thermal_gradient=sim_state.get('T_sun', 25.0) - sim_state.get('T_shade', 20.0),
                thermal_bending=sim_state.get('thermal_bending', 0.0),
                vibration_amplitude=sim_state.get('vib_amp', 0.0),
                joint_gap=sim_state.get('joint_gap', 20.0),
                bearing_stress=sim_state.get('bearing_stress', 20.0),
                bearing_locked=sim_state.get('bearing_locked', False),
                ground_acceleration=sim_state.get('ground_accel', 0.0),
                froude_number=sim_state.get('fr', 0.3),
                reynolds_number=sim_state.get('Re', 1e6),
                water_volume=sim_state.get('h', 4.5) * self.physics.width * self.physics.length,
                energy_dissipation=0.0,
                active_scenarios=sim_state.get('scenarios', []),
                control_mode=sim_state.get('mode', 'AUTO'),
                risk_level=sim_state.get('risk_level', 'LOW')
            )

            # Compute additional physics
            hydraulics = self.physics.compute_hydraulics(
                state.water_level, state.flow_rate_in,
                state.flow_rate_out, 0.5
            )
            state.energy_dissipation = hydraulics['energy_dissipation']

            self.current_state = state
            self.state_history.append(state)

            # Update component states
            self._update_component_states(sim_state)

            # Update virtual sensors
            self._update_virtual_sensors(sim_state)

            # Update sync status
            self.last_sync_time = datetime.now()
            self.sync_state = TwinSyncState.SYNCHRONIZED

    def _update_component_states(self, sim_state: Dict[str, Any]):
        """Update individual component states"""
        h = sim_state.get('h', 4.5)
        T_sun = sim_state.get('T_sun', 25.0)
        T_shade = sim_state.get('T_shade', 20.0)
        joint_gap = sim_state.get('joint_gap', 20.0)
        bearing_stress = sim_state.get('bearing_stress', 20.0)
        bearing_locked = sim_state.get('bearing_locked', False)

        # Update sections with water level profile
        for comp_id, comp in self.components.items():
            if comp.component_type == ComponentType.AQUEDUCT_SECTION:
                # Simplified linear water surface
                x = comp.position.x
                level_variation = 0.1 * math.sin(2 * math.pi * x / self.physics.length)
                comp.state['water_level'] = h + level_variation
                comp.state['temperature'] = T_sun - (T_sun - T_shade) * 0.5

            elif comp.component_type == ComponentType.JOINT:
                comp.state['gap'] = joint_gap
                comp.state['movement'] = joint_gap - 20.0

            elif comp.component_type == ComponentType.BEARING:
                comp.state['stress'] = bearing_stress
                comp.state['locked'] = bearing_locked

            elif comp.component_type == ComponentType.GATE:
                comp.state['water_level_upstream'] = h
                comp.state['water_level_downstream'] = h * 0.95

            comp.last_updated = datetime.now()

    def _update_virtual_sensors(self, sim_state: Dict[str, Any]):
        """Update virtual sensor values through interpolation"""
        h = sim_state.get('h', 4.5)
        T_sun = sim_state.get('T_sun', 25.0)
        T_shade = sim_state.get('T_shade', 20.0)

        for vs_id, vs in self.virtual_sensors.items():
            x = vs.position.x

            if vs.measurement_type == 'water_level':
                # Linear interpolation with wave effect
                wave = 0.1 * math.sin(2 * math.pi * x / 200)
                vs.current_value = h + wave
                vs.confidence = 0.95

            elif vs.measurement_type == 'temperature':
                # Temperature varies with position and exposure
                exposure = 0.5 + 0.5 * math.sin(2 * math.pi * x / self.physics.length)
                vs.current_value = T_shade + (T_sun - T_shade) * exposure
                vs.confidence = 0.9

    def run_what_if_scenario(self, scenario: Dict[str, Any],
                            duration: float = 300.0,
                            dt: float = 1.0) -> List[Dict[str, Any]]:
        """Run a what-if simulation scenario"""
        results = []

        # Initialize from current state or default
        if self.current_state:
            h = self.current_state.water_level
            v = self.current_state.flow_velocity
            T_sun = self.current_state.temperature_sun
            T_shade = self.current_state.temperature_shade
            bending = self.current_state.thermal_bending
        else:
            h = 4.5
            v = 2.0
            T_sun = 25.0
            T_shade = 20.0
            bending = 0.0

        # Apply scenario modifications
        Q_in = scenario.get('Q_in', 85.0)
        Q_out = scenario.get('Q_out', 85.0)
        if 'temperature_change' in scenario:
            T_sun += scenario['temperature_change']

        t = 0
        while t < duration:
            # Compute hydraulics
            hydraulics = self.physics.compute_hydraulics(h, Q_in, Q_out, dt)

            # Update water level (mass balance)
            dh = (Q_in - Q_out) * dt / (self.physics.width * self.physics.length)
            h = max(0.5, min(7.0, h + dh))

            # Compute thermal effects
            thermal = self.physics.compute_thermal(T_sun, T_shade, bending, dt)
            bending = thermal['bending']

            # Record results
            results.append({
                'time': t,
                'water_level': h,
                'velocity': hydraulics['velocity'],
                'froude_number': hydraulics['froude_number'],
                'thermal_bending': bending,
                'thermal_stress': thermal['thermal_stress'],
                'volume': hydraulics['volume']
            })

            t += dt

        return results

    def get_component(self, component_id: str) -> Optional[PhysicalComponent]:
        """Get component by ID"""
        return self.components.get(component_id)

    def get_components_by_type(self, comp_type: ComponentType) -> List[PhysicalComponent]:
        """Get all components of a specific type"""
        return [c for c in self.components.values() if c.component_type == comp_type]

    def get_virtual_sensor(self, sensor_id: str) -> Optional[VirtualSensor]:
        """Get virtual sensor by ID"""
        return self.virtual_sensors.get(sensor_id)

    def get_virtual_sensor_readings(self) -> Dict[str, float]:
        """Get all virtual sensor readings"""
        return {vs_id: vs.current_value for vs_id, vs in self.virtual_sensors.items()}

    def get_state_at_time(self, timestamp: datetime) -> Optional[SimulationState]:
        """Get historical state at specific time"""
        for state in reversed(self.state_history):
            if state.timestamp <= timestamp:
                return state
        return None

    def get_state_history(self, duration_minutes: float = 60) -> List[Dict[str, Any]]:
        """Get state history for specified duration"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        return [s.to_dict() for s in self.state_history if s.timestamp > cutoff]

    def get_3d_model_data(self) -> Dict[str, Any]:
        """Get data for 3D visualization"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'aqueduct': {
                'length': self.physics.length,
                'width': self.physics.width,
                'depth': self.physics.depth
            },
            'components': {cid: c.to_dict() for cid, c in self.components.items()},
            'virtual_sensors': {vid: v.to_dict() for vid, v in self.virtual_sensors.items()},
            'current_state': self.current_state.to_dict() if self.current_state else None,
            'sync_state': self.sync_state.value,
            'last_sync': self.last_sync_time.isoformat()
        }

    def get_cross_section_at(self, chainage: float) -> Dict[str, Any]:
        """Get cross-section data at specified chainage"""
        if not self.current_state:
            return {}

        # Find relevant section
        section = None
        for comp in self.components.values():
            if comp.component_type == ComponentType.AQUEDUCT_SECTION:
                start = comp.properties.get('chainage_start', 0)
                end = comp.properties.get('chainage_end', 0)
                if start <= chainage <= end:
                    section = comp
                    break

        water_level = self.current_state.water_level
        if section and 'water_level' in section.state:
            water_level = section.state['water_level']

        return {
            'chainage': chainage,
            'water_level': water_level,
            'channel_width': self.physics.width,
            'channel_depth': self.physics.depth,
            'wet_area': water_level * self.physics.width,
            'wet_perimeter': self.physics.width + 2 * water_level,
            'hydraulic_radius': (water_level * self.physics.width) / (self.physics.width + 2 * water_level),
            'section': section.to_dict() if section else None
        }

    def export_to_json(self, filepath: str = None) -> str:
        """Export model to JSON"""
        data = self.get_3d_model_data()
        data['exported_at'] = datetime.now().isoformat()

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        return json.dumps(data, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get digital twin status"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'resolution': self.resolution.value,
            'sync_state': self.sync_state.value,
            'last_sync': self.last_sync_time.isoformat(),
            'sync_delay_ms': self.sync_delay_ms,
            'components_count': len(self.components),
            'virtual_sensors_count': len(self.virtual_sensors),
            'state_history_size': len(self.state_history),
            'current_state': self.current_state.to_dict() if self.current_state else None
        }


class DigitalTwinManager:
    """
    Digital Twin management system
    """

    def __init__(self, data_dir: str = None):
        self.model = DigitalTwinModel(data_dir)
        self.running = False
        self.update_thread = None
        self.lock = threading.Lock()

        # Scenarios for analysis
        self.saved_scenarios: Dict[str, Dict[str, Any]] = {}

        # Callbacks
        self.on_state_update: Optional[Callable] = None
        self.on_alert: Optional[Callable] = None

    def start(self):
        """Start the digital twin manager"""
        self.running = True
        print("[DigitalTwin] Started")

    def stop(self):
        """Stop the digital twin manager"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

    def update_from_real_time(self, data: Dict[str, Any]):
        """Update from real-time data"""
        self.model.update_from_simulation(data)

        if self.on_state_update:
            self.on_state_update(self.model.current_state)

    def run_scenario_analysis(self, name: str, scenario: Dict[str, Any],
                             duration: float = 300.0) -> Dict[str, Any]:
        """Run and save a scenario analysis"""
        results = self.model.run_what_if_scenario(scenario, duration)

        analysis = {
            'name': name,
            'scenario': scenario,
            'duration': duration,
            'results': results,
            'summary': self._summarize_results(results),
            'timestamp': datetime.now().isoformat()
        }

        self.saved_scenarios[name] = analysis
        return analysis

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize scenario results"""
        if not results:
            return {}

        levels = [r['water_level'] for r in results]
        velocities = [r['velocity'] for r in results]
        froudes = [r['froude_number'] for r in results]

        return {
            'water_level': {
                'min': min(levels),
                'max': max(levels),
                'mean': sum(levels) / len(levels),
                'final': levels[-1]
            },
            'velocity': {
                'min': min(velocities),
                'max': max(velocities),
                'mean': sum(velocities) / len(velocities)
            },
            'froude_number': {
                'max': max(froudes),
                'critical_exceeded': any(f > 0.9 for f in froudes)
            },
            'duration_simulated': results[-1]['time'] if results else 0
        }

    def get_saved_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get all saved scenarios"""
        return self.saved_scenarios

    def compare_scenarios(self, scenario_names: List[str]) -> Dict[str, Any]:
        """Compare multiple scenarios"""
        comparison = {}
        for name in scenario_names:
            if name in self.saved_scenarios:
                comparison[name] = self.saved_scenarios[name]['summary']
        return comparison

    def get_status(self) -> Dict[str, Any]:
        """Get digital twin manager status"""
        return {
            'running': self.running,
            'model_status': self.model.get_status(),
            'saved_scenarios': len(self.saved_scenarios),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_twin_manager = None


def get_twin_manager() -> DigitalTwinManager:
    """Get global digital twin manager"""
    global _twin_manager
    if _twin_manager is None:
        _twin_manager = DigitalTwinManager()
    return _twin_manager


if __name__ == "__main__":
    # Test digital twin
    print("=== Digital Twin Test ===")

    manager = DigitalTwinManager()
    manager.start()

    # Update from simulation
    print("\n1. Updating from simulation data...")
    manager.update_from_real_time({
        'h': 4.5,
        'v': 2.0,
        'Q_in': 85.0,
        'Q_out': 85.0,
        'T_sun': 30.0,
        'T_shade': 22.0,
        'thermal_bending': 2.5,
        'vib_amp': 3.0,
        'joint_gap': 22.0,
        'bearing_stress': 25.0,
        'bearing_locked': False,
        'ground_accel': 0.0,
        'fr': 0.32,
        'mode': 'AUTO',
        'risk_level': 'LOW'
    })

    # Get model status
    print("\n2. Model Status:")
    status = manager.get_status()
    print(f"   Components: {status['model_status']['components_count']}")
    print(f"   Virtual Sensors: {status['model_status']['virtual_sensors_count']}")
    print(f"   Sync State: {status['model_status']['sync_state']}")

    # Get component info
    print("\n3. Component Sample:")
    comp = manager.model.get_component('section_010')
    if comp:
        print(f"   {comp.name}: water_level={comp.state.get('water_level', 'N/A'):.2f}m")

    # Virtual sensors
    print("\n4. Virtual Sensor Readings (sample):")
    readings = manager.model.get_virtual_sensor_readings()
    for vs_id, value in list(readings.items())[:5]:
        print(f"   {vs_id}: {value:.3f}")

    # Cross-section
    print("\n5. Cross-Section at 500m:")
    cs = manager.model.get_cross_section_at(500)
    print(f"   Water Level: {cs.get('water_level', 0):.2f}m")
    print(f"   Wet Area: {cs.get('wet_area', 0):.2f}m2")

    # What-if scenario
    print("\n6. What-If Scenario Analysis:")
    analysis = manager.run_scenario_analysis(
        'high_inflow',
        {'Q_in': 120.0, 'Q_out': 85.0},
        duration=60.0
    )
    print(f"   Water Level Range: {analysis['summary']['water_level']['min']:.2f} - {analysis['summary']['water_level']['max']:.2f}m")
    print(f"   Critical Froude: {analysis['summary']['froude_number']['critical_exceeded']}")

    manager.stop()
    print("\nDigital Twin test completed!")
