"""
TAOS V3.10 Digital Twin Module
数字孪生模块

Features:
- 3D scene management
- Real-time state synchronization
- Virtual simulation environment
- Multi-layer visualization
- Animation and effects
- Interactive control
"""

import uuid
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ObjectType(Enum):
    """3D object types"""
    CHANNEL = "channel"
    GATE = "gate"
    PUMP = "pump"
    SENSOR = "sensor"
    RESERVOIR = "reservoir"
    PIPE = "pipe"
    VALVE = "valve"
    BUILDING = "building"
    TERRAIN = "terrain"
    WATER = "water"
    ANNOTATION = "annotation"


class MaterialType(Enum):
    """Material types"""
    CONCRETE = "concrete"
    METAL = "metal"
    WATER = "water"
    GLASS = "glass"
    TERRAIN = "terrain"
    DEFAULT = "default"


class AnimationType(Enum):
    """Animation types"""
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALE = "scale"
    COLOR = "color"
    OPACITY = "opacity"
    FLOW = "flow"


class LayerType(Enum):
    """Visualization layer types"""
    STRUCTURE = "structure"
    EQUIPMENT = "equipment"
    SENSORS = "sensors"
    FLOW = "flow"
    ALARMS = "alarms"
    ANNOTATIONS = "annotations"
    TERRAIN = "terrain"


@dataclass
class Vector3:
    """3D vector"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        l = self.length()
        if l > 0:
            return Vector3(self.x/l, self.y/l, self.z/l)
        return Vector3()


@dataclass
class Quaternion:
    """Quaternion for rotation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float):
        """Create quaternion from Euler angles (radians)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy
        )


@dataclass
class Transform:
    """3D transform"""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    def to_dict(self) -> Dict:
        return {
            'position': self.position.to_dict(),
            'rotation': self.rotation.to_dict(),
            'scale': self.scale.to_dict()
        }


@dataclass
class Material:
    """3D material"""
    material_type: MaterialType = MaterialType.DEFAULT
    color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'type': self.material_type.value,
            'color': list(self.color),
            'metallic': self.metallic,
            'roughness': self.roughness,
            'emissive': list(self.emissive),
            'texture_id': self.texture_id
        }


@dataclass
class Geometry:
    """3D geometry definition"""
    geometry_type: str = "box"  # box, cylinder, sphere, plane, mesh
    dimensions: Dict[str, float] = field(default_factory=dict)
    mesh_id: Optional[str] = None
    lod_levels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'type': self.geometry_type,
            'dimensions': self.dimensions,
            'mesh_id': self.mesh_id,
            'lod_levels': self.lod_levels
        }


@dataclass
class SceneObject:
    """3D scene object"""
    object_id: str
    name: str
    object_type: ObjectType
    transform: Transform = field(default_factory=Transform)
    geometry: Optional[Geometry] = None
    material: Optional[Material] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    layer: LayerType = LayerType.STRUCTURE
    visible: bool = True
    interactive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Dynamic state binding
    bound_entity_id: Optional[str] = None
    state_mappings: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'object_id': self.object_id,
            'name': self.name,
            'object_type': self.object_type.value,
            'transform': self.transform.to_dict(),
            'geometry': self.geometry.to_dict() if self.geometry else None,
            'material': self.material.to_dict() if self.material else None,
            'parent_id': self.parent_id,
            'children': self.children,
            'layer': self.layer.value,
            'visible': self.visible,
            'interactive': self.interactive,
            'metadata': self.metadata,
            'bound_entity_id': self.bound_entity_id
        }


@dataclass
class Animation:
    """Animation definition"""
    animation_id: str
    target_object_id: str
    animation_type: AnimationType
    duration: float = 1.0
    loop: bool = False
    start_value: Any = None
    end_value: Any = None
    easing: str = "linear"
    playing: bool = False

    def to_dict(self) -> Dict:
        return {
            'animation_id': self.animation_id,
            'target_object_id': self.target_object_id,
            'animation_type': self.animation_type.value,
            'duration': self.duration,
            'loop': self.loop,
            'easing': self.easing,
            'playing': self.playing
        }


@dataclass
class Camera:
    """Camera configuration"""
    camera_id: str
    name: str
    position: Vector3 = field(default_factory=Vector3)
    target: Vector3 = field(default_factory=Vector3)
    fov: float = 60.0
    near: float = 0.1
    far: float = 10000.0
    is_orthographic: bool = False

    def to_dict(self) -> Dict:
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'position': self.position.to_dict(),
            'target': self.target.to_dict(),
            'fov': self.fov,
            'near': self.near,
            'far': self.far,
            'is_orthographic': self.is_orthographic
        }


@dataclass
class Light:
    """Light source"""
    light_id: str
    name: str
    light_type: str = "directional"  # directional, point, spot, ambient
    position: Vector3 = field(default_factory=Vector3)
    direction: Vector3 = field(default_factory=lambda: Vector3(0, -1, 0))
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    cast_shadows: bool = True

    def to_dict(self) -> Dict:
        return {
            'light_id': self.light_id,
            'name': self.name,
            'light_type': self.light_type,
            'position': self.position.to_dict(),
            'direction': self.direction.to_dict(),
            'color': list(self.color),
            'intensity': self.intensity,
            'cast_shadows': self.cast_shadows
        }


class Scene:
    """3D Scene container"""

    def __init__(self, scene_id: str, name: str):
        self.scene_id = scene_id
        self.name = name
        self.objects: Dict[str, SceneObject] = {}
        self.cameras: Dict[str, Camera] = {}
        self.lights: Dict[str, Light] = {}
        self.animations: Dict[str, Animation] = {}
        self.active_camera_id: Optional[str] = None
        self.layer_visibility: Dict[LayerType, bool] = {lt: True for lt in LayerType}
        self.metadata: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def add_object(self, obj: SceneObject) -> str:
        """Add object to scene"""
        with self._lock:
            self.objects[obj.object_id] = obj
            if obj.parent_id and obj.parent_id in self.objects:
                parent = self.objects[obj.parent_id]
                if obj.object_id not in parent.children:
                    parent.children.append(obj.object_id)
            return obj.object_id

    def remove_object(self, object_id: str) -> bool:
        """Remove object from scene"""
        with self._lock:
            if object_id not in self.objects:
                return False
            obj = self.objects[object_id]
            # Remove from parent
            if obj.parent_id and obj.parent_id in self.objects:
                parent = self.objects[obj.parent_id]
                if object_id in parent.children:
                    parent.children.remove(object_id)
            # Remove children recursively
            for child_id in list(obj.children):
                self.remove_object(child_id)
            del self.objects[object_id]
            return True

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID"""
        return self.objects.get(object_id)

    def update_object_transform(self, object_id: str, transform: Transform) -> bool:
        """Update object transform"""
        with self._lock:
            if object_id in self.objects:
                self.objects[object_id].transform = transform
                return True
            return False

    def add_camera(self, camera: Camera) -> str:
        """Add camera to scene"""
        with self._lock:
            self.cameras[camera.camera_id] = camera
            if not self.active_camera_id:
                self.active_camera_id = camera.camera_id
            return camera.camera_id

    def add_light(self, light: Light) -> str:
        """Add light to scene"""
        with self._lock:
            self.lights[light.light_id] = light
            return light.light_id

    def add_animation(self, animation: Animation) -> str:
        """Add animation"""
        with self._lock:
            self.animations[animation.animation_id] = animation
            return animation.animation_id

    def set_layer_visibility(self, layer: LayerType, visible: bool):
        """Set layer visibility"""
        with self._lock:
            self.layer_visibility[layer] = visible
            for obj in self.objects.values():
                if obj.layer == layer:
                    obj.visible = visible

    def get_objects_by_type(self, object_type: ObjectType) -> List[SceneObject]:
        """Get all objects of a type"""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def get_objects_by_layer(self, layer: LayerType) -> List[SceneObject]:
        """Get all objects in a layer"""
        return [obj for obj in self.objects.values() if obj.layer == layer]

    def to_dict(self) -> Dict:
        """Export scene to dictionary"""
        return {
            'scene_id': self.scene_id,
            'name': self.name,
            'objects': {k: v.to_dict() for k, v in self.objects.items()},
            'cameras': {k: v.to_dict() for k, v in self.cameras.items()},
            'lights': {k: v.to_dict() for k, v in self.lights.items()},
            'animations': {k: v.to_dict() for k, v in self.animations.items()},
            'active_camera_id': self.active_camera_id,
            'layer_visibility': {k.value: v for k, v in self.layer_visibility.items()},
            'metadata': self.metadata
        }


class StateSync:
    """Real-time state synchronization"""

    def __init__(self):
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.bindings: Dict[str, List[Tuple[str, str, Callable]]] = {}  # entity_id -> [(object_id, property, transform_fn)]
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[str, Dict], None]] = []

    def update_state(self, entity_id: str, state: Dict[str, Any]):
        """Update entity state"""
        with self._lock:
            self.state_cache[entity_id] = state
            # Notify bindings
            if entity_id in self.bindings:
                for object_id, prop, transform_fn in self.bindings[entity_id]:
                    if prop in state:
                        transformed = transform_fn(state[prop]) if transform_fn else state[prop]
                        for callback in self._callbacks:
                            callback(object_id, {prop: transformed})

    def bind_property(self, entity_id: str, object_id: str, property_name: str,
                      transform_fn: Callable = None):
        """Bind entity property to scene object"""
        with self._lock:
            if entity_id not in self.bindings:
                self.bindings[entity_id] = []
            self.bindings[entity_id].append((object_id, property_name, transform_fn))

    def unbind(self, entity_id: str, object_id: str = None):
        """Unbind entity or specific object"""
        with self._lock:
            if entity_id in self.bindings:
                if object_id:
                    self.bindings[entity_id] = [
                        b for b in self.bindings[entity_id] if b[0] != object_id
                    ]
                else:
                    del self.bindings[entity_id]

    def register_callback(self, callback: Callable[[str, Dict], None]):
        """Register state update callback"""
        self._callbacks.append(callback)

    def get_state(self, entity_id: str) -> Optional[Dict]:
        """Get cached state"""
        return self.state_cache.get(entity_id)


class WaterSimulation:
    """Water flow visualization simulation"""

    def __init__(self):
        self.flow_particles: Dict[str, List[Dict]] = {}
        self.flow_rates: Dict[str, float] = {}
        self.water_levels: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_flow_rate(self, channel_id: str, flow_rate: float):
        """Set flow rate for channel"""
        self.flow_rates[channel_id] = flow_rate

    def set_water_level(self, channel_id: str, level: float):
        """Set water level for channel"""
        self.water_levels[channel_id] = level

    def get_particle_positions(self, channel_id: str) -> List[Dict]:
        """Get particle positions for visualization"""
        return self.flow_particles.get(channel_id, [])

    def start_simulation(self):
        """Start flow simulation"""
        self._running = True
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()

    def stop_simulation(self):
        """Stop flow simulation"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _simulation_loop(self):
        """Simulation loop for particle updates"""
        while self._running:
            for channel_id, flow_rate in self.flow_rates.items():
                if channel_id not in self.flow_particles:
                    self.flow_particles[channel_id] = []
                # Update particle positions based on flow rate
                particles = self.flow_particles[channel_id]
                for p in particles:
                    p['position'] += flow_rate * 0.1
                    if p['position'] > 100:
                        p['position'] = 0
                # Add new particles if needed
                if len(particles) < int(abs(flow_rate) * 10):
                    particles.append({'position': 0, 'speed': flow_rate})
            time.sleep(0.05)


class DigitalTwinManager:
    """
    Digital Twin Manager
    数字孪生管理器
    """

    def __init__(self):
        self.scenes: Dict[str, Scene] = {}
        self.active_scene_id: Optional[str] = None
        self.state_sync = StateSync()
        self.water_sim = WaterSimulation()
        self._lock = threading.Lock()
        self._update_callbacks: List[Callable[[str, Dict], None]] = []

        # Register state sync callback
        self.state_sync.register_callback(self._on_state_update)

    def create_scene(self, name: str) -> Scene:
        """Create new scene"""
        with self._lock:
            scene_id = str(uuid.uuid4())
            scene = Scene(scene_id, name)
            self.scenes[scene_id] = scene
            if not self.active_scene_id:
                self.active_scene_id = scene_id
            return scene

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Get scene by ID"""
        return self.scenes.get(scene_id)

    def get_active_scene(self) -> Optional[Scene]:
        """Get active scene"""
        if self.active_scene_id:
            return self.scenes.get(self.active_scene_id)
        return None

    def set_active_scene(self, scene_id: str) -> bool:
        """Set active scene"""
        if scene_id in self.scenes:
            self.active_scene_id = scene_id
            return True
        return False

    def delete_scene(self, scene_id: str) -> bool:
        """Delete scene"""
        with self._lock:
            if scene_id in self.scenes:
                del self.scenes[scene_id]
                if self.active_scene_id == scene_id:
                    self.active_scene_id = next(iter(self.scenes.keys()), None)
                return True
            return False

    def create_channel_object(self, scene: Scene, channel_id: str,
                               name: str, start: Vector3, end: Vector3,
                               width: float, depth: float) -> SceneObject:
        """Create channel visualization object"""
        # Calculate transform
        direction = end - start
        length = direction.length()
        center = Vector3(
            (start.x + end.x) / 2,
            (start.y + end.y) / 2,
            (start.z + end.z) / 2
        )

        # Calculate rotation
        angle = math.atan2(direction.x, direction.z)
        rotation = Quaternion.from_euler(0, angle, 0)

        obj = SceneObject(
            object_id=f"channel_{channel_id}",
            name=name,
            object_type=ObjectType.CHANNEL,
            transform=Transform(
                position=center,
                rotation=rotation,
                scale=Vector3(width, depth, length)
            ),
            geometry=Geometry(
                geometry_type="box",
                dimensions={'width': width, 'height': depth, 'length': length}
            ),
            material=Material(
                material_type=MaterialType.CONCRETE,
                color=(0.6, 0.6, 0.6, 1.0)
            ),
            layer=LayerType.STRUCTURE,
            bound_entity_id=channel_id
        )

        scene.add_object(obj)

        # Add water surface
        water_obj = SceneObject(
            object_id=f"water_{channel_id}",
            name=f"{name} Water",
            object_type=ObjectType.WATER,
            transform=Transform(
                position=Vector3(center.x, center.y + depth * 0.3, center.z),
                rotation=rotation,
                scale=Vector3(width * 0.95, 0.1, length)
            ),
            geometry=Geometry(
                geometry_type="box",
                dimensions={'width': width * 0.95, 'height': 0.1, 'length': length}
            ),
            material=Material(
                material_type=MaterialType.WATER,
                color=(0.2, 0.4, 0.8, 0.7)
            ),
            layer=LayerType.FLOW,
            parent_id=obj.object_id,
            bound_entity_id=channel_id,
            state_mappings={'water_level': 'scale.y'}
        )

        scene.add_object(water_obj)
        return obj

    def create_gate_object(self, scene: Scene, gate_id: str,
                            name: str, position: Vector3,
                            width: float, height: float) -> SceneObject:
        """Create gate visualization object"""
        obj = SceneObject(
            object_id=f"gate_{gate_id}",
            name=name,
            object_type=ObjectType.GATE,
            transform=Transform(
                position=position,
                scale=Vector3(width, height, 0.3)
            ),
            geometry=Geometry(
                geometry_type="box",
                dimensions={'width': width, 'height': height, 'depth': 0.3}
            ),
            material=Material(
                material_type=MaterialType.METAL,
                color=(0.3, 0.3, 0.35, 1.0),
                metallic=0.8,
                roughness=0.3
            ),
            layer=LayerType.EQUIPMENT,
            interactive=True,
            bound_entity_id=gate_id,
            state_mappings={'position': 'transform.position.y'}
        )

        scene.add_object(obj)
        return obj

    def create_sensor_object(self, scene: Scene, sensor_id: str,
                              name: str, position: Vector3,
                              sensor_type: str) -> SceneObject:
        """Create sensor visualization object"""
        obj = SceneObject(
            object_id=f"sensor_{sensor_id}",
            name=name,
            object_type=ObjectType.SENSOR,
            transform=Transform(
                position=position,
                scale=Vector3(0.2, 0.5, 0.2)
            ),
            geometry=Geometry(
                geometry_type="cylinder",
                dimensions={'radius': 0.1, 'height': 0.5}
            ),
            material=Material(
                material_type=MaterialType.METAL,
                color=(0.9, 0.9, 0.2, 1.0),
                metallic=0.5
            ),
            layer=LayerType.SENSORS,
            interactive=True,
            bound_entity_id=sensor_id,
            metadata={'sensor_type': sensor_type}
        )

        scene.add_object(obj)
        return obj

    def update_entity_state(self, entity_id: str, state: Dict[str, Any]):
        """Update entity state for synchronization"""
        self.state_sync.update_state(entity_id, state)

    def _on_state_update(self, object_id: str, state: Dict):
        """Handle state updates"""
        scene = self.get_active_scene()
        if not scene:
            return

        obj = scene.get_object(object_id)
        if not obj:
            return

        # Apply state mappings
        for prop, value in state.items():
            if prop in obj.state_mappings:
                target = obj.state_mappings[prop]
                self._apply_property(obj, target, value)

        # Notify external callbacks
        for callback in self._update_callbacks:
            callback(object_id, state)

    def _apply_property(self, obj: SceneObject, target: str, value: Any):
        """Apply property value to object"""
        parts = target.split('.')
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part, None)
            if current is None:
                return
        setattr(current, parts[-1], value)

    def register_update_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for state updates"""
        self._update_callbacks.append(callback)

    def start_water_simulation(self):
        """Start water flow simulation"""
        self.water_sim.start_simulation()

    def stop_water_simulation(self):
        """Stop water flow simulation"""
        self.water_sim.stop_simulation()

    def export_scene(self, scene_id: str) -> Optional[Dict]:
        """Export scene to dictionary"""
        scene = self.scenes.get(scene_id)
        return scene.to_dict() if scene else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        total_objects = sum(len(s.objects) for s in self.scenes.values())
        return {
            'total_scenes': len(self.scenes),
            'active_scene': self.active_scene_id,
            'total_objects': total_objects,
            'bindings': len(self.state_sync.bindings)
        }


# Singleton instance
_dt_manager: Optional[DigitalTwinManager] = None


def get_digital_twin_manager() -> DigitalTwinManager:
    """Get singleton digital twin manager"""
    global _dt_manager
    if _dt_manager is None:
        _dt_manager = DigitalTwinManager()
    return _dt_manager


def create_sample_aqueduct_scene() -> Scene:
    """Create sample aqueduct scene"""
    manager = get_digital_twin_manager()
    scene = manager.create_scene("Aqueduct System")

    # Add main camera
    camera = Camera(
        camera_id="main_camera",
        name="Main Camera",
        position=Vector3(50, 30, 50),
        target=Vector3(0, 0, 0),
        fov=60
    )
    scene.add_camera(camera)

    # Add lights
    sun = Light(
        light_id="sun",
        name="Sun",
        light_type="directional",
        direction=Vector3(-0.5, -1, -0.3),
        intensity=1.0
    )
    scene.add_light(sun)

    ambient = Light(
        light_id="ambient",
        name="Ambient",
        light_type="ambient",
        intensity=0.3
    )
    scene.add_light(ambient)

    # Create channels
    manager.create_channel_object(
        scene, "ch_main", "Main Channel",
        Vector3(-50, 0, 0), Vector3(50, 0, 0),
        width=5, depth=3
    )

    manager.create_channel_object(
        scene, "ch_branch1", "Branch 1",
        Vector3(50, 0, 0), Vector3(80, 0, 30),
        width=3, depth=2
    )

    # Create gates
    manager.create_gate_object(
        scene, "gate_1", "Main Gate",
        Vector3(0, 0, 0), width=5, height=3
    )

    manager.create_gate_object(
        scene, "gate_2", "Branch Gate",
        Vector3(50, 0, 0), width=3, height=2
    )

    # Create sensors
    manager.create_sensor_object(
        scene, "sensor_wl1", "Water Level Sensor 1",
        Vector3(-25, 2, 0), "water_level"
    )

    manager.create_sensor_object(
        scene, "sensor_flow1", "Flow Sensor 1",
        Vector3(25, 2, 0), "flow_rate"
    )

    return scene
