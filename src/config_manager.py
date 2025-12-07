#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - Configuration Management System
团河渡槽自主运行系统 - 配置管理模块

Features:
- YAML/JSON configuration file support
- Runtime configuration updates
- Configuration validation
- Configuration versioning
- Default value management
- Environment variable override
"""

import os
import json
import yaml
import copy
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import threading


@dataclass
class ConfigVersion:
    """Configuration version record"""
    version: int
    timestamp: str
    hash: str
    changes: Dict[str, Any]
    author: str = "system"


@dataclass
class ValidationRule:
    """Validation rule for configuration values"""
    name: str
    validator: Callable[[Any], bool]
    message: str


@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Simulation
    simulation_dt: float = 0.5
    simulation_speed: float = 1.0
    max_simulation_time: float = 86400.0

    # Data
    data_retention_hours: int = 24
    data_aggregation_enabled: bool = True
    history_max_points: int = 10000

    # Logging
    log_level: str = "INFO"
    log_file_enabled: bool = True
    log_max_size_mb: int = 100
    log_backup_count: int = 5

    # API
    api_port: int = 5000
    api_rate_limit: int = 100
    api_key_enabled: bool = False
    cors_enabled: bool = True

    # Performance
    thread_pool_size: int = 4
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60


@dataclass
class ControlConfig:
    """Control system configuration"""
    # PID parameters
    kp_level: float = 0.5
    ki_level: float = 0.1
    kd_level: float = 0.05
    kp_velocity: float = 0.3
    ki_velocity: float = 0.05
    kd_velocity: float = 0.02

    # Setpoints
    target_level: float = 4.0
    target_velocity: float = 2.0
    target_froude: float = 0.32

    # Limits
    level_min: float = 2.0
    level_max: float = 7.0
    velocity_min: float = 0.5
    velocity_max: float = 4.0

    # Control modes
    auto_control_enabled: bool = True
    adaptive_control_enabled: bool = True
    predictive_control_enabled: bool = True

    # Actuator limits
    gate_position_min: float = 0.0
    gate_position_max: float = 100.0
    gate_rate_limit: float = 5.0  # % per second


@dataclass
class SafetyConfig:
    """Safety system configuration"""
    # Thresholds - Hydraulic
    level_high_warning: float = 6.0
    level_high_alarm: float = 6.5
    level_high_critical: float = 7.0
    level_low_warning: float = 3.0
    level_low_alarm: float = 2.5
    level_low_critical: float = 2.0
    froude_warning: float = 0.6
    froude_alarm: float = 0.8
    froude_critical: float = 1.0

    # Thresholds - Structural
    vibration_warning: float = 5.0
    vibration_alarm: float = 10.0
    vibration_critical: float = 15.0
    joint_gap_min_warning: float = 8.0
    joint_gap_min_alarm: float = 5.0
    joint_gap_max_warning: float = 30.0
    joint_gap_max_alarm: float = 35.0
    deflection_warning: float = 30.0
    deflection_alarm: float = 50.0

    # Thresholds - Thermal
    temp_diff_warning: float = 15.0
    temp_diff_alarm: float = 20.0
    temp_diff_critical: float = 25.0
    temp_max_warning: float = 40.0
    temp_max_alarm: float = 50.0

    # Interlock settings
    interlocks_enabled: bool = True
    emergency_stop_enabled: bool = True
    auto_recovery_enabled: bool = True
    recovery_delay_seconds: float = 60.0

    # Redundancy
    redundant_control_enabled: bool = True
    voting_threshold: int = 2  # Out of 3


@dataclass
class SensorConfig:
    """Sensor configuration"""
    # Sampling
    sample_rate_hz: float = 10.0
    filter_enabled: bool = True
    filter_cutoff_hz: float = 1.0

    # Calibration
    level_offset: float = 0.0
    level_scale: float = 1.0
    velocity_offset: float = 0.0
    velocity_scale: float = 1.0
    temp_offset: float = 0.0
    temp_scale: float = 1.0

    # Fault detection
    sensor_timeout_ms: int = 1000
    outlier_detection_enabled: bool = True
    outlier_threshold_sigma: float = 3.0

    # Virtual sensors
    virtual_sensors_enabled: bool = True
    sensor_fusion_enabled: bool = True


@dataclass
class ScadaConfig:
    """SCADA integration configuration"""
    # OPC-UA
    opcua_enabled: bool = True
    opcua_endpoint: str = "opc.tcp://localhost:4840"
    opcua_namespace: str = "http://tuanhe.aqueduct/TAOS"
    opcua_security_mode: str = "None"

    # Modbus
    modbus_enabled: bool = True
    modbus_port: int = 502
    modbus_slave_id: int = 1
    modbus_timeout_ms: int = 1000

    # Data historian
    historian_enabled: bool = True
    historian_compression_enabled: bool = True
    historian_deadband_percent: float = 1.0


class ConfigManager:
    """
    Configuration Manager for TAOS

    Manages all system configuration with:
    - File-based configuration (YAML/JSON)
    - Runtime updates
    - Validation
    - Versioning
    - Environment variable override
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.lock = threading.RLock()

        # Configuration sections
        self.system = SystemConfig()
        self.control = ControlConfig()
        self.safety = SafetyConfig()
        self.sensor = SensorConfig()
        self.scada = ScadaConfig()

        # Version tracking
        self.versions: List[ConfigVersion] = []
        self.current_version = 0

        # Validation rules
        self.validators: Dict[str, List[ValidationRule]] = {}
        self._setup_validators()

        # Change callbacks
        self.change_callbacks: List[Callable[[str, Any, Any], None]] = []

        # Load configuration
        self._load_config()
        self._apply_env_overrides()

    def _default_config_path(self) -> str:
        """Get default configuration file path"""
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "config" / "taos_config.yaml")

    def _setup_validators(self):
        """Setup validation rules for configuration values"""
        self.validators["control"] = [
            ValidationRule(
                "kp_positive",
                lambda v: v.get("kp_level", 0) >= 0,
                "kp_level must be non-negative"
            ),
            ValidationRule(
                "ki_positive",
                lambda v: v.get("ki_level", 0) >= 0,
                "ki_level must be non-negative"
            ),
            ValidationRule(
                "level_range",
                lambda v: v.get("level_min", 0) < v.get("level_max", 10),
                "level_min must be less than level_max"
            ),
            ValidationRule(
                "target_in_range",
                lambda v: v.get("level_min", 0) <= v.get("target_level", 4) <= v.get("level_max", 10),
                "target_level must be within level limits"
            ),
        ]

        self.validators["safety"] = [
            ValidationRule(
                "warning_less_than_alarm",
                lambda v: v.get("level_high_warning", 6) < v.get("level_high_alarm", 6.5),
                "warning threshold must be less than alarm threshold"
            ),
            ValidationRule(
                "alarm_less_than_critical",
                lambda v: v.get("level_high_alarm", 6.5) < v.get("level_high_critical", 7),
                "alarm threshold must be less than critical threshold"
            ),
        ]

        self.validators["system"] = [
            ValidationRule(
                "dt_positive",
                lambda v: v.get("simulation_dt", 0.5) > 0,
                "simulation_dt must be positive"
            ),
            ValidationRule(
                "port_valid",
                lambda v: 1024 <= v.get("api_port", 5000) <= 65535,
                "api_port must be between 1024 and 65535"
            ),
        ]

    def _load_config(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            # Create default configuration
            self._save_config()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f)

            # Apply loaded configuration
            if "system" in data:
                self._update_dataclass(self.system, data["system"])
            if "control" in data:
                self._update_dataclass(self.control, data["control"])
            if "safety" in data:
                self._update_dataclass(self.safety, data["safety"])
            if "sensor" in data:
                self._update_dataclass(self.sensor, data["sensor"])
            if "scada" in data:
                self._update_dataclass(self.scada, data["scada"])

            # Load version history
            if "versions" in data:
                self.versions = [
                    ConfigVersion(**v) for v in data["versions"][-100:]  # Keep last 100
                ]
                self.current_version = len(self.versions)

        except Exception as e:
            print(f"Warning: Failed to load config: {e}")

    def _save_config(self):
        """Save configuration to file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        data = {
            "system": asdict(self.system),
            "control": asdict(self.control),
            "safety": asdict(self.safety),
            "sensor": asdict(self.sensor),
            "scada": asdict(self.scada),
            "versions": [asdict(v) for v in self.versions[-100:]]
        }

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")

    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            "TAOS_API_PORT": ("system", "api_port", int),
            "TAOS_LOG_LEVEL": ("system", "log_level", str),
            "TAOS_SIMULATION_DT": ("system", "simulation_dt", float),
            "TAOS_TARGET_LEVEL": ("control", "target_level", float),
            "TAOS_TARGET_VELOCITY": ("control", "target_velocity", float),
            "TAOS_AUTO_CONTROL": ("control", "auto_control_enabled", lambda x: x.lower() == "true"),
            "TAOS_INTERLOCKS": ("safety", "interlocks_enabled", lambda x: x.lower() == "true"),
            "TAOS_OPCUA_ENDPOINT": ("scada", "opcua_endpoint", str),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    converted = converter(value)
                    config_obj = getattr(self, section)
                    setattr(config_obj, key, converted)
                except Exception:
                    pass

    def _compute_hash(self) -> str:
        """Compute hash of current configuration"""
        data = self.get_all()
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def validate(self, section: str, data: Dict[str, Any]) -> List[str]:
        """Validate configuration data"""
        errors = []

        if section in self.validators:
            for rule in self.validators[section]:
                try:
                    if not rule.validator(data):
                        errors.append(rule.message)
                except Exception as e:
                    errors.append(f"Validation error in {rule.name}: {e}")

        return errors

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        with self.lock:
            return {
                "system": asdict(self.system),
                "control": asdict(self.control),
                "safety": asdict(self.safety),
                "sensor": asdict(self.sensor),
                "scada": asdict(self.scada),
            }

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        with self.lock:
            obj = getattr(self, section, None)
            if obj:
                return asdict(obj)
            return {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get single configuration value"""
        with self.lock:
            obj = getattr(self, section, None)
            if obj:
                return getattr(obj, key, default)
            return default

    def set(self, section: str, key: str, value: Any, author: str = "user") -> bool:
        """Set single configuration value"""
        with self.lock:
            obj = getattr(self, section, None)
            if not obj or not hasattr(obj, key):
                return False

            old_value = getattr(obj, key)
            if old_value == value:
                return True

            # Validate
            test_data = asdict(obj)
            test_data[key] = value
            errors = self.validate(section, test_data)
            if errors:
                print(f"Validation failed: {errors}")
                return False

            # Apply change
            setattr(obj, key, value)

            # Record version
            self._record_version({f"{section}.{key}": {"old": old_value, "new": value}}, author)

            # Notify callbacks
            for callback in self.change_callbacks:
                try:
                    callback(f"{section}.{key}", old_value, value)
                except Exception:
                    pass

            # Save to file
            self._save_config()

            return True

    def update_section(self, section: str, data: Dict[str, Any], author: str = "user") -> bool:
        """Update entire configuration section"""
        with self.lock:
            obj = getattr(self, section, None)
            if not obj:
                return False

            # Get current values
            current = asdict(obj)

            # Merge with new data
            merged = {**current, **data}

            # Validate
            errors = self.validate(section, merged)
            if errors:
                print(f"Validation failed: {errors}")
                return False

            # Track changes
            changes = {}
            for key, new_value in data.items():
                if hasattr(obj, key):
                    old_value = getattr(obj, key)
                    if old_value != new_value:
                        changes[f"{section}.{key}"] = {"old": old_value, "new": new_value}
                        setattr(obj, key, new_value)

            if changes:
                self._record_version(changes, author)

                # Notify callbacks
                for change_key, change_data in changes.items():
                    for callback in self.change_callbacks:
                        try:
                            callback(change_key, change_data["old"], change_data["new"])
                        except Exception:
                            pass

                # Save to file
                self._save_config()

            return True

    def _record_version(self, changes: Dict[str, Any], author: str):
        """Record configuration version"""
        self.current_version += 1
        version = ConfigVersion(
            version=self.current_version,
            timestamp=datetime.now().isoformat(),
            hash=self._compute_hash(),
            changes=changes,
            author=author
        )
        self.versions.append(version)

        # Keep only last 100 versions in memory
        if len(self.versions) > 100:
            self.versions = self.versions[-100:]

    def get_version_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get configuration version history"""
        with self.lock:
            return [asdict(v) for v in self.versions[-limit:]]

    def rollback(self, version: int) -> bool:
        """Rollback to a specific version"""
        with self.lock:
            # Find version
            target = None
            for v in self.versions:
                if v.version == version:
                    target = v
                    break

            if not target:
                return False

            # Apply inverse of all changes after target version
            for v in reversed(self.versions):
                if v.version <= version:
                    break

                for change_key, change_data in v.changes.items():
                    section, key = change_key.split(".", 1)
                    obj = getattr(self, section, None)
                    if obj and hasattr(obj, key):
                        setattr(obj, key, change_data["old"])

            self._record_version({"rollback_to": version}, "system")
            self._save_config()
            return True

    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """Add callback for configuration changes"""
        self.change_callbacks.append(callback)

    def export_config(self, format: str = "yaml") -> str:
        """Export configuration as string"""
        data = self.get_all()

        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    def import_config(self, config_str: str, format: str = "yaml", author: str = "import") -> bool:
        """Import configuration from string"""
        try:
            if format == "json":
                data = json.loads(config_str)
            else:
                data = yaml.safe_load(config_str)

            # Update each section
            for section in ["system", "control", "safety", "sensor", "scada"]:
                if section in data:
                    self.update_section(section, data[section], author)

            return True

        except Exception as e:
            print(f"Import failed: {e}")
            return False

    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults"""
        with self.lock:
            if section:
                if section == "system":
                    self.system = SystemConfig()
                elif section == "control":
                    self.control = ControlConfig()
                elif section == "safety":
                    self.safety = SafetyConfig()
                elif section == "sensor":
                    self.sensor = SensorConfig()
                elif section == "scada":
                    self.scada = ScadaConfig()
            else:
                self.system = SystemConfig()
                self.control = ControlConfig()
                self.safety = SafetyConfig()
                self.sensor = SensorConfig()
                self.scada = ScadaConfig()

            self._record_version({"reset": section or "all"}, "system")
            self._save_config()


class ConfigWatcher:
    """
    Watch configuration file for changes
    """

    def __init__(self, config_manager: ConfigManager, check_interval: float = 5.0):
        self.config_manager = config_manager
        self.check_interval = check_interval
        self.last_mtime = 0.0
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start watching for changes"""
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop watching"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.check_interval + 1)

    def _watch_loop(self):
        """Watch loop for file changes"""
        import time

        while self.running:
            try:
                if os.path.exists(self.config_manager.config_path):
                    mtime = os.path.getmtime(self.config_manager.config_path)
                    if mtime > self.last_mtime:
                        if self.last_mtime > 0:
                            # Reload configuration
                            self.config_manager._load_config()
                            print(f"Configuration reloaded from {self.config_manager.config_path}")
                        self.last_mtime = mtime
            except Exception as e:
                print(f"Config watch error: {e}")

            time.sleep(self.check_interval)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager"""
    return config_manager


# Convenience functions
def get_system_config() -> Dict[str, Any]:
    return config_manager.get_section("system")


def get_control_config() -> Dict[str, Any]:
    return config_manager.get_section("control")


def get_safety_config() -> Dict[str, Any]:
    return config_manager.get_section("safety")


def get_sensor_config() -> Dict[str, Any]:
    return config_manager.get_section("sensor")


def get_scada_config() -> Dict[str, Any]:
    return config_manager.get_section("scada")


if __name__ == "__main__":
    # Test configuration manager
    cm = ConfigManager()

    print("=== Configuration Manager Test ===")
    print(f"Config path: {cm.config_path}")

    # Get all config
    all_config = cm.get_all()
    print(f"\nSections: {list(all_config.keys())}")

    # Test get/set
    print(f"\nTarget level: {cm.get('control', 'target_level')}")
    cm.set("control", "target_level", 4.5)
    print(f"Updated target level: {cm.get('control', 'target_level')}")

    # Test version history
    print(f"\nVersion history: {len(cm.get_version_history())} entries")

    # Export
    print("\n=== YAML Export ===")
    print(cm.export_config("yaml")[:500] + "...")

    print("\nConfiguration manager test completed!")
