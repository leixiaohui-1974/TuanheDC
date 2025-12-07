#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - API Documentation System
团河渡槽自主运行系统 - API文档模块

Features:
- OpenAPI 3.0 specification
- Swagger UI integration
- Auto-generated documentation
- API versioning
- Request/Response schemas
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class APIParameter:
    """API parameter definition"""
    name: str
    location: str  # query, path, header, cookie
    description: str
    required: bool = False
    schema: Dict[str, Any] = None
    example: Any = None


@dataclass
class APIResponse:
    """API response definition"""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Dict[str, Any] = None
    example: Any = None


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[APIParameter] = None
    request_body: Dict[str, Any] = None
    responses: List[APIResponse] = None
    security: List[Dict[str, List[str]]] = None
    deprecated: bool = False


class OpenAPIGenerator:
    """OpenAPI 3.0 specification generator"""

    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.security_schemes: Dict[str, Dict[str, Any]] = {}

        # Initialize with TAOS API endpoints
        self._define_schemas()
        self._define_security()
        self._define_endpoints()

    def _define_schemas(self):
        """Define data schemas"""
        self.schemas = {
            "SystemState": {
                "type": "object",
                "description": "Current system state",
                "properties": {
                    "time": {"type": "number", "description": "Simulation time (seconds)"},
                    "h": {"type": "number", "description": "Water level (m)"},
                    "v": {"type": "number", "description": "Flow velocity (m/s)"},
                    "fr": {"type": "number", "description": "Froude number"},
                    "Q_in": {"type": "number", "description": "Inlet flow rate (m³/s)"},
                    "Q_out": {"type": "number", "description": "Outlet flow rate (m³/s)"},
                    "T_sun": {"type": "number", "description": "Sun-side temperature (°C)"},
                    "T_shade": {"type": "number", "description": "Shade-side temperature (°C)"},
                    "T_amb": {"type": "number", "description": "Ambient temperature (°C)"},
                    "solar_rad": {"type": "number", "description": "Solar radiation (kW/m²)"},
                    "joint_gap": {"type": "number", "description": "Expansion joint gap (mm)"},
                    "bending_deflection": {"type": "number", "description": "Bending deflection (mm)"},
                    "vib_amp": {"type": "number", "description": "Vibration amplitude (mm)"},
                    "vib_freq": {"type": "number", "description": "Vibration frequency (Hz)"},
                    "active_scenarios": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "status": {"type": "string", "enum": ["NORMAL", "WARNING", "ALARM", "CRITICAL"]}
                }
            },
            "ControlAction": {
                "type": "object",
                "description": "Control action result",
                "properties": {
                    "action_type": {"type": "string"},
                    "target": {"type": "string"},
                    "value": {"type": "number"},
                    "reason": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            },
            "ScenarioRequest": {
                "type": "object",
                "required": ["scenario_id"],
                "properties": {
                    "scenario_id": {"type": "string", "description": "Scenario ID to inject"},
                    "intensity": {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
                    "duration": {"type": "number", "description": "Duration in seconds"}
                }
            },
            "SafetyStatus": {
                "type": "object",
                "properties": {
                    "safety_level": {"type": "string", "enum": ["NORMAL", "CAUTION", "WARNING", "DANGER", "EMERGENCY"]},
                    "interlocks_triggered": {"type": "array", "items": {"type": "string"}},
                    "faults": {"type": "array", "items": {"$ref": "#/components/schemas/Fault"}},
                    "actions": {"type": "array", "items": {"type": "string"}}
                }
            },
            "Fault": {
                "type": "object",
                "properties": {
                    "fault_id": {"type": "string"},
                    "fault_type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                    "description": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            },
            "ConfigSection": {
                "type": "object",
                "properties": {
                    "system": {"$ref": "#/components/schemas/SystemConfig"},
                    "control": {"$ref": "#/components/schemas/ControlConfig"},
                    "safety": {"$ref": "#/components/schemas/SafetyConfig"},
                    "sensor": {"$ref": "#/components/schemas/SensorConfig"},
                    "scada": {"$ref": "#/components/schemas/ScadaConfig"}
                }
            },
            "SystemConfig": {
                "type": "object",
                "properties": {
                    "simulation_dt": {"type": "number", "default": 0.5},
                    "simulation_speed": {"type": "number", "default": 1.0},
                    "data_retention_hours": {"type": "integer", "default": 24},
                    "log_level": {"type": "string", "default": "INFO"},
                    "api_port": {"type": "integer", "default": 5000}
                }
            },
            "ControlConfig": {
                "type": "object",
                "properties": {
                    "kp_level": {"type": "number", "default": 0.5},
                    "ki_level": {"type": "number", "default": 0.1},
                    "kd_level": {"type": "number", "default": 0.05},
                    "target_level": {"type": "number", "default": 4.0},
                    "target_velocity": {"type": "number", "default": 2.0},
                    "auto_control_enabled": {"type": "boolean", "default": True}
                }
            },
            "SafetyConfig": {
                "type": "object",
                "properties": {
                    "level_high_warning": {"type": "number", "default": 6.0},
                    "level_high_alarm": {"type": "number", "default": 6.5},
                    "vibration_warning": {"type": "number", "default": 5.0},
                    "vibration_alarm": {"type": "number", "default": 10.0},
                    "interlocks_enabled": {"type": "boolean", "default": True}
                }
            },
            "SensorConfig": {
                "type": "object",
                "properties": {
                    "sample_rate_hz": {"type": "number", "default": 10.0},
                    "filter_enabled": {"type": "boolean", "default": True},
                    "outlier_detection_enabled": {"type": "boolean", "default": True}
                }
            },
            "ScadaConfig": {
                "type": "object",
                "properties": {
                    "opcua_enabled": {"type": "boolean", "default": True},
                    "opcua_endpoint": {"type": "string"},
                    "modbus_enabled": {"type": "boolean", "default": True},
                    "modbus_port": {"type": "integer", "default": 502}
                }
            },
            "HistoryQuery": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "aggregation": {"type": "string", "enum": ["none", "1min", "5min", "1hour", "1day"]},
                    "limit": {"type": "integer", "default": 1000}
                }
            },
            "LogEntry": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "category": {"type": "string"},
                    "source": {"type": "string"},
                    "message": {"type": "string"},
                    "data": {"type": "object"}
                }
            },
            "AuditEntry": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "category": {"type": "string"},
                    "action": {"type": "string"},
                    "actor": {"type": "string"},
                    "target": {"type": "string"},
                    "details": {"type": "object"},
                    "result": {"type": "string"}
                }
            },
            "Prediction": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "current_value": {"type": "number"},
                    "predicted_value": {"type": "number"},
                    "prediction_time": {"type": "number"},
                    "confidence": {"type": "number"},
                    "trend": {"type": "string", "enum": ["rising", "stable", "falling"]}
                }
            },
            "Anomaly": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "anomaly_type": {"type": "string"},
                    "severity": {"type": "string"},
                    "value": {"type": "number"},
                    "expected_range": {"type": "object"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "code": {"type": "integer"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            }
        }

    def _define_security(self):
        """Define security schemes"""
        self.security_schemes = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token"
            }
        }

    def _define_endpoints(self):
        """Define API endpoints"""
        self.endpoints = [
            # === System State ===
            APIEndpoint(
                path="/api/state",
                method="GET",
                summary="Get current system state",
                description="Returns the current state of the aqueduct system including all sensor readings and computed values.",
                tags=["State"],
                responses=[
                    APIResponse(200, "Current system state", schema={"$ref": "#/components/schemas/SystemState"})
                ]
            ),
            APIEndpoint(
                path="/api/history",
                method="GET",
                summary="Get state history",
                description="Returns historical state data for the specified time range.",
                tags=["State"],
                parameters=[
                    APIParameter("minutes", "query", "Minutes of history to retrieve", required=False,
                               schema={"type": "integer", "default": 5}),
                    APIParameter("limit", "query", "Maximum number of records", required=False,
                               schema={"type": "integer", "default": 1000})
                ],
                responses=[
                    APIResponse(200, "Historical state data", schema={"type": "array", "items": {"$ref": "#/components/schemas/SystemState"}})
                ]
            ),

            # === Control ===
            APIEndpoint(
                path="/api/control/state",
                method="GET",
                summary="Get control system state",
                description="Returns current control system status including mode, setpoints, and active control actions.",
                tags=["Control"],
                responses=[
                    APIResponse(200, "Control system state")
                ]
            ),
            APIEndpoint(
                path="/api/control/action",
                method="POST",
                summary="Execute manual control action",
                description="Execute a manual control action. Requires appropriate permissions.",
                tags=["Control"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["action_type", "value"],
                                "properties": {
                                    "action_type": {"type": "string", "enum": ["gate_position", "pump_speed", "valve"]},
                                    "target": {"type": "string"},
                                    "value": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                responses=[
                    APIResponse(200, "Action executed", schema={"$ref": "#/components/schemas/ControlAction"}),
                    APIResponse(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"}),
                    APIResponse(403, "Permission denied", schema={"$ref": "#/components/schemas/Error"})
                ],
                security=[{"ApiKeyAuth": []}]
            ),

            # === Scenarios ===
            APIEndpoint(
                path="/api/scenario",
                method="POST",
                summary="Inject scenario",
                description="Inject a test scenario into the simulation.",
                tags=["Scenarios"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ScenarioRequest"}
                        }
                    }
                },
                responses=[
                    APIResponse(200, "Scenario injected successfully"),
                    APIResponse(400, "Invalid scenario", schema={"$ref": "#/components/schemas/Error"})
                ]
            ),
            APIEndpoint(
                path="/api/scenarios",
                method="GET",
                summary="List available scenarios",
                description="Returns a list of all available test scenarios.",
                tags=["Scenarios"],
                responses=[
                    APIResponse(200, "List of scenarios")
                ]
            ),
            APIEndpoint(
                path="/api/scenario/active",
                method="GET",
                summary="Get active scenarios",
                description="Returns currently active scenarios.",
                tags=["Scenarios"],
                responses=[
                    APIResponse(200, "Active scenarios")
                ]
            ),

            # === Safety ===
            APIEndpoint(
                path="/api/safety/status",
                method="GET",
                summary="Get safety system status",
                description="Returns comprehensive safety system status including faults, interlocks, and safety level.",
                tags=["Safety"],
                responses=[
                    APIResponse(200, "Safety status", schema={"$ref": "#/components/schemas/SafetyStatus"})
                ]
            ),
            APIEndpoint(
                path="/api/safety/faults",
                method="GET",
                summary="Get active faults",
                description="Returns list of currently active faults.",
                tags=["Safety"],
                responses=[
                    APIResponse(200, "Active faults")
                ]
            ),
            APIEndpoint(
                path="/api/safety/interlocks",
                method="GET",
                summary="Get interlock status",
                description="Returns status of all safety interlocks.",
                tags=["Safety"],
                responses=[
                    APIResponse(200, "Interlock status")
                ]
            ),
            APIEndpoint(
                path="/api/emergency",
                method="POST",
                summary="Trigger emergency action",
                description="Trigger an emergency response action. Requires elevated permissions.",
                tags=["Safety"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["action"],
                                "properties": {
                                    "action": {"type": "string", "enum": ["STOP", "DRAIN", "ISOLATE", "RESET"]},
                                    "reason": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                responses=[
                    APIResponse(200, "Emergency action executed"),
                    APIResponse(403, "Permission denied", schema={"$ref": "#/components/schemas/Error"})
                ],
                security=[{"ApiKeyAuth": []}]
            ),

            # === Intelligence ===
            APIEndpoint(
                path="/api/intelligence/predictions",
                method="GET",
                summary="Get predictions",
                description="Returns AI-based predictions for system parameters.",
                tags=["Intelligence"],
                parameters=[
                    APIParameter("variables", "query", "Variables to predict (comma-separated)", required=False,
                               schema={"type": "string"})
                ],
                responses=[
                    APIResponse(200, "Predictions", schema={"type": "array", "items": {"$ref": "#/components/schemas/Prediction"}})
                ]
            ),
            APIEndpoint(
                path="/api/intelligence/anomalies",
                method="GET",
                summary="Get detected anomalies",
                description="Returns list of detected anomalies.",
                tags=["Intelligence"],
                responses=[
                    APIResponse(200, "Anomalies", schema={"type": "array", "items": {"$ref": "#/components/schemas/Anomaly"}})
                ]
            ),
            APIEndpoint(
                path="/api/intelligence/patterns",
                method="GET",
                summary="Get recognized patterns",
                description="Returns recognized operational patterns.",
                tags=["Intelligence"],
                responses=[
                    APIResponse(200, "Patterns")
                ]
            ),

            # === Configuration ===
            APIEndpoint(
                path="/api/config",
                method="GET",
                summary="Get all configuration",
                description="Returns complete system configuration.",
                tags=["Configuration"],
                responses=[
                    APIResponse(200, "Configuration", schema={"$ref": "#/components/schemas/ConfigSection"})
                ]
            ),
            APIEndpoint(
                path="/api/config/{section}",
                method="GET",
                summary="Get configuration section",
                description="Returns specific configuration section.",
                tags=["Configuration"],
                parameters=[
                    APIParameter("section", "path", "Configuration section name", required=True,
                               schema={"type": "string", "enum": ["system", "control", "safety", "sensor", "scada"]})
                ],
                responses=[
                    APIResponse(200, "Configuration section"),
                    APIResponse(404, "Section not found", schema={"$ref": "#/components/schemas/Error"})
                ]
            ),
            APIEndpoint(
                path="/api/config/{section}",
                method="PUT",
                summary="Update configuration section",
                description="Update a configuration section. Changes are validated before applying.",
                tags=["Configuration"],
                parameters=[
                    APIParameter("section", "path", "Configuration section name", required=True,
                               schema={"type": "string"})
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"}
                        }
                    }
                },
                responses=[
                    APIResponse(200, "Configuration updated"),
                    APIResponse(400, "Validation failed", schema={"$ref": "#/components/schemas/Error"}),
                    APIResponse(403, "Permission denied", schema={"$ref": "#/components/schemas/Error"})
                ],
                security=[{"ApiKeyAuth": []}]
            ),

            # === Logging ===
            APIEndpoint(
                path="/api/logs",
                method="GET",
                summary="Get logs",
                description="Returns system logs with optional filtering.",
                tags=["Logging"],
                parameters=[
                    APIParameter("level", "query", "Minimum log level", required=False,
                               schema={"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}),
                    APIParameter("category", "query", "Log category filter", required=False,
                               schema={"type": "string"}),
                    APIParameter("limit", "query", "Maximum entries", required=False,
                               schema={"type": "integer", "default": 100}),
                    APIParameter("search", "query", "Search term", required=False,
                               schema={"type": "string"})
                ],
                responses=[
                    APIResponse(200, "Log entries", schema={"type": "array", "items": {"$ref": "#/components/schemas/LogEntry"}})
                ]
            ),
            APIEndpoint(
                path="/api/logs/statistics",
                method="GET",
                summary="Get log statistics",
                description="Returns log statistics for the specified period.",
                tags=["Logging"],
                parameters=[
                    APIParameter("hours", "query", "Hours to analyze", required=False,
                               schema={"type": "integer", "default": 24})
                ],
                responses=[
                    APIResponse(200, "Log statistics")
                ]
            ),

            # === Audit ===
            APIEndpoint(
                path="/api/audit",
                method="GET",
                summary="Get audit trail",
                description="Returns audit trail entries with optional filtering.",
                tags=["Audit"],
                parameters=[
                    APIParameter("category", "query", "Audit category", required=False,
                               schema={"type": "string"}),
                    APIParameter("actor", "query", "Actor filter", required=False,
                               schema={"type": "string"}),
                    APIParameter("limit", "query", "Maximum entries", required=False,
                               schema={"type": "integer", "default": 100})
                ],
                responses=[
                    APIResponse(200, "Audit entries", schema={"type": "array", "items": {"$ref": "#/components/schemas/AuditEntry"}})
                ],
                security=[{"ApiKeyAuth": []}]
            ),

            # === Data Export ===
            APIEndpoint(
                path="/api/export",
                method="GET",
                summary="Export data",
                description="Export historical data in various formats.",
                tags=["Data"],
                parameters=[
                    APIParameter("format", "query", "Export format", required=True,
                               schema={"type": "string", "enum": ["csv", "json", "xlsx"]}),
                    APIParameter("start_time", "query", "Start time (ISO format)", required=False,
                               schema={"type": "string", "format": "date-time"}),
                    APIParameter("end_time", "query", "End time (ISO format)", required=False,
                               schema={"type": "string", "format": "date-time"}),
                    APIParameter("variables", "query", "Variables to export", required=False,
                               schema={"type": "string"})
                ],
                responses=[
                    APIResponse(200, "Export file", content_type="application/octet-stream")
                ]
            ),

            # === SCADA ===
            APIEndpoint(
                path="/api/scada/values",
                method="GET",
                summary="Get SCADA values",
                description="Returns current SCADA data point values.",
                tags=["SCADA"],
                responses=[
                    APIResponse(200, "SCADA values")
                ]
            ),
            APIEndpoint(
                path="/api/scada/write",
                method="POST",
                summary="Write SCADA value",
                description="Write a value to a SCADA data point.",
                tags=["SCADA"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["point_id", "value"],
                                "properties": {
                                    "point_id": {"type": "string"},
                                    "value": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                responses=[
                    APIResponse(200, "Value written"),
                    APIResponse(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"}),
                    APIResponse(403, "Write not permitted", schema={"$ref": "#/components/schemas/Error"})
                ],
                security=[{"ApiKeyAuth": []}]
            ),

            # === Dashboard ===
            APIEndpoint(
                path="/api/dashboard/gauges",
                method="GET",
                summary="Get dashboard gauge data",
                description="Returns gauge data for dashboard display.",
                tags=["Dashboard"],
                responses=[
                    APIResponse(200, "Gauge data")
                ]
            ),
            APIEndpoint(
                path="/api/dashboard/charts/{chart_type}",
                method="GET",
                summary="Get chart data",
                description="Returns chart data for the specified chart type.",
                tags=["Dashboard"],
                parameters=[
                    APIParameter("chart_type", "path", "Chart type", required=True,
                               schema={"type": "string", "enum": ["temperature", "hydraulic", "structural", "safety"]}),
                    APIParameter("minutes", "query", "Minutes of data", required=False,
                               schema={"type": "integer", "default": 10})
                ],
                responses=[
                    APIResponse(200, "Chart data")
                ]
            ),

            # === System ===
            APIEndpoint(
                path="/api/health",
                method="GET",
                summary="Health check",
                description="Returns system health status.",
                tags=["System"],
                responses=[
                    APIResponse(200, "System healthy"),
                    APIResponse(503, "System unhealthy")
                ]
            ),
            APIEndpoint(
                path="/api/version",
                method="GET",
                summary="Get version info",
                description="Returns system version information.",
                tags=["System"],
                responses=[
                    APIResponse(200, "Version info")
                ]
            ),
            APIEndpoint(
                path="/api/metrics",
                method="GET",
                summary="Get system metrics",
                description="Returns system performance metrics.",
                tags=["System"],
                responses=[
                    APIResponse(200, "System metrics")
                ]
            ),
        ]

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification"""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "TAOS - Tuanhe Aqueduct Autonomous Operation System API",
                "description": """
## 团河渡槽自主运行系统 API

TAOS (Tuanhe Aqueduct Autonomous Operation System) provides a comprehensive API for monitoring and controlling the aqueduct system.

### Features
- Real-time system state monitoring
- Scenario injection for testing
- Safety system management
- AI-based predictions and anomaly detection
- Configuration management
- Historical data access
- SCADA integration

### Authentication
Most read endpoints are publicly accessible. Write operations and sensitive data require API key authentication.

### Rate Limiting
- Anonymous: 60 requests/minute
- Authenticated: 1000 requests/minute

### Versioning
This documentation is for API v1.0. The API version is included in all responses.
                """,
                "version": "1.0.0",
                "contact": {
                    "name": "TAOS Support",
                    "email": "taos-support@example.com"
                },
                "license": {
                    "name": "Proprietary",
                    "url": "https://example.com/license"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:5000",
                    "description": "Development server"
                },
                {
                    "url": "https://taos.example.com",
                    "description": "Production server"
                }
            ],
            "tags": [
                {"name": "State", "description": "System state and history"},
                {"name": "Control", "description": "Control system operations"},
                {"name": "Scenarios", "description": "Test scenario management"},
                {"name": "Safety", "description": "Safety system monitoring"},
                {"name": "Intelligence", "description": "AI predictions and analytics"},
                {"name": "Configuration", "description": "System configuration"},
                {"name": "Logging", "description": "System logs"},
                {"name": "Audit", "description": "Audit trail"},
                {"name": "Data", "description": "Data export and import"},
                {"name": "SCADA", "description": "SCADA integration"},
                {"name": "Dashboard", "description": "Dashboard data"},
                {"name": "System", "description": "System health and metrics"}
            ],
            "paths": {},
            "components": {
                "schemas": self.schemas,
                "securitySchemes": self.security_schemes,
                "responses": {
                    "UnauthorizedError": {
                        "description": "API key is missing or invalid",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    },
                    "NotFoundError": {
                        "description": "Resource not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        }
                    }
                }
            }
        }

        # Add paths
        for endpoint in self.endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}

            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "operationId": f"{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}",
                "responses": {}
            }

            # Add parameters
            if endpoint.parameters:
                operation["parameters"] = []
                for param in endpoint.parameters:
                    p = {
                        "name": param.name,
                        "in": param.location,
                        "description": param.description,
                        "required": param.required
                    }
                    if param.schema:
                        p["schema"] = param.schema
                    if param.example:
                        p["example"] = param.example
                    operation["parameters"].append(p)

            # Add request body
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body

            # Add responses
            if endpoint.responses:
                for resp in endpoint.responses:
                    resp_obj = {"description": resp.description}
                    if resp.schema:
                        resp_obj["content"] = {
                            resp.content_type: {"schema": resp.schema}
                        }
                    if resp.example:
                        if "content" not in resp_obj:
                            resp_obj["content"] = {resp.content_type: {}}
                        resp_obj["content"][resp.content_type]["example"] = resp.example
                    operation["responses"][str(resp.status_code)] = resp_obj

            # Add security
            if endpoint.security:
                operation["security"] = endpoint.security

            # Mark deprecated
            if endpoint.deprecated:
                operation["deprecated"] = True

            spec["paths"][endpoint.path][endpoint.method.lower()] = operation

        return spec

    def get_openapi_json(self) -> str:
        """Get OpenAPI spec as JSON string"""
        return json.dumps(self.generate_openapi_spec(), indent=2, ensure_ascii=False)

    def get_openapi_yaml(self) -> str:
        """Get OpenAPI spec as YAML string"""
        try:
            import yaml
            return yaml.dump(self.generate_openapi_spec(), default_flow_style=False, allow_unicode=True)
        except ImportError:
            return self.get_openapi_json()


def get_swagger_ui_html(openapi_url: str = "/api/openapi.json") -> str:
    """Generate Swagger UI HTML page"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>TAOS API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html {{ box-sizing: border-box; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin: 0; background: #fafafa; }}
        .swagger-ui .topbar {{ display: none; }}
        .swagger-ui .info {{ margin: 20px 0; }}
        .swagger-ui .info .title {{ color: #1976d2; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: "{openapi_url}",
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout",
                deepLinking: true,
                showExtensions: true,
                showCommonExtensions: true,
                docExpansion: "list",
                filter: true,
                syntaxHighlight: {{
                    activate: true,
                    theme: "monokai"
                }}
            }});
        }};
    </script>
</body>
</html>
"""


def get_redoc_html(openapi_url: str = "/api/openapi.json") -> str:
    """Generate ReDoc HTML page"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>TAOS API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <redoc spec-url="{openapi_url}"></redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>
"""


# Global generator instance
_generator = None


def get_generator() -> OpenAPIGenerator:
    """Get OpenAPI generator instance"""
    global _generator
    if _generator is None:
        _generator = OpenAPIGenerator()
    return _generator


if __name__ == "__main__":
    # Generate and print OpenAPI spec
    generator = OpenAPIGenerator()

    print("=== TAOS OpenAPI Specification ===\n")

    spec = generator.generate_openapi_spec()
    print(f"Title: {spec['info']['title']}")
    print(f"Version: {spec['info']['version']}")
    print(f"Endpoints: {len(generator.endpoints)}")
    print(f"Schemas: {len(generator.schemas)}")

    print("\n=== Tags ===")
    for tag in spec['tags']:
        print(f"  - {tag['name']}: {tag['description']}")

    print("\n=== Sample Paths ===")
    for path, methods in list(spec['paths'].items())[:5]:
        for method, details in methods.items():
            print(f"  {method.upper()} {path}: {details['summary']}")

    # Save to file
    output_path = Path(__file__).parent / "openapi.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generator.get_openapi_json())
    print(f"\n=== Saved to {output_path} ===")
