#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - WebSocket Real-time Communication
团河渡槽自主运行系统 - WebSocket实时通信模块

Features:
- Real-time state broadcasting
- Bidirectional communication
- Event subscription system
- Connection management
- Message queuing
"""

import json
import threading
import time
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Set, Optional, Callable
from enum import Enum
import hashlib
import secrets


class MessageType(Enum):
    """WebSocket message types"""
    STATE_UPDATE = "state_update"
    ALARM = "alarm"
    SCENARIO = "scenario"
    CONTROL = "control"
    SAFETY = "safety"
    PREDICTION = "prediction"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ACK = "ack"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: str
    data: Dict[str, Any]
    timestamp: str = None
    id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat(timespec='milliseconds')
        if self.id is None:
            self.id = secrets.token_hex(8)

    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp,
            'id': self.id
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        data = json.loads(json_str)
        return cls(
            type=data.get('type', 'unknown'),
            data=data.get('data', {}),
            timestamp=data.get('timestamp'),
            id=data.get('id')
        )


@dataclass
class ClientConnection:
    """WebSocket client connection"""
    client_id: str
    connected_at: str
    subscriptions: Set[str]
    last_heartbeat: float
    user_agent: str = ""
    ip_address: str = ""
    authenticated: bool = False
    user_id: Optional[str] = None

    def is_subscribed(self, message_type: str) -> bool:
        return message_type in self.subscriptions or '*' in self.subscriptions


class WebSocketHub:
    """
    WebSocket Hub for managing real-time connections

    This is a simplified WebSocket hub that can be integrated with
    Flask-SocketIO, websockets, or other WebSocket libraries.
    """

    def __init__(self):
        self.clients: Dict[str, ClientConnection] = {}
        self.message_queue: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.running = False
        self.broadcast_thread: Optional[threading.Thread] = None

        # Message handlers
        self.handlers: Dict[str, List[Callable]] = {}

        # Statistics
        self.stats = {
            'total_connections': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'peak_connections': 0
        }

        # Heartbeat settings
        self.heartbeat_interval = 30.0  # seconds
        self.heartbeat_timeout = 90.0   # seconds

    def start(self):
        """Start the WebSocket hub"""
        self.running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()

    def stop(self):
        """Stop the WebSocket hub"""
        self.running = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=5.0)

    def connect(self, client_id: str, user_agent: str = "", ip_address: str = "") -> ClientConnection:
        """Register a new client connection"""
        with self.lock:
            if client_id in self.clients:
                # Reconnection - update existing
                self.clients[client_id].last_heartbeat = time.time()
                return self.clients[client_id]

            client = ClientConnection(
                client_id=client_id,
                connected_at=datetime.now().isoformat(),
                subscriptions={'state_update', 'alarm', 'system'},  # Default subscriptions
                last_heartbeat=time.time(),
                user_agent=user_agent,
                ip_address=ip_address
            )
            self.clients[client_id] = client

            # Update stats
            self.stats['total_connections'] += 1
            if len(self.clients) > self.stats['peak_connections']:
                self.stats['peak_connections'] = len(self.clients)

            # Send welcome message
            self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.SYSTEM.value,
                data={
                    'event': 'connected',
                    'client_id': client_id,
                    'message': 'Connected to TAOS WebSocket Hub'
                }
            ))

            return client

    def disconnect(self, client_id: str):
        """Remove a client connection"""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]

    def subscribe(self, client_id: str, message_types: List[str]):
        """Subscribe client to message types"""
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id].subscriptions.update(message_types)
                return True
        return False

    def unsubscribe(self, client_id: str, message_types: List[str]):
        """Unsubscribe client from message types"""
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id].subscriptions -= set(message_types)
                return True
        return False

    def heartbeat(self, client_id: str):
        """Update client heartbeat"""
        with self.lock:
            if client_id in self.clients:
                self.clients[client_id].last_heartbeat = time.time()

    def send_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific client"""
        # In a real implementation, this would send via the actual WebSocket connection
        # Here we queue it for the integration layer to handle
        self.message_queue.put(('client', client_id, message))
        self.stats['total_messages_sent'] += 1
        return True

    def broadcast(self, message: WebSocketMessage, exclude: Set[str] = None):
        """Broadcast message to all subscribed clients"""
        with self.lock:
            for client_id, client in self.clients.items():
                if exclude and client_id in exclude:
                    continue
                if client.is_subscribed(message.type):
                    self.send_to_client(client_id, message)

    def broadcast_state(self, state: Dict[str, Any]):
        """Broadcast state update"""
        message = WebSocketMessage(
            type=MessageType.STATE_UPDATE.value,
            data=state
        )
        self.broadcast(message)

    def broadcast_alarm(self, alarm_data: Dict[str, Any]):
        """Broadcast alarm"""
        message = WebSocketMessage(
            type=MessageType.ALARM.value,
            data=alarm_data
        )
        self.broadcast(message)

    def broadcast_scenario(self, scenario_data: Dict[str, Any]):
        """Broadcast scenario event"""
        message = WebSocketMessage(
            type=MessageType.SCENARIO.value,
            data=scenario_data
        )
        self.broadcast(message)

    def broadcast_safety(self, safety_data: Dict[str, Any]):
        """Broadcast safety status"""
        message = WebSocketMessage(
            type=MessageType.SAFETY.value,
            data=safety_data
        )
        self.broadcast(message)

    def broadcast_prediction(self, prediction_data: Dict[str, Any]):
        """Broadcast prediction"""
        message = WebSocketMessage(
            type=MessageType.PREDICTION.value,
            data=prediction_data
        )
        self.broadcast(message)

    def handle_message(self, client_id: str, message: WebSocketMessage):
        """Handle incoming message from client"""
        self.stats['total_messages_received'] += 1

        if message.type == MessageType.SUBSCRIBE.value:
            types = message.data.get('types', [])
            self.subscribe(client_id, types)
            self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.ACK.value,
                data={'action': 'subscribe', 'types': types, 'success': True}
            ))

        elif message.type == MessageType.UNSUBSCRIBE.value:
            types = message.data.get('types', [])
            self.unsubscribe(client_id, types)
            self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.ACK.value,
                data={'action': 'unsubscribe', 'types': types, 'success': True}
            ))

        elif message.type == MessageType.HEARTBEAT.value:
            self.heartbeat(client_id)
            self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.HEARTBEAT.value,
                data={'pong': True}
            ))

        elif message.type == MessageType.CONTROL.value:
            # Handle control commands - delegate to registered handlers
            if MessageType.CONTROL.value in self.handlers:
                for handler in self.handlers[MessageType.CONTROL.value]:
                    try:
                        handler(client_id, message.data)
                    except Exception as e:
                        self.send_to_client(client_id, WebSocketMessage(
                            type=MessageType.ERROR.value,
                            data={'error': str(e)}
                        ))

    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)

    def _broadcast_loop(self):
        """Background loop for processing broadcasts and cleanup"""
        while self.running:
            # Check for stale connections
            self._cleanup_stale_connections()

            # Send heartbeats
            self._send_heartbeats()

            time.sleep(self.heartbeat_interval)

    def _cleanup_stale_connections(self):
        """Remove stale connections"""
        now = time.time()
        stale = []

        with self.lock:
            for client_id, client in self.clients.items():
                if now - client.last_heartbeat > self.heartbeat_timeout:
                    stale.append(client_id)

        for client_id in stale:
            self.disconnect(client_id)

    def _send_heartbeats(self):
        """Send heartbeat to all clients"""
        message = WebSocketMessage(
            type=MessageType.HEARTBEAT.value,
            data={'ping': True, 'server_time': datetime.now().isoformat()}
        )

        with self.lock:
            for client_id in self.clients:
                self.send_to_client(client_id, message)

    def get_stats(self) -> Dict[str, Any]:
        """Get hub statistics"""
        with self.lock:
            return {
                **self.stats,
                'active_connections': len(self.clients),
                'clients': [
                    {
                        'client_id': c.client_id,
                        'connected_at': c.connected_at,
                        'subscriptions': list(c.subscriptions),
                        'authenticated': c.authenticated
                    }
                    for c in self.clients.values()
                ]
            }

    def get_pending_messages(self, limit: int = 100) -> List[tuple]:
        """Get pending messages from queue (for integration layer)"""
        messages = []
        try:
            while len(messages) < limit:
                msg = self.message_queue.get_nowait()
                messages.append(msg)
        except queue.Empty:
            pass
        return messages


class EventEmitter:
    """Event emitter for internal pub/sub"""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()

    def on(self, event: str, callback: Callable):
        """Register event listener"""
        with self.lock:
            if event not in self.listeners:
                self.listeners[event] = []
            self.listeners[event].append(callback)

    def off(self, event: str, callback: Callable):
        """Remove event listener"""
        with self.lock:
            if event in self.listeners:
                self.listeners[event] = [cb for cb in self.listeners[event] if cb != callback]

    def emit(self, event: str, *args, **kwargs):
        """Emit event to all listeners"""
        with self.lock:
            callbacks = self.listeners.get(event, []).copy()

        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Event handler error: {e}")

    def once(self, event: str, callback: Callable):
        """Register one-time event listener"""
        def wrapper(*args, **kwargs):
            self.off(event, wrapper)
            callback(*args, **kwargs)
        self.on(event, wrapper)


class RealtimeStateManager:
    """
    Manages real-time state updates and broadcasting
    """

    def __init__(self, hub: WebSocketHub):
        self.hub = hub
        self.last_state: Dict[str, Any] = {}
        self.state_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.update_interval = 0.5  # seconds
        self.last_update = 0.0
        self.events = EventEmitter()

        # Throttling settings
        self.throttle_enabled = True
        self.min_change_threshold = {
            'h': 0.01,      # 1cm water level change
            'v': 0.01,      # 0.01 m/s velocity change
            'fr': 0.001,    # 0.001 Froude number change
            'T_sun': 0.1,   # 0.1°C temperature change
            'T_shade': 0.1,
            'joint_gap': 0.1,  # 0.1mm gap change
            'vib_amp': 0.1     # 0.1mm vibration change
        }

    def update_state(self, state: Dict[str, Any]):
        """Update state and broadcast if changed significantly"""
        now = time.time()

        # Check if we should broadcast
        should_broadcast = False

        if not self.throttle_enabled:
            should_broadcast = True
        elif now - self.last_update >= self.update_interval:
            # Time-based update
            should_broadcast = True
        elif self._has_significant_change(state):
            # Change-based update
            should_broadcast = True

        if should_broadcast:
            self.hub.broadcast_state(state)
            self.last_state = state.copy()
            self.last_update = now

            # Buffer for history
            self.state_buffer.append({
                'timestamp': datetime.now().isoformat(),
                **state
            })
            if len(self.state_buffer) > self.buffer_size:
                self.state_buffer.pop(0)

            # Emit event
            self.events.emit('state_update', state)

    def _has_significant_change(self, state: Dict[str, Any]) -> bool:
        """Check if state has changed significantly"""
        if not self.last_state:
            return True

        for key, threshold in self.min_change_threshold.items():
            old_val = self.last_state.get(key)
            new_val = state.get(key)

            if old_val is not None and new_val is not None:
                if abs(new_val - old_val) >= threshold:
                    return True

        return False

    def get_recent_states(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent state history from buffer"""
        return self.state_buffer[-count:]


# Global instances
_hub = None
_state_manager = None


def get_websocket_hub() -> WebSocketHub:
    """Get global WebSocket hub"""
    global _hub
    if _hub is None:
        _hub = WebSocketHub()
    return _hub


def get_state_manager() -> RealtimeStateManager:
    """Get global state manager"""
    global _state_manager
    if _state_manager is None:
        _state_manager = RealtimeStateManager(get_websocket_hub())
    return _state_manager


# Flask-SocketIO integration helper
def create_socketio_handlers(socketio):
    """
    Create Flask-SocketIO event handlers

    Usage:
        from flask_socketio import SocketIO
        socketio = SocketIO(app)
        create_socketio_handlers(socketio)
    """
    hub = get_websocket_hub()

    @socketio.on('connect')
    def handle_connect():
        from flask import request
        client_id = request.sid
        hub.connect(
            client_id,
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr
        )

    @socketio.on('disconnect')
    def handle_disconnect():
        from flask import request
        hub.disconnect(request.sid)

    @socketio.on('subscribe')
    def handle_subscribe(data):
        from flask import request
        types = data.get('types', [])
        hub.subscribe(request.sid, types)
        return {'success': True, 'subscribed': types}

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        from flask import request
        types = data.get('types', [])
        hub.unsubscribe(request.sid, types)
        return {'success': True, 'unsubscribed': types}

    @socketio.on('heartbeat')
    def handle_heartbeat():
        from flask import request
        hub.heartbeat(request.sid)
        return {'pong': True}

    @socketio.on('control')
    def handle_control(data):
        from flask import request
        message = WebSocketMessage(
            type=MessageType.CONTROL.value,
            data=data
        )
        hub.handle_message(request.sid, message)
        return {'received': True}

    return hub


if __name__ == "__main__":
    # Test WebSocket hub
    print("=== WebSocket Hub Test ===")

    hub = WebSocketHub()
    hub.start()

    # Simulate client connection
    client1 = hub.connect("client1", "Test Agent", "127.0.0.1")
    client2 = hub.connect("client2", "Test Agent 2", "127.0.0.2")

    print(f"Connected clients: {len(hub.clients)}")

    # Subscribe to additional types
    hub.subscribe("client1", ["prediction", "safety"])
    print(f"Client1 subscriptions: {hub.clients['client1'].subscriptions}")

    # Broadcast state
    hub.broadcast_state({
        'h': 4.0,
        'v': 2.0,
        'fr': 0.32,
        'time': 100.0
    })

    # Broadcast alarm
    hub.broadcast_alarm({
        'level': 'WARNING',
        'message': 'High water level detected',
        'value': 6.2
    })

    # Get stats
    stats = hub.get_stats()
    print(f"Hub stats: {stats}")

    # Get pending messages
    messages = hub.get_pending_messages()
    print(f"Pending messages: {len(messages)}")

    hub.stop()
    print("\nWebSocket Hub test completed!")
