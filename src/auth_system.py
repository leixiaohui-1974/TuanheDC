#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAOS V3.5 - User Authentication System
团河渡槽自主运行系统 - 用户认证模块

Features:
- User management (CRUD)
- Role-based access control (RBAC)
- JWT token authentication
- API key management
- Session management
- Password hashing
- Audit logging
"""

import json
import sqlite3
import hashlib
import secrets
import hmac
import base64
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from functools import wraps
from pathlib import Path


class UserRole(Enum):
    """User roles"""
    VIEWER = "viewer"           # Read-only access
    OPERATOR = "operator"       # Can operate controls
    ENGINEER = "engineer"       # Can modify configurations
    ADMIN = "admin"             # Full access
    SYSTEM = "system"           # System account


class Permission(Enum):
    """System permissions"""
    # View permissions
    VIEW_STATE = "view_state"
    VIEW_HISTORY = "view_history"
    VIEW_CONFIG = "view_config"
    VIEW_LOGS = "view_logs"
    VIEW_REPORTS = "view_reports"

    # Control permissions
    CONTROL_MANUAL = "control_manual"
    CONTROL_SCENARIO = "control_scenario"
    CONTROL_EMERGENCY = "control_emergency"

    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_CONFIG = "manage_config"
    MANAGE_SYSTEM = "manage_system"

    # Special permissions
    API_ACCESS = "api_access"
    EXPORT_DATA = "export_data"


# Role-Permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.VIEWER: {
        Permission.VIEW_STATE,
        Permission.VIEW_HISTORY,
        Permission.VIEW_REPORTS,
    },
    UserRole.OPERATOR: {
        Permission.VIEW_STATE,
        Permission.VIEW_HISTORY,
        Permission.VIEW_CONFIG,
        Permission.VIEW_LOGS,
        Permission.VIEW_REPORTS,
        Permission.CONTROL_MANUAL,
        Permission.CONTROL_SCENARIO,
        Permission.API_ACCESS,
    },
    UserRole.ENGINEER: {
        Permission.VIEW_STATE,
        Permission.VIEW_HISTORY,
        Permission.VIEW_CONFIG,
        Permission.VIEW_LOGS,
        Permission.VIEW_REPORTS,
        Permission.CONTROL_MANUAL,
        Permission.CONTROL_SCENARIO,
        Permission.MANAGE_CONFIG,
        Permission.API_ACCESS,
        Permission.EXPORT_DATA,
    },
    UserRole.ADMIN: {
        p for p in Permission
    },
    UserRole.SYSTEM: {
        p for p in Permission
    },
}


@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    password_hash: str
    role: UserRole
    email: str = ""
    full_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a permission"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
        }
        if include_sensitive:
            data['api_key'] = self.api_key
        return data


@dataclass
class Session:
    """Session data structure"""
    session_id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: str = ""
    user_agent: str = ""
    is_valid: bool = True

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class APIKey:
    """API key data structure"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    permissions: Set[Permission] = field(default_factory=set)


class PasswordHasher:
    """Secure password hashing"""

    @staticmethod
    def hash(password: str, salt: str = None) -> str:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use PBKDF2-like derivation
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${base64.b64encode(key).decode('utf-8')}"

    @staticmethod
    def verify(password: str, hash_str: str) -> bool:
        """Verify password against hash"""
        try:
            salt, _ = hash_str.split('$')
            return hmac.compare_digest(
                PasswordHasher.hash(password, salt),
                hash_str
            )
        except Exception:
            return False


class TokenManager:
    """JWT-like token management"""

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry = timedelta(hours=8)

    def create_token(self, user_id: str, extra_data: Dict = None) -> str:
        """Create authentication token"""
        payload = {
            'user_id': user_id,
            'iat': datetime.now().isoformat(),
            'exp': (datetime.now() + self.token_expiry).isoformat(),
            **(extra_data or {})
        }

        # Encode payload
        payload_str = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()

        # Create signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload_str}.{signature}"

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify token and return payload"""
        try:
            parts = token.split('.')
            if len(parts) != 2:
                return None

            payload_str, signature = parts

            # Verify signature
            expected_sig = hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return None

            # Decode payload
            payload = json.loads(
                base64.urlsafe_b64decode(payload_str.encode())
            )

            # Check expiry
            exp = datetime.fromisoformat(payload['exp'])
            if datetime.now() > exp:
                return None

            return payload

        except Exception:
            return None


class UserStorage:
    """SQLite-based user storage"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                api_key TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_valid INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used TEXT,
                is_active INTEGER DEFAULT 1,
                permissions TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                ip_address TEXT,
                details TEXT,
                success INTEGER
            )
        """)

        conn.commit()
        conn.close()

        # Create default admin user if not exists
        self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user"""
        if self.get_user_by_username('admin') is None:
            admin = User(
                user_id=secrets.token_hex(8),
                username='admin',
                password_hash=PasswordHasher.hash('admin123'),  # Change in production!
                role=UserRole.ADMIN,
                email='admin@taos.local',
                full_name='System Administrator'
            )
            self.save_user(admin)

    def save_user(self, user: User):
        """Save user to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO users
            (user_id, username, password_hash, role, email, full_name,
             created_at, last_login, is_active, api_key, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user.user_id,
            user.username,
            user.password_hash,
            user.role.value,
            user.email,
            user.full_name,
            user.created_at.isoformat(),
            user.last_login.isoformat() if user.last_login else None,
            1 if user.is_active else 0,
            user.api_key,
            json.dumps(user.metadata)
        ))

        conn.commit()
        conn.close()

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_user(row)
        return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_user(row)
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE api_key = ?", (api_key,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_user(row)
        return None

    def get_all_users(self) -> List[User]:
        """Get all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users ORDER BY created_at")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_user(row) for row in rows]

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return deleted

    def save_session(self, session: Session):
        """Save session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, user_id, token, created_at, expires_at,
             ip_address, user_agent, is_valid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.user_id,
            session.token,
            session.created_at.isoformat(),
            session.expires_at.isoformat(),
            session.ip_address,
            session.user_agent,
            1 if session.is_valid else 0
        ))

        conn.commit()
        conn.close()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Session(
                session_id=row[0],
                user_id=row[1],
                token=row[2],
                created_at=datetime.fromisoformat(row[3]),
                expires_at=datetime.fromisoformat(row[4]),
                ip_address=row[5] or "",
                user_agent=row[6] or "",
                is_valid=bool(row[7])
            )
        return None

    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE sessions SET is_valid = 0 WHERE session_id = ?",
            (session_id,)
        )

        conn.commit()
        conn.close()

    def log_auth_event(self, user_id: str, action: str, ip_address: str,
                       details: str = None, success: bool = True):
        """Log authentication event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO auth_audit
            (timestamp, user_id, action, ip_address, details, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            action,
            ip_address,
            details,
            1 if success else 0
        ))

        conn.commit()
        conn.close()

    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row[0],
            username=row[1],
            password_hash=row[2],
            role=UserRole(row[3]),
            email=row[4] or "",
            full_name=row[5] or "",
            created_at=datetime.fromisoformat(row[6]),
            last_login=datetime.fromisoformat(row[7]) if row[7] else None,
            is_active=bool(row[8]),
            api_key=row[9],
            metadata=json.loads(row[10]) if row[10] else {}
        )


class AuthManager:
    """
    Main authentication manager for TAOS
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "data" / "auth.db")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.storage = UserStorage(db_path)
        self.token_manager = TokenManager()
        self.lock = threading.RLock()

        # Active sessions cache
        self.session_cache: Dict[str, Session] = {}

        # Rate limiting for login attempts
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)

    def authenticate(self, username: str, password: str,
                    ip_address: str = "", user_agent: str = "") -> Optional[Session]:
        """Authenticate user with username and password"""
        # Check rate limiting
        if self._is_locked_out(username):
            self.storage.log_auth_event(
                username, "login_lockout", ip_address,
                "Account locked due to too many failed attempts", False
            )
            return None

        # Get user
        user = self.storage.get_user_by_username(username)
        if not user or not user.is_active:
            self._record_failed_attempt(username)
            self.storage.log_auth_event(
                username, "login_failed", ip_address,
                "User not found or inactive", False
            )
            return None

        # Verify password
        if not PasswordHasher.verify(password, user.password_hash):
            self._record_failed_attempt(username)
            self.storage.log_auth_event(
                user.user_id, "login_failed", ip_address,
                "Invalid password", False
            )
            return None

        # Create session
        session = self._create_session(user, ip_address, user_agent)

        # Update last login
        user.last_login = datetime.now()
        self.storage.save_user(user)

        # Clear failed attempts
        self.login_attempts.pop(username, None)

        # Log success
        self.storage.log_auth_event(
            user.user_id, "login_success", ip_address, success=True
        )

        return session

    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate using token"""
        payload = self.token_manager.verify_token(token)
        if not payload:
            return None

        user = self.storage.get_user(payload['user_id'])
        if user and user.is_active:
            return user
        return None

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        user = self.storage.get_user_by_api_key(api_key)
        if user and user.is_active:
            return user
        return None

    def logout(self, session_id: str, ip_address: str = ""):
        """Logout and invalidate session"""
        session = self.session_cache.pop(session_id, None)
        if session:
            self.storage.invalidate_session(session_id)
            self.storage.log_auth_event(
                session.user_id, "logout", ip_address, success=True
            )

    def create_user(self, username: str, password: str, role: UserRole,
                   email: str = "", full_name: str = "",
                   created_by: str = "system") -> Optional[User]:
        """Create new user"""
        # Check if username exists
        if self.storage.get_user_by_username(username):
            return None

        user = User(
            user_id=secrets.token_hex(8),
            username=username,
            password_hash=PasswordHasher.hash(password),
            role=role,
            email=email,
            full_name=full_name
        )

        self.storage.save_user(user)
        self.storage.log_auth_event(
            user.user_id, "user_created", "",
            f"Created by {created_by}", True
        )

        return user

    def update_user(self, user_id: str, updates: Dict[str, Any],
                   updated_by: str = "system") -> bool:
        """Update user"""
        user = self.storage.get_user(user_id)
        if not user:
            return False

        for key, value in updates.items():
            if key == 'password':
                user.password_hash = PasswordHasher.hash(value)
            elif key == 'role':
                user.role = UserRole(value) if isinstance(value, str) else value
            elif hasattr(user, key):
                setattr(user, key, value)

        self.storage.save_user(user)
        self.storage.log_auth_event(
            user_id, "user_updated", "",
            f"Updated by {updated_by}: {list(updates.keys())}", True
        )

        return True

    def delete_user(self, user_id: str, deleted_by: str = "system") -> bool:
        """Delete user"""
        if self.storage.delete_user(user_id):
            self.storage.log_auth_event(
                user_id, "user_deleted", "",
                f"Deleted by {deleted_by}", True
            )
            return True
        return False

    def change_password(self, user_id: str, old_password: str,
                       new_password: str) -> bool:
        """Change user password"""
        user = self.storage.get_user(user_id)
        if not user:
            return False

        if not PasswordHasher.verify(old_password, user.password_hash):
            return False

        user.password_hash = PasswordHasher.hash(new_password)
        self.storage.save_user(user)

        self.storage.log_auth_event(
            user_id, "password_changed", "", success=True
        )

        return True

    def generate_api_key(self, user_id: str) -> Optional[str]:
        """Generate API key for user"""
        user = self.storage.get_user(user_id)
        if not user:
            return None

        api_key = secrets.token_urlsafe(32)
        user.api_key = api_key
        self.storage.save_user(user)

        self.storage.log_auth_event(
            user_id, "api_key_generated", "", success=True
        )

        return api_key

    def revoke_api_key(self, user_id: str) -> bool:
        """Revoke user's API key"""
        user = self.storage.get_user(user_id)
        if not user:
            return False

        user.api_key = None
        self.storage.save_user(user)

        self.storage.log_auth_event(
            user_id, "api_key_revoked", "", success=True
        )

        return True

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.storage.get_user(user_id)

    def get_all_users(self) -> List[User]:
        """Get all users"""
        return self.storage.get_all_users()

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission"""
        return user.has_permission(permission)

    def _create_session(self, user: User, ip_address: str,
                       user_agent: str) -> Session:
        """Create new session"""
        session_id = secrets.token_hex(16)
        token = self.token_manager.create_token(user.user_id, {
            'username': user.username,
            'role': user.role.value
        })

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            token=token,
            created_at=datetime.now(),
            expires_at=datetime.now() + self.token_manager.token_expiry,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.session_cache[session_id] = session
        self.storage.save_session(session)

        return session

    def _is_locked_out(self, username: str) -> bool:
        """Check if user is locked out"""
        attempts = self.login_attempts.get(username, [])
        recent = [a for a in attempts
                 if datetime.now() - a < self.lockout_duration]

        return len(recent) >= self.max_login_attempts

    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.login_attempts:
            self.login_attempts[username] = []
        self.login_attempts[username].append(datetime.now())


def require_auth(permission: Permission = None):
    """Decorator for requiring authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # This would integrate with Flask request context
            # For now, just a placeholder
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# Global instance
_auth_manager = None


def get_auth_manager() -> AuthManager:
    """Get global authentication manager"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


if __name__ == "__main__":
    # Test authentication system
    print("=== Authentication System Test ===")

    auth = AuthManager()

    # Test user creation
    print("\n1. Creating test user...")
    user = auth.create_user(
        username="testuser",
        password="testpass123",
        role=UserRole.OPERATOR,
        email="test@example.com",
        full_name="Test User"
    )
    if user:
        print(f"   Created: {user.username} ({user.role.value})")

    # Test authentication
    print("\n2. Testing authentication...")
    session = auth.authenticate("testuser", "testpass123", "127.0.0.1")
    if session:
        print(f"   Session created: {session.session_id[:16]}...")
        print(f"   Token: {session.token[:32]}...")

    # Test token verification
    print("\n3. Testing token verification...")
    if session:
        verified_user = auth.authenticate_token(session.token)
        if verified_user:
            print(f"   Token verified for: {verified_user.username}")

    # Test API key
    print("\n4. Testing API key...")
    if user:
        api_key = auth.generate_api_key(user.user_id)
        if api_key:
            print(f"   API key: {api_key[:20]}...")
            api_user = auth.authenticate_api_key(api_key)
            if api_user:
                print(f"   API key verified for: {api_user.username}")

    # Test permissions
    print("\n5. Testing permissions...")
    if user:
        can_control = auth.check_permission(user, Permission.CONTROL_MANUAL)
        can_manage = auth.check_permission(user, Permission.MANAGE_USERS)
        print(f"   Can control: {can_control}")
        print(f"   Can manage users: {can_manage}")

    # List users
    print("\n6. All users:")
    for u in auth.get_all_users():
        print(f"   - {u.username} ({u.role.value})")

    # Cleanup
    if user:
        auth.delete_user(user.user_id)

    print("\nAuthentication system test completed!")
