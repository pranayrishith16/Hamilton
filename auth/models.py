"""
SQLAlchemy models for user authentication.
Uses pymssql driver (no ODBC required).
"""

from sqlalchemy import ForeignKey, create_engine, Column, String, Integer, Boolean, DateTime, Text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import urllib.parse
import uuid
import dotenv
import json

dotenv.load_dotenv()

Base = declarative_base()

# Global engine instance (singleton)
_engine = None
_SessionLocal = None

class User(Base):
    """User accounts with subscription tiers"""
    __tablename__ = "users"
    
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    company = Column(String(255))

    # Email verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255), unique=True, nullable=True)
    email_verified_at = Column(DateTime, nullable=True)
    
    # Subscription tier
    tier = Column(String(50), default="free")
    daily_query_limit = Column(Integer, default=5)
    queries_today = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Roles (relationship)
    roles = relationship("Role", secondary="user_roles", back_populates="users")
    
    # Password reset tokens (relationship)
    password_resets = relationship("PasswordReset", back_populates="user")
    
    # Refresh tokens (relationship)
    refresh_tokens = relationship("RefreshToken", back_populates="user")

class Role(Base):
    """User roles (admin, editor, viewer, etc.)"""
    
    __tablename__ = "roles"
    
    role_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255))
    permissions = Column(Text)  # JSON string of permissions
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to users
    users = relationship("User", secondary="user_roles", back_populates="roles")


class UserRole(Base):
    """Association table for User-Role many-to-many relationship"""
    
    __tablename__ = "user_roles"
    
    user_id = Column(String(36), ForeignKey("users.user_id"), primary_key=True)
    role_id = Column(String(36), ForeignKey("roles.role_id"), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)


class PasswordReset(Base):
    """Password reset tokens with expiration"""
    
    __tablename__ = "password_resets"
    
    reset_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    reset_token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to user
    user = relationship("User", back_populates="password_resets")


class RefreshToken(Base):
    """Refresh tokens for token rotation and long-lived sessions"""
    
    __tablename__ = "refresh_tokens"
    
    token_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    refresh_token = Column(String(500), unique=True, nullable=False)
    access_token = Column(String(500), nullable=True)  # Associated access token
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime, nullable=True)
    rotation_count = Column(Integer, default=0)  # Track token rotations
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    
    # Relationship back to user
    user = relationship("User", back_populates="refresh_tokens")


class QueryLog(Base):
    """Query history for rate limiting and auditing"""
    
    __tablename__ = "query_logs"
    
    log_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    query_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(50))
    response_time_ms = Column(Integer, nullable=True)

class AuditLog(Base):
    """Security audit log for all auth events"""
    
    __tablename__ = "audit_logs"
    
    audit_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False)  # login, logout, register, password_reset, email_verify, etc.
    event_details = Column(Text)  # JSON string
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    status = Column(String(20), default="success")  # success, failure
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def get_engine():
    """Get SQLAlchemy engine using pymssql driver"""
    global _engine

    if _engine is not None:
        return _engine
    
    # Extract from your connection string
    # Format: Server=tcp:server.database.windows.net,1433;Initial Catalog=db;User ID=user;Password=pass
    connection_string = os.getenv("AZURE_SQL_CONNECTION_STRING")
    
    print(f"DEBUG: connection_string = {connection_string}")
    if not connection_string:
        raise ValueError("AZURE_SQL_CONNECTION_STRING not set in .env")
    
    # Parse connection string manually
    parts = {}
    for part in connection_string.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            parts[key.strip()] = value.strip()
    
    server = parts.get('Server', '').replace('tcp:', '').split(',')[0]
    database = parts.get('Initial Catalog', '')
    user = parts.get('User ID', '')
    password = parts.get('Password', '')
    
    # URL encode password if it has special characters
    password_encoded = urllib.parse.quote_plus(password)
    
    # Build pymssql connection string
    connection_uri = f"mssql+pymssql://{user}:{password_encoded}@{server}:1433/{database}"
    
    print(f"Connecting to: {server}/{database}")
    
    _engine = create_engine(connection_uri, pool_size=10, max_overflow=20)
    return _engine


def get_db_session():
    """Get database session"""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    
    return _SessionLocal()

def init_database():
    """
    Initialize database schema safely (IDEMPOTENT).
    Creates tables in correct dependency order.
    """
    try:
        engine = get_engine()
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        
        # Define table creation order (dependencies first)
        table_creation_order = [
            "roles",              # No dependencies
            "users",              # No dependencies
            "user_roles",         # Depends on users, roles
            "refresh_tokens",     # Depends on users
            "password_resets",    # Depends on users
            "query_logs",         # Depends on users
            "audit_logs",         # Depends on users (optional)
        ]
        
        # Create tables in correct order
        for table_name in table_creation_order:
            if table_name in Base.metadata.tables:
                table = Base.metadata.tables[table_name]
                
                if table_name not in existing_tables:
                    print(f"Creating table: {table_name}")
                    table.create(engine, checkfirst=True)
                    existing_tables.add(table_name)
                else:
                    print(f"Table already exists: {table_name}")
        
        # Create default roles
        _create_default_roles()
        
        print("✓ Database initialization completed successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def _create_default_roles():
    """Create default roles if they don't exist"""
    session = get_db_session()
    try:
        # Check existing roles
        existing_roles = {role.name for role in session.query(Role).all()}
        
        default_roles = [
            {
                "name": "admin",
                "description": "Full system access",
                "permissions": json.dumps(["*"])
            },
            {
                "name": "editor",
                "description": "Can manage documents and queries",
                "permissions": json.dumps(["query", "upload", "edit", "delete_own"])
            },
            {
                "name": "viewer",
                "description": "Read-only access",
                "permissions": json.dumps(["query", "view"])
            }
        ]
        
        # Only add roles that don't exist
        created = []
        for role_data in default_roles:
            if role_data["name"] not in existing_roles:
                role = Role(**role_data)
                session.add(role)
                created.append(role_data["name"])
        
        if created:
            session.commit()
            print(f"✓ Created default roles: {created}")
        else:
            print("✓ Default roles already exist")
            
    except Exception as e:
        print(f"⚠ Error managing default roles: {e}")
        session.rollback()
    finally:
        session.close()