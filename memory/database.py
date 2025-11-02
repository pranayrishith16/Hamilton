"""
Database setup and connection management for memory system.

This module handles:
- SQLAlchemy engine creation
- Session management
- Connection pooling configuration
- Database initialization
"""

from sqlalchemy import create_engine, event, inspect, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import os
from typing import AsyncGenerator, Generator
import logging

import urllib.parse
from memory.models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connections"""
    
    def __init__(self):
        # Read from environment (same connection string as auth)
        self.connection_string = os.getenv("AZURE_SQL_CONNECTION_STRING")
        
        if not self.connection_string:
            # Fallback to local PostgreSQL for development
            self.connection_string = os.getenv(
                "MEMORY_DATABASE_URL",
                "postgresql://veritly:password@localhost/veritly_db"
            )
        
        # Connection pooling
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Echo SQL for debugging (set False in production)
        self.echo = os.getenv("DB_ECHO", "False").lower() == "true"
        
        logger.info(f"Memory Database config: pool_size={self.pool_size}, max_overflow={self.max_overflow}")


class DatabaseManager:
    """
    Manages database connections and session lifecycle.
    
    Usage:
        db_manager = DatabaseManager()
        with db_manager.get_session() as session:
            # Do database operations
            pass
    """
    
    _engine = None
    _SessionLocal = None
    
    @classmethod
    def initialize(cls, config: DatabaseConfig = None):
        """
        Initialize database engine and session factory.
        Call this once at application startup.
        
        Args:
            config: DatabaseConfig object (uses defaults if None)
        """
        if cls._engine is not None:
            logger.warning("DatabaseManager already initialized")
            return
        
        if config is None:
            config = DatabaseConfig()
        
        logger.info(f"Initializing database with URL: {config.connection_string[:50]}...")

        connection_uri = cls._build_connection_uri(config.connection_string)
        
        # Create engine with connection pooling
        cls._engine = create_engine(
            connection_uri,
            poolclass=pool.QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_recycle=config.pool_recycle,
            echo=config.echo,
        )
        
        # Create session factory
        cls._SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=cls._engine,
            expire_on_commit=False
        )
        
        # Set up event listeners for PostgreSQL extensions
        if "postgresql" in connection_uri.lower():
            cls._setup_postgresql_listeners()
        
        # Create tables with proper ordering
        cls.create_tables()
        
        logger.info("✓ Memory database initialized successfully")
    
    @classmethod
    def _build_connection_uri(cls, connection_string: str) -> str:
        """
        Build SQLAlchemy connection URI from connection string.
        Supports both Azure SQL (pymssql) and PostgreSQL.
        
        Args:
            connection_string: Either Azure SQL or PostgreSQL connection string
        
        Returns:
            SQLAlchemy connection URI
        """
        if "postgres" in connection_string.lower():
            # PostgreSQL connection string (for development)
            return connection_string
        
        # Parse Azure SQL connection string
        # Format: Server=tcp:server.database.windows.net,1433;Initial Catalog=db;User ID=user;Password=pass
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
        
        logger.info(f"Connecting to Azure SQL: {server}/{database}")
        return connection_uri

    @classmethod
    def _setup_postgresql_listeners(cls):
        """Configure PostgreSQL-specific event listeners"""
        
        @event.listens_for(cls._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Enable pgvector and pgcrypto extensions on connection"""
            try:
                cursor = dbapi_conn.cursor()
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                dbapi_conn.commit()
                cursor.close()
                logger.debug("PostgreSQL extensions enabled")
            except Exception as e:
                logger.warning(f"Could not enable PostgreSQL extensions: {e}")
        
        @event.listens_for(cls._engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log SQL statements in debug mode"""
            if os.getenv("DEBUG", "False").lower() == "true":
                logger.debug(f"QUERY: {statement}")

    @classmethod
    def create_tables(cls):
        """
        Create all tables if they don't exist (IDEMPOTENT).
        Check existing tables first, only create missing ones.
        """
        if cls._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        try:
            from sqlalchemy import inspect
            
            inspector = inspect(cls._engine)
            existing_tables = set(inspector.get_table_names())
            print(existing_tables)
            
            # Define table creation order (dependencies first)
            table_creation_order = [
                "conversations",              # No dependencies
                "chat_messages",              # Depends on conversations
                "conversation_metadata",      # Depends on conversations
            ]
            
            # Create only missing tables
            for table_name in table_creation_order:
                if table_name in Base.metadata.tables:
                    table = Base.metadata.tables[table_name]
                    if table_name not in existing_tables:
                        print(f'creating table: {table_name}')
                        table.create(cls._engine, checkfirst=True)
                        existing_tables.add(table_name)
                        logger.info(f"✓ Created table: {table_name}")
                    else:
                        print(f"✓ Table already exists: {table_name}")
            
            print("✓ Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            import traceback
            traceback.print_exc()
            raise

    @classmethod
    def drop_tables(cls):
        """
        Drop all tables. USE WITH CAUTION (for testing only).
        """
        if cls._engine is None:
            raise RuntimeError("Database not initialized")
        
        logger.warning("DROPPING ALL MEMORY TABLES - THIS IS DESTRUCTIVE")
        Base.metadata.drop_all(bind=cls._engine)
        logger.info("All memory tables dropped")

    @classmethod
    def get_session(cls) -> Generator[Session, None, None]:
        """
        FastAPI dependency for getting a database session.
        
        Usage in FastAPI:
            @app.post("/api/query")
            async def query(db: Session = Depends(DatabaseManager.get_session)):
                ...
        
        Yields:
            SQLAlchemy Session
        """
        if cls._SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = cls._SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def health_check(cls) -> bool:
        try:
            session = cls._SessionLocal()
            session.execute(text("SELECT 1"))  # Use text() from SQLAlchemy
            session.close()
            logger.info("✓ Database health check passed")
            return True
        except SQLAlchemyError as e:
            logger.error(f"❌ Database health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error in health check: {e}")
            return False

    @classmethod
    def get_engine(cls):
        """Get the SQLAlchemy engine (for migrations, admin tasks)"""
        if cls._engine is None:
            raise RuntimeError("Database not initialized")
        return cls._engine


# ============ FastAPI Dependencies ============

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.post("/api/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    session = DatabaseManager._SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        session.close()
