"""
Database setup and connection management for memory system.

This module handles:
- SQLAlchemy engine creation
- Session management
- Connection pooling configuration
- Database initialization
"""

from sqlalchemy import create_engine, event, inspect, pool, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import os
from typing import AsyncGenerator, Generator
import logging

import urllib.parse
from memory.models import Base

logger = logging.getLogger(__name__)

import dotenv

dotenv.load_dotenv()


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
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1500"))
        
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
    _db_type = None
    
    @classmethod
    def initialize(cls, config: DatabaseConfig = None):
        """
        Initialize database engine and session factory.
        
        ✅ FIXED: Tries Azure SQL first, falls back to SQLite if unavailable
        ✅ App can start even if DB is unreachable
        """
        
        print('in database manager')
        
        if cls._engine is not None:
            logger.warning("DatabaseManager already initialized")
            return

        if config is None:
            config = DatabaseConfig()

        logger.info(f"Initializing memory database...")
        
        # Try Azure SQL first
        if "mssql" in config.connection_string.lower() or "azure" in config.connection_string.lower():
            try:
                connection_uri = cls._build_connection_uri(config.connection_string)
                cls._engine = cls._create_engine(connection_uri, config)
                cls._db_type = "azure_sql"
                cls._SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=cls._engine,
                    expire_on_commit=False
                )
                logger.info("✓ Using Azure SQL Server")
                cls.create_tables()
                cls._validate_schema()
                logger.info("✓ Memory database initialized successfully")
                return
            except Exception as e:
                logger.error(f"❌ Azure SQL connection failed: {e}")
                logger.warning("⚠️  Falling back to SQLite in-memory...")

        # Fallback to SQLite for development
        logger.info("🔄 Using SQLite in-memory database (development mode)")
        cls._engine = create_engine(
            "sqlite:///:memory:",
            echo=config.echo,
            connect_args={"check_same_thread": False}
        )
        cls._db_type = "sqlite"
        
        cls._SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=cls._engine,
            expire_on_commit=False
        )
        
        # Create tables
        cls.create_tables()
        logger.info("✓ Memory database initialized successfully (SQLite in-memory)")

    @classmethod
    def _create_engine(cls, connection_uri: str, config: DatabaseConfig):
        """Create SQLAlchemy engine with pooling"""
        return create_engine(
            connection_uri,
            poolclass=pool.QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_recycle=3600,
            echo=False,
            pool_timeout=30,
            connect_args={"timeout": 10}
        )

    @classmethod
    def _validate_schema(cls):
        """
        ✅ FIXED: Validate database schema matches SQLAlchemy models
        """
        if cls._engine is None:
            logger.warning("Cannot validate schema: engine not initialized")
            return

        try:
            inspector = inspect(cls._engine)
            
            for model_table_name, model_table in Base.metadata.tables.items():
                if model_table_name not in inspector.get_table_names():
                    logger.debug(f"Table not yet created: {model_table_name}")
                    continue

                # Get existing columns from database
                existing_columns = {
                    col['name'] for col in inspector.get_columns(model_table_name)
                }

                # Get expected columns from model
                model_columns = {col.name for col in model_table.columns}

                # Check for mismatches
                missing_in_db = model_columns - existing_columns
                extra_in_db = existing_columns - model_columns

                if missing_in_db or extra_in_db:
                    error_msg = f"Schema mismatch for table '{model_table_name}':\n"
                    if missing_in_db:
                        error_msg += f" Missing in DB: {missing_in_db}\n"
                    if extra_in_db:
                        error_msg += f" Extra in DB: {extra_in_db}\n"
                    error_msg += " Action: Run database migrations or recreate the table."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            logger.info("✓ Database schema validation passed")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            # Don't fail on validation error for SQLite
            if cls._db_type == "sqlite":
                logger.warning("Skipping schema validation for SQLite")
            else:
                raise RuntimeError(f"Failed to validate database schema: {e}")

    @classmethod
    def _build_connection_uri(cls, connection_string: str) -> str:
        """
        Build SQLAlchemy connection URI from connection string.
        Supports Azure SQL (pymssql) and PostgreSQL.
        """
        if not connection_string:
            raise ValueError("Connection string is empty")

        # PostgreSQL
        if "postgres" in connection_string.lower():
            return connection_string

        # SQLite
        if "sqlite" in connection_string.lower():
            return connection_string

        # Azure SQL - parse connection string
        parts = {}
        for part in connection_string.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                parts[key.strip()] = value.strip()

        # Validate required fields
        required = ['Server', 'Initial Catalog', 'User ID', 'Password']
        missing = [f for f in required if f not in parts]

        if missing:
            raise ValueError(f"Invalid connection string: missing {missing}")

        server = parts['Server'].replace('tcp:', '').split(',')[0]
        database = parts['Initial Catalog']
        user = parts['User ID']
        password = parts['Password']

        password_encoded = urllib.parse.quote_plus(password)
        connection_uri = f"mssql+pymssql://{user}:{password_encoded}@{server}:1433/{database}"

        return connection_uri

    @classmethod
    def create_tables(cls):
        """
        ✅ FIXED: Create all tables if they don't exist (IDEMPOTENT)
        Works with both Azure SQL and SQLite
        """
        if cls._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        try:
            inspector = inspect(cls._engine)
            existing_tables = set(inspector.get_table_names())

            # Define table creation order (dependencies first)
            table_creation_order = [
                "conversations",  # No dependencies
                "chat_messages",  # Depends on conversations
                "conversation_metadata",  # Depends on conversations
            ]

            # Create only missing tables
            for table_name in table_creation_order:
                if table_name in Base.metadata.tables:
                    table = Base.metadata.tables[table_name]
                    
                    if table_name not in existing_tables:
                        table.create(cls._engine, checkfirst=True)
                        existing_tables.add(table_name)
                        logger.info(f"✓ Created table: {table_name}")
                    else:
                        logger.info(f"✓ Table already exists: {table_name}")

            logger.info("✓ Database initialization completed successfully")

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
            try:
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Commit failed: {e}")
                raise
        except Exception as e:
            try:
                session.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            logger.error(f"Session error: {e}")
            raise
        finally:
            try:
                session.close()
            except Exception as close_error:
                logger.error(f"Failed to close session: {close_error}")

    @classmethod
    def health_check(cls) -> bool:
        """Check if database is healthy"""
        try:
            if cls._SessionLocal is None:
                return False
                
            session = cls._SessionLocal()
            session.execute(text("SELECT 1"))
            session.close()
            logger.info("✓ Database health check passed")
            return True
        except SQLAlchemyError as e:
            logger.error(f"❌ Database health check failed: {e}")
            return False

    @classmethod
    def get_engine(cls):
        """Get the SQLAlchemy engine (for migrations, admin tasks)"""
        if cls._engine is None:
            raise RuntimeError("Database not initialized")
        return cls._engine

    @classmethod
    def is_using_sqlite(cls) -> bool:
        """Check if currently using SQLite (development)"""
        return cls._db_type == "sqlite"

    @classmethod
    def is_using_azure_sql(cls) -> bool:
        """Check if currently using Azure SQL (production)"""
        return cls._db_type == "azure_sql"


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
