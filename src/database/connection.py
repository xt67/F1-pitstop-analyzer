"""
Database connection management for F1 Pit Stop Analyzer.
Supports PostgreSQL and SQLite.
"""

import os
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base


class DatabaseConnection:
    """
    Manages database connections with support for PostgreSQL and SQLite.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        use_sqlite: bool = False,
        sqlite_path: str = 'f1_pitstop.db',
        echo: bool = False
    ):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string 
                Format: postgresql://user:password@host:port/database
            use_sqlite: If True, use SQLite instead of PostgreSQL
            sqlite_path: Path to SQLite database file
            echo: If True, log all SQL statements
        """
        self.echo = echo
        
        if use_sqlite or connection_string is None:
            # Use SQLite for development/testing
            self.connection_string = f"sqlite:///{sqlite_path}"
            self.is_postgres = False
        else:
            # Use PostgreSQL for production
            self.connection_string = connection_string
            self.is_postgres = True
        
        self._engine = None
        self._session_factory = None
        
    def _create_engine(self):
        """Create the SQLAlchemy engine."""
        if self.is_postgres:
            self._engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=self.echo
            )
        else:
            # SQLite doesn't support pool settings
            self._engine = create_engine(
                self.connection_string,
                echo=self.echo,
                connect_args={"check_same_thread": False}  # For SQLite
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        return self._engine
    
    @property
    def engine(self):
        """Get or create the database engine."""
        if self._engine is None:
            self._create_engine()
        return self._engine
    
    @property
    def session_factory(self):
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        print(f"Database tables created successfully")
    
    def drop_tables(self):
        """Drop all tables from the database."""
        Base.metadata.drop_all(self.engine)
        print("Database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope for database operations.
        
        Usage:
            with db.session_scope() as session:
                session.add(new_object)
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.session_scope() as session:
                if self.is_postgres:
                    session.execute("SELECT 1")
                else:
                    session.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def close(self):
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


def get_database_from_env() -> DatabaseConnection:
    """
    Create a database connection from environment variables.
    
    Environment variables:
        DATABASE_URL: Full PostgreSQL connection string
        DB_HOST: PostgreSQL host
        DB_PORT: PostgreSQL port (default: 5432)
        DB_NAME: Database name
        DB_USER: Database user
        DB_PASSWORD: Database password
        USE_SQLITE: If 'true', use SQLite instead
        SQLITE_PATH: Path to SQLite file
    """
    # Check for full connection string
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        return DatabaseConnection(connection_string=database_url)
    
    # Check for SQLite preference
    use_sqlite = os.getenv('USE_SQLITE', 'true').lower() == 'true'
    if use_sqlite:
        sqlite_path = os.getenv('SQLITE_PATH', 'data/f1_pitstop.db')
        os.makedirs(os.path.dirname(sqlite_path) or '.', exist_ok=True)
        return DatabaseConnection(use_sqlite=True, sqlite_path=sqlite_path)
    
    # Build PostgreSQL connection string from parts
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    name = os.getenv('DB_NAME', 'f1_pitstop')
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASSWORD', '')
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return DatabaseConnection(connection_string=connection_string)
