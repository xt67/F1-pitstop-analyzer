"""
SQLAlchemy models for F1 Pit Stop Analyzer.
Supports PostgreSQL for production and SQLite for development.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    ForeignKey, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Team(Base):
    """F1 Team/Constructor."""
    __tablename__ = 'teams'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(String(20))
    color: Mapped[Optional[str]] = mapped_column(String(7))  # Hex color
    country: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Relationships
    drivers: Mapped[List["Driver"]] = relationship(back_populates="team")
    pit_stops: Mapped[List["PitStop"]] = relationship(back_populates="team")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Team(name='{self.name}')>"


class Driver(Base):
    """F1 Driver."""
    __tablename__ = 'drivers'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(3), nullable=False)  # e.g., 'VER', 'HAM'
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    number: Mapped[Optional[int]] = mapped_column(Integer)
    nationality: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Current team (can be null for retired drivers)
    team_id: Mapped[Optional[int]] = mapped_column(ForeignKey('teams.id'))
    team: Mapped[Optional["Team"]] = relationship(back_populates="drivers")
    
    # Relationships
    pit_stops: Mapped[List["PitStop"]] = relationship(back_populates="driver")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_drivers_code', 'code'),
    )
    
    def __repr__(self):
        return f"<Driver(code='{self.code}', name='{self.first_name} {self.last_name}')>"


class Race(Base):
    """F1 Race/Grand Prix."""
    __tablename__ = 'races'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    circuit: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(50))
    date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    total_laps: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Race metadata
    is_sprint: Mapped[bool] = mapped_column(Boolean, default=False)
    weather_conditions: Mapped[Optional[str]] = mapped_column(String(50))
    track_temp: Mapped[Optional[float]] = mapped_column(Float)
    air_temp: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    sessions: Mapped[List["Session"]] = relationship(back_populates="race")
    pit_stops: Mapped[List["PitStop"]] = relationship(back_populates="race")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('year', 'round_number', name='uq_race_year_round'),
        Index('ix_races_year', 'year'),
    )
    
    def __repr__(self):
        return f"<Race(name='{self.name}', year={self.year})>"


class Session(Base):
    """F1 Session (Practice, Qualifying, Race, Sprint)."""
    __tablename__ = 'sessions'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey('races.id'), nullable=False)
    session_type: Mapped[str] = mapped_column(String(20), nullable=False)  # FP1, FP2, FP3, Q, S, R
    date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Session data cache
    data_cached: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_path: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Relationships
    race: Mapped["Race"] = relationship(back_populates="sessions")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        UniqueConstraint('race_id', 'session_type', name='uq_session_race_type'),
    )
    
    def __repr__(self):
        return f"<Session(race_id={self.race_id}, type='{self.session_type}')>"


class PitStop(Base):
    """Individual pit stop record."""
    __tablename__ = 'pit_stops'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Foreign keys
    race_id: Mapped[int] = mapped_column(ForeignKey('races.id'), nullable=False)
    driver_id: Mapped[int] = mapped_column(ForeignKey('drivers.id'), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey('teams.id'), nullable=False)
    
    # Pit stop details
    lap_number: Mapped[int] = mapped_column(Integer, nullable=False)
    stop_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1st, 2nd, 3rd stop
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Tyre information
    tyre_compound_old: Mapped[Optional[str]] = mapped_column(String(20))
    tyre_compound_new: Mapped[Optional[str]] = mapped_column(String(20))
    tyre_life: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Additional context
    position_before: Mapped[Optional[int]] = mapped_column(Integer)
    position_after: Mapped[Optional[int]] = mapped_column(Integer)
    is_under_safety_car: Mapped[bool] = mapped_column(Boolean, default=False)
    is_under_vsc: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps
    pit_in_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    pit_out_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # ML features cache
    features_json: Mapped[Optional[str]] = mapped_column(JSON)
    
    # Relationships
    race: Mapped["Race"] = relationship(back_populates="pit_stops")
    driver: Mapped["Driver"] = relationship(back_populates="pit_stops")
    team: Mapped["Team"] = relationship(back_populates="pit_stops")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="pit_stop")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_pit_stops_race', 'race_id'),
        Index('ix_pit_stops_driver', 'driver_id'),
        Index('ix_pit_stops_team', 'team_id'),
        Index('ix_pit_stops_duration', 'duration_seconds'),
    )
    
    def __repr__(self):
        return f"<PitStop(driver_id={self.driver_id}, lap={self.lap_number}, duration={self.duration_seconds:.2f}s)>"


class Prediction(Base):
    """ML model prediction for a pit stop."""
    __tablename__ = 'predictions'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    
    # Foreign key to pit stop (optional - can be for future predictions)
    pit_stop_id: Mapped[Optional[int]] = mapped_column(ForeignKey('pit_stops.id'))
    
    # Prediction details
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    predicted_duration: Mapped[float] = mapped_column(Float, nullable=False)
    actual_duration: Mapped[Optional[float]] = mapped_column(Float)
    uncertainty: Mapped[Optional[float]] = mapped_column(Float)
    
    # Prediction error metrics
    absolute_error: Mapped[Optional[float]] = mapped_column(Float)
    percentage_error: Mapped[Optional[float]] = mapped_column(Float)
    
    # Anomaly detection
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Feature importance for this prediction
    feature_contributions: Mapped[Optional[str]] = mapped_column(JSON)
    
    # Relationships
    pit_stop: Mapped[Optional["PitStop"]] = relationship(back_populates="predictions")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_predictions_model', 'model_name'),
        Index('ix_predictions_pit_stop', 'pit_stop_id'),
    )
    
    def __repr__(self):
        return f"<Prediction(model='{self.model_name}', predicted={self.predicted_duration:.2f}s)>"


class ModelMetrics(Base):
    """Track ML model performance over time."""
    __tablename__ = 'model_metrics'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Training info
    training_samples: Mapped[int] = mapped_column(Integer)
    training_date: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Performance metrics
    mae: Mapped[float] = mapped_column(Float)  # Mean Absolute Error
    rmse: Mapped[float] = mapped_column(Float)  # Root Mean Square Error
    r2_score: Mapped[float] = mapped_column(Float)
    
    # Cross-validation scores
    cv_mean: Mapped[Optional[float]] = mapped_column(Float)
    cv_std: Mapped[Optional[float]] = mapped_column(Float)
    
    # Model file path
    model_path: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Additional metadata
    hyperparameters: Mapped[Optional[str]] = mapped_column(JSON)
    feature_names: Mapped[Optional[str]] = mapped_column(JSON)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('ix_model_metrics_name_version', 'model_name', 'model_version'),
    )
    
    def __repr__(self):
        return f"<ModelMetrics(model='{self.model_name}', mae={self.mae:.3f})>"
