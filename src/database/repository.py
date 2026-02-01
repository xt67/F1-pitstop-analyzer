"""
Database repository for F1 Pit Stop Analyzer.
Provides high-level data access methods.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import Session, joinedload

from .models import Race, Driver, Team, PitStop, Prediction, Session as RaceSession, ModelMetrics
from .connection import DatabaseConnection


class DatabaseRepository:
    """
    Repository for database operations.
    Provides CRUD operations and complex queries.
    """
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    # ==================== Team Operations ====================
    
    def get_or_create_team(self, session: Session, name: str, **kwargs) -> Team:
        """Get existing team or create new one."""
        team = session.query(Team).filter(Team.name == name).first()
        if not team:
            team = Team(name=name, **kwargs)
            session.add(team)
            session.flush()
        return team
    
    def get_all_teams(self) -> List[Team]:
        """Get all teams."""
        with self.db.session_scope() as session:
            return session.query(Team).all()
    
    # ==================== Driver Operations ====================
    
    def get_or_create_driver(
        self, 
        session: Session, 
        code: str, 
        first_name: str, 
        last_name: str,
        team: Optional[Team] = None,
        **kwargs
    ) -> Driver:
        """Get existing driver or create new one."""
        driver = session.query(Driver).filter(Driver.code == code).first()
        if not driver:
            driver = Driver(
                code=code,
                first_name=first_name,
                last_name=last_name,
                team=team,
                **kwargs
            )
            session.add(driver)
            session.flush()
        return driver
    
    def get_driver_by_code(self, code: str) -> Optional[Driver]:
        """Get driver by 3-letter code."""
        with self.db.session_scope() as session:
            return session.query(Driver).filter(Driver.code == code).first()
    
    # ==================== Race Operations ====================
    
    def get_or_create_race(
        self,
        session: Session,
        year: int,
        round_number: int,
        name: str,
        **kwargs
    ) -> Race:
        """Get existing race or create new one."""
        race = session.query(Race).filter(
            and_(Race.year == year, Race.round_number == round_number)
        ).first()
        if not race:
            race = Race(year=year, round_number=round_number, name=name, **kwargs)
            session.add(race)
            session.flush()
        return race
    
    def get_race(self, year: int, round_number: int) -> Optional[Race]:
        """Get a specific race."""
        with self.db.session_scope() as session:
            return session.query(Race).filter(
                and_(Race.year == year, Race.round_number == round_number)
            ).first()
    
    def get_races_by_year(self, year: int) -> List[Race]:
        """Get all races for a given year."""
        with self.db.session_scope() as session:
            return session.query(Race).filter(Race.year == year).order_by(Race.round_number).all()
    
    # ==================== Pit Stop Operations ====================
    
    def add_pit_stop(
        self,
        session: Session,
        race: Race,
        driver: Driver,
        team: Team,
        lap_number: int,
        stop_number: int,
        duration_seconds: float,
        **kwargs
    ) -> PitStop:
        """Add a new pit stop record."""
        pit_stop = PitStop(
            race=race,
            driver=driver,
            team=team,
            lap_number=lap_number,
            stop_number=stop_number,
            duration_seconds=duration_seconds,
            **kwargs
        )
        session.add(pit_stop)
        session.flush()
        return pit_stop
    
    def get_pit_stops_for_race(self, race_id: int) -> List[PitStop]:
        """Get all pit stops for a race."""
        with self.db.session_scope() as session:
            return session.query(PitStop).filter(
                PitStop.race_id == race_id
            ).options(
                joinedload(PitStop.driver),
                joinedload(PitStop.team)
            ).order_by(PitStop.lap_number).all()
    
    def get_pit_stops_by_driver(self, driver_code: str, limit: int = 100) -> List[PitStop]:
        """Get pit stops for a specific driver."""
        with self.db.session_scope() as session:
            driver = session.query(Driver).filter(Driver.code == driver_code).first()
            if not driver:
                return []
            
            return session.query(PitStop).filter(
                PitStop.driver_id == driver.id
            ).options(
                joinedload(PitStop.race),
                joinedload(PitStop.team)
            ).order_by(desc(PitStop.id)).limit(limit).all()
    
    def get_pit_stops_by_team(self, team_name: str, limit: int = 100) -> List[PitStop]:
        """Get pit stops for a specific team."""
        with self.db.session_scope() as session:
            team = session.query(Team).filter(Team.name == team_name).first()
            if not team:
                return []
            
            return session.query(PitStop).filter(
                PitStop.team_id == team.id
            ).options(
                joinedload(PitStop.race),
                joinedload(PitStop.driver)
            ).order_by(desc(PitStop.id)).limit(limit).all()
    
    def get_pit_stops_dataframe(
        self,
        years: Optional[List[int]] = None,
        teams: Optional[List[str]] = None,
        min_duration: float = 15.0,
        max_duration: float = 60.0
    ) -> pd.DataFrame:
        """
        Get pit stops as a pandas DataFrame for ML training.
        
        Args:
            years: Filter by years (None = all years)
            teams: Filter by team names (None = all teams)
            min_duration: Minimum pit stop duration
            max_duration: Maximum pit stop duration
        
        Returns:
            DataFrame with pit stop data
        """
        with self.db.session_scope() as session:
            query = session.query(
                PitStop.id,
                PitStop.lap_number.label('PitLap'),
                PitStop.stop_number.label('StopNumber'),
                PitStop.duration_seconds.label('PitDuration_Seconds'),
                PitStop.tyre_compound_new.label('Compound'),
                PitStop.tyre_life.label('TyreLife'),
                PitStop.is_under_safety_car,
                PitStop.is_under_vsc,
                Driver.code.label('Driver'),
                Team.name.label('Team'),
                Race.year.label('Year'),
                Race.name.label('Race'),
                Race.total_laps.label('TotalLaps'),
                Race.date.label('Date')
            ).join(Driver).join(Team).join(Race)
            
            # Apply filters
            query = query.filter(
                PitStop.duration_seconds >= min_duration,
                PitStop.duration_seconds <= max_duration
            )
            
            if years:
                query = query.filter(Race.year.in_(years))
            
            if teams:
                query = query.filter(Team.name.in_(teams))
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            return pd.DataFrame(results, columns=[
                'id', 'PitLap', 'StopNumber', 'PitDuration_Seconds', 'Compound',
                'TyreLife', 'is_under_safety_car', 'is_under_vsc',
                'Driver', 'Team', 'Year', 'Race', 'TotalLaps', 'Date'
            ])
    
    # ==================== Prediction Operations ====================
    
    def add_prediction(
        self,
        session: Session,
        pit_stop: Optional[PitStop],
        model_name: str,
        model_version: str,
        predicted_duration: float,
        **kwargs
    ) -> Prediction:
        """Add a new prediction record."""
        prediction = Prediction(
            pit_stop=pit_stop,
            model_name=model_name,
            model_version=model_version,
            predicted_duration=predicted_duration,
            **kwargs
        )
        session.add(prediction)
        session.flush()
        return prediction
    
    def update_prediction_with_actual(
        self,
        prediction_id: int,
        actual_duration: float
    ) -> None:
        """Update a prediction with the actual duration."""
        with self.db.session_scope() as session:
            prediction = session.query(Prediction).get(prediction_id)
            if prediction:
                prediction.actual_duration = actual_duration
                prediction.absolute_error = abs(actual_duration - prediction.predicted_duration)
                if actual_duration > 0:
                    prediction.percentage_error = (prediction.absolute_error / actual_duration) * 100
    
    # ==================== Analytics Queries ====================
    
    def get_team_performance_stats(
        self,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """Get team pit stop performance statistics."""
        with self.db.session_scope() as session:
            query = session.query(
                Team.name,
                func.count(PitStop.id).label('total_stops'),
                func.avg(PitStop.duration_seconds).label('avg_duration'),
                func.min(PitStop.duration_seconds).label('best_duration'),
                func.max(PitStop.duration_seconds).label('worst_duration'),
                func.stddev(PitStop.duration_seconds).label('std_duration')
            ).join(PitStop).join(Race)
            
            if year:
                query = query.filter(Race.year == year)
            
            query = query.group_by(Team.name).order_by(func.avg(PitStop.duration_seconds))
            
            results = query.all()
            
            return pd.DataFrame(results, columns=[
                'team', 'total_stops', 'avg_duration', 'best_duration', 
                'worst_duration', 'std_duration'
            ])
    
    def get_driver_performance_stats(
        self,
        year: Optional[int] = None
    ) -> pd.DataFrame:
        """Get driver pit stop performance statistics."""
        with self.db.session_scope() as session:
            query = session.query(
                Driver.code,
                Team.name.label('team'),
                func.count(PitStop.id).label('total_stops'),
                func.avg(PitStop.duration_seconds).label('avg_duration'),
                func.min(PitStop.duration_seconds).label('best_duration')
            ).join(PitStop).join(Team).join(Race)
            
            if year:
                query = query.filter(Race.year == year)
            
            query = query.group_by(Driver.code, Team.name).order_by(func.avg(PitStop.duration_seconds))
            
            results = query.all()
            
            return pd.DataFrame(results, columns=[
                'driver', 'team', 'total_stops', 'avg_duration', 'best_duration'
            ])
    
    def get_model_accuracy_stats(self, model_name: str) -> Dict[str, Any]:
        """Get accuracy statistics for a model."""
        with self.db.session_scope() as session:
            predictions = session.query(Prediction).filter(
                and_(
                    Prediction.model_name == model_name,
                    Prediction.actual_duration.isnot(None)
                )
            ).all()
            
            if not predictions:
                return {}
            
            errors = [p.absolute_error for p in predictions if p.absolute_error is not None]
            
            return {
                'model_name': model_name,
                'total_predictions': len(predictions),
                'mae': sum(errors) / len(errors) if errors else None,
                'max_error': max(errors) if errors else None,
                'min_error': min(errors) if errors else None,
            }
    
    # ==================== Bulk Operations ====================
    
    def import_pit_stops_from_dataframe(
        self,
        df: pd.DataFrame,
        progress_callback=None
    ) -> Tuple[int, int]:
        """
        Import pit stops from a DataFrame.
        
        Args:
            df: DataFrame with pit stop data
            progress_callback: Optional callback for progress updates
        
        Returns:
            Tuple of (imported_count, skipped_count)
        """
        imported = 0
        skipped = 0
        
        with self.db.session_scope() as session:
            for i, row in df.iterrows():
                try:
                    # Get or create team
                    team = self.get_or_create_team(session, row['Team'])
                    
                    # Get or create driver
                    driver = self.get_or_create_driver(
                        session,
                        code=row['Driver'],
                        first_name=row.get('FirstName', row['Driver']),
                        last_name=row.get('LastName', ''),
                        team=team
                    )
                    
                    # Get or create race
                    race = self.get_or_create_race(
                        session,
                        year=row['Year'],
                        round_number=row.get('Round', 1),
                        name=row['Race'],
                        total_laps=row.get('TotalLaps')
                    )
                    
                    # Add pit stop
                    self.add_pit_stop(
                        session,
                        race=race,
                        driver=driver,
                        team=team,
                        lap_number=row['PitLap'],
                        stop_number=row.get('StopNumber', 1),
                        duration_seconds=row['PitDuration_Seconds'],
                        tyre_compound_new=row.get('Compound'),
                        tyre_life=row.get('TyreLife')
                    )
                    
                    imported += 1
                    
                    if progress_callback and i % 100 == 0:
                        progress_callback(f"Imported {imported} pit stops...")
                        
                except Exception as e:
                    skipped += 1
                    if progress_callback:
                        progress_callback(f"Error on row {i}: {str(e)[:50]}")
        
        return imported, skipped
    
    # ==================== Model Metrics ====================
    
    def save_model_metrics(
        self,
        model_name: str,
        model_version: str,
        training_samples: int,
        mae: float,
        rmse: float,
        r2_score: float,
        **kwargs
    ) -> ModelMetrics:
        """Save model training metrics."""
        with self.db.session_scope() as session:
            metrics = ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                training_samples=training_samples,
                mae=mae,
                rmse=rmse,
                r2_score=r2_score,
                **kwargs
            )
            session.add(metrics)
            session.flush()
            return metrics
    
    def get_latest_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get the latest metrics for a model."""
        with self.db.session_scope() as session:
            return session.query(ModelMetrics).filter(
                ModelMetrics.model_name == model_name
            ).order_by(desc(ModelMetrics.training_date)).first()
