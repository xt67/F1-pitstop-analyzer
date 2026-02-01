"""
Enhanced Machine Learning module for F1 Pit Stop Analysis.
Includes:
- Pit stop duration prediction
- Tyre degradation modeling (Bayesian-inspired)
- Lap time prediction
- Anomaly detection
- Optimal pit window suggestions
- Driver performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import os
import warnings

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    IsolationForest
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import scipy.stats as stats


class TyreCategory(Enum):
    """Tyre category classifications."""
    DRY_PRIME = "dry_prime"      # Hard
    DRY_OPTION = "dry_option"    # Medium  
    DRY_QUALIFIER = "dry_qual"   # Soft
    INTERMEDIATE = "intermediate"
    WET = "wet"


@dataclass
class TyreProfile:
    """Profile for a specific tyre compound."""
    compound: str
    category: TyreCategory
    base_degradation: float       # Base degradation rate per lap
    degradation_rate: float       # Fitted degradation rate
    max_degradation: float        # Max degradation before cliff
    grip_level: float             # Relative grip (1.0 = baseline)
    warmup_laps: int              # Laps needed for optimal temp
    
    def get_effective_degradation(self, track_abrasion: float = 1.0) -> float:
        """Calculate effective degradation considering track conditions."""
        return self.degradation_rate * track_abrasion


# Default tyre profiles (can be updated with fitted data)
DEFAULT_TYRE_PROFILES = {
    'SOFT': TyreProfile(
        compound='SOFT',
        category=TyreCategory.DRY_QUALIFIER,
        base_degradation=0.08,
        degradation_rate=0.08,
        max_degradation=3.0,
        grip_level=1.0,
        warmup_laps=1
    ),
    'MEDIUM': TyreProfile(
        compound='MEDIUM',
        category=TyreCategory.DRY_OPTION,
        base_degradation=0.05,
        degradation_rate=0.05,
        max_degradation=4.0,
        grip_level=0.95,
        warmup_laps=2
    ),
    'HARD': TyreProfile(
        compound='HARD',
        category=TyreCategory.DRY_PRIME,
        base_degradation=0.03,
        degradation_rate=0.03,
        max_degradation=5.0,
        grip_level=0.90,
        warmup_laps=3
    ),
    'INTERMEDIATE': TyreProfile(
        compound='INTERMEDIATE',
        category=TyreCategory.INTERMEDIATE,
        base_degradation=0.04,
        degradation_rate=0.04,
        max_degradation=4.5,
        grip_level=0.85,
        warmup_laps=2
    ),
    'WET': TyreProfile(
        compound='WET',
        category=TyreCategory.WET,
        base_degradation=0.02,
        degradation_rate=0.02,
        max_degradation=6.0,
        grip_level=0.75,
        warmup_laps=1
    ),
}


class TyreDegradationModel:
    """
    Bayesian-inspired tyre degradation model.
    Predicts tyre performance and suggests optimal pit windows.
    """
    
    def __init__(self):
        self.tyre_profiles = DEFAULT_TYRE_PROFILES.copy()
        self.track_abrasion = 1.0
        self.fuel_effect = 0.035  # Seconds per kg of fuel
        self.starting_fuel = 110.0  # kg
        self.fuel_burn_rate = 1.5  # kg per lap
        self._fitted = False
        
        # State space model parameters
        self.sigma_eta = 0.5    # State noise
        self.sigma_epsilon = 0.3  # Observation noise
        
        # Driver-specific latent states
        self._latent_states: Dict[str, List[float]] = {}
        self._latent_uncertainty: Dict[str, List[float]] = {}
        
    def fit(self, laps_df: pd.DataFrame, driver: Optional[str] = None) -> None:
        """
        Fit the degradation model to lap data.
        
        Args:
            laps_df: DataFrame with lap times and tyre info
            driver: Optional specific driver to fit
        """
        if driver:
            laps_df = laps_df[laps_df['Driver'] == driver]
        
        laps_clean = self._prepare_data(laps_df)
        
        if laps_clean.empty:
            print("Warning: No valid laps after data preparation")
            return
        
        # Estimate track-specific abrasion
        self.track_abrasion = self._estimate_track_abrasion(laps_clean)
        
        # Estimate compound-specific degradation rates
        self._estimate_degradation_rates(laps_clean)
        
        # Compute latent driver states
        self._compute_latent_states(laps_clean)
        
        self._fitted = True
        
    def _prepare_data(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate lap data."""
        laps = laps_df.copy()
        
        # Ensure required columns
        required_cols = ['LapNumber', 'LapTime', 'Driver', 'Compound', 'TyreLife']
        for col in required_cols:
            if col not in laps.columns:
                return pd.DataFrame()
        
        # Convert lap time to seconds if needed
        if 'LapTimeSeconds' not in laps.columns:
            laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        # Filter invalid laps
        is_pit_out = laps.get('PitOutTime', pd.Series([None] * len(laps))).notna()
        is_pit_in = laps.get('PitInTime', pd.Series([None] * len(laps))).notna()
        
        valid_laps = laps[
            (laps['LapNumber'] > 1) &
            ~is_pit_in &
            ~is_pit_out &
            (laps['LapTimeSeconds'] > 60) &
            (laps['LapTimeSeconds'] < 180)
        ]
        
        return valid_laps
    
    def _estimate_track_abrasion(self, laps_df: pd.DataFrame) -> float:
        """Estimate track abrasion factor from lap data."""
        abrasion_samples = []
        
        for compound in laps_df['Compound'].unique():
            if compound not in self.tyre_profiles:
                continue
                
            base_rate = self.tyre_profiles[compound].base_degradation
            compound_laps = laps_df[laps_df['Compound'] == compound]
            
            for driver in compound_laps['Driver'].unique():
                driver_laps = compound_laps[compound_laps['Driver'] == driver]
                driver_laps = driver_laps.sort_values('LapNumber')
                
                if len(driver_laps) < 5:
                    continue
                
                # Calculate fuel-corrected lap times
                fuel_correction = self.fuel_effect * (
                    self.starting_fuel - 
                    (driver_laps['LapNumber'] - 1) * self.fuel_burn_rate
                ).clip(lower=0)
                
                corrected = driver_laps['LapTimeSeconds'] - fuel_correction
                delta = corrected - corrected.iloc[0]
                
                if delta.std() > 0:
                    slope, _, _, _ = stats.theilslopes(
                        delta.values,
                        driver_laps['TyreLife'].values
                    )
                    if slope > 0:
                        abrasion_samples.append(slope / base_rate)
        
        if len(abrasion_samples) < 3:
            return 1.0
        
        return float(np.clip(np.median(abrasion_samples), 0.7, 1.4))
    
    def _estimate_degradation_rates(self, laps_df: pd.DataFrame) -> None:
        """Estimate compound-specific degradation rates."""
        for compound in laps_df['Compound'].unique():
            if compound not in self.tyre_profiles:
                continue
            
            compound_laps = laps_df[laps_df['Compound'] == compound]
            
            degradation_samples = []
            
            for driver in compound_laps['Driver'].unique():
                driver_laps = compound_laps[compound_laps['Driver'] == driver]
                driver_laps = driver_laps.sort_values('TyreLife')
                
                if len(driver_laps) < 3:
                    continue
                
                # Regress lap time against tyre life
                X = driver_laps['TyreLife'].values.reshape(-1, 1)
                y = driver_laps['LapTimeSeconds'].values
                
                try:
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression().fit(X, y)
                    if reg.coef_[0] > 0:
                        degradation_samples.append(reg.coef_[0])
                except:
                    pass
            
            if degradation_samples:
                self.tyre_profiles[compound].degradation_rate = np.median(degradation_samples)
    
    def _compute_latent_states(self, laps_df: pd.DataFrame) -> None:
        """Compute latent driver performance states using Kalman-like filtering."""
        for driver in laps_df['Driver'].unique():
            driver_laps = laps_df[laps_df['Driver'] == driver].sort_values('LapNumber')
            
            states = []
            uncertainties = []
            
            # Initialize
            alpha_0 = driver_laps['LapTimeSeconds'].median()
            P_0 = 2.0
            
            alpha_t = alpha_0
            P_t = P_0
            
            for _, lap in driver_laps.iterrows():
                # Prediction step
                alpha_pred = alpha_t
                P_pred = P_t + self.sigma_eta ** 2
                
                # Update step
                y = lap['LapTimeSeconds']
                K = P_pred / (P_pred + self.sigma_epsilon ** 2)
                
                alpha_t = alpha_pred + K * (y - alpha_pred)
                P_t = (1 - K) * P_pred
                
                states.append(alpha_t)
                uncertainties.append(P_t)
            
            self._latent_states[driver] = states
            self._latent_uncertainty[driver] = uncertainties
    
    def predict_tyre_health(
        self,
        compound: str,
        laps_on_tyre: int,
        track_condition: str = 'DRY'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict current tyre health percentage.
        
        Returns:
            Tuple of (health_percentage, info_dict)
        """
        if compound not in self.tyre_profiles:
            return 100.0, {'error': 'Unknown compound'}
        
        tyre = self.tyre_profiles[compound]
        effective_deg = tyre.get_effective_degradation(self.track_abrasion)
        
        # Calculate cumulative degradation
        total_deg = laps_on_tyre * effective_deg
        
        # Calculate health (100% = fresh, 0% = cliff)
        max_laps = tyre.max_degradation / max(effective_deg, 0.001)
        health = max(0, min(100, 100 * (1 - laps_on_tyre / max_laps)))
        
        # Estimate laps remaining
        laps_remaining = max(0, max_laps - laps_on_tyre)
        
        info = {
            'compound': compound,
            'laps_on_tyre': laps_on_tyre,
            'degradation_rate': effective_deg,
            'total_degradation': total_deg,
            'laps_remaining': int(laps_remaining),
            'cliff_approaching': health < 25,
            'track_abrasion': self.track_abrasion,
        }
        
        return health, info
    
    def suggest_pit_window(
        self,
        current_lap: int,
        total_laps: int,
        current_compound: str,
        laps_on_tyre: int,
        traffic_density: float = 0.5
    ) -> Dict[str, Any]:
        """
        Suggest optimal pit window based on current race state.
        
        Returns:
            Dictionary with pit window suggestions
        """
        health, _ = self.predict_tyre_health(current_compound, laps_on_tyre)
        
        remaining_laps = total_laps - current_lap
        
        # Calculate optimal strategies
        strategies = []
        
        for target_compound in ['SOFT', 'MEDIUM', 'HARD']:
            if target_compound not in self.tyre_profiles:
                continue
            
            tyre = self.tyre_profiles[target_compound]
            max_stint = int(tyre.max_degradation / tyre.degradation_rate)
            
            # Can this compound finish the race?
            can_finish = remaining_laps <= max_stint * 0.9
            
            strategies.append({
                'compound': target_compound,
                'max_stint': max_stint,
                'can_finish': can_finish,
                'recommended': can_finish and health < 50
            })
        
        # Determine urgency
        if health < 15:
            urgency = 'CRITICAL'
            window = (current_lap, current_lap + 2)
        elif health < 30:
            urgency = 'HIGH'
            window = (current_lap, current_lap + 5)
        elif health < 50:
            urgency = 'MEDIUM'
            window = (current_lap + 3, current_lap + 10)
        else:
            urgency = 'LOW'
            window = (current_lap + 5, current_lap + 15)
        
        return {
            'current_health': health,
            'urgency': urgency,
            'suggested_window': window,
            'strategies': strategies,
            'traffic_factor': traffic_density,
            'undercut_opportunity': traffic_density > 0.6 and health > 40
        }


class EnhancedPitStopPredictor:
    """
    Enhanced ML model for pit stop duration prediction.
    Uses ensemble methods and feature engineering.
    """
    
    def __init__(self, models_dir: str = 'ml_models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Ensemble of models
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'bayesian': BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )
        }
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self._trained = False
        self.model_weights = {'rf': 0.4, 'gbm': 0.4, 'bayesian': 0.2}
        
        # Tyre degradation model
        self.tyre_model = TyreDegradationModel()
        
    def prepare_features(
        self,
        data: pd.DataFrame,
        fit_encoders: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix from pit stop data.
        """
        features = pd.DataFrame()
        
        # Categorical features
        categorical_cols = ['Team', 'Compound', 'Driver']
        for col in categorical_cols:
            if col in data.columns:
                if fit_encoders:
                    self.encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        data[col].fillna('Unknown')
                    )
                elif col in self.encoders:
                    features[f'{col}_encoded'] = self.encoders[col].transform(
                        data[col].fillna('Unknown').apply(
                            lambda x: x if x in self.encoders[col].classes_ else 'Unknown'
                        )
                    )
        
        # Numerical features
        numerical_cols = ['PitLap', 'StopNumber', 'TyreLife']
        for col in numerical_cols:
            if col in data.columns:
                features[col] = data[col].fillna(0)
        
        # Derived features
        if 'PitLap' in data.columns:
            features['lap_percentage'] = data['PitLap'] / data.get('TotalLaps', 60)
            features['lap_squared'] = data['PitLap'] ** 2
            features['is_early_stop'] = (data['PitLap'] < 15).astype(int)
            features['is_late_stop'] = (data['PitLap'] > 45).astype(int)
        
        if 'StopNumber' in data.columns:
            features['is_first_stop'] = (data['StopNumber'] == 1).astype(int)
            features['multi_stop'] = (data['StopNumber'] > 2).astype(int)
        
        self.feature_names = list(features.columns)
        
        return features.values
    
    def train(
        self,
        training_data: pd.DataFrame,
        target_col: str = 'PitDuration_Seconds'
    ) -> Dict[str, float]:
        """
        Train the ensemble model.
        
        Returns:
            Dictionary with training metrics
        """
        if training_data is None or len(training_data) < 50:
            print("Insufficient training data")
            return {'error': 'Insufficient data'}
        
        # Filter valid data
        valid_data = training_data[
            (training_data[target_col] > 15) &
            (training_data[target_col] < 60)
        ].copy()
        
        if len(valid_data) < 50:
            return {'error': 'Insufficient valid data after filtering'}
        
        # Prepare features
        X = self.prepare_features(valid_data, fit_encoders=True)
        y = valid_data[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        metrics = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics[f'{name}_mae'] = mae
            metrics[f'{name}_r2'] = r2
            
            print(f"  {name}: MAE={mae:.3f}s, R²={r2:.3f}")
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        self._trained = True
        metrics['n_samples'] = len(valid_data)
        metrics['n_features'] = X.shape[1]
        
        return metrics
    
    def predict(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pit stop durations with uncertainty.
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.prepare_features(data, fit_encoders=False)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Weighted ensemble
        ensemble_pred = sum(
            self.model_weights[name] * pred 
            for name, pred in predictions.items()
        )
        
        # Estimate uncertainty from model disagreement
        pred_array = np.array(list(predictions.values()))
        uncertainty = pred_array.std(axis=0)
        
        return ensemble_pred, uncertainty
    
    def detect_anomalies(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect anomalous pit stops.
        
        Returns:
            DataFrame with anomaly labels and scores
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before anomaly detection")
        
        X = self.prepare_features(data, fit_encoders=False)
        X_scaled = self.scaler.transform(X)
        
        anomaly_labels = self.anomaly_detector.predict(X_scaled)
        anomaly_scores = self.anomaly_detector.score_samples(X_scaled)
        
        result = data.copy()
        result['is_anomaly'] = anomaly_labels == -1
        result['anomaly_score'] = anomaly_scores
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest model."""
        if not self._trained:
            return pd.DataFrame()
        
        importance = self.models['rf'].feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filename: str = 'enhanced_pitstop_model.pkl') -> str:
        """Save trained model to file."""
        filepath = os.path.join(self.models_dir, filename)
        
        model_state = {
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'anomaly_detector': self.anomaly_detector,
            'trained': self._trained,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filename: str = 'enhanced_pitstop_model.pkl') -> bool:
        """Load trained model from file."""
        filepath = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            self.models = model_state['models']
            self.encoders = model_state['encoders']
            self.scaler = model_state['scaler']
            self.feature_names = model_state['feature_names']
            self.model_weights = model_state['model_weights']
            self.anomaly_detector = model_state['anomaly_detector']
            self._trained = model_state['trained']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class DriverPerformanceAnalyzer:
    """Analyzes driver and team pit stop performance over time."""
    
    @staticmethod
    def analyze_team_trends(
        pit_stops: pd.DataFrame,
        window: int = 5
    ) -> pd.DataFrame:
        """
        Analyze team pit stop performance trends.
        
        Args:
            pit_stops: Historical pit stop data
            window: Rolling window for trend analysis
        
        Returns:
            DataFrame with team performance trends
        """
        results = []
        
        for team in pit_stops['Team'].unique():
            team_data = pit_stops[pit_stops['Team'] == team].sort_values('Date')
            
            if len(team_data) < window:
                continue
            
            # Calculate rolling statistics
            rolling_mean = team_data['PitDuration_Seconds'].rolling(window).mean()
            rolling_std = team_data['PitDuration_Seconds'].rolling(window).std()
            
            # Trend direction
            recent_mean = team_data['PitDuration_Seconds'].tail(window).mean()
            overall_mean = team_data['PitDuration_Seconds'].mean()
            
            trend = 'improving' if recent_mean < overall_mean else 'declining'
            
            results.append({
                'team': team,
                'overall_mean': overall_mean,
                'recent_mean': recent_mean,
                'best_time': team_data['PitDuration_Seconds'].min(),
                'worst_time': team_data['PitDuration_Seconds'].max(),
                'consistency': team_data['PitDuration_Seconds'].std(),
                'total_stops': len(team_data),
                'trend': trend,
                'improvement': overall_mean - recent_mean
            })
        
        return pd.DataFrame(results).sort_values('recent_mean')
    
    @staticmethod
    def compare_drivers(
        pit_stops: pd.DataFrame,
        driver1: str,
        driver2: str
    ) -> Dict[str, Any]:
        """Compare pit stop performance between two drivers."""
        d1_data = pit_stops[pit_stops['Driver'] == driver1]
        d2_data = pit_stops[pit_stops['Driver'] == driver2]
        
        return {
            'driver1': {
                'name': driver1,
                'mean': d1_data['PitDuration_Seconds'].mean(),
                'best': d1_data['PitDuration_Seconds'].min(),
                'count': len(d1_data)
            },
            'driver2': {
                'name': driver2,
                'mean': d2_data['PitDuration_Seconds'].mean(),
                'best': d2_data['PitDuration_Seconds'].min(),
                'count': len(d2_data)
            },
            'advantage': d1_data['PitDuration_Seconds'].mean() - d2_data['PitDuration_Seconds'].mean()
        }


def collect_training_data(
    years: List[int],
    max_races_per_year: int = 10,
    progress_callback=None
) -> pd.DataFrame:
    """
    Collect training data from multiple seasons.
    
    Args:
        years: List of years to collect data from
        max_races_per_year: Maximum races to load per year
        progress_callback: Optional callback for progress updates
    
    Returns:
        DataFrame with collected pit stop data
    """
    import fastf1
    
    all_data = []
    
    for year in years:
        if progress_callback:
            progress_callback(f"Loading {year} season schedule...")
        
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint_qualifying', 'sprint_shootout', 'sprint'])]
            
            race_count = 0
            for _, event in races.iterrows():
                if race_count >= max_races_per_year:
                    break
                
                round_num = event['RoundNumber']
                race_name = event['EventName']
                
                if progress_callback:
                    progress_callback(f"Loading {year} {race_name}...")
                
                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load()
                    
                    laps = session.laps
                    
                    # Extract pit stops
                    pit_laps = laps[laps['PitInTime'].notna()].copy()
                    
                    for _, lap in pit_laps.iterrows():
                        pit_time = lap.get('PitOutTime', None) 
                        if pit_time is not None and lap.get('PitInTime') is not None:
                            duration = (pit_time - lap['PitInTime']).total_seconds()
                            if 15 < duration < 60:
                                all_data.append({
                                    'Year': year,
                                    'Race': race_name,
                                    'Date': event['EventDate'],
                                    'Driver': lap['Driver'],
                                    'Team': lap['Team'],
                                    'PitLap': lap['LapNumber'],
                                    'Compound': lap.get('Compound', 'UNKNOWN'),
                                    'TyreLife': lap.get('TyreLife', 0),
                                    'StopNumber': lap.get('Stint', 1),
                                    'PitDuration_Seconds': duration,
                                    'TotalLaps': laps['LapNumber'].max()
                                })
                    
                    race_count += 1
                    
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error loading {race_name}: {str(e)[:50]}")
                    continue
                    
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error loading {year} schedule: {str(e)[:50]}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    if progress_callback:
        progress_callback(f"Collected {len(df)} pit stops from {len(df['Race'].unique())} races")
    
    return df
