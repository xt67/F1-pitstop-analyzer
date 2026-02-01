"""
ML Pit Stop Predictor for F1 Pit Stop Analyzer.
Uses scikit-learn for prediction and anomaly detection.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Callable

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

import fastf1

from src.lib.settings import get_settings


class PitStopPredictor:
    """ML model for pit stop duration prediction and analysis."""

    def __init__(self):
        self.model = None
        self.anomaly_detector = None
        self.team_encoder = LabelEncoder()
        self.compound_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None

    def prepare_features(self, pit_data, laps_data=None):
        """Prepare features for ML model from pit stop data."""
        df = pit_data.copy()

        # Basic features
        df['Team_Encoded'] = self.team_encoder.fit_transform(df['Team'].astype(str))
        df['Compound_Encoded'] = self.compound_encoder.fit_transform(df['Compound'].astype(str))

        # Pit lap as percentage of race
        if laps_data is not None:
            total_laps = laps_data['LapNumber'].max()
            df['PitLap_Pct'] = df['PitLap'] / total_laps
        else:
            df['PitLap_Pct'] = df['PitLap'] / 60  # Assume 60 lap race

        # Stint number (derived from sequential pit stops per driver)
        df['StintNumber'] = df.groupby('Driver').cumcount() + 1

        # Feature columns
        feature_cols = ['Team_Encoded', 'Compound_Encoded', 'PitLap', 'PitLap_Pct', 'StintNumber']

        return df, feature_cols

    def train(self, pit_data, laps_data=None):
        """Train the pit stop duration prediction model."""
        if len(pit_data) < 10:
            print("⚠️ Not enough data to train model (need at least 10 pit stops)")
            return False

        df, feature_cols = self.prepare_features(pit_data, laps_data)

        # Filter valid data
        valid_data = df[df['PitDuration_Seconds'].notna() & (df['PitDuration_Seconds'] > 0)]

        if len(valid_data) < 10:
            print("⚠️ Not enough valid pit stop data for training")
            return False

        X = valid_data[feature_cols].values
        y = valid_data['PitDuration_Seconds'].values

        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # Train anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector.fit(X)

        # Store feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        self.is_trained = True

        return True

    def predict(self, team, compound, pit_lap, stint_number=1, total_laps=60):
        """Predict pit stop duration for given parameters."""
        if not self.is_trained:
            return None

        try:
            team_encoded = self.team_encoder.transform([team])[0]
        except ValueError:
            team_encoded = 0  # Unknown team

        try:
            compound_encoded = self.compound_encoder.transform([compound])[0]
        except ValueError:
            compound_encoded = 0  # Unknown compound

        pit_lap_pct = pit_lap / total_laps

        features = np.array([[team_encoded, compound_encoded, pit_lap, pit_lap_pct, stint_number]])
        prediction = self.model.predict(features)[0]

        return prediction

    def detect_anomalies(self, pit_data, laps_data=None):
        """Detect anomalous pit stops (unusually slow or problematic)."""
        if not self.is_trained or self.anomaly_detector is None:
            return pit_data

        df, feature_cols = self.prepare_features(pit_data, laps_data)
        valid_mask = df['PitDuration_Seconds'].notna() & (df['PitDuration_Seconds'] > 0)

        if valid_mask.sum() == 0:
            return pit_data

        X = df.loc[valid_mask, feature_cols].values
        anomaly_scores = self.anomaly_detector.predict(X)

        # -1 = anomaly, 1 = normal
        df.loc[valid_mask, 'IsAnomaly'] = anomaly_scores == -1
        df['IsAnomaly'] = df['IsAnomaly'].fillna(False)

        return df

    def get_optimal_pit_windows(self, total_laps, team, compound_sequence=['MEDIUM', 'HARD']):
        """Suggest optimal pit windows based on trained model."""
        if not self.is_trained:
            return None

        predictions = []
        for lap in range(1, total_laps + 1):
            for stint, compound in enumerate(compound_sequence, 1):
                pred = self.predict(team, compound, lap, stint, total_laps)
                if pred:
                    predictions.append({
                        'Lap': lap,
                        'Compound': compound,
                        'Stint': stint,
                        'PredictedDuration': pred
                    })

        if not predictions:
            return None

        pred_df = pd.DataFrame(predictions)

        optimal_windows = []
        for stint in pred_df['Stint'].unique():
            stint_data = pred_df[pred_df['Stint'] == stint]
            if len(stint_data) >= 5:
                stint_data = stint_data.sort_values('Lap')
                rolling_avg = stint_data['PredictedDuration'].rolling(5, center=True).mean()
                best_idx = rolling_avg.idxmin()
                if pd.notna(best_idx):
                    best_lap = stint_data.loc[best_idx, 'Lap']
                    optimal_windows.append({
                        'Stint': stint,
                        'OptimalLap': int(best_lap),
                        'WindowStart': max(1, int(best_lap) - 2),
                        'WindowEnd': min(total_laps, int(best_lap) + 2),
                        'ExpectedDuration': stint_data.loc[best_idx, 'PredictedDuration']
                    })

        return optimal_windows

    def save_model(self, filepath=None):
        """Save trained model to disk."""
        if not self.is_trained:
            print("⚠️ No trained model to save")
            return False

        settings = get_settings()
        ml_path = settings.ml_models_path

        if filepath is None:
            if not os.path.exists(ml_path):
                os.makedirs(ml_path)
            filepath = os.path.join(ml_path, 'pitstop_predictor.pkl')

        model_data = {
            'model': self.model,
            'anomaly_detector': self.anomaly_detector,
            'team_encoder': self.team_encoder,
            'compound_encoder': self.compound_encoder,
            'feature_importance': self.feature_importance
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved: {filepath}")
        return True

    def load_model(self, filepath=None):
        """Load trained model from disk."""
        settings = get_settings()

        if filepath is None:
            filepath = os.path.join(settings.ml_models_path, 'pitstop_predictor.pkl')

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.anomaly_detector = model_data['anomaly_detector']
            self.team_encoder = model_data['team_encoder']
            self.compound_encoder = model_data['compound_encoder']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = True

            print(f"✅ Model loaded: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def collect_training_data(
    years: List[int] = None,
    max_races_per_year: int = 5,
    progress_callback: Callable[[str], None] = None
) -> Optional[pd.DataFrame]:
    """Collect pit stop data from multiple races for ML training."""
    if years is None:
        years = [2023, 2024, 2025]

    all_pit_data = []

    # Import here to avoid circular imports
    from src.f1_data import get_season_schedule, analyze_pit_stops

    msg = f"Collecting data from {len(years)} seasons..."
    if progress_callback:
        progress_callback(msg)
    else:
        print(f"\n🔄 {msg}")

    fastf1.Cache.enable_cache('cache')

    for year in years:
        try:
            schedule = get_season_schedule(year)
            if schedule is None or len(schedule) == 0:
                continue

            races_to_load = min(len(schedule), max_races_per_year)

            for idx in range(races_to_load):
                race = schedule.iloc[idx]
                round_num = race['RoundNumber']
                race_name = race['EventName']

                msg = f"Loading {year} {race_name}..."
                if progress_callback:
                    progress_callback(msg)
                else:
                    print(f"   {msg}", end=" ")

                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load(laps=True, telemetry=False, weather=False, messages=False)

                    pit_data, laps = analyze_pit_stops(session, year, race_name)

                    if len(pit_data) > 0:
                        pit_data['Year'] = year
                        pit_data['Race'] = race_name
                        pit_data['TotalLaps'] = laps['LapNumber'].max()
                        all_pit_data.append(pit_data)
                        if not progress_callback:
                            print(f"✅ ({len(pit_data)} stops)")
                    else:
                        if not progress_callback:
                            print("⚠️ No pit data")

                except Exception as e:
                    if not progress_callback:
                        print(f"❌ {str(e)[:30]}")

        except Exception as e:
            if not progress_callback:
                print(f"   ❌ Error with {year}: {e}")

    if all_pit_data:
        combined_data = pd.concat(all_pit_data, ignore_index=True)
        msg = f"Collected {len(combined_data)} pit stops from {len(all_pit_data)} races"
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"\n✅ {msg}")
        return combined_data

    return None
