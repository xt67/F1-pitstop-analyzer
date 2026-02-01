"""
F1 Data Loading and Processing for Pit Stop Analyzer.
Handles session loading, pit stop extraction, and data caching.
"""

import fastf1
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from src.lib.settings import get_settings


def enable_cache():
    """Enable FastF1 cache."""
    settings = get_settings()
    fastf1.Cache.enable_cache(settings.cache_path)


def get_available_seasons():
    """Get list of available F1 seasons (2018 onwards has good data)."""
    current_year = datetime.now().year
    max_year = min(current_year, 2025)
    return list(range(2018, max_year + 1))


def get_season_schedule(year: int) -> Optional[pd.DataFrame]:
    """Fetch the race schedule for a given season."""
    try:
        enable_cache()
        schedule = fastf1.get_event_schedule(year)
        # Filter to only show races (not testing sessions)
        races = schedule[schedule['EventFormat'].notna()]
        # Filter out future races (no data available)
        today = pd.Timestamp.now()
        past_races = races[races['EventDate'] < today]
        return past_races
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")
        return None


def load_session(year: int, round_number: int, session_type: str = 'R'):
    """Load a FastF1 session."""
    enable_cache()
    session = fastf1.get_session(year, round_number, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def analyze_pit_stops(session, year: int, race_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze pit stops for the given session."""
    laps = session.laps

    # Get laps where car EXITED the pit (PitOutTime exists = this is an "out lap")
    out_laps = laps[laps['PitOutTime'].notna()][
        ['Driver', 'Team', 'LapNumber', 'PitOutTime', 'Compound', 'Stint']
    ].copy()

    # Get laps where car ENTERED the pit (PitInTime exists = this is an "in lap")
    # IMPORTANT: Exclude Lap 1 - those aren't real pit stops, just race start data
    in_laps = laps[(laps['PitInTime'].notna()) & (laps['LapNumber'] > 1)][
        ['Driver', 'LapNumber', 'PitInTime']
    ].copy()
    in_laps['NextLap'] = in_laps['LapNumber'] + 1  # The out lap is the next lap

    # Merge: match each pit entry with its pit exit
    pit_data = pd.merge(
        in_laps,
        out_laps,
        left_on=['Driver', 'NextLap'],
        right_on=['Driver', 'LapNumber'],
        suffixes=('_in', '_out')
    )

    # Calculate pit stop duration
    pit_data['PitDuration'] = pit_data['PitOutTime'] - pit_data['PitInTime']
    pit_data['PitDuration_Seconds'] = pit_data['PitDuration'].dt.total_seconds()

    # Clean up columns
    pit_data = pit_data[
        ['Driver', 'Team', 'LapNumber_in', 'PitInTime', 'PitOutTime', 
         'PitDuration_Seconds', 'Compound', 'Stint']
    ]
    pit_data = pit_data.rename(columns={'LapNumber_in': 'PitLap'})

    return pit_data, laps


def get_tire_strategy(laps: pd.DataFrame) -> pd.DataFrame:
    """Extract tire strategy for each driver."""
    strategy = laps.groupby(['Driver', 'Stint']).agg({
        'Compound': 'first',
        'LapNumber': ['min', 'max', 'count'],
        'Team': 'first'
    }).reset_index()

    strategy.columns = ['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'LapCount', 'Team']
    return strategy
