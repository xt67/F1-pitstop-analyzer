"""
Time formatting utilities for F1 data.
"""

from datetime import timedelta
from typing import Union


def format_pit_duration(seconds: float) -> str:
    """Format pit stop duration in seconds to a readable string."""
    if seconds is None:
        return "N/A"
    return f"{seconds:.2f}s"


def format_lap_time(seconds: float) -> str:
    """Format lap time from seconds to M:SS.mmm format."""
    if seconds is None:
        return "N/A"
    
    minutes = int(seconds // 60)
    remaining = seconds % 60
    
    return f"{minutes}:{remaining:06.3f}"


def format_time_delta(delta: Union[timedelta, float]) -> str:
    """Format a time delta to a readable string."""
    if isinstance(delta, timedelta):
        total_seconds = delta.total_seconds()
    else:
        total_seconds = delta
    
    if total_seconds is None:
        return "N/A"
    
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    
    if total_seconds < 60:
        return f"{sign}{total_seconds:.3f}s"
    
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{sign}{minutes}:{seconds:06.3f}"
