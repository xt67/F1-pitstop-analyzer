"""
F1 team colors and tire compound colors for visualization.
"""

# F1 Team Colors (2024-2025 liveries)
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Mercedes': '#27F4D2',
    'Ferrari': '#E8002D',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'RB': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
    # Historical teams
    'Alfa Romeo': '#C92D4B',
    'AlphaTauri': '#5E8FAA',
    'Racing Point': '#F596C8',
    'Renault': '#FFF500',
    'Toro Rosso': '#469BFF',
}

# Tire compound colors
COMPOUND_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFF200',
    'HARD': '#EBEBEB',
    'INTERMEDIATE': '#43B02A',
    'WET': '#0067AD',
    'UNKNOWN': '#888888',
    'TEST_UNKNOWN': '#888888',
}

# Compound emoji mappings
COMPOUND_EMOJI = {
    'SOFT': '🔴',
    'MEDIUM': '🟡',
    'HARD': '⚪',
    'INTERMEDIATE': '🟢',
    'WET': '🔵',
}


def get_team_color(team_name: str, default: str = '#888888') -> str:
    """Get the color for a team, with fallback."""
    return TEAM_COLORS.get(team_name, default)


def get_compound_color(compound: str, default: str = '#888888') -> str:
    """Get the color for a tire compound, with fallback."""
    return COMPOUND_COLORS.get(compound, default)


def get_compound_emoji(compound: str, default: str = '⚫') -> str:
    """Get the emoji for a tire compound."""
    return COMPOUND_EMOJI.get(compound, default)
