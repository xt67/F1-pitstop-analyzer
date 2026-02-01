"""
CLI Race Selection for F1 Pit Stop Analyzer.
Uses questionary and rich for interactive terminal UI.
"""

import sys
import os
from datetime import datetime

from questionary import Style, select, Choice
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

import fastf1
import pandas as pd


# F1-themed style for questionary
F1_STYLE = Style([
    ("pointer", "fg:#e10600 bold"),
    ("selected", "noinherit fg:#64eb34 bold"),
    ("highlighted", "fg:#e10600 bold"),
    ("answer", "fg:#64eb34 bold"),
    ("question", "fg:white bold"),
])


def cli_main():
    """Run the CLI interface for race selection."""
    console = Console()
    
    # Header
    console.print(Panel.fit(
        "[bold red]🏎️ F1 PIT STOP ANALYZER[/bold red]\n"
        "[dim]Analyze pit stops from any F1 race[/dim]",
        border_style="red"
    ))
    
    # Enable cache
    fastf1.Cache.enable_cache('cache')
    
    # Year selection
    current_year = min(datetime.now().year, 2025)
    years = [str(year) for year in range(current_year, 2017, -1)]
    
    year = select(
        "Select season:",
        choices=years,
        qmark="🗓️ ",
        style=F1_STYLE
    ).ask()
    
    if not year:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    year = int(year)
    
    # Load schedule with progress
    with Progress(
        SpinnerColumn(style="bold red"),
        TextColumn("[bold]Loading {task.description}..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("race schedule", total=None)
        
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'].notna()]
            today = pd.Timestamp.now()
            past_races = races[races['EventDate'] < today]
        except Exception as e:
            console.print(f"[red]Error loading schedule: {e}[/red]")
            sys.exit(1)
    
    if len(past_races) == 0:
        console.print(f"[yellow]No completed races found for {year}[/yellow]")
        sys.exit(0)
    
    # Display race table
    table = Table(title=f"🏁 {year} Season - {len(past_races)} Races Available", border_style="red")
    table.add_column("Round", style="cyan", justify="center")
    table.add_column("Race", style="white")
    table.add_column("Date", style="dim")
    
    for _, race in past_races.head(10).iterrows():
        date_str = race['EventDate'].strftime('%b %d') if pd.notna(race['EventDate']) else 'TBD'
        table.add_row(str(race['RoundNumber']), race['EventName'], date_str)
    
    if len(past_races) > 10:
        table.add_row("...", f"[dim]and {len(past_races) - 10} more[/dim]", "")
    
    console.print(table)
    console.print()
    
    # Race selection
    race_choices = [
        Choice(
            title=f"Round {race['RoundNumber']}: {race['EventName']}",
            value={'round': race['RoundNumber'], 'name': race['EventName']}
        )
        for _, race in past_races.iterrows()
    ]
    
    selected = select(
        "Select race to analyze:",
        choices=race_choices,
        qmark="🏎️ ",
        style=F1_STYLE
    ).ask()
    
    if not selected:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    round_number = selected['round']
    race_name = selected['name']
    
    # Analysis options
    console.print()
    analysis_mode = select(
        "Analysis type:",
        choices=[
            Choice(title="📊 Standard Analysis", value="standard"),
            Choice(title="🤖 Analysis with ML Predictions", value="ml"),
        ],
        qmark="📈 ",
        style=F1_STYLE
    ).ask()
    
    if not analysis_mode:
        console.print("[yellow]Cancelled.[/yellow]")
        sys.exit(0)
    
    # Load session with progress
    console.print()
    with Progress(
        SpinnerColumn(style="bold red"),
        TextColumn("[bold]Loading {task.description}..."),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"{race_name} data", total=None)
        
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as e:
            console.print(f"[red]Error loading session: {e}[/red]")
            sys.exit(1)
    
    console.print(f"[green]✅ Loaded {race_name} {year}[/green]")
    console.print()
    
    # Run analysis
    from src.analysis import run_analysis
    run_analysis(session, year, race_name, use_ml=(analysis_mode == "ml"))


def cli_train_model():
    """CLI interface for training ML model."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🤖 ML MODEL TRAINING[/bold blue]\n"
        "[dim]Train pit stop prediction model[/dim]",
        border_style="blue"
    ))
    
    # Training preset selection
    preset = select(
        "Select training preset:",
        choices=[
            Choice(title="⚡ Quick (2024-2025, 3 races each)", value="quick"),
            Choice(title="📊 Standard (2023-2025, 5 races each)", value="standard"),
            Choice(title="🔬 Comprehensive (2022-2025, 8 races each)", value="comprehensive"),
        ],
        qmark="📈 ",
        style=F1_STYLE
    ).ask()
    
    if not preset:
        console.print("[yellow]Cancelled.[/yellow]")
        return
    
    presets = {
        'quick': ([2024, 2025], 3),
        'standard': ([2023, 2024, 2025], 5),
        'comprehensive': ([2022, 2023, 2024, 2025], 8),
    }
    
    years, max_races = presets[preset]
    
    console.print(f"\n[bold]Training on {len(years)} seasons, {max_races} races each...[/bold]\n")
    
    from src.ml.predictor import PitStopPredictor, collect_training_data
    
    # Collect data
    with Progress(
        SpinnerColumn(style="bold blue"),
        TextColumn("[bold]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Collecting training data...", total=None)
        
        training_data = collect_training_data(years, max_races)
        
        if training_data is None or len(training_data) == 0:
            console.print("[red]❌ No training data collected[/red]")
            return
        
        progress.update(task, description=f"Training on {len(training_data)} pit stops...")
        
        predictor = PitStopPredictor()
        success = predictor.train(training_data)
    
    if success:
        predictor.save_model()
        console.print(f"[green]✅ Model trained successfully on {len(training_data)} pit stops![/green]")
        
        # Show feature importance
        console.print("\n[bold]Feature Importance:[/bold]")
        for feature, importance in sorted(predictor.feature_importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(importance * 30)
            console.print(f"  {feature:<20} {importance:.3f} [blue]{bar}[/blue]")
    else:
        console.print("[red]❌ Model training failed[/red]")


if __name__ == "__main__":
    cli_main()
