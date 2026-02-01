"""
F1 Pit Stop Analyzer - Main Entry Point

A comprehensive tool for analyzing Formula 1 pit stop data with ML predictions.

Usage:
    python main.py              # Run GUI (default)
    python main.py --cli        # Run CLI interface
    python main.py --train-ml   # Train ML model via CLI
    python main.py --help       # Show help
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.lib.settings import get_settings


def show_help():
    """Display help message."""
    print("""
🏎️ F1 PIT STOP ANALYZER
========================

Usage:
  python main.py              Run the graphical user interface (default)
  python main.py --cli        Run the command-line interface
  python main.py --train-ml   Train the ML model via CLI
  python main.py --legacy     Run the original legacy interface
  python main.py --help       Show this help message

Options:
  --year <YEAR>       Specify season year (e.g., --year 2024)
  --round <ROUND>     Specify round number (e.g., --round 12)
  --no-ml             Skip ML analysis in legacy mode

Examples:
  python main.py                           # Start GUI
  python main.py --cli                     # Interactive CLI
  python main.py --legacy --year 2024      # Legacy mode for 2024
""")


def run_gui():
    """Run the PyQt GUI interface."""
    from src.gui.race_selection import run_gui as gui_main
    gui_main()


def run_cli():
    """Run the CLI interface."""
    from src.cli.race_selection import cli_main
    cli_main()


def run_cli_train():
    """Run CLI ML training."""
    from src.cli.race_selection import cli_train_model
    cli_train_model()


def run_legacy():
    """Run the legacy terminal-based interface."""
    # Import legacy main module functions
    import fastf1
    from src.lib.settings import get_settings
    from src.f1_data import get_available_seasons, get_season_schedule, analyze_pit_stops, get_tire_strategy
    from src.analysis import display_analysis, create_visualizations, run_ml_analysis
    
    settings = get_settings()
    fastf1.Cache.enable_cache(settings.cache_path)
    
    def select_season():
        seasons = get_available_seasons()
        print("\n🏎️  F1 PIT STOP ANALYZER 🏎️")
        print("="*40)
        print("\n📅 AVAILABLE SEASONS:")
        print(", ".join(map(str, seasons)))
        
        while True:
            try:
                year = int(input(f"\nEnter season year ({seasons[0]}-{seasons[-1]}): "))
                if year in seasons:
                    return year
                print(f"❌ Please enter a year between {seasons[0]} and {seasons[-1]}")
            except ValueError:
                print("❌ Please enter a valid year number")
    
    def display_race_options(schedule):
        import pandas as pd
        print(f"\n{'='*70}")
        print(f"{'#':<4} {'Round':<6} {'Race Name':<40} {'Date'}")
        print(f"{'='*70}")
        
        for idx, (_, race) in enumerate(schedule.iterrows(), 1):
            race_name = race['EventName'][:40]
            race_date = race['EventDate'].strftime('%Y-%m-%d') if pd.notna(race['EventDate']) else 'TBD'
            round_num = race['RoundNumber']
            print(f"{idx:<4} {round_num:<6} {race_name:<40} {race_date}")
        
        print(f"{'='*70}")
    
    def select_race(schedule):
        display_race_options(schedule)
        
        while True:
            try:
                choice = int(input(f"\nEnter race number (1-{len(schedule)}): "))
                if 1 <= choice <= len(schedule):
                    selected_race = schedule.iloc[choice - 1]
                    return selected_race['RoundNumber'], selected_race['EventName']
                print(f"❌ Please enter a number between 1 and {len(schedule)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def show_main_menu():
        print("\n" + "🏁"*20)
        print("   FORMULA 1 PIT STOP ANALYZER")
        print("🏁"*20)
        print("\n📋 MAIN MENU:")
        print("   1. Analyze a specific race")
        print("   2. Analyze race with ML predictions 🤖")
        print("   3. Train ML model on historical data")
        print("   4. Exit")
        
        while True:
            try:
                choice = int(input("\nEnter your choice (1-4): "))
                if choice in [1, 2, 3, 4]:
                    return choice
                print("❌ Please enter 1, 2, 3, or 4")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def analyze_race(use_ml=False):
        year = select_season()
        
        print(f"\n⏳ Loading {year} season schedule...")
        schedule = get_season_schedule(year)
        
        if schedule is None or len(schedule) == 0:
            print(f"❌ Could not load schedule for {year}")
            return
        
        round_number, race_name = select_race(schedule)
        
        print(f"\n⏳ Loading race data for {race_name} {year}...")
        print("   (First load downloads data, subsequent loads use cache)")
        
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            print("   This race may not have occurred yet or data is unavailable.")
            return
        
        pit_data, laps = analyze_pit_stops(session, year, race_name)
        
        if len(pit_data) == 0:
            print("\n⚠️ No pit stop data found for this race")
            return
        
        valid_pits, strategy, total_laps = display_analysis(pit_data, laps, year, race_name, session)
        
        if use_ml:
            run_ml_analysis(valid_pits, laps, year, race_name, total_laps)
        
        create_visualizations(valid_pits, strategy, year, race_name, total_laps)
    
    def train_ml_model():
        from src.ml.predictor import PitStopPredictor, collect_training_data
        
        print("\n" + "="*60)
        print("🤖 MACHINE LEARNING MODEL TRAINING")
        print("="*60)
        
        predictor = PitStopPredictor()
        model_exists = predictor.load_model()
        
        if model_exists:
            print("\n📦 Existing model found!")
            choice = input("Retrain with new data? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing model.")
                return predictor
        
        print("\n📅 Select training data range:")
        print("   1. Quick (2024-2025, 3 races each)")
        print("   2. Standard (2023-2025, 5 races each)")
        print("   3. Comprehensive (2022-2025, 8 races each)")
        
        try:
            choice = int(input("\nChoice (1-3): "))
        except ValueError:
            choice = 1
        
        if choice == 1:
            years = [2024, 2025]
            max_races = 3
        elif choice == 2:
            years = [2023, 2024, 2025]
            max_races = 5
        else:
            years = [2022, 2023, 2024, 2025]
            max_races = 8
        
        training_data = collect_training_data(years, max_races)
        
        if training_data is None or len(training_data) == 0:
            print("❌ No training data collected")
            return None
        
        print("\n🔧 Training model...")
        predictor = PitStopPredictor()
        success = predictor.train(training_data)
        
        if success:
            print("\n✅ Model trained successfully!")
            print("\n📊 Feature Importance:")
            for feature, importance in sorted(predictor.feature_importance.items(), key=lambda x: -x[1]):
                bar = '█' * int(importance * 50)
                print(f"   {feature:<20}: {importance:.3f} {bar}")
            
            save = input("\n💾 Save model for future use? (y/n): ").strip().lower()
            if save == 'y':
                predictor.save_model()
        else:
            print("❌ Model training failed")
            return None
        
        return predictor
    
    # Main loop
    while True:
        choice = show_main_menu()
        
        if choice == 1:
            analyze_race(use_ml=False)
        elif choice == 2:
            analyze_race(use_ml=True)
        elif choice == 3:
            train_ml_model()
        elif choice == 4:
            print("\n👋 Thanks for using F1 Pit Stop Analyzer! Goodbye!")
            break
        
        print("\n" + "-"*40)
        input("Press Enter to continue...")


def main():
    """Main entry point."""
    # Parse command line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    if "--cli" in sys.argv:
        run_cli()
    elif "--train-ml" in sys.argv:
        run_cli_train()
    elif "--legacy" in sys.argv:
        run_legacy()
    else:
        # Default: Run GUI
        run_gui()


if __name__ == "__main__":
    main()
