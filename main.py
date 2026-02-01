import fastf1
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Enable cache for faster subsequent loads
fastf1.Cache.enable_cache('cache')

# ML Model storage path
ML_MODEL_PATH = 'ml_models'

# F1 Team Colors (for visualization)
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6', 'Mercedes': '#27F4D2', 'Ferrari': '#E8002D',
    'McLaren': '#FF8000', 'Aston Martin': '#229971', 'Alpine': '#FF87BC',
    'Williams': '#64C4FF', 'RB': '#6692FF', 'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD', 'Alfa Romeo': '#C92D4B', 'AlphaTauri': '#5E8FAA',
    'Racing Point': '#F596C8', 'Renault': '#FFF500', 'Toro Rosso': '#469BFF'
}

# Tire compound colors
COMPOUND_COLORS = {
    'SOFT': '#FF3333', 'MEDIUM': '#FFF200', 'HARD': '#EBEBEB',
    'INTERMEDIATE': '#43B02A', 'WET': '#0067AD',
    'UNKNOWN': '#888888', 'TEST_UNKNOWN': '#888888'
}


def get_available_seasons():
    """Get list of available F1 seasons (2018 onwards has good data)"""
    current_year = datetime.now().year
    # Cap at 2025 since 2026 season hasn't started yet
    max_year = min(current_year, 2025)
    return list(range(2018, max_year + 1))


def get_season_schedule(year):
    """Fetch the race schedule for a given season"""
    try:
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


def display_race_options(schedule):
    """Display available races in a formatted list"""
    print(f"\n{'='*70}")
    print(f"{'#':<4} {'Round':<6} {'Race Name':<40} {'Date'}")
    print(f"{'='*70}")
    
    for idx, (_, race) in enumerate(schedule.iterrows(), 1):
        race_name = race['EventName'][:40]
        race_date = race['EventDate'].strftime('%Y-%m-%d') if pd.notna(race['EventDate']) else 'TBD'
        round_num = race['RoundNumber']
        print(f"{idx:<4} {round_num:<6} {race_name:<40} {race_date}")
    
    print(f"{'='*70}")


def select_season():
    """Let user select a season"""
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


def select_race(schedule):
    """Let user select a race from the schedule"""
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


def analyze_pit_stops(session, year, race_name):
    """Analyze pit stops for the given session"""
    laps = session.laps
    
    # Get laps where car EXITED the pit (PitOutTime exists = this is an "out lap")
    out_laps = laps[laps['PitOutTime'].notna()][['Driver', 'Team', 'LapNumber', 'PitOutTime', 'Compound', 'Stint']].copy()
    
    # Get laps where car ENTERED the pit (PitInTime exists = this is an "in lap")
    # IMPORTANT: Exclude Lap 1 - those aren't real pit stops, just race start data
    in_laps = laps[(laps['PitInTime'].notna()) & (laps['LapNumber'] > 1)][['Driver', 'LapNumber', 'PitInTime']].copy()
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
    pit_data = pit_data[['Driver', 'Team', 'LapNumber_in', 'PitInTime', 'PitOutTime', 'PitDuration_Seconds', 'Compound', 'Stint']]
    pit_data = pit_data.rename(columns={'LapNumber_in': 'PitLap'})
    
    return pit_data, laps


def get_tire_strategy(laps):
    """Extract tire strategy for each driver"""
    strategy = laps.groupby(['Driver', 'Stint']).agg({
        'Compound': 'first',
        'LapNumber': ['min', 'max', 'count'],
        'Team': 'first'
    }).reset_index()
    
    strategy.columns = ['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'LapCount', 'Team']
    return strategy


def display_analysis(pit_data, laps, year, race_name, session):
    """Display comprehensive pit stop analysis results"""
    total_laps = int(laps['LapNumber'].max())
    
    print(f"\n{'='*70}")
    print(f"{'🏁 PIT STOP ANALYSIS: ' + race_name + ' ' + str(year):^70}")
    print(f"{'='*70}")
    
    # === RACE OVERVIEW ===
    print(f"\n📋 RACE OVERVIEW:")
    print(f"   Total Laps: {total_laps}")
    print(f"   Total Pit Stops: {len(pit_data)}")
    print(f"   Drivers: {pit_data['Driver'].nunique()}")
    print(f"   Average Stops per Driver: {len(pit_data) / pit_data['Driver'].nunique():.1f}")
    
    valid_pits = pit_data[pit_data['PitDuration_Seconds'].notna()].copy()
    
    if len(valid_pits) == 0:
        print("\n⚠️ No valid pit stop data available for analysis")
        return valid_pits, None, total_laps
    
    # === PIT STOP RECORDS ===
    print(f"\n{'─'*70}")
    print("🏆 PIT STOP RECORDS:")
    print(f"{'─'*70}")
    
    # Fastest pit stop
    fastest = valid_pits.loc[valid_pits['PitDuration_Seconds'].idxmin()]
    print(f"   🥇 Fastest: {fastest['Driver']} ({fastest['Team']}) - {fastest['PitDuration_Seconds']:.2f}s on Lap {int(fastest['PitLap'])}")
    
    # 2nd and 3rd fastest
    top3 = valid_pits.nsmallest(3, 'PitDuration_Seconds')
    if len(top3) >= 2:
        second = top3.iloc[1]
        print(f"   🥈 2nd: {second['Driver']} ({second['Team']}) - {second['PitDuration_Seconds']:.2f}s on Lap {int(second['PitLap'])}")
    if len(top3) >= 3:
        third = top3.iloc[2]
        print(f"   🥉 3rd: {third['Driver']} ({third['Team']}) - {third['PitDuration_Seconds']:.2f}s on Lap {int(third['PitLap'])}")
    
    # Slowest pit stop
    slowest = valid_pits.loc[valid_pits['PitDuration_Seconds'].idxmax()]
    print(f"\n   🐢 Slowest: {slowest['Driver']} ({slowest['Team']}) - {slowest['PitDuration_Seconds']:.2f}s on Lap {int(slowest['PitLap'])}")
    
    # === TEAM PERFORMANCE ===
    print(f"\n{'─'*70}")
    print("🏎️ TEAM PIT STOP PERFORMANCE (Ranked by Average):")
    print(f"{'─'*70}")
    
    team_stats = valid_pits.groupby('Team').agg({
        'PitDuration_Seconds': ['mean', 'min', 'max', 'std', 'count']
    }).round(2)
    team_stats.columns = ['Avg', 'Best', 'Worst', 'StdDev', 'Count']
    team_stats = team_stats.sort_values('Avg')
    
    print(f"   {'Team':<25} {'Avg':>8} {'Best':>8} {'Worst':>8} {'StdDev':>8} {'Stops':>6}")
    print(f"   {'-'*63}")
    for team, row in team_stats.iterrows():
        std = f"{row['StdDev']:.2f}" if pd.notna(row['StdDev']) else "N/A"
        print(f"   {team:<25} {row['Avg']:>7.2f}s {row['Best']:>7.2f}s {row['Worst']:>7.2f}s {std:>8} {int(row['Count']):>6}")
    
    # === DRIVER PIT STOP SUMMARY ===
    print(f"\n{'─'*70}")
    print("👨‍✈️ DRIVER PIT STOP SUMMARY:")
    print(f"{'─'*70}")
    
    driver_stats = valid_pits.groupby(['Driver', 'Team']).agg({
        'PitDuration_Seconds': ['mean', 'min', 'count'],
        'PitLap': list
    }).reset_index()
    driver_stats.columns = ['Driver', 'Team', 'Avg', 'Best', 'Stops', 'PitLaps']
    driver_stats = driver_stats.sort_values('Avg')
    
    print(f"   {'Driver':<8} {'Team':<20} {'Avg':>8} {'Best':>8} {'Stops':>6} {'Pit Laps'}")
    print(f"   {'-'*70}")
    for _, row in driver_stats.iterrows():
        pit_laps_str = ", ".join([str(int(x)) for x in row['PitLaps']])
        print(f"   {row['Driver']:<8} {row['Team']:<20} {row['Avg']:>7.2f}s {row['Best']:>7.2f}s {row['Stops']:>6} [{pit_laps_str}]")
    
    # === PIT WINDOW ANALYSIS ===
    print(f"\n{'─'*70}")
    print("⏱️ PIT WINDOW ANALYSIS:")
    print(f"{'─'*70}")
    
    # Group pit stops by lap ranges
    valid_pits['PitWindow'] = pd.cut(valid_pits['PitLap'], 
                                      bins=[0, 15, 30, 45, 60, 100],
                                      labels=['Laps 1-15', 'Laps 16-30', 'Laps 31-45', 'Laps 46-60', 'Laps 60+'])
    window_counts = valid_pits['PitWindow'].value_counts().sort_index()
    
    for window, count in window_counts.items():
        pct = count / len(valid_pits) * 100
        bar = '█' * int(pct / 2)
        print(f"   {window}: {count:>3} stops ({pct:>5.1f}%) {bar}")
    
    # === TIRE STRATEGY ANALYSIS ===
    print(f"\n{'─'*70}")
    print("🔵🟡🔴 TIRE COMPOUND USAGE:")
    print(f"{'─'*70}")
    
    compound_counts = valid_pits['Compound'].value_counts()
    for compound, count in compound_counts.items():
        pct = count / len(valid_pits) * 100
        emoji = {'SOFT': '🔴', 'MEDIUM': '🟡', 'HARD': '⚪', 'INTERMEDIATE': '🟢', 'WET': '🔵'}.get(compound, '⚫')
        bar = '█' * int(pct / 2)
        print(f"   {emoji} {compound:<12}: {count:>3} ({pct:>5.1f}%) {bar}")
    
    # === TIRE STRATEGY PER DRIVER ===
    strategy = get_tire_strategy(laps)
    print(f"\n{'─'*70}")
    print("📊 TIRE STRATEGIES BY DRIVER:")
    print(f"{'─'*70}")
    
    for driver in strategy['Driver'].unique():
        driver_strat = strategy[strategy['Driver'] == driver].sort_values('Stint')
        stints = []
        for _, stint in driver_strat.iterrows():
            compound = stint['Compound'] if pd.notna(stint['Compound']) else 'UNK'
            emoji = {'SOFT': '🔴', 'MEDIUM': '🟡', 'HARD': '⚪', 'INTERMEDIATE': '🟢', 'WET': '🔵'}.get(compound, '⚫')
            stints.append(f"{emoji}{compound[:3]}({int(stint['LapCount'])})")
        team = driver_strat['Team'].iloc[0]
        print(f"   {driver:<5} ({team[:15]:<15}): {' → '.join(stints)}")
    
    # === STATISTICS SUMMARY ===
    print(f"\n{'─'*70}")
    print("📈 STATISTICAL SUMMARY:")
    print(f"{'─'*70}")
    print(f"   Mean Pit Duration:   {valid_pits['PitDuration_Seconds'].mean():.2f} seconds")
    print(f"   Median Pit Duration: {valid_pits['PitDuration_Seconds'].median():.2f} seconds")
    print(f"   Std Deviation:       {valid_pits['PitDuration_Seconds'].std():.2f} seconds")
    print(f"   Fastest Stop:        {valid_pits['PitDuration_Seconds'].min():.2f} seconds")
    print(f"   Slowest Stop:        {valid_pits['PitDuration_Seconds'].max():.2f} seconds")
    
    # Pit stops under 25 seconds (exceptional)
    fast_stops = valid_pits[valid_pits['PitDuration_Seconds'] < 25]
    print(f"\n   ⚡ Sub-25s stops: {len(fast_stops)} ({len(fast_stops)/len(valid_pits)*100:.1f}%)")
    
    # Pit stops over 30 seconds (problematic)
    slow_stops = valid_pits[valid_pits['PitDuration_Seconds'] > 30]
    if len(slow_stops) > 0:
        print(f"   ⚠️ Stops over 30s: {len(slow_stops)} (potential issues)")
        for _, stop in slow_stops.iterrows():
            print(f"      - {stop['Driver']} ({stop['Team']}): {stop['PitDuration_Seconds']:.2f}s on Lap {int(stop['PitLap'])}")
    
    return valid_pits, strategy, total_laps


def create_visualizations(valid_pits, strategy, year, race_name, total_laps):
    """Create all pit stop visualizations in a single scrollable window and auto-save to charts/ directory."""
    if len(valid_pits) == 0:
        print("\n⚠️ No data to visualize")
        return

    import matplotlib.gridspec as gridspec
    import os
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'font.family': 'sans-serif'
    })

    num_drivers = valid_pits['Driver'].nunique()
    num_teams = valid_pits['Team'].nunique()
    avg_time = valid_pits['PitDuration_Seconds'].mean()
    safe_race_name = race_name.replace(' ', '_').replace('/', '-')

    # Calculate dynamic figure height based on content
    nrows = 4 if strategy is not None and len(strategy) > 0 else 3
    fig_height = max(28, num_drivers * 1.8 + 20)
    
    # Create the figure
    fig = Figure(figsize=(16, fig_height), facecolor='white')
    gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.35, height_ratios=[1.2, 1, 1, 1.2] if nrows == 4 else [1.2, 1, 1])
    fig.suptitle(f'🏎️ F1 Pit Stop Analysis: {race_name} {year}', fontsize=22, fontweight='bold', y=0.995)

    # 1. Average Pit Stop by Driver
    ax1 = fig.add_subplot(gs[0, 0])
    driver_pits = valid_pits.groupby('Driver').agg({
        'PitDuration_Seconds': 'mean',
        'Team': 'first'
    }).reset_index().sort_values('PitDuration_Seconds')
    bar_colors = [TEAM_COLORS.get(team, '#888888') for team in driver_pits['Team']]
    y_pos = np.arange(len(driver_pits))
    bars = ax1.barh(y_pos, driver_pits['PitDuration_Seconds'], color=bar_colors, edgecolor='white', height=0.75)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(driver_pits['Driver'], fontsize=11)
    ax1.axvline(x=avg_time, color='#e74c3c', linestyle='--', linewidth=2, label=f'Avg: {avg_time:.1f}s')
    ax1.set_xlabel("Average Pit Duration (seconds)", fontsize=12, fontweight='bold')
    ax1.set_title("📊 Average Pit Stop Duration by Driver", fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(left=0)
    for bar, val in zip(bars, driver_pits['PitDuration_Seconds']):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}s', va='center', fontsize=9, color='#333333')

    # Compound Distribution (Pie Chart) as inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    compound_counts = valid_pits['Compound'].value_counts()
    colors = [COMPOUND_COLORS.get(c, '#888888') for c in compound_counts.index]
    inset_ax = inset_axes(ax1, width="28%", height="55%", loc='upper left', bbox_to_anchor=(0.01, 0.38, 0.5, 0.5), bbox_transform=ax1.transAxes)
    wedges, texts, autotexts = inset_ax.pie(
        compound_counts.values,
        labels=compound_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.7,
        labeldistance=1.15,
        explode=[0.02] * len(compound_counts)
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('#333333')
    inset_ax.set_title('🔵 Compound Usage', fontsize=10, fontweight='bold', pad=6)

    # 2. Pit Stop Timeline
    ax3 = fig.add_subplot(gs[1, 0])
    teams_in_race = valid_pits['Team'].unique()
    for team in teams_in_race:
        team_stops = valid_pits[valid_pits['Team'] == team]
        color = TEAM_COLORS.get(team, '#888888')
        ax3.scatter(team_stops['PitLap'], team_stops['PitDuration_Seconds'], c=color, s=120, alpha=0.85, edgecolors='#333333', linewidth=1, label=team, zorder=5)
    ax3.axhline(y=avg_time, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg: {avg_time:.1f}s')
    ax3.set_xlabel("Lap Number", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Pit Duration (seconds)", fontsize=12, fontweight='bold')
    ax3.set_title("⏱️ Pit Stops Timeline by Lap", fontsize=15, fontweight='bold', pad=10)
    ax3.set_xlim(0, total_laps + 5)
    ax3.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # 3. Team Consistency (Box Plot)
    ax4 = fig.add_subplot(gs[2, 0])
    teams_sorted = valid_pits.groupby('Team')['PitDuration_Seconds'].mean().sort_values().index.tolist()
    box_data = [valid_pits[valid_pits['Team'] == team]['PitDuration_Seconds'].values for team in teams_sorted]
    bp = ax4.boxplot(box_data, vert=False, patch_artist=True)
    ax4.set_yticklabels(teams_sorted, fontsize=10)
    for patch, team in zip(bp['boxes'], teams_sorted):
        patch.set_facecolor(TEAM_COLORS.get(team, '#888888'))
        patch.set_alpha(0.8)
    plt.setp(bp['medians'], color='#e74c3c', linewidth=2)
    ax4.set_xlabel("Pit Duration (seconds)", fontsize=12, fontweight='bold')
    ax4.set_title("🏎️ Team Pit Stop Consistency", fontsize=15, fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)

    # 4. Tire Strategy (if available)
    if strategy is not None and len(strategy) > 0:
        ax5 = fig.add_subplot(gs[3, 0])
        drivers = list(strategy['Driver'].unique())
        y_positions = {driver: i for i, driver in enumerate(drivers)}
        for _, stint in strategy.iterrows():
            driver = stint['Driver']
            y = y_positions[driver]
            start = stint['StartLap']
            end = stint['EndLap']
            compound = stint['Compound'] if pd.notna(stint['Compound']) else 'UNKNOWN'
            color = COMPOUND_COLORS.get(compound, '#888888')
            ax5.barh(y, end - start + 1, left=start, height=0.7, color=color, edgecolor='#333333', linewidth=0.8, alpha=0.9)
            mid_point = start + (end - start) / 2
            if end - start > 4:
                ax5.text(mid_point, y, compound[:3], ha='center', va='center', fontsize=9, fontweight='bold', color='#333333')
        ax5.set_yticks(list(y_positions.values()))
        ax5.set_yticklabels(list(y_positions.keys()), fontsize=11)
        ax5.set_xlabel("Lap Number", fontsize=12, fontweight='bold')
        ax5.set_xlim(0, total_laps + 2)
        ax5.set_ylim(-0.5, len(drivers) - 0.5)
        ax5.grid(axis='x', alpha=0.3)
        from matplotlib.patches import Patch
        used_compounds = strategy['Compound'].dropna().unique()
        legend_elements = [Patch(facecolor=COMPOUND_COLORS.get(c, '#888888'), edgecolor='#333333', label=c) for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'] if c in used_compounds]
        ax5.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax5.set_title("🔴🟡⚪ Tire Strategy by Driver", fontsize=15, fontweight='bold', pad=10)

    fig.subplots_adjust(top=0.96, bottom=0.03, left=0.08, right=0.95, hspace=0.35)

    # Prepare save path
    charts_dir = os.path.join(os.getcwd(), 'charts')
    filename = f'charts/pitstop_analysis_{safe_race_name}_{year}.png'

    # Create scrollable Tkinter window
    root = tk.Tk()
    root.title(f"F1 Pit Stop Analysis - {race_name} {year}")
    root.geometry("1400x800")
    root.configure(bg='white')
    
    # Create main frame with scrollbar
    main_frame = tk.Frame(root, bg='white')
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create canvas with scrollbar
    canvas = tk.Canvas(main_frame, bg='white', highlightthickness=0)
    scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_x = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
    
    # Configure canvas scrolling
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    
    # Pack scrollbars and canvas
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Create frame inside canvas to hold the figure
    chart_frame = tk.Frame(canvas, bg='white')
    canvas_window = canvas.create_window((0, 0), window=chart_frame, anchor='nw')
    
    # Embed matplotlib figure in Tkinter
    figure_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar
    toolbar_frame = tk.Frame(root, bg='white')
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    toolbar = NavigationToolbar2Tk(figure_canvas, toolbar_frame)
    toolbar.update()
    
    # Update scroll region when frame size changes
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    chart_frame.bind('<Configure>', configure_scroll_region)
    
    # Enable mouse wheel scrolling
    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def on_shift_mouse_wheel(event):
        canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Shift-MouseWheel>", on_shift_mouse_wheel)
    
    # Close button and save button
    def on_close():
        root.quit()
        root.destroy()
    
    def on_save():
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        fig.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
        save_btn.config(text="✅ Saved!", state=tk.DISABLED, bg='#27ae60')
        print(f"\n✅ Chart saved: {filename}")
    
    # Button frame at the bottom
    btn_frame = tk.Frame(root, bg='white')
    btn_frame.pack(side=tk.BOTTOM, pady=10)
    
    save_btn = tk.Button(btn_frame, text="💾 Save Chart", command=on_save, font=('Arial', 11, 'bold'), 
                         bg='#3498db', fg='white', padx=20, pady=5, cursor='hand2')
    save_btn.pack(side=tk.LEFT, padx=10)
    
    close_btn = tk.Button(btn_frame, text="✖ Close", command=on_close, font=('Arial', 11, 'bold'), 
                          bg='#e74c3c', fg='white', padx=20, pady=5, cursor='hand2')
    close_btn.pack(side=tk.LEFT, padx=10)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    print("📊 Opening scrollable chart viewer...")
    print("   💡 Use mouse wheel to scroll vertically")
    print("   💡 Use Shift + mouse wheel to scroll horizontally")
    print("   💡 Use the toolbar to zoom/pan")
    print("   💡 Click 'Save Chart' button to save to charts/ folder")
    
    root.mainloop()


# ============================================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================================

class PitStopPredictor:
    """ML model for pit stop duration prediction and analysis"""
    
    def __init__(self):
        self.model = None
        self.anomaly_detector = None
        self.team_encoder = LabelEncoder()
        self.compound_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
        
    def prepare_features(self, pit_data, laps_data=None):
        """Prepare features for ML model from pit stop data"""
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
        """Train the pit stop duration prediction model"""
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
        """Predict pit stop duration for given parameters"""
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
        """Detect anomalous pit stops (unusually slow or problematic)"""
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
        """Suggest optimal pit windows based on trained model"""
        if not self.is_trained:
            return None
        
        # Analyze predicted pit times across all laps
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
        
        # Find optimal windows (lowest predicted pit times)
        # Group by stint and find the lap range with lowest predictions
        optimal_windows = []
        for stint in pred_df['Stint'].unique():
            stint_data = pred_df[pred_df['Stint'] == stint]
            # Find the window of 5 laps with lowest average prediction
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
        """Save trained model to disk"""
        if not self.is_trained:
            print("⚠️ No trained model to save")
            return False
        
        if filepath is None:
            if not os.path.exists(ML_MODEL_PATH):
                os.makedirs(ML_MODEL_PATH)
            filepath = os.path.join(ML_MODEL_PATH, 'pitstop_predictor.pkl')
        
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
        """Load trained model from disk"""
        if filepath is None:
            filepath = os.path.join(ML_MODEL_PATH, 'pitstop_predictor.pkl')
        
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


def collect_training_data(years=None, max_races_per_year=5):
    """Collect pit stop data from multiple races for ML training"""
    if years is None:
        years = [2023, 2024, 2025]
    
    all_pit_data = []
    
    print(f"\n🔄 Collecting training data from {len(years)} seasons...")
    
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
                
                print(f"   Loading {year} {race_name}...", end=" ")
                
                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load(laps=True, telemetry=False, weather=False, messages=False)
                    
                    pit_data, laps = analyze_pit_stops(session, year, race_name)
                    
                    if len(pit_data) > 0:
                        pit_data['Year'] = year
                        pit_data['Race'] = race_name
                        pit_data['TotalLaps'] = laps['LapNumber'].max()
                        all_pit_data.append(pit_data)
                        print(f"✅ ({len(pit_data)} stops)")
                    else:
                        print("⚠️ No pit data")
                        
                except Exception as e:
                    print(f"❌ {str(e)[:30]}")
                    
        except Exception as e:
            print(f"   ❌ Error with {year}: {e}")
    
    if all_pit_data:
        combined_data = pd.concat(all_pit_data, ignore_index=True)
        print(f"\n✅ Collected {len(combined_data)} pit stops from {len(all_pit_data)} races")
        return combined_data
    
    return None


def train_ml_model():
    """Train ML model on historical data"""
    print("\n" + "="*60)
    print("🤖 MACHINE LEARNING MODEL TRAINING")
    print("="*60)
    
    # Check for existing model
    predictor = PitStopPredictor()
    model_exists = predictor.load_model()
    
    if model_exists:
        print("\n📦 Existing model found!")
        choice = input("Retrain with new data? (y/n): ").strip().lower()
        if choice != 'y':
            print("Using existing model.")
            return predictor
    
    # Select years for training
    print("\n📅 Select training data range:")
    print("   1. Quick (2024-2025, 3 races each)")
    print("   2. Standard (2023-2025, 5 races each)")
    print("   3. Comprehensive (2022-2025, 8 races each)")
    print("   4. Custom")
    
    try:
        choice = int(input("\nChoice (1-4): "))
    except ValueError:
        choice = 1
    
    if choice == 1:
        years = [2024, 2025]
        max_races = 3
    elif choice == 2:
        years = [2023, 2024, 2025]
        max_races = 5
    elif choice == 3:
        years = [2022, 2023, 2024, 2025]
        max_races = 8
    else:
        years_input = input("Enter years (comma-separated, e.g., 2023,2024,2025): ")
        years = [int(y.strip()) for y in years_input.split(',')]
        max_races = int(input("Races per year: "))
    
    # Collect data
    training_data = collect_training_data(years, max_races)
    
    if training_data is None or len(training_data) == 0:
        print("❌ No training data collected")
        return None
    
    # Train model
    print("\n🔧 Training model...")
    predictor = PitStopPredictor()
    success = predictor.train(training_data)
    
    if success:
        print("\n✅ Model trained successfully!")
        print("\n📊 Feature Importance:")
        for feature, importance in sorted(predictor.feature_importance.items(), key=lambda x: -x[1]):
            bar = '█' * int(importance * 50)
            print(f"   {feature:<20}: {importance:.3f} {bar}")
        
        # Save model
        save = input("\n💾 Save model for future use? (y/n): ").strip().lower()
        if save == 'y':
            predictor.save_model()
    else:
        print("❌ Model training failed")
        return None
    
    return predictor


def ml_analysis(valid_pits, laps, year, race_name, total_laps):
    """Perform ML-based analysis on pit stop data"""
    print(f"\n{'='*60}")
    print("🤖 MACHINE LEARNING ANALYSIS")
    print(f"{'='*60}")
    
    predictor = PitStopPredictor()
    
    # Try to load existing model
    if not predictor.load_model():
        print("\n📝 No pre-trained model found. Training on current race data...")
        predictor.train(valid_pits, laps)
    
    if not predictor.is_trained:
        print("⚠️ Could not train model with available data")
        return
    
    # 1. Anomaly Detection
    print(f"\n{'─'*60}")
    print("🔍 ANOMALY DETECTION (Unusual Pit Stops):")
    print(f"{'─'*60}")
    
    analyzed_data = predictor.detect_anomalies(valid_pits, laps)
    anomalies = analyzed_data[analyzed_data['IsAnomaly'] == True]
    
    if len(anomalies) > 0:
        print(f"   Found {len(anomalies)} anomalous pit stop(s):")
        for _, stop in anomalies.iterrows():
            print(f"   ⚠️ {stop['Driver']} ({stop['Team']}): {stop['PitDuration_Seconds']:.2f}s on Lap {int(stop['PitLap'])}")
    else:
        print("   ✅ No anomalous pit stops detected")
    
    # 2. Predictions vs Actuals
    print(f"\n{'─'*60}")
    print("📈 PREDICTED vs ACTUAL PIT TIMES:")
    print(f"{'─'*60}")
    
    predictions = []
    for _, row in valid_pits.iterrows():
        pred = predictor.predict(row['Team'], row['Compound'], row['PitLap'], 
                                  stint_number=1, total_laps=total_laps)
        if pred:
            predictions.append({
                'Driver': row['Driver'],
                'Team': row['Team'],
                'Actual': row['PitDuration_Seconds'],
                'Predicted': pred,
                'Difference': row['PitDuration_Seconds'] - pred
            })
    
    if predictions:
        pred_df = pd.DataFrame(predictions)
        mae = mean_absolute_error(pred_df['Actual'], pred_df['Predicted'])
        
        print(f"   Model Mean Absolute Error: {mae:.2f} seconds")
        print(f"\n   {'Driver':<8} {'Actual':>10} {'Predicted':>10} {'Diff':>10}")
        print(f"   {'-'*40}")
        
        for _, row in pred_df.head(10).iterrows():
            diff_symbol = "⬆️" if row['Difference'] > 0.5 else ("⬇️" if row['Difference'] < -0.5 else "✓")
            print(f"   {row['Driver']:<8} {row['Actual']:>9.2f}s {row['Predicted']:>9.2f}s {row['Difference']:>+9.2f}s {diff_symbol}")
    
    # 3. Optimal Pit Windows
    print(f"\n{'─'*60}")
    print("🎯 OPTIMAL PIT WINDOWS (ML Suggestions):")
    print(f"{'─'*60}")
    
    teams = valid_pits['Team'].unique()[:5]  # Top 5 teams
    for team in teams:
        windows = predictor.get_optimal_pit_windows(total_laps, team)
        if windows:
            print(f"\n   {team}:")
            for w in windows:
                print(f"      Stint {w['Stint']}: Lap {w['WindowStart']}-{w['WindowEnd']} (optimal: Lap {w['OptimalLap']}, ~{w['ExpectedDuration']:.1f}s)")


def pre_cache_races(year, schedule):
    """Pre-download and cache all races for a season"""
    print(f"\n📥 PRE-CACHING ALL {year} RACES")
    print("="*50)
    print("This will download data for all races in the season.")
    print("⚠️ This may take 15-30 minutes depending on your connection.\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    successful = 0
    failed = 0
    
    for idx, (_, race) in enumerate(schedule.iterrows(), 1):
        race_name = race['EventName']
        round_num = race['RoundNumber']
        
        print(f"\n[{idx}/{len(schedule)}] Caching: {race_name}...")
        
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            print(f"   ✅ Cached successfully!")
            successful += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 CACHING COMPLETE!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"{'='*50}")


def show_main_menu():
    """Display main menu options"""
    print("\n" + "🏁"*20)
    print("   FORMULA 1 PIT STOP ANALYZER")
    print("🏁"*20)
    print("\n📋 MAIN MENU:")
    print("   1. Analyze a specific race")
    print("   2. Analyze race with ML predictions 🤖")
    print("   3. Train ML model on historical data")
    print("   4. Pre-cache all races for a season")
    print("   5. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if choice in [1, 2, 3, 4, 5]:
                return choice
            print("❌ Please enter 1, 2, 3, 4, or 5")
        except ValueError:
            print("❌ Please enter a valid number")


def analyze_race():
    """Run the race analysis workflow"""
    # Step 1: Select season
    year = select_season()
    
    # Step 2: Get schedule for that season
    print(f"\n⏳ Loading {year} season schedule...")
    schedule = get_season_schedule(year)
    
    if schedule is None or len(schedule) == 0:
        print(f"❌ Could not load schedule for {year}")
        return
    
    # Step 3: Select race
    round_number, race_name = select_race(schedule)
    
    # Step 4: Load session data
    print(f"\n⏳ Loading race data for {race_name} {year}...")
    print("   (First load downloads data, subsequent loads use cache)")
    
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"❌ Error loading session: {e}")
        print("   This race may not have occurred yet or data is unavailable.")
        return
    
    # Step 5: Analyze pit stops
    pit_data, laps = analyze_pit_stops(session, year, race_name)
    
    if len(pit_data) == 0:
        print("\n⚠️ No pit stop data found for this race")
        return
    
    # Step 6: Display comprehensive analysis
    valid_pits, strategy, total_laps = display_analysis(pit_data, laps, year, race_name, session)
    
    # Step 7: Create visualizations
    create_visualizations(valid_pits, strategy, year, race_name, total_laps)
    
    return valid_pits, laps, year, race_name, total_laps


def cache_season():
    """Pre-cache all races for a season"""
    year = select_season()
    
    print(f"\n⏳ Loading {year} season schedule...")
    schedule = get_season_schedule(year)
    
    if schedule is None or len(schedule) == 0:
        print(f"❌ Could not load schedule for {year}")
        return
    
    print(f"\n📅 Found {len(schedule)} completed races in {year}")
    pre_cache_races(year, schedule)


def analyze_race_with_ml():
    """Run race analysis with ML predictions"""
    result = analyze_race()
    
    if result is None:
        return
    
    valid_pits, laps, year, race_name, total_laps = result
    
    # Run ML analysis
    ml_analysis(valid_pits, laps, year, race_name, total_laps)


def main():
    """Main function to run the pit stop analyzer"""
    while True:
        choice = show_main_menu()
        
        if choice == 1:
            analyze_race()
        elif choice == 2:
            analyze_race_with_ml()
        elif choice == 3:
            train_ml_model()
        elif choice == 4:
            cache_season()
        elif choice == 5:
            print("\n👋 Thanks for using F1 PitLab! Goodbye!")
            break
        
        print("\n" + "-"*40)
        input("Press Enter to continue...")


if __name__ == "__main__":
    # Launch GUI by default
    from src.gui.race_selection import run_gui
    run_gui()
