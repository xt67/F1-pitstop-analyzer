import fastf1
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from datetime import datetime

# Enable cache for faster subsequent loads
fastf1.Cache.enable_cache('cache')

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
    """Create all pit stop visualizations in a single scrollable figure (vertical subplots) and optionally save to charts/ directory."""
    if len(valid_pits) == 0:
        print("\n⚠️ No data to visualize")
        return

    import matplotlib.gridspec as gridspec
    import os
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

    nrows = 4 if strategy is not None and len(strategy) > 0 else 3
    fig = plt.figure(constrained_layout=False, figsize=(18, max(24, num_drivers * 1.5)))
    gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.45)
    fig.suptitle(f'F1 Pit Stop Analysis: {race_name} {year}', fontsize=20, fontweight='bold', y=0.995)

    # ...existing code for all subplots...

    # 1. Average Pit Stop by Driver and Team
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
    ax1.set_title("Average Pit Stop by Driver", fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_xlim(left=0)
    for bar, val in zip(bars, driver_pits['PitDuration_Seconds']):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}s', va='center', fontsize=9, color='#333333')

    # 2. Average Pit Stop by Team
    ax2 = ax1.twiny()
    team_avg = valid_pits.groupby('Team')['PitDuration_Seconds'].mean().sort_values()
    bar_colors_team = [TEAM_COLORS.get(t, '#888888') for t in team_avg.index]
    y_pos_team = np.arange(len(team_avg))
    bars2 = ax2.barh(y_pos_team, team_avg.values, color=bar_colors_team, edgecolor='white', height=0.25, alpha=0.4)
    ax2.set_yticks([])
    ax2.set_xticks([])

    # 3. Pit Stop Timeline & Team Consistency
    ax3 = fig.add_subplot(gs[1, 0])
    teams_in_race = valid_pits['Team'].unique()
    for team in teams_in_race:
        team_stops = valid_pits[valid_pits['Team'] == team]
        color = TEAM_COLORS.get(team, '#888888')
        ax3.scatter(team_stops['PitLap'], team_stops['PitDuration_Seconds'], c=color, s=120, alpha=0.85, edgecolors='#333333', linewidth=1, label=team, zorder=5)
    ax3.axhline(y=avg_time, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel("Lap Number", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Pit Duration (seconds)", fontsize=12, fontweight='bold')
    ax3.set_title("Pit Stops by Lap (Timeline)", fontsize=15, fontweight='bold', pad=10)
    ax3.set_xlim(0, total_laps + 5)
    ax3.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # 4. Team Consistency (Box Plot)
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
    ax4.set_title("Team Consistency (Box Plot)", fontsize=15, fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)

    # 5. Tire Strategy (if available)
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
        ax5.set_title("Tire Strategy by Driver", fontsize=15, fontweight='bold', pad=10)

    # 6. Compound Distribution (Pie Chart) as inset in the first plot
    from matplotlib.transforms import Bbox
    compound_counts = valid_pits['Compound'].value_counts()
    colors = [COMPOUND_COLORS.get(c, '#888888') for c in compound_counts.index]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax1, width="30%", height="60%", loc='upper left', bbox_to_anchor=(0.01, 0.35, 0.5, 0.5), bbox_transform=ax1.transAxes)
    wedges, texts, autotexts = inset_ax.pie(
        compound_counts.values,
        labels=compound_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.7,
        labeldistance=1.1,
        explode=[0.02] * len(compound_counts)
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('#333333')
    inset_ax.set_title('Compound Usage', fontsize=11, fontweight='bold', pad=8)

    plt.subplots_adjust(top=0.96, hspace=0.38)
    plt.show()

    # Prompt user to save the figure
    save = input("\n💾 Do you want to save this chart to the charts/ folder? (y/n): ").strip().lower()
    if save == 'y':
        charts_dir = os.path.join(os.getcwd(), 'charts')
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        filename = f'charts/pitstop_analysis_{safe_race_name}_{year}.png'
        fig.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
        print(f"\n✅ Chart saved: {filename}")
    else:
        print("\nℹ️ Chart not saved.")


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
    print("   2. Pre-cache all races for a season (faster future loads)")
    print("   3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            print("❌ Please enter 1, 2, or 3")
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


def main():
    """Main function to run the pit stop analyzer"""
    while True:
        choice = show_main_menu()
        
        if choice == 1:
            analyze_race()
        elif choice == 2:
            cache_season()
        elif choice == 3:
            print("\n👋 Thanks for using F1 Pit Stop Analyzer! Goodbye!")
            break
        
        print("\n" + "-"*40)
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
