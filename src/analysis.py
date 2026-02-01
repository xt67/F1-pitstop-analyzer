"""
Main Analysis Module for F1 Pit Stop Analyzer.
Handles pit stop analysis, display, and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.f1_data import analyze_pit_stops, get_tire_strategy
from src.lib.colors import TEAM_COLORS, COMPOUND_COLORS, get_compound_emoji
from src.lib.settings import get_settings


def display_analysis(pit_data, laps, year, race_name, session):
    """Display comprehensive pit stop analysis results."""
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

    fastest = valid_pits.loc[valid_pits['PitDuration_Seconds'].idxmin()]
    print(f"   🥇 Fastest: {fastest['Driver']} ({fastest['Team']}) - {fastest['PitDuration_Seconds']:.2f}s on Lap {int(fastest['PitLap'])}")

    top3 = valid_pits.nsmallest(3, 'PitDuration_Seconds')
    if len(top3) >= 2:
        second = top3.iloc[1]
        print(f"   🥈 2nd: {second['Driver']} ({second['Team']}) - {second['PitDuration_Seconds']:.2f}s on Lap {int(second['PitLap'])}")
    if len(top3) >= 3:
        third = top3.iloc[2]
        print(f"   🥉 3rd: {third['Driver']} ({third['Team']}) - {third['PitDuration_Seconds']:.2f}s on Lap {int(third['PitLap'])}")

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

    # === STATISTICS SUMMARY ===
    print(f"\n{'─'*70}")
    print("📈 STATISTICAL SUMMARY:")
    print(f"{'─'*70}")
    print(f"   Mean Pit Duration:   {valid_pits['PitDuration_Seconds'].mean():.2f} seconds")
    print(f"   Median Pit Duration: {valid_pits['PitDuration_Seconds'].median():.2f} seconds")
    print(f"   Std Deviation:       {valid_pits['PitDuration_Seconds'].std():.2f} seconds")

    strategy = get_tire_strategy(laps)

    return valid_pits, strategy, total_laps


def create_visualizations(valid_pits, strategy, year, race_name, total_laps):
    """Create all pit stop visualizations in a single scrollable window."""
    if len(valid_pits) == 0:
        print("\n⚠️ No data to visualize")
        return

    settings = get_settings()

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
    avg_time = valid_pits['PitDuration_Seconds'].mean()
    safe_race_name = race_name.replace(' ', '_').replace('/', '-')

    # Calculate dynamic figure height
    nrows = 4 if strategy is not None and len(strategy) > 0 else 3
    fig_height = max(28, num_drivers * 1.8 + 20)

    # Create the figure
    fig = Figure(figsize=(16, fig_height), facecolor='white')
    gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.35, 
                           height_ratios=[1.2, 1, 1, 1.2] if nrows == 4 else [1.2, 1, 1])
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
    compound_counts = valid_pits['Compound'].value_counts()
    colors = [COMPOUND_COLORS.get(c, '#888888') for c in compound_counts.index]
    inset_ax = inset_axes(ax1, width="28%", height="55%", loc='upper left', 
                          bbox_to_anchor=(0.01, 0.38, 0.5, 0.5), bbox_transform=ax1.transAxes)
    wedges, texts, autotexts = inset_ax.pie(
        compound_counts.values, labels=compound_counts.index, colors=colors,
        autopct='%1.1f%%', textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.7, labeldistance=1.15, explode=[0.02] * len(compound_counts)
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
        ax3.scatter(team_stops['PitLap'], team_stops['PitDuration_Seconds'], c=color, s=120, 
                    alpha=0.85, edgecolors='#333333', linewidth=1, label=team, zorder=5)
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
            ax5.barh(y, end - start + 1, left=start, height=0.7, color=color, 
                     edgecolor='#333333', linewidth=0.8, alpha=0.9)
            mid_point = start + (end - start) / 2
            if end - start > 4:
                ax5.text(mid_point, y, compound[:3], ha='center', va='center', 
                         fontsize=9, fontweight='bold', color='#333333')
        ax5.set_yticks(list(y_positions.values()))
        ax5.set_yticklabels(list(y_positions.keys()), fontsize=11)
        ax5.set_xlabel("Lap Number", fontsize=12, fontweight='bold')
        ax5.set_xlim(0, total_laps + 2)
        ax5.set_ylim(-0.5, len(drivers) - 0.5)
        ax5.grid(axis='x', alpha=0.3)
        used_compounds = strategy['Compound'].dropna().unique()
        legend_elements = [Patch(facecolor=COMPOUND_COLORS.get(c, '#888888'), edgecolor='#333333', label=c) 
                           for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'] if c in used_compounds]
        ax5.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax5.set_title("🔴🟡⚪ Tire Strategy by Driver", fontsize=15, fontweight='bold', pad=10)

    fig.subplots_adjust(top=0.96, bottom=0.03, left=0.08, right=0.95, hspace=0.35)

    # Prepare save path
    charts_dir = settings.charts_path
    filename = os.path.join(charts_dir, f'pitstop_analysis_{safe_race_name}_{year}.png')

    # Create scrollable Tkinter window
    root = tk.Tk()
    root.title(f"F1 Pit Stop Analysis - {race_name} {year}")
    root.geometry(f"{settings.get('window_width', 1400)}x{settings.get('window_height', 900)}")
    root.configure(bg='white')

    main_frame = tk.Frame(root, bg='white')
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame, bg='white', highlightthickness=0)
    scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_x = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)

    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    chart_frame = tk.Frame(canvas, bg='white')
    canvas_window = canvas.create_window((0, 0), window=chart_frame, anchor='nw')

    figure_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    figure_canvas.draw()
    figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar_frame = tk.Frame(root, bg='white')
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    toolbar = NavigationToolbar2Tk(figure_canvas, toolbar_frame)
    toolbar.update()

    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    chart_frame.bind('<Configure>', configure_scroll_region)

    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def on_shift_mouse_wheel(event):
        canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)
    canvas.bind_all("<Shift-MouseWheel>", on_shift_mouse_wheel)

    def on_close():
        root.quit()
        root.destroy()

    def on_save():
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        fig.savefig(filename, dpi=150, facecolor='white', bbox_inches='tight')
        save_btn.config(text="✅ Saved!", state=tk.DISABLED, bg='#27ae60')
        print(f"\n✅ Chart saved: {filename}")

    btn_frame = tk.Frame(root, bg='white')
    btn_frame.pack(side=tk.BOTTOM, pady=10)

    save_btn = tk.Button(btn_frame, text="💾 Save Chart", command=on_save, font=('Arial', 11, 'bold'),
                         bg='#3498db', fg='white', padx=20, pady=5, cursor='hand2')
    save_btn.pack(side=tk.LEFT, padx=10)

    close_btn = tk.Button(btn_frame, text="✖ Close", command=on_close, font=('Arial', 11, 'bold'),
                          bg='#e74c3c', fg='white', padx=20, pady=5, cursor='hand2')
    close_btn.pack(side=tk.LEFT, padx=10)

    root.protocol("WM_DELETE_WINDOW", on_close)

    print("📊 Opening scrollable chart viewer...")
    print("   💡 Use mouse wheel to scroll vertically")
    print("   💡 Use Shift + mouse wheel to scroll horizontally")
    print("   💡 Use the toolbar to zoom/pan")
    print("   💡 Click 'Save Chart' button to save to charts/ folder")

    root.mainloop()


def run_analysis(session, year: int, race_name: str, use_ml: bool = False):
    """Run full pit stop analysis for a session."""
    # Analyze pit stops
    pit_data, laps = analyze_pit_stops(session, year, race_name)

    if len(pit_data) == 0:
        print("\n⚠️ No pit stop data found for this race")
        return

    # Display analysis
    valid_pits, strategy, total_laps = display_analysis(pit_data, laps, year, race_name, session)

    # ML Analysis if requested
    if use_ml:
        run_ml_analysis(valid_pits, laps, year, race_name, total_laps)

    # Create visualizations
    create_visualizations(valid_pits, strategy, year, race_name, total_laps)


def run_ml_analysis(valid_pits, laps, year, race_name, total_laps):
    """Perform ML-based analysis on pit stop data."""
    from src.ml.predictor import PitStopPredictor

    print(f"\n{'='*60}")
    print("🤖 MACHINE LEARNING ANALYSIS")
    print(f"{'='*60}")

    predictor = PitStopPredictor()

    if not predictor.load_model():
        print("\n📝 No pre-trained model found. Training on current race data...")
        predictor.train(valid_pits, laps)

    if not predictor.is_trained:
        print("⚠️ Could not train model with available data")
        return

    # Anomaly Detection
    print(f"\n{'─'*60}")
    print("🔍 ANOMALY DETECTION (Unusual Pit Stops):")
    print(f"{'─'*60}")

    analyzed_data = predictor.detect_anomalies(valid_pits, laps)
    anomalies = analyzed_data[analyzed_data.get('IsAnomaly', False) == True]

    if len(anomalies) > 0:
        print(f"   Found {len(anomalies)} anomalous pit stop(s):")
        for _, stop in anomalies.iterrows():
            print(f"   ⚠️ {stop['Driver']} ({stop['Team']}): {stop['PitDuration_Seconds']:.2f}s on Lap {int(stop['PitLap'])}")
    else:
        print("   ✅ No anomalous pit stops detected")

    # Optimal Pit Windows
    print(f"\n{'─'*60}")
    print("🎯 OPTIMAL PIT WINDOWS (ML Suggestions):")
    print(f"{'─'*60}")

    teams = valid_pits['Team'].unique()[:5]
    for team in teams:
        windows = predictor.get_optimal_pit_windows(total_laps, team)
        if windows:
            print(f"\n   {team}:")
            for w in windows:
                print(f"      Stint {w['Stint']}: Lap {w['WindowStart']}-{w['WindowEnd']} (optimal: Lap {w['OptimalLap']}, ~{w['ExpectedDuration']:.1f}s)")
