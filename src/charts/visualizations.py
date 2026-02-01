"""
Professional F1-styled chart visualizations using Plotly.
Creates interactive, publication-quality charts inspired by official F1 graphics.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os

# F1 2024 Team Colors (official)
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E80020',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'Racing Bulls': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
}

# Tire compound colors (official F1)
COMPOUND_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFD700', 
    'HARD': '#FFFFFF',
    'INTERMEDIATE': '#39B54A',
    'WET': '#00AEEF',
    'UNKNOWN': '#888888',
}

# F1 Dark theme settings
F1_THEME = {
    'bg_color': '#15151E',
    'plot_bg': '#1E1E2E',
    'grid_color': '#2D2D3D',
    'text_color': '#FFFFFF',
    'accent_red': '#E10600',
    'accent_white': '#FFFFFF',
    'font_family': 'Formula1, Arial, sans-serif',
}


class ChartGenerator:
    """
    Generates professional F1-styled interactive charts using Plotly.
    """
    
    def __init__(self, charts_dir: str = 'charts'):
        self.charts_dir = charts_dir
        os.makedirs(charts_dir, exist_ok=True)
        
    def _apply_f1_theme(self, fig: go.Figure, title: str = '') -> go.Figure:
        """Apply F1 dark theme styling to a figure."""
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color=F1_THEME['text_color'], family=F1_THEME['font_family']),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor=F1_THEME['bg_color'],
            plot_bgcolor=F1_THEME['plot_bg'],
            font=dict(color=F1_THEME['text_color'], family=F1_THEME['font_family']),
            legend=dict(
                bgcolor='rgba(30, 30, 46, 0.8)',
                bordercolor=F1_THEME['grid_color'],
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60),
        )
        fig.update_xaxes(
            gridcolor=F1_THEME['grid_color'],
            showline=True,
            linecolor=F1_THEME['grid_color'],
            tickfont=dict(size=12)
        )
        fig.update_yaxes(
            gridcolor=F1_THEME['grid_color'],
            showline=True,
            linecolor=F1_THEME['grid_color'],
            tickfont=dict(size=12)
        )
        return fig

    def pit_stop_duration_chart(
        self,
        pit_stops: pd.DataFrame,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create an interactive bar chart showing pit stop durations by driver.
        """
        # Sort by average duration
        driver_stats = pit_stops.groupby(['Driver', 'Team']).agg({
            'PitDuration_Seconds': ['mean', 'min', 'max', 'count']
        }).round(2)
        driver_stats.columns = ['avg', 'best', 'worst', 'count']
        driver_stats = driver_stats.reset_index().sort_values('avg')
        
        fig = go.Figure()
        
        # Add bars with team colors
        for _, row in driver_stats.iterrows():
            color = TEAM_COLORS.get(row['Team'], '#888888')
            fig.add_trace(go.Bar(
                y=[row['Driver']],
                x=[row['avg']],
                orientation='h',
                name=row['Driver'],
                marker=dict(
                    color=color,
                    line=dict(color='white', width=1)
                ),
                text=f"{row['avg']:.2f}s",
                textposition='outside',
                hovertemplate=(
                    f"<b>{row['Driver']}</b> ({row['Team']})<br>"
                    f"Average: {row['avg']:.2f}s<br>"
                    f"Best: {row['best']:.2f}s<br>"
                    f"Worst: {row['worst']:.2f}s<br>"
                    f"Stops: {int(row['count'])}<extra></extra>"
                ),
                showlegend=False
            ))
        
        # Add average line
        avg_duration = pit_stops['PitDuration_Seconds'].mean()
        fig.add_vline(
            x=avg_duration,
            line=dict(color=F1_THEME['accent_red'], width=2, dash='dash'),
            annotation_text=f"Avg: {avg_duration:.2f}s",
            annotation_position="top"
        )
        
        fig = self._apply_f1_theme(
            fig, 
            f"🏎️ Pit Stop Durations - {race_name} {year}"
        )
        
        fig.update_layout(
            xaxis_title="Duration (seconds)",
            yaxis_title="Driver",
            height=max(400, len(driver_stats) * 35),
            bargap=0.15,
        )
        
        return fig

    def pit_stop_timeline(
        self,
        pit_stops: pd.DataFrame,
        total_laps: int,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create a scatter plot timeline of pit stops during the race.
        """
        fig = go.Figure()
        
        teams = pit_stops['Team'].unique()
        
        for team in teams:
            team_data = pit_stops[pit_stops['Team'] == team]
            color = TEAM_COLORS.get(team, '#888888')
            
            fig.add_trace(go.Scatter(
                x=team_data['PitLap'],
                y=team_data['PitDuration_Seconds'],
                mode='markers',
                name=team,
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(color='white', width=1.5),
                    symbol='circle'
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Team: %{customdata[1]}<br>"
                    "Lap: %{x}<br>"
                    "Duration: %{y:.2f}s<br>"
                    "Compound: %{customdata[2]}<extra></extra>"
                ),
                customdata=team_data[['Driver', 'Team', 'Compound']].values
            ))
        
        # Add pit windows highlight
        avg_duration = pit_stops['PitDuration_Seconds'].mean()
        fig.add_hline(
            y=avg_duration,
            line=dict(color=F1_THEME['accent_red'], width=2, dash='dash'),
            annotation_text=f"Avg: {avg_duration:.2f}s"
        )
        
        # Highlight common pit windows
        pit_windows = pit_stops.groupby(pd.cut(pit_stops['PitLap'], bins=5)).size()
        
        fig = self._apply_f1_theme(
            fig,
            f"⏱️ Pit Stop Timeline - {race_name} {year}"
        )
        
        fig.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Pit Duration (seconds)",
            xaxis=dict(range=[0, total_laps + 2]),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig

    def team_performance_radar(
        self,
        pit_stops: pd.DataFrame,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create a radar chart comparing team pit stop performance.
        """
        team_stats = pit_stops.groupby('Team').agg({
            'PitDuration_Seconds': ['mean', 'std', 'min', 'count']
        }).round(2)
        team_stats.columns = ['avg', 'consistency', 'best', 'total_stops']
        team_stats = team_stats.reset_index()
        
        # Normalize metrics (higher is better for radar)
        team_stats['speed_score'] = 100 - ((team_stats['avg'] - team_stats['avg'].min()) / 
                                           (team_stats['avg'].max() - team_stats['avg'].min() + 0.001) * 50)
        team_stats['consistency_score'] = 100 - ((team_stats['consistency'] - team_stats['consistency'].min()) / 
                                                  (team_stats['consistency'].max() - team_stats['consistency'].min() + 0.001) * 50)
        team_stats['best_stop_score'] = 100 - ((team_stats['best'] - team_stats['best'].min()) / 
                                                (team_stats['best'].max() - team_stats['best'].min() + 0.001) * 50)
        
        categories = ['Speed', 'Consistency', 'Best Stop', 'Volume']
        
        fig = go.Figure()
        
        for _, row in team_stats.iterrows():
            color = TEAM_COLORS.get(row['Team'], '#888888')
            values = [
                row['speed_score'],
                row['consistency_score'],
                row['best_stop_score'],
                min(100, row['total_stops'] * 15)  # Normalize volume
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the shape
                theta=categories + [categories[0]],
                name=row['Team'],
                line=dict(color=color, width=2),
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
            ))
        
        fig = self._apply_f1_theme(
            fig,
            f"📊 Team Pit Crew Performance - {race_name} {year}"
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=F1_THEME['grid_color'],
                    linecolor=F1_THEME['grid_color']
                ),
                angularaxis=dict(
                    gridcolor=F1_THEME['grid_color'],
                    linecolor=F1_THEME['grid_color']
                ),
                bgcolor=F1_THEME['plot_bg']
            ),
            height=600
        )
        
        return fig

    def tire_strategy_gantt(
        self,
        strategy: pd.DataFrame,
        total_laps: int,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create a Gantt-style chart showing tire strategies.
        """
        if strategy is None or len(strategy) == 0:
            return None
            
        drivers = strategy['Driver'].unique().tolist()
        
        fig = go.Figure()
        
        for i, driver in enumerate(drivers):
            driver_stints = strategy[strategy['Driver'] == driver].sort_values('StartLap')
            
            for _, stint in driver_stints.iterrows():
                compound = stint['Compound'] if pd.notna(stint['Compound']) else 'UNKNOWN'
                color = COMPOUND_COLORS.get(compound, '#888888')
                start = stint['StartLap']
                end = stint['EndLap']
                laps_on_tire = end - start + 1
                
                fig.add_trace(go.Bar(
                    y=[driver],
                    x=[laps_on_tire],
                    base=start,
                    orientation='h',
                    name=compound,
                    marker=dict(
                        color=color,
                        line=dict(color='#333', width=1)
                    ),
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        f"Compound: {compound}<br>"
                        f"Laps {start} - {end} ({laps_on_tire} laps)<extra></extra>"
                    ),
                    showlegend=False,
                    text=compound[:3] if laps_on_tire > 4 else '',
                    textposition='inside',
                    textfont=dict(color='#333', size=11, family='Arial Black')
                ))
        
        # Add compound legend
        for compound, color in COMPOUND_COLORS.items():
            if compound in strategy['Compound'].values:
                fig.add_trace(go.Bar(
                    y=[None], x=[None],
                    marker=dict(color=color),
                    name=compound,
                    showlegend=True
                ))
        
        fig = self._apply_f1_theme(
            fig,
            f"🔴🟡⚪ Tire Strategy - {race_name} {year}"
        )
        
        fig.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Driver",
            xaxis=dict(range=[0, total_laps + 2]),
            barmode='stack',
            height=max(400, len(drivers) * 30),
            bargap=0.2,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig

    def pit_window_heatmap(
        self,
        pit_stops: pd.DataFrame,
        total_laps: int,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create a heatmap showing pit stop density by lap and team.
        """
        # Create lap bins
        lap_bins = list(range(0, total_laps + 5, 5))
        pit_stops['LapBin'] = pd.cut(pit_stops['PitLap'], bins=lap_bins, labels=lap_bins[:-1])
        
        # Create pivot table
        heatmap_data = pit_stops.pivot_table(
            values='PitDuration_Seconds',
            index='Team',
            columns='LapBin',
            aggfunc='count',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"Lap {int(l)}-{int(l)+4}" for l in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale=[
                [0, F1_THEME['plot_bg']],
                [0.5, '#FFD700'],
                [1, F1_THEME['accent_red']]
            ],
            hovertemplate="Team: %{y}<br>%{x}<br>Stops: %{z}<extra></extra>"
        ))
        
        fig = self._apply_f1_theme(
            fig,
            f"🗓️ Pit Window Activity - {race_name} {year}"
        )
        
        fig.update_layout(
            xaxis_title="Lap Window",
            yaxis_title="Team",
            height=max(400, len(heatmap_data) * 35)
        )
        
        return fig

    def ml_prediction_comparison(
        self,
        actual_times: List[float],
        predicted_times: List[float],
        drivers: List[str],
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Create a comparison chart of predicted vs actual pit times.
        """
        fig = go.Figure()
        
        # Actual times
        fig.add_trace(go.Bar(
            name='Actual',
            x=drivers,
            y=actual_times,
            marker_color=F1_THEME['accent_red'],
            text=[f"{t:.2f}s" for t in actual_times],
            textposition='outside'
        ))
        
        # Predicted times
        fig.add_trace(go.Bar(
            name='ML Predicted',
            x=drivers,
            y=predicted_times,
            marker_color='#27F4D2',
            text=[f"{t:.2f}s" for t in predicted_times],
            textposition='outside'
        ))
        
        fig = self._apply_f1_theme(
            fig,
            f"🤖 ML Prediction Accuracy - {race_name} {year}"
        )
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Driver",
            yaxis_title="Pit Duration (seconds)",
            height=500
        )
        
        return fig

    def anomaly_detection_chart(
        self,
        pit_stops: pd.DataFrame,
        anomalies: pd.DataFrame,
        race_name: str,
        year: int
    ) -> go.Figure:
        """
        Visualize detected anomalies in pit stops.
        """
        fig = go.Figure()
        
        # Normal stops
        normal = pit_stops[~pit_stops.index.isin(anomalies.index)]
        fig.add_trace(go.Scatter(
            x=normal['PitLap'],
            y=normal['PitDuration_Seconds'],
            mode='markers',
            name='Normal',
            marker=dict(
                size=12,
                color='#27F4D2',
                line=dict(color='white', width=1)
            ),
            hovertemplate="<b>%{customdata}</b><br>Lap: %{x}<br>Duration: %{y:.2f}s<extra></extra>",
            customdata=normal['Driver']
        ))
        
        # Anomalies
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['PitLap'],
                y=anomalies['PitDuration_Seconds'],
                mode='markers',
                name='Anomaly Detected',
                marker=dict(
                    size=16,
                    color=F1_THEME['accent_red'],
                    symbol='x',
                    line=dict(color='white', width=2)
                ),
                hovertemplate="<b>⚠️ %{customdata}</b><br>Lap: %{x}<br>Duration: %{y:.2f}s<extra></extra>",
                customdata=anomalies['Driver']
            ))
        
        fig = self._apply_f1_theme(
            fig,
            f"⚠️ Anomaly Detection - {race_name} {year}"
        )
        
        fig.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Pit Duration (seconds)",
            height=500
        )
        
        return fig

    def create_dashboard(
        self,
        pit_stops: pd.DataFrame,
        strategy: pd.DataFrame,
        total_laps: int,
        race_name: str,
        year: int,
        predictions: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Pit Stop Durations by Driver',
                'Pit Stop Timeline',
                'Team Consistency (Box Plot)',
                'Tire Strategy',
                'Pit Window Heatmap',
                'Compound Distribution'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "pie"}]
            ]
        )
        
        # This is a simplified version - full implementation would add all traces
        # For now, generate individual charts that can be displayed together
        
        fig.update_layout(
            height=1200,
            title_text=f"🏁 F1 Pit Stop Analysis Dashboard - {race_name} {year}",
            showlegend=True
        )
        
        fig = self._apply_f1_theme(fig, f"🏁 Pit Stop Dashboard - {race_name} {year}")
        
        return fig

    def save_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html'
    ) -> str:
        """
        Save a chart to file.
        
        Args:
            fig: Plotly figure
            filename: Base filename (without extension)
            format: 'html', 'png', 'pdf', or 'svg'
        
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.charts_dir, f"{filename}.{format}")
        
        if format == 'html':
            fig.write_html(filepath, include_plotlyjs=True)
        else:
            fig.write_image(filepath, scale=2)
        
        return filepath

    def show_all_charts(
        self,
        pit_stops: pd.DataFrame,
        strategy: pd.DataFrame,
        total_laps: int,
        race_name: str,
        year: int,
        save: bool = True
    ) -> List[go.Figure]:
        """
        Generate and optionally save all available charts.
        
        Returns:
            List of generated figures
        """
        charts = []
        safe_name = race_name.replace(' ', '_')
        
        # 1. Duration chart
        duration_chart = self.pit_stop_duration_chart(pit_stops, race_name, year)
        charts.append(('duration', duration_chart))
        
        # 2. Timeline
        timeline = self.pit_stop_timeline(pit_stops, total_laps, race_name, year)
        charts.append(('timeline', timeline))
        
        # 3. Radar
        radar = self.team_performance_radar(pit_stops, race_name, year)
        charts.append(('radar', radar))
        
        # 4. Strategy
        if strategy is not None and len(strategy) > 0:
            strat_chart = self.tire_strategy_gantt(strategy, total_laps, race_name, year)
            if strat_chart:
                charts.append(('strategy', strat_chart))
        
        # 5. Heatmap
        heatmap = self.pit_window_heatmap(pit_stops, total_laps, race_name, year)
        charts.append(('heatmap', heatmap))
        
        if save:
            for name, fig in charts:
                self.save_chart(fig, f"{safe_name}_{year}_{name}", 'html')
        
        return [fig for _, fig in charts]
