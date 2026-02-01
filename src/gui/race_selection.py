"""
Race Selection GUI Window for F1 Pit Stop Analyzer.
Built with PySide6 for modern, cross-platform GUI.
"""

import sys
import os
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea, QFrame, QGridLayout,
    QProgressDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import fastf1
import pandas as pd


class DataLoaderThread(QThread):
    """Background thread for loading F1 data."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, year, round_number):
        super().__init__()
        self.year = year
        self.round_number = round_number

    def run(self):
        try:
            session = fastf1.get_session(self.year, self.round_number, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            self.finished.emit(session)
        except Exception as e:
            self.error.emit(str(e))


class RaceCard(QFrame):
    """Widget representing a single race in the grid."""
    
    def __init__(self, race_data: dict, parent=None):
        super().__init__(parent)
        self.race_data = race_data
        self._setup_ui()
        
    def _setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            RaceCard {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 10px;
            }
            RaceCard:hover {
                border: 2px solid #e10600;
                background-color: #3d3d3d;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Round number
        round_label = QLabel(f"Round {self.race_data.get('round', '?')}")
        round_label.setStyleSheet("color: #e10600; font-weight: bold; font-size: 11px;")
        layout.addWidget(round_label)
        
        # Race name
        name_label = QLabel(self.race_data.get('name', 'Unknown'))
        name_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Date
        date_label = QLabel(self.race_data.get('date', ''))
        date_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(date_label)
        
        # Analyze button
        self.analyze_btn = QPushButton("📊 Analyze")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #e10600;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff1a1a;
            }
            QPushButton:pressed {
                background-color: #b30000;
            }
        """)
        layout.addWidget(self.analyze_btn)


class RaceSelectionWindow(QMainWindow):
    """Main window for selecting races to analyze."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏎️ F1 PitLab")
        self.setMinimumSize(1200, 800)
        self._setup_ui()
        self._load_schedule()

    def _setup_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #e10600;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #e10600;
            }
        """)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🏎️ F1 PitLab")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: white;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)
        
        # ML Training button
        ml_btn = QPushButton("🤖 Train ML Model")
        ml_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e88e5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #42a5f5;
            }
        """)
        ml_btn.clicked.connect(self._open_ml_training)
        header_layout.addWidget(ml_btn)
        
        main_layout.addLayout(header_layout)

        # Year selection
        year_layout = QHBoxLayout()
        year_label = QLabel("Select Season:")
        year_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        year_layout.addWidget(year_label)
        
        self.year_combo = QComboBox()
        current_year = min(datetime.now().year, 2025)
        for year in range(current_year, 2017, -1):
            self.year_combo.addItem(str(year))
        self.year_combo.currentTextChanged.connect(self._on_year_changed)
        year_layout.addWidget(self.year_combo)
        
        year_layout.addStretch()
        
        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #888; font-size: 12px;")
        year_layout.addWidget(self.stats_label)
        
        main_layout.addLayout(year_layout)

        # Race grid scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.races_container = QWidget()
        self.races_layout = QGridLayout(self.races_container)
        self.races_layout.setSpacing(15)
        scroll_area.setWidget(self.races_container)
        
        main_layout.addWidget(scroll_area)

        # Footer
        footer_label = QLabel("F1 PitLab | Powered by FastF1 & Machine Learning")
        footer_label.setStyleSheet("color: #666; font-size: 11px;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)

    def _load_schedule(self):
        """Load the race schedule for the selected year."""
        year = int(self.year_combo.currentText())
        
        try:
            fastf1.Cache.enable_cache('cache')
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'].notna()]
            
            # Filter past races
            today = pd.Timestamp.now()
            past_races = races[races['EventDate'] < today]
            
            self._populate_race_grid(past_races)
            self.stats_label.setText(f"📅 {len(past_races)} races available")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load schedule: {e}")

    def _populate_race_grid(self, races):
        """Populate the grid with race cards."""
        # Clear existing cards
        while self.races_layout.count():
            item = self.races_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add race cards
        cols = 4
        for idx, (_, race) in enumerate(races.iterrows()):
            race_data = {
                'round': race['RoundNumber'],
                'name': race['EventName'],
                'date': race['EventDate'].strftime('%B %d, %Y') if pd.notna(race['EventDate']) else 'TBD',
            }
            
            card = RaceCard(race_data)
            card.analyze_btn.clicked.connect(
                lambda checked, r=race['RoundNumber'], n=race['EventName']: 
                self._analyze_race(r, n)
            )
            
            row = idx // cols
            col = idx % cols
            self.races_layout.addWidget(card, row, col)

    def _on_year_changed(self, year: str):
        """Handle year selection change."""
        self._load_schedule()

    def _analyze_race(self, round_number: int, race_name: str):
        """Start analysis for the selected race."""
        year = int(self.year_combo.currentText())
        
        # Show loading dialog
        progress = QProgressDialog("Loading race data...", None, 0, 0, self)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        # Load in background thread
        self.loader_thread = DataLoaderThread(year, round_number)
        self.loader_thread.finished.connect(
            lambda session: self._on_session_loaded(session, year, race_name, progress)
        )
        self.loader_thread.error.connect(
            lambda err: self._on_load_error(err, progress)
        )
        self.loader_thread.start()

    def _on_session_loaded(self, session, year, race_name, progress):
        """Handle successful session load."""
        progress.close()
        
        # Import analysis module and run analysis
        from src.analysis import run_analysis
        run_analysis(session, year, race_name)

    def _on_load_error(self, error: str, progress):
        """Handle session load error."""
        progress.close()
        QMessageBox.critical(self, "Error", f"Failed to load session:\n{error}")

    def _open_settings(self):
        """Open settings dialog."""
        QMessageBox.information(self, "Settings", "Settings dialog coming soon!")

    def _open_ml_training(self):
        """Open ML training dialog."""
        from src.gui.ml_training import MLTrainingWindow
        self.ml_window = MLTrainingWindow(self)
        self.ml_window.show()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = RaceSelectionWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
