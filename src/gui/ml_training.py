"""
ML Training Window for F1 Pit Stop Analyzer.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar, QTextEdit,
    QSpinBox, QCheckBox, QGroupBox, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class MLTrainingThread(QThread):
    """Background thread for ML model training."""
    progress = Signal(str)
    metrics = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, years, max_races, use_enhanced=True):
        super().__init__()
        self.years = years
        self.max_races = max_races
        self.use_enhanced = use_enhanced

    def run(self):
        try:
            if self.use_enhanced:
                from src.ml.enhanced_predictor import EnhancedPitStopPredictor, collect_training_data
                
                self.progress.emit("Collecting training data from FastF1...")
                training_data = collect_training_data(
                    self.years, 
                    self.max_races, 
                    progress_callback=self.progress.emit
                )
                
                if training_data is None or len(training_data) == 0:
                    self.finished.emit(False, "No training data collected")
                    return
                
                self.progress.emit(f"Training enhanced ensemble model on {len(training_data)} pit stops...")
                predictor = EnhancedPitStopPredictor()
                metrics = predictor.train(training_data)
                
                if 'error' not in metrics:
                    predictor.save_model()
                    self.metrics.emit(metrics)
                    self.finished.emit(True, f"Model trained successfully!\n\nSamples: {metrics.get('n_samples', 0)}\nRF MAE: {metrics.get('rf_mae', 0):.3f}s\nGBM MAE: {metrics.get('gbm_mae', 0):.3f}s")
                else:
                    self.finished.emit(False, metrics.get('error', 'Training failed'))
            else:
                from src.ml.predictor import PitStopPredictor, collect_training_data
                
                self.progress.emit("Collecting training data...")
                training_data = collect_training_data(
                    self.years, 
                    self.max_races, 
                    progress_callback=self.progress.emit
                )
                
                if training_data is None or len(training_data) == 0:
                    self.finished.emit(False, "No training data collected")
                    return
                
                self.progress.emit(f"Training model on {len(training_data)} pit stops...")
                predictor = PitStopPredictor()
                success = predictor.train(training_data)
                
                if success:
                    predictor.save_model()
                    self.finished.emit(True, f"Model trained on {len(training_data)} pit stops!")
                else:
                    self.finished.emit(False, "Model training failed")
                    
        except Exception as e:
            import traceback
            self.progress.emit(f"Error: {str(e)}")
            self.progress.emit(traceback.format_exc()[:500])
            self.finished.emit(False, str(e))


class MLTrainingWindow(QMainWindow):
    """Window for configuring and running ML model training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 ML Model Training")
        self.setMinimumSize(600, 500)
        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: white;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSpinBox, QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #e10600;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                color: #0f0;
                font-family: Consolas, monospace;
            }
        """)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("🤖 Machine Learning Model Training")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Year range
        years_layout = QHBoxLayout()
        years_layout.addWidget(QLabel("Years to include:"))
        
        self.year_checks = {}
        for year in range(2025, 2019, -1):
            cb = QCheckBox(str(year))
            cb.setChecked(year >= 2023)
            self.year_checks[year] = cb
            years_layout.addWidget(cb)
        
        years_layout.addStretch()
        config_layout.addLayout(years_layout)
        
        # Races per year
        races_layout = QHBoxLayout()
        races_layout.addWidget(QLabel("Races per year:"))
        self.races_spin = QSpinBox()
        self.races_spin.setRange(1, 24)
        self.races_spin.setValue(5)
        races_layout.addWidget(self.races_spin)
        races_layout.addStretch()
        config_layout.addLayout(races_layout)
        
        # Model type selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model type:"))
        self.enhanced_check = QCheckBox("Use Enhanced Ensemble Model")
        self.enhanced_check.setChecked(True)
        self.enhanced_check.setToolTip("Uses Random Forest + Gradient Boosting + Bayesian Ridge ensemble")
        model_layout.addWidget(self.enhanced_check)
        model_layout.addStretch()
        config_layout.addLayout(model_layout)
        self.races_spin.setRange(1, 24)
        self.races_spin.setValue(5)
        races_layout.addWidget(self.races_spin)
        races_layout.addStretch()
        config_layout.addLayout(races_layout)
        
        layout.addWidget(config_group)

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        progress_layout.addWidget(self.log_text)
        
        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #e10600;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff1a1a;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Results section (hidden initially)
        self.results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(self.results_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                color: white;
            }
            QHeaderView::section {
                background-color: #e10600;
                color: white;
                padding: 5px;
                border: none;
            }
        """)
        results_layout.addWidget(self.metrics_table)
        
        self.results_group.setVisible(False)
        layout.addWidget(self.results_group)
        
        layout.addStretch()

    def _start_training(self):
        """Start the ML training process."""
        years = [year for year, cb in self.year_checks.items() if cb.isChecked()]
        
        if not years:
            QMessageBox.warning(self, "Error", "Please select at least one year.")
            return
        
        max_races = self.races_spin.value()
        use_enhanced = self.enhanced_check.isChecked()
        
        self.log_text.clear()
        model_type = "enhanced ensemble" if use_enhanced else "basic"
        self.log_text.append(f"🚀 Starting {model_type} ML model training...")
        self.log_text.append(f"📅 Years: {sorted(years)}")
        self.log_text.append(f"🏎️ Max races per year: {max_races}")
        self.progress_bar.setVisible(True)
        self.train_btn.setEnabled(False)
        self.results_group.setVisible(False)
        
        self.training_thread = MLTrainingThread(years, max_races, use_enhanced)
        self.training_thread.progress.connect(self._on_progress)
        self.training_thread.metrics.connect(self._on_metrics)
        self.training_thread.finished.connect(self._on_finished)
        self.training_thread.start()

    def _on_progress(self, message: str):
        """Handle progress updates."""
        self.log_text.append(f"📝 {message}")
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _on_metrics(self, metrics: dict):
        """Display training metrics in the table."""
        self.metrics_table.setRowCount(len(metrics))
        
        for i, (key, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key.replace('_', ' ').title()))
            if isinstance(value, float):
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                self.metrics_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.results_group.setVisible(True)

    def _on_finished(self, success: bool, message: str):
        """Handle training completion."""
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        
        if success:
            self.log_text.append(f"\n✅ SUCCESS!")
            self.log_text.append(f"Model saved to ml_models/ directory")
            QMessageBox.information(self, "Training Complete", message)
        else:
            self.log_text.append(f"\n❌ FAILED: {message}")
            QMessageBox.critical(self, "Training Failed", message)
