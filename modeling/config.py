"""
Configuration module for mill modeling pipeline.

Centralizes all configuration parameters for data preparation and model training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Mill and date range
    mill_number: int
    start_date: str
    end_date: str
    resample_freq: str = '1min'
    
    # Feature definitions
    mv_features: List[str] = field(default_factory=lambda: ['Ore', 'WaterMill', 'WaterZumpf', 'MotorAmp'])
    cv_features: List[str] = field(default_factory=lambda: ['DensityHC', 'PulpHC', 'PressureHC'])
    dv_features: List[str] = field(default_factory=lambda: ['Class_15', 'Daiki', 'FE'])  
    target: str = 'PSI200'
    
    # Data filtering thresholds
    filter_thresholds: Dict[str, tuple] = field(default_factory=lambda: {
        'Ore': (100, 220),
        'PulpHC': (400, 600),
        'DensityHC': (1600, 1800),
    })
    
    def get_all_features(self) -> List[str]:
        """Get all features (MV + CV + DV)."""
        return self.mv_features + self.cv_features + self.dv_features
    
    def get_all_columns(self) -> List[str]:
        """Get all required columns including target."""
        return self.get_all_features() + [self.target]


@dataclass
class MotifConfig:
    """Configuration for motif discovery."""
    
    # Motif discovery for MV features
    mv_window_size: int = 60  # minutes
    mv_max_motifs: int = 20
    mv_max_instances_per_motif: int = 1000
    mv_radius: float = 4.5
    
    # Motif discovery for density analysis
    density_window_size: int = 60
    density_max_motifs: int = 15
    density_radius: float = 4.5
    
    # Correlation filtering
    apply_correlation_filter: bool = True
    correlation_rules: Dict[tuple, str] = field(default_factory=lambda: {
        ('PressureHC', 'PulpHC'): 'pos',
        ('WaterZumpf', 'PressureHC'): 'pos',
    })
    min_correlation_strength: float = 0.1
    filter_level: str = 'instance'  # 'instance' or 'motif'
    
    # Visualization
    top_motifs_to_plot: int = 10


@dataclass
class ModelConfig:
    """Configuration for XGBoost model training."""
    
    # Train/test split
    test_size: float = 0.2
    
    # XGBoost base parameters
    objective: str = "reg:squarederror"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    
    # Hyperparameter tuning grid
    param_grid: Dict = field(default_factory=lambda: {
        "n_estimators": [150, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 8],
        "subsample": [0.6, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
    })
    
    # Cross-validation
    cv_splits: int = 5
    
    # Cascade validation
    cascade_validation_samples: int = 200


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    
    # Output directories
    output_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    analysis_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.output_dir = self.base_dir / "output"
        self.models_dir = self.base_dir / "models"
        self.analysis_dir = self.output_dir / "analysis"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def get_mill_model_dir(self, mill_number: int) -> Path:
        """Get model directory for specific mill."""
        mill_dir = self.models_dir / f"mill_{mill_number}"
        mill_dir.mkdir(parents=True, exist_ok=True)
        return mill_dir
    
    def get_mill_plots_dir(self, mill_number: int) -> Path:
        """Get plots directory for specific mill."""
        mill_dir = self.plots_dir / f"mill_{mill_number}"
        mill_dir.mkdir(parents=True, exist_ok=True)
        return mill_dir


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    data: DataConfig
    motif: MotifConfig = field(default_factory=MotifConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Database configuration (loaded from settings)
    use_database: bool = True
    use_cached_data: bool = False  # If True, skip DB and use cached CSV
    
    @classmethod
    def create_default(cls, mill_number: int, start_date: str, end_date: str) -> 'PipelineConfig':
        """Create default configuration for a mill."""
        data_config = DataConfig(
            mill_number=mill_number,
            start_date=start_date,
            end_date=end_date
        )
        return cls(data=data_config)
    
    def summary(self) -> str:
        """Generate configuration summary."""
        lines = [
            "=" * 60,
            "PIPELINE CONFIGURATION SUMMARY",
            "=" * 60,
            f"\nMill Number: {self.data.mill_number}",
            f"Date Range: {self.data.start_date} to {self.data.end_date}",
            f"\nFeatures:",
            f"  MV: {self.data.mv_features}",
            f"  CV: {self.data.cv_features}",
            f"  DV: {self.data.dv_features}",
            f"  Target: {self.data.target}",
            f"\nMotif Discovery:",
            f"  Window Size: {self.motif.mv_window_size} min",
            f"  Max Motifs: {self.motif.mv_max_motifs}",
            f"  Radius: {self.motif.mv_radius}",
            f"  Correlation Filter: {self.motif.apply_correlation_filter}",
            f"\nModel Training:",
            f"  Test Size: {self.model.test_size}",
            f"  CV Splits: {self.model.cv_splits}",
            f"\nOutput Paths:",
            f"  Output: {self.paths.output_dir}",
            f"  Models: {self.paths.models_dir}",
            f"  Analysis: {self.paths.analysis_dir}",
            f"  Plots: {self.paths.plots_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)
