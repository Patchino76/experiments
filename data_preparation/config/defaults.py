"""
Default configuration classes for the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    mill_number: int
    start_date: str
    end_date: str
    resample_freq: str = '1min'
    
    # Feature definitions
    mv_features: List[str] = field(default_factory=lambda: ['Ore', 'WaterMill', 'WaterZumpf'])
    cv_features: List[str] = field(default_factory=lambda: ['DensityHC', 'PulpHC', 'PressureHC', 'CirculativeLoad'])
    dv_features: List[str] = field(default_factory=lambda: ['Class_15', 'Daiki', 'FE', 'MotorAmp'])
    target: str = 'PSI200'
    
    # Data filtering thresholds
    filter_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Ore': (130, 220),
        'PulpHC': (350, 600),
        'DensityHC': (1600, 1920),
    })
    
    def get_all_features(self) -> List[str]:
        """Get all features (MV + CV + DV)."""
        return self.mv_features + self.cv_features + self.dv_features
    
    def get_all_columns(self) -> List[str]:
        """Get all required columns including target."""
        return self.get_all_features() + [self.target]


@dataclass
class PatternConfig:
    """
    Configuration for a single pattern.
    
    This is a flexible configuration that can represent any pattern type.
    """
    
    name: str
    type: str  # 'mv' or 'constraint'
    enabled: bool = True
    
    # Common parameters
    window_size: int = 60
    max_motifs: int = 15
    radius: float = 4.5
    max_instances_per_motif: int = 20
    
    # Pattern-specific parameters
    features: List[str] = field(default_factory=list)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Additional options
    save_analysis: bool = True
    save_plots: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for pattern creation."""
        return {
            'name': self.name,
            'type': self.type,
            'enabled': self.enabled,
            'window_size': self.window_size,
            'max_motifs': self.max_motifs,
            'radius': self.radius,
            'max_instances_per_motif': self.max_instances_per_motif,
            'features': self.features,
            'constraints': self.constraints,
            'save_analysis': self.save_analysis,
            'save_plots': self.save_plots
        }


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Output directories
    output_dir: Path = field(init=False)
    analysis_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.output_dir = self.base_dir / "output"
        self.analysis_dir = self.output_dir / "analysis"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def get_mill_plots_dir(self, mill_number: int) -> Path:
        """Get plots directory for specific mill."""
        mill_dir = self.plots_dir / f"mill_{mill_number}"
        mill_dir.mkdir(parents=True, exist_ok=True)
        return mill_dir


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    data: DataConfig
    patterns: List[PatternConfig] = field(default_factory=list)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Database configuration
    use_database: bool = True
    
    # Output options
    save_mv_only: bool = True  # Save MV motifs separately for training
    save_combined: bool = True  # Save all motifs combined
    save_to_database: bool = False  # Save to database
    
    @classmethod
    def create_default(
        cls,
        mill_number: int,
        start_date: str,
        end_date: str,
        patterns: List[PatternConfig] = None
    ) -> 'PipelineConfig':
        """
        Create default configuration.
        
        Args:
            mill_number: Mill identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            patterns: List of pattern configs (uses defaults if None)
            
        Returns:
            PipelineConfig instance
        """
        data_config = DataConfig(
            mill_number=mill_number,
            start_date=start_date,
            end_date=end_date
        )
        
        if patterns is None:
            from .pattern_configs import get_default_patterns
            patterns = get_default_patterns()
        
        return cls(data=data_config, patterns=patterns)
    
    def get_enabled_patterns(self) -> List[PatternConfig]:
        """Get list of enabled patterns."""
        return [p for p in self.patterns if p.enabled]
    
    def summary(self) -> str:
        """Generate configuration summary."""
        enabled = self.get_enabled_patterns()
        
        lines = [
            "=" * 70,
            "DATA PREPARATION PIPELINE CONFIGURATION",
            "=" * 70,
            f"\nMill Number: {self.data.mill_number}",
            f"Date Range: {self.data.start_date} to {self.data.end_date}",
            f"\nFeatures:",
            f"  MV: {self.data.mv_features}",
            f"  CV: {self.data.cv_features}",
            f"  DV: {self.data.dv_features}",
            f"  Target: {self.data.target}",
            f"\nPatterns ({len(enabled)} enabled):"
        ]
        
        for pattern in enabled:
            lines.append(f"  • {pattern.name} ({pattern.type})")
            lines.append(f"    Window: {pattern.window_size}, Max: {pattern.max_motifs}, Radius: {pattern.radius}")
            if pattern.constraints:
                stable = [k for k, v in pattern.constraints.items() if v.get('type') == 'stable']
                varying = [k for k, v in pattern.constraints.items() if v.get('type') == 'varying']
                if stable:
                    lines.append(f"    Stable: {stable}")
                if varying:
                    lines.append(f"    Varying: {varying}")
        
        lines.extend([
            f"\nOutput Options:",
            f"  Save MV only: {self.save_mv_only}",
            f"  Save combined: {self.save_combined}",
            f"  Save to database: {self.save_to_database}",
            f"  Use database: {self.use_database}",
            f"\nOutput Paths:",
            f"  Output: {self.paths.output_dir}",
            f"  Analysis: {self.paths.analysis_dir}",
            f"  Plots: {self.paths.plots_dir}",
            "=" * 70
        ])
        
        return "\n".join(lines)
