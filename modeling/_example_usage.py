"""
Example Usage Script

Demonstrates how to use the modeling pipeline with custom configuration.
"""

from config import PipelineConfig, DataConfig, MotifConfig, ModelConfig
from prepare_data import DataPreparationPipeline
from train_models import CascadeModelTrainer


def example_basic_usage():
    """Basic usage with default configuration."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    # Create default configuration
    config = PipelineConfig.create_default(
        mill_number=6,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Run data preparation
    print("\nRunning data preparation...")
    prep_pipeline = DataPreparationPipeline(config)
    prep_pipeline.run()
    
    # Run model training
    print("\nRunning model training...")
    train_pipeline = CascadeModelTrainer(config)
    train_pipeline.run()
    
    print("\n✓ Basic pipeline completed!")


def example_custom_configuration():
    """Advanced usage with custom configuration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 80)
    
    # Create custom data configuration
    data_config = DataConfig(
        mill_number=6,
        start_date="2024-01-01",
        end_date="2024-06-30",
        mv_features=['Ore', 'WaterMill', 'WaterZumpf', 'MotorAmp'],
        cv_features=['DensityHC', 'PulpHC', 'PressureHC'],
        dv_features=[],  # Can add: ['Class_15', 'Shisti', 'Daiki', 'FE']
        target='PSI200',
        filter_thresholds={
            'Ore': (100, 220),
            'PulpHC': (400, 600),
            'DensityHC': (1600, 1800),
        }
    )
    
    # Create custom motif configuration
    motif_config = MotifConfig(
        mv_window_size=60,
        mv_max_motifs=25,  # Increased from default 20
        mv_max_instances_per_motif=1000,
        mv_radius=4.0,  # Stricter than default 4.5
        apply_correlation_filter=True,
        correlation_rules={
            ('PressureHC', 'PulpHC'): 'pos',
            ('WaterZumpf', 'PressureHC'): 'pos',
        },
        min_correlation_strength=0.15,  # Increased from 0.1
        top_motifs_to_plot=15  # Plot more motifs
    )
    
    # Create custom model configuration
    model_config = ModelConfig(
        test_size=0.25,  # Larger test set
        n_estimators=400,  # More trees
        learning_rate=0.03,  # Lower learning rate
        cv_splits=3  # Fewer CV splits for faster training
    )
    
    # Combine into pipeline configuration
    config = PipelineConfig(
        data=data_config,
        motif=motif_config,
        model=model_config
    )
    
    # Print configuration summary
    print(config.summary())
    
    # Run pipelines
    print("\nRunning data preparation with custom config...")
    prep_pipeline = DataPreparationPipeline(config)
    prep_pipeline.run()
    
    print("\nRunning model training with custom config...")
    train_pipeline = CascadeModelTrainer(config)
    train_pipeline.run()
    
    print("\n✓ Custom pipeline completed!")


def example_cached_data():
    """Using cached data instead of database."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Using Cached Data")
    print("=" * 80)
    
    # Create configuration
    config = PipelineConfig.create_default(
        mill_number=6,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Disable database, use cached data
    config.use_database = False
    config.use_cached_data = True
    
    print("\nConfiguration:")
    print(f"  Use database: {config.use_database}")
    print(f"  Use cached data: {config.use_cached_data}")
    
    # Run data preparation (will use cached initial_data.csv)
    print("\nRunning data preparation with cached data...")
    prep_pipeline = DataPreparationPipeline(config)
    prep_pipeline.run()
    
    print("\n✓ Cached data pipeline completed!")


def example_data_preparation_only():
    """Run only data preparation without training."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Data Preparation Only")
    print("=" * 80)
    
    config = PipelineConfig.create_default(
        mill_number=6,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Run only data preparation
    print("\nRunning data preparation only...")
    prep_pipeline = DataPreparationPipeline(config)
    prep_pipeline.run()
    
    print("\n✓ Data preparation completed!")
    print("  You can now inspect the outputs before training models.")


def example_model_training_only():
    """Run only model training (assumes data is already prepared)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Model Training Only")
    print("=" * 80)
    
    config = PipelineConfig.create_default(
        mill_number=6,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Run only model training
    print("\nRunning model training only...")
    print("  (Assumes segmented_motifsMV.csv already exists)")
    
    train_pipeline = CascadeModelTrainer(config)
    train_pipeline.run()
    
    print("\n✓ Model training completed!")


if __name__ == "__main__":
    # Choose which example to run
    print("Mill Modeling Pipeline - Example Usage\n")
    print("Select an example to run:")
    print("  1. Basic usage (default configuration)")
    print("  2. Custom configuration")
    print("  3. Using cached data")
    print("  4. Data preparation only")
    print("  5. Model training only")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_custom_configuration()
    elif choice == "3":
        example_cached_data()
    elif choice == "4":
        example_data_preparation_only()
    elif choice == "5":
        example_model_training_only()
    else:
        print("Invalid choice. Running basic usage example...")
        example_basic_usage()
