import os
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application info
    APP_NAME: str = "Mills XGBoost API"
    APP_VERSION: str = "1.0.0"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    
    # Default database connection parameters
    # These will be overridden by environment variables or direct API calls
    DB_HOST: str = os.environ.get("DB_HOST", "em-m-db4.ellatzite-med.com")
    DB_PORT: int = int(os.environ.get("DB_PORT", "5432"))
    DB_NAME: str = os.environ.get("DB_NAME", "em_pulse_data")
    DB_USER: str = os.environ.get("DB_USER", "s.lyubenov")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "tP9uB7sH7mK6zA7t")
    
    # Default paths
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    # Data processing settings
    RESAMPLE_FREQUENCY: str = "1min"  # For mill sensor data, which is at 1-minute intervals
    
    # Feature sets - these are the most commonly used features for different targets
    FEATURE_SETS: Dict[str, Dict[str, Any]] = {
        "PSI80": {
            "features": [
                "Ore", "WaterMill", "WaterZumpf", "PressureHC", 
                "DensityHC", "MotorAmp", "Shisti", "Daiki"
            ],
            "default_bounds": {
                'Ore': (160.0, 200.0),
                'WaterMill': (12.0, 18.0),
                'WaterZumpf': (140.0, 240.0),
                'PressureHC': (0.3, 0.5),
                'DensityHC': (1500, 1900),
                'MotorAmp': (170.0, 220.0)
            }
        },
        "FR200": {
            "features": [
                "Ore", "WaterMill", "WaterZumpf", "PressureHC", 
                "DensityHC", "MotorAmp", "Shisti", "Daiki"
            ],
            "default_bounds": {
                'Ore': (160.0, 200.0),
                'WaterMill': (12.0, 18.0),
                'WaterZumpf': (140.0, 240.0),
                'PressureHC': (0.3, 0.5),
                'DensityHC': (1500, 1900),
                'MotorAmp': (170.0, 220.0)
            }
        }
    }
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        case_sensitive = True  # Important because PostgreSQL column names are case-sensitive

settings = Settings()
