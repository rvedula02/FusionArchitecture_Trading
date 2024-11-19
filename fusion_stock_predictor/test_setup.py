import os
import logging
from pathlib import Path
from config.config import Config
from data.collectors.yahoo_collector import YahooDataCollector
from data.processor import DataProcessor
from utils.metrics import EvaluationMetrics
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent

def test_configuration():
    """Test configuration loading"""
    try:
        config_path = PROJECT_ROOT / 'config' / 'base_config.yaml'
        if not config_path.exists():
            logger.error(f"Config file not found at: {config_path}")
            return None
        
        config = Config(str(config_path))
        logger.info("Configuration loaded successfully")
        logger.info(f"Number of symbols: {len(config.get('data.symbols'))}")
        return config
    except Exception as e:
        logger.error(f"Configuration loading failed: {str(e)}")
        return None

def test_data_pipeline(config):
    """Test entire data pipeline"""
    try:
        processor = DataProcessor(config.config)
        train_loader, val_loader, test_loader = processor.prepare_data()
        
        logger.info("Data processing completed successfully")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Test a single batch
        X, y = next(iter(train_loader))
        logger.info(f"Sample batch shapes - X: {X.shape}, y: {y.shape}")
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise  # This will show the full error traceback
        return None

def main():
    """Run all tests"""
    logger.info("Starting setup tests...")
    
    # Test configuration
    config = test_configuration()
    if config is None:
        return
    
    # Test data pipeline
    loaders = test_data_pipeline(config)
    if loaders is None:
        return
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()
