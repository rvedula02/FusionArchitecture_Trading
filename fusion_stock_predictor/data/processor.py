import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .collectors.yahoo_collector import YahooDataCollector
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.collector = YahooDataCollector(config)
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation and test dataloaders"""
        # Fetch and process data
        raw_data = self.collector.fetch_data()
        processed_data = self._process_raw_data(raw_data)
        
        # Create sequences
        X, y = self._create_sequences(processed_data)
        
        # Split data
        train_data, val_data, test_data = self._split_data(X, y)
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_data[0], train_data[1], is_train=True)
        val_loader = self._create_dataloader(val_data[0], val_data[1], is_train=False)
        test_loader = self._create_dataloader(test_data[0], test_data[1], is_train=False)
        
        return train_loader, val_loader, test_loader
    
    def _process_raw_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process raw data and combine all features"""
        processed_dfs = []
        
        for symbol, df in raw_data.items():
            # Add symbol prefix to columns
            df = df.copy()
            df.columns = [f"{symbol}_{col}" for col in df.columns]
            processed_dfs.append(df)
        
        # Combine all processed dataframes
        combined_df = pd.concat(processed_dfs, axis=1)
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        sequence_length = self.config['data']['sequence_length']
        prediction_horizon = self.config['data']['prediction_horizon']
        
        # Get all return columns
        return_columns = [col for col in data.columns if col.endswith('_returns')]
        logger.info(f"Return columns found: {return_columns}")
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence: all features
            X.append(data.iloc[i:i+sequence_length].values)
            
            # Target: returns for all stocks for the next 'prediction_horizon' days
            target = data.iloc[i+sequence_length:i+sequence_length+prediction_horizon][return_columns].values
            y.append(target)
            
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Split data into train, validation and test sets"""
        train_split = self.config['data']['train_test_split']
        val_split = self.config['data']['validation_split']
        
        n = len(X)
        train_idx = int(n * train_split)
        val_idx = int(n * (train_split + val_split))
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, is_train: bool) -> DataLoader:
        """Create dataloader"""
        dataset = StockDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=is_train,
            num_workers=4
        )
