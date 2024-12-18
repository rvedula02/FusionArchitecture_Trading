import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooDataCollector:
    def __init__(self, config: Dict):
        self.symbols = config['data']['symbols']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.sequence_length = config['data']['sequence_length']
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols"""
        data = {}
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d'
                )
                # Calculate basic features right after fetching
                df = self.calculate_technical_indicators(df)
                data[symbol] = df
                logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a single stock"""
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close']/df['Close'].shift(1))
            
            # Volume features
            df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
            
            # Price moving averages
            df['ma5'] = df['Close'].rolling(window=5).mean()
            df['ma20'] = df['Close'].rolling(window=20).mean()
            df['ma50'] = df['Close'].rolling(window=50).mean()
            
            # Volatility
            df['daily_volatility'] = df['returns'].rolling(window=20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Price momentum
            df['momentum'] = df['Close'] / df['Close'].shift(5) - 1
            
            # Drop any NaN values that resulted from calculations
            df = df.dropna()
            
            logger.debug(f"Calculated technical indicators. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
