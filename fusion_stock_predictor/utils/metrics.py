import numpy as np
import torch
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate various evaluation metrics"""
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'directional_accuracy': EvaluationMetrics._directional_accuracy(y_true, y_pred)
        }
    
    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy of predictions"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return np.mean(true_direction == pred_direction)
    
    @staticmethod
    def calculate_trading_metrics(returns: np.ndarray) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        return {
            'total_return': np.prod(1 + returns) - 1,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),  # Annualized
            'max_drawdown': EvaluationMetrics._calculate_max_drawdown(returns),
            'volatility': np.std(returns) * np.sqrt(252)  # Annualized
        }
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        return np.min(drawdowns)
