import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns


# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fusion_stock_predictor.data.processor import DataProcessor
from fusion_stock_predictor.models.fusion_architectures.hierarchical_fusion import MarketAwareHierarchicalFusion
from fusion_stock_predictor.models.fusion_architectures.parallel_fusion import ParallelFusion
from fusion_stock_predictor.config.config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_returns_to_prices(returns: torch.Tensor, last_prices: torch.Tensor) -> torch.Tensor:
    """Convert returns to price predictions with safety checks"""
    # Clip returns to prevent extreme values
    returns = torch.clamp(returns, min=-0.1, max=0.1)  # More conservative clipping
    
    # Debug logging
    logger.info(f"Returns range after clipping: [{returns.min().item():.4f}, {returns.max().item():.4f}]")
    
    # Convert returns to prices
    predicted_prices = last_prices * (1 + returns)
    
    # Safety checks
    if torch.isnan(predicted_prices).any() or torch.isinf(predicted_prices).any():
        logger.error("NaN or Inf detected in price predictions!")
        predicted_prices = torch.nan_to_num(predicted_prices, nan=last_prices.mean())
    
    # Ensure predictions don't deviate too far from last prices
    max_change = 0.2  # 20% maximum change
    lower_bound = last_prices * (1 - max_change)
    upper_bound = last_prices * (1 + max_change)
    predicted_prices = torch.clamp(predicted_prices, min=lower_bound, max=upper_bound)
    
    logger.info(f"Predicted prices range: [{predicted_prices.min().item():.2f}, {predicted_prices.max().item():.2f}]")
    
    return predicted_prices

def calculate_normalized_metrics(pred: torch.Tensor, target: torch.Tensor, last_prices: torch.Tensor):
    """Calculate normalized metrics to handle different price scales
    Args:
        pred: Predicted prices tensor [batch_size, sequence_length]
        target: Target prices tensor [batch_size, sequence_length]
        last_prices: Last known prices tensor [batch_size, num_stocks]
    Returns:
        Dictionary of normalized metrics
    """
    # Add debug logging
    logger.info(f"Prediction range: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
    logger.info(f"Target range: [{target.min().item():.2f}, {target.max().item():.2f}]")
    logger.info(f"Last prices range: [{last_prices.min().item():.2f}, {last_prices.max().item():.2f}]")
    
    # Check if targets are returns instead of prices
    if target.abs().max() < 1.0:
        logger.info("Converting targets from returns to prices")
        target = last_prices * (1 + target)
    
    # Normalize predictions and targets by last known price
    norm_pred = pred / last_prices.mean()
    norm_target = target / last_prices.mean()
    
    # Calculate basic error metrics
    mse = torch.nn.functional.mse_loss(norm_pred, norm_target)  # Keep as tensor
    mae = torch.nn.functional.l1_loss(norm_pred, norm_target)   # Keep as tensor
    
    # Calculate MAPE with safety checks
    epsilon = 1e-7  # Small constant to prevent division by zero
    abs_percentage_error = torch.abs((pred - target) / (torch.abs(target) + epsilon))
    # Clip extreme values
    abs_percentage_error = torch.clamp(abs_percentage_error, min=0.0, max=1.0)
    mape = torch.mean(abs_percentage_error) * 100
    
    # Directional accuracy
    pred_direction = (pred[:, 1:] > pred[:, :-1]).float()
    true_direction = (target[:, 1:] > target[:, :-1]).float()
    dir_acc = (pred_direction == true_direction).float().mean()
    
    # Calculate RMSE
    rmse = torch.sqrt(mse)
    
    # Convert all metrics to Python floats
    return {
        'normalized_mse': mse.item(),
        'normalized_mae': mae.item(),
        'rmse': rmse.item(),
        'mape': mape.item(),
        'directional_accuracy': dir_acc.item()
    }

def plot_comparison_predictions(h_prices, p_prices, true_prices, last_prices, stock_names, save_path=None):
    """Plot both hierarchical and parallel predictions vs actual prices for each stock"""
    # Debug shapes
    logger.info(f"Plotting - hierarchical shape: {h_prices.shape}")
    logger.info(f"Plotting - parallel shape: {p_prices.shape}")
    logger.info(f"Plotting - true shape: {true_prices.shape}")
    logger.info(f"Plotting - last shape: {last_prices.shape}")
    
    # Convert tensors to numpy for plotting
    h_np = h_prices.detach().cpu().numpy()
    p_np = p_prices.detach().cpu().numpy()
    true_np = true_prices.detach().cpu().numpy()
    last_np = last_prices.detach().cpu().numpy()
    
    # Set style to a built-in style
    plt.style.use('bmh')  # Alternative clean style
    
    # Create subplot for each stock
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, stock in enumerate(stock_names):
        if i >= len(axes):  # Skip if we have more stocks than subplot spaces
            break
            
        ax = axes[i]
        
        # Extract data for this stock
        h_stock = h_np[:, i]    # Hierarchical predictions
        p_stock = p_np[:, i]    # Parallel predictions
        true_stock = true_np[:, i]  # True values
        last_price = last_np[0, i]  # Last known price
        
        # Plot predictions and actual prices
        time_steps = range(len(true_stock))
        ax.plot(time_steps, h_stock, label='Hierarchical', color='#1f77b4', linestyle='--', alpha=0.8)
        ax.plot(time_steps, p_stock, label='Parallel', color='#d62728', linestyle='--', alpha=0.8)
        ax.plot(time_steps, true_stock, label='Actual', color='#2ca02c', linewidth=2)
        ax.axhline(y=last_price, color='#7f7f7f', linestyle=':', label='Last Price', alpha=0.5)
        
        # Calculate and display MAPE for each method
        h_mape = np.mean(np.abs((h_stock - true_stock) / true_stock)) * 100
        p_mape = np.mean(np.abs((p_stock - true_stock) / true_stock)) * 100
        
        ax.set_title(f'{stock} Price Predictions\nMAPE - Hierarchical: {h_mape:.2f}%, Parallel: {p_mape:.2f}%')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add price labels on y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
    
    plt.suptitle('Comparison of Hierarchical and Parallel Fusion Predictions', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

def compare_fusion_methods(hierarchical_model, parallel_model, test_loader):
    """Compare hierarchical and parallel fusion performance"""
    hierarchical_metrics = {
        'normalized_mse': [],
        'normalized_mae': [],
        'rmse': [],
        'mape': [],
        'directional_accuracy': []
    }
    
    parallel_metrics = {
        'normalized_mse': [],
        'normalized_mae': [],
        'rmse': [],
        'mape': [],
        'directional_accuracy': []
    }
    
    stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    for batch_idx, (batch_data, targets) in enumerate(test_loader):
        # Extract last known prices
        last_prices = batch_data[:, -1, ::22]  # Every 22nd feature is a price
        
        # Extract market information from the last timestep
        market_info = batch_data[:, -1, :]
        
        # Get hierarchical predictions
        h_pred_tuple = hierarchical_model(batch_data, market_info)
        h_pred = h_pred_tuple[0] if isinstance(h_pred_tuple, tuple) else h_pred_tuple
        h_prices = convert_returns_to_prices(h_pred, last_prices)
        
        # Get parallel predictions
        p_pred_tuple = parallel_model(batch_data, market_info)
        p_pred = p_pred_tuple[0] if isinstance(p_pred_tuple, tuple) else p_pred_tuple
        p_prices = convert_returns_to_prices(p_pred, last_prices)
        
        # Plot first batch predictions
        if batch_idx == 0:
            # Convert targets to prices if they're returns
            if targets.abs().max() < 1.0:
                target_prices = last_prices * (1 + targets[..., 0])
            else:
                target_prices = targets[..., 0]
            
            logger.info("Plotting predictions comparison")
            
            # Plot both predictions on the same graphs
            plot_comparison_predictions(
                h_prices,
                p_prices, 
                target_prices,
                last_prices,
                stock_names,
                save_path='results/fusion_comparison_predictions.png'
            )
        
        # Calculate metrics
        h_metrics = calculate_normalized_metrics(h_prices, targets[..., 0], last_prices)
        p_metrics = calculate_normalized_metrics(p_prices, targets[..., 0], last_prices)
        
        # Store metrics
        for metric_name in hierarchical_metrics:
            hierarchical_metrics[metric_name].append(h_metrics[metric_name])
            parallel_metrics[metric_name].append(p_metrics[metric_name])
    
    # Average metrics
    results = {
        'Hierarchical': {k: np.mean(v) for k, v in hierarchical_metrics.items()},
        'Parallel': {k: np.mean(v) for k, v in parallel_metrics.items()}
    }
    
    # Create DataFrame and add interpretation
    df = pd.DataFrame(results).round(4)
    df['Difference (%)'] = ((df['Parallel'] - df['Hierarchical']) / df['Hierarchical'] * 100).round(2)
    
    print("\nModel Comparison Results:")
    print("=" * 80)
    print(df)
    print("\nMetric Explanations:")
    print("-" * 80)
    print("Normalized MSE: Mean Squared Error normalized by square of average price")
    print("Normalized MAE: Mean Absolute Error normalized by average price")
    print("MAPE: Mean Absolute Percentage Error")
    print("Directional Accuracy: Percentage of correct trend predictions")
    
    return df

def visualize_market_impact(hierarchical_outputs, market_gates, save_path=None):
    """Visualize the impact of market gating across hierarchical levels"""
    plt.figure(figsize=(15, 10))
    
    n_levels = len(hierarchical_outputs)
    
    for level in range(n_levels):
        plt.subplot(n_levels, 1, level + 1)
        
        # Plot original and gated features
        features = hierarchical_outputs[level][0, :, 0].detach().cpu().numpy()
        gate_value = market_gates[level].mean().item()
        
        plt.plot(features, label='Original Features', alpha=0.7)
        plt.plot(features * gate_value, label=f'Gated Features (gate={gate_value:.3f})', 
                linestyle='--', alpha=0.7)
        
        plt.title(f'Level {level + 1} Market Impact')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Configuration and model setup
    config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    config = Config(str(config_path)).config
    
    # Add debug info
    logger.info("Loading configuration from: %s", config_path)
    logger.info("Model configuration: %s", config['model'])
    
    # Initialize models
    hierarchical_model = MarketAwareHierarchicalFusion(config)
    parallel_model = ParallelFusion(config)
    
    # Load data
    data_processor = DataProcessor(config)
    _, _, test_loader = data_processor.prepare_data()
    
    # Compare models
    results_df = compare_fusion_methods(hierarchical_model, parallel_model, test_loader)
    
    # Save results
    results_path = Path(__file__).parent / 'results'
    results_path.mkdir(exist_ok=True)
    results_df.to_csv(results_path / 'fusion_comparison_results.csv')
    logger.info("Results saved to: %s", results_path / 'fusion_comparison_results.csv')

if __name__ == "__main__":
    main()
