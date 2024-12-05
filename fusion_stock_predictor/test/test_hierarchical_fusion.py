import torch
import sys
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np
import torch.nn.functional as F

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fusion_stock_predictor.data.processor import DataProcessor
from fusion_stock_predictor.models.fusion_architectures.hierarchical_fusion import (
    HierarchicalFusion, TimeScaleAttention, HierarchicalProcessor
)
from fusion_stock_predictor.config.config import Config
from fusion_stock_predictor.analysis.component_analysis import ComponentAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_hierarchical_outputs(outputs: List[torch.Tensor], actual_data: torch.Tensor, level_names=None):
    """
    Visualize outputs from different hierarchical levels and compare with actual prices
    
    Args:
        outputs: List of tensor outputs from hierarchical processor
        actual_data: Original stock price data
        level_names: Names for each hierarchical level
    """
    if level_names is None:
        level_names = [f'Level {i+1}' for i in range(len(outputs))]
    
    plt.figure(figsize=(15, 5 * len(outputs)))
    
    # Get actual closing prices for the first stock
    actual_prices = actual_data[0, :, 0].detach().cpu().numpy()
    
    for idx, (output, name) in enumerate(zip(outputs, level_names)):
        plt.subplot(len(outputs), 1, idx + 1)
        
        # Convert predictions to numpy
        output_np = output[0, :, 0].detach().cpu().numpy()
        
        # Calculate the step size for this level
        step_size = len(actual_prices) // len(output_np)
        # Downsample actual prices to match the current level's sequence length
        actual_prices_downsampled = actual_prices[::step_size][:len(output_np)]
        
        # Plot actual prices
        plt.plot(range(0, len(actual_prices), step_size)[:len(output_np)], 
                actual_prices_downsampled, 
                label='Actual Stock Price', 
                color='green', 
                alpha=0.7)
        
        # Plot model output
        plt.plot(range(0, len(actual_prices), step_size)[:len(output_np)], 
                output_np, 
                label='Model Output', 
                color='blue', 
                alpha=0.7)
        
        # Add rolling average
        window = min(5, len(output_np) // 4)  # Smaller window for shorter sequences
        if window > 0:
            rolling_avg = np.convolve(output_np, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(output_np)), rolling_avg, 
                    label=f'Rolling Average (window={window})', 
                    color='red', 
                    linestyle='--')
        
        # Calculate error metrics using downsampled actual prices
        mse = np.mean((output_np - actual_prices_downsampled) ** 2)
        mae = np.mean(np.abs(output_np - actual_prices_downsampled))
        
        plt.title(f'{name} - MSE: {mse:.4f}, MAE: {mae:.4f}')
        plt.xlabel('Time Steps (Days)')
        plt.ylabel('Stock Price (Normalized)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'plots' / 'hierarchical_patterns.png')
    plt.close()

def compare_fusion_methods(hierarchical_model, parallel_model, test_loader):
    """Compare hierarchical and parallel fusion performance"""
    hierarchical_metrics = {
        'mse': [],
        'mae': [],
        'directional_accuracy': []
    }
    
    parallel_metrics = {
        'mse': [],
        'mae': [],
        'directional_accuracy': []
    }
    
    for batch_data, targets in test_loader:
        # Hierarchical predictions
        h_pred, _ = hierarchical_model(batch_data)
        
        # Parallel predictions
        p_pred = parallel_model(batch_data)
        
        # Calculate metrics
        for pred, metrics in [(h_pred, hierarchical_metrics), (p_pred, parallel_metrics)]:
            mse = F.mse_loss(pred, targets).item()
            mae = F.l1_loss(pred, targets).item()
            
            # Directional accuracy (whether price movement direction was predicted correctly)
            pred_direction = (pred[:, 1:] > pred[:, :-1]).float()
            true_direction = (targets[:, 1:] > targets[:, :-1]).float()
            dir_acc = (pred_direction == true_direction).float().mean().item()
            
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['directional_accuracy'].append(dir_acc)
    
    # Average metrics
    results = {
        'Hierarchical': {k: np.mean(v) for k, v in hierarchical_metrics.items()},
        'Parallel': {k: np.mean(v) for k, v in parallel_metrics.items()}
    }
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    
    metrics = ['mse', 'mae', 'directional_accuracy']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [results['Hierarchical'][m] for m in metrics], width, label='Hierarchical')
    plt.bar(x + width/2, [results['Parallel'][m] for m in metrics], width, label='Parallel')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Fusion Methods Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'plots' / 'fusion_comparison.png')
    plt.close()
    
    return results

def test_time_scale_attention():
    """Test the time-scale attention mechanism"""
    config = {
        'model': {
            'd_model': 32,
            'n_heads': 4
        }
    }
    
    # Create component
    time_attn = TimeScaleAttention(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads']
    )
    
    # Test input
    batch_size, seq_len = 16, 30
    x = torch.randn(batch_size, seq_len, config['model']['d_model'])
    
    # Forward pass
    output = time_attn(x)
    
    # Assertions
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    logger.info("TimeScaleAttention test passed")
    return output

def test_hierarchical_processor():
    """Test the hierarchical processing component with real stock data"""
    config = {
        'model': {
            'd_feat': 110,
            'd_model': 32,
            'n_heads': 4,
            'n_levels': 3
        },
        'data': {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'selected_stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'features': ['Close', 'Volume', 'High', 'Low', 'Open'],
            'target_columns': ['Close'],
            'sequence_length': 60,
            'prediction_horizon': 1,
            'train_test_split': 0.7,
            'validation_split': 0.1,
            'start_date': '2020-01-01',
            'end_date': '2023-01-01'
        },
        'training': {
            'batch_size': 16
        }
    }
    
    # Create component
    processor = HierarchicalProcessor(
        d_feat=config['model']['d_feat'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_levels=config['model']['n_levels']
    )
    
    # Load real stock data
    data_processor = DataProcessor(config)
    
    # Get a batch of real data
    train_loader, _, test_loader = data_processor.prepare_data()
    batch_data, _ = next(iter(train_loader))
    x = batch_data
    
    # Forward pass
    outputs = processor(x)
    
    # Visualize outputs with actual stock prices
    visualize_hierarchical_outputs(outputs, x, level_names=[
        f'Level 1 (Daily)',
        f'Level 2 (Weekly)',
        f'Level 3 (Monthly)'
    ])
    
    # Assertions
    assert len(outputs) == config['model']['n_levels'], \
        f"Expected {config['model']['n_levels']} outputs, got {len(outputs)}"
    
    # Calculate expected lengths
    expected_lengths = []
    current_len = x.size(1)  # 60 days
    for i in range(config['model']['n_levels']):
        expected_lengths.append(current_len)
        current_len = current_len // 2
    
    # Check sequence lengths
    for i, (output, exp_len) in enumerate(zip(outputs, expected_lengths)):
        assert output.size(1) == exp_len, \
            f"Level {i}: Expected sequence length {exp_len}, got {output.size(1)}"
        logger.info(f"Level {i}: Sequence length {output.size(1)} matches expected {exp_len}")
    
    logger.info("HierarchicalProcessor test passed")
    return outputs

def test_full_hierarchical_fusion():
    """Test the complete hierarchical fusion model"""
    config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    config = Config(str(config_path)).config
    
    # Create model
    model = HierarchicalFusion(config)
    
    # Test data
    batch_size = 32
    seq_len = 60
    d_feat = config['model']['d_feat']
    
    x = torch.randn(batch_size, seq_len, d_feat)
    
    # Forward pass
    prediction, attention_info = model(x)
    
    # Test loss computation
    dummy_targets = torch.randn_like(prediction)
    losses = model.compute_losses(prediction, dummy_targets)
    
    logger.info(f"Computed losses: {losses}")
    
    # Assertions
    assert prediction.shape == (batch_size, 5), \
        f"Expected prediction shape {(batch_size, 5)}, got {prediction.shape}"
    
    logger.info(f"Prediction shape: {prediction.shape}")
    logger.info("Full HierarchicalFusion test passed")
    return prediction, attention_info

def main():
    """Run all tests"""
    logger.info("Starting hierarchical fusion tests...")
    
    try:
        # Test individual components
        test_time_scale_attention()
        test_hierarchical_processor()
        
        # Test full model
        test_full_hierarchical_fusion()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()