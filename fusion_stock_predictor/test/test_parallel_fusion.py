import torch
import sys
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fusion_stock_predictor.models.fusion_architectures.parallel_fusion import (
    ParallelFusion, IntraStockAttention, MarketGating, CrossTimeAttention
)
from fusion_stock_predictor.config.config import Config
from fusion_stock_predictor.data.processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_market_gating(x: torch.Tensor, output: torch.Tensor, gates: torch.Tensor):
    """Visualize the effect of market gating"""
    # Plot sample of original vs gated features
    plt.figure(figsize=(12, 6))
    
    # Sample first sequence from batch
    x_sample = x[0, :, 0].detach().numpy()  # First feature dimension
    output_sample = output[0, :, 0].detach().numpy()
    gates_sample = gates[0].mean().item()
    
    plt.plot(x_sample, label='Original', alpha=0.7)
    plt.plot(output_sample, label=f'Gated (avg gate: {gates_sample:.3f})', alpha=0.7)
    plt.title('Effect of Market Gating on Features')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'market_gating.png')
    plt.close()

def visualize_predictions(y_true: torch.Tensor, predictions: torch.Tensor):
    """Visualize model predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy for plotting
    y_true = y_true.detach().numpy()
    predictions = predictions.detach().numpy()
    
    # Plot first target variable
    plt.plot(y_true[:, 0], label='Actual', alpha=0.7)
    plt.plot(predictions[:, 0], label='Predicted', alpha=0.7)
    plt.title('Model Predictions vs Actual Values')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'predictions.png')
    plt.close()

def test_intra_stock_attention():
    """Test the intra-stock attention mechanism"""
    config = {
        'model': {
            'd_feat': 10,
            'd_model': 32,
            'n_heads': 4
        }
    }
    
    # Create component
    intra_attn = IntraStockAttention(
        d_feat=config['model']['d_feat'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads']
    )
    
    # Test input
    batch_size, seq_len = 16, 30
    x = torch.randn(batch_size, seq_len, config['model']['d_feat'])
    
    # Forward pass
    output = intra_attn(x)
    
    # Assertions
    assert output.shape == (batch_size, seq_len, config['model']['d_model']), \
        f"Expected shape {(batch_size, seq_len, config['model']['d_model'])}, got {output.shape}"
    
    logger.info("IntraStockAttention test passed")
    return output

def test_market_gating():
    """Test the market gating mechanism"""
    config = {
        'model': {
            'd_feat': 10,
            'd_model': 32
        }
    }
    
    # Create component
    market_gate = MarketGating(
        d_feat=config['model']['d_feat'],
        d_model=config['model']['d_model']
    )
    
    # Test inputs
    batch_size, seq_len = 16, 30
    x = torch.randn(batch_size, seq_len, config['model']['d_model'])
    market_info = torch.randn(batch_size, config['model']['d_feat'])
    
    # Forward pass
    output = market_gate(x, market_info)
    
    # Assertions
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    # Check that gating values are between 0 and 1
    gates = market_gate.gate(market_gate.market_projection(market_info))
    assert torch.all((gates >= 0) & (gates <= 1)), "Gate values should be between 0 and 1"
    
    # Check that the output is modified from the input
    assert not torch.allclose(output, x), "Gating should modify the input features"
    
    # Check that the gating operation preserves the sign of the input
    assert torch.all(torch.sign(output) == torch.sign(x)), "Gating should preserve sign of input"
    
    # Visualize the gating effect
    visualize_market_gating(x, output, gates)
    
    logger.info("MarketGating test passed")
    logger.info(f"Average gate value: {gates.mean().item():.3f}")
    return output

def test_cross_time_attention():
    """Test the cross-time attention mechanism"""
    config = {
        'model': {
            'd_model': 32,
            'n_heads': 4
        }
    }
    
    # Create component
    cross_attn = CrossTimeAttention(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads']
    )
    
    # Test input
    batch_size, seq_len = 16, 30
    x = torch.randn(batch_size, seq_len, config['model']['d_model'])
    
    # Forward pass
    output = cross_attn(x)
    
    # Assertions
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    logger.info("CrossTimeAttention test passed")
    return output

def test_full_parallel_fusion():
    """Test the complete parallel fusion model"""
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    config = Config(str(config_path)).config
    
    # Create model
    model = ParallelFusion(config)
    
    # Test data
    batch_size = 32
    seq_len = 60
    d_feat = config['model']['d_feat']
    
    x = torch.randn(batch_size, seq_len, d_feat)
    market_info = torch.randn(batch_size, d_feat)
    
    # Forward pass
    prediction, attention_weights = model(x, market_info)
    
    # Assertions
    assert prediction.shape == (batch_size, 1), \
        f"Expected prediction shape {(batch_size, 1)}, got {prediction.shape}"
    assert 'temporal_weight' in attention_weights, "Missing temporal weight"
    assert 'market_weight' in attention_weights, "Missing market weight"
    assert abs(attention_weights['temporal_weight'] + attention_weights['market_weight'] - 1) < 1e-6, \
        "Attention weights should sum to 1"
    
    logger.info(f"Prediction shape: {prediction.shape}")
    logger.info(f"Attention weights: {attention_weights}")
    logger.info("Full ParallelFusion test passed")
    return prediction, attention_weights

def test_with_real_data():
    """Test the model with real data from DataProcessor"""
    config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    config = Config(str(config_path))
    
    # Get real data
    processor = DataProcessor(config.config)
    train_loader, val_loader, test_loader = processor.prepare_data()
    
    # Create model with correct dimensions
    model = ParallelFusion(config.config)
    
    # Test with one batch
    x, y = next(iter(train_loader))
    logger.info(f"Input data shape: {x.shape}")
    logger.info(f"Target data shape: {y.shape}")
    
    # Use last timestep as market info
    market_info = x[:, -1, :]
    
    # Forward pass
    prediction, attention_weights = model(x, market_info)
    
    # Assertions
    assert prediction.shape[0] == y.shape[0], "Batch size mismatch"
    assert prediction.shape[1] == 1, "Prediction should be single value"
    
    logger.info(f"Real data test - Prediction shape: {prediction.shape}")
    logger.info(f"Real data test - Target shape: {y.shape}")
    logger.info(f"Real data test - Attention weights: {attention_weights}")
    
    # Log model statistics
    logger.info("\nModel Statistics:")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Input feature dimension: {config.config['model']['d_feat']}")
    logger.info(f"Model hidden dimension: {config.config['model']['d_model']}")
    logger.info(f"Number of attention heads: {config.config['model']['n_heads']}")
    
    # Log prediction statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"Mean prediction: {prediction.mean().item():.4f}")
    logger.info(f"Std prediction: {prediction.std().item():.4f}")
    logger.info(f"Mean target: {y.mean().item():.4f}")
    logger.info(f"Std target: {y.std().item():.4f}")
    
    # Visualize predictions
    visualize_predictions(y, prediction)
    
    logger.info("Real data test passed")

def main():
    """Run all tests"""
    logger.info("Starting parallel fusion tests...")
    
    try:
        # Test individual components
        test_intra_stock_attention()
        test_market_gating()
        test_cross_time_attention()
        
        # Test full model
        test_full_parallel_fusion()
        
        # Test with real data
        test_with_real_data()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
