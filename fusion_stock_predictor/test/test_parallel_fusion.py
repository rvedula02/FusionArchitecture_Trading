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
from fusion_stock_predictor.analysis.component_analysis import ComponentAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_market_gating(x: torch.Tensor, output: torch.Tensor, gates: torch.Tensor):
    """Visualize the effect of market gating"""
    plt.figure(figsize=(15, 8))
    
    # Create subplots
    plt.subplot(2, 1, 1)
    x_sample = x[0, :, 0].detach().numpy()
    output_sample = output[0, :, 0].detach().numpy()
    gates_sample = gates[0].detach().numpy()
    
    # Plot features
    plt.plot(x_sample, label='Original Features', alpha=0.7)
    plt.plot(output_sample, label='Gated Features', alpha=0.7)
    plt.title('Effect of Market Gating on Features')
    plt.legend()
    plt.grid(True)
    
    # Plot gate values
    plt.subplot(2, 1, 2)
    plt.plot(gates_sample, label='Gate Values', color='green')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.title('Market Gate Values')
    plt.ylabel('Gate Strength')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'plots' / 'market_gating.png')
    plt.close()

def visualize_predictions(y_true: torch.Tensor, predictions: torch.Tensor, stock_names=None, return_types=None):
    """Visualize model predictions vs actual values for each stock"""
    if stock_names is None:
        stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    if return_types is None:
        return_types = ['returns', 'log_returns']
    
    # Reshape y_true if it's 3D (batch, stocks, returns)
    if len(y_true.shape) == 3:
        batch_size, num_stocks, num_returns = y_true.shape
        y_true = y_true.reshape(batch_size, num_stocks * num_returns)
    
    num_stocks = len(stock_names)
    fig, axes = plt.subplots(num_stocks, 1, figsize=(15, 6*num_stocks))
    if num_stocks == 1:
        axes = [axes]
    
    # Convert to numpy for plotting
    y_true = y_true.detach().numpy()
    predictions = predictions.detach().numpy()
    
    colors = {'returns': 'blue', 'log_returns': 'green', 'predicted': 'red'}
    
    for idx, (ax, stock) in enumerate(zip(axes, stock_names)):
        # Plot each type of return
        for j, return_type in enumerate(return_types):
            return_idx = idx * 2 + j  # Each stock has 2 return types
            ax.plot(y_true[:, return_idx], 
                   label=f'Actual {stock} ({return_type})', 
                   color=colors[return_type], 
                   alpha=0.7)
        
        # Plot predictions for this stock
        ax.plot(predictions[:, idx], 
                label=f'Predicted {stock}', 
                color=colors['predicted'], 
                linestyle='--', 
                alpha=0.7)
        
        ax.set_title(f'{stock} Stock Returns - Predictions vs Actual')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Returns')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        # Add statistics to the plot
        stats_text = []
        for j, return_type in enumerate(return_types):
            return_idx = idx * 2 + j
            stats_text.append(f'{return_type}:\n')
            stats_text.append(f'  Mean: {y_true[:, return_idx].mean():.4f}\n')
            stats_text.append(f'  Std: {y_true[:, return_idx].std():.4f}\n')
        
        stats_text.append(f'Predictions:\n')
        stats_text.append(f'  Mean: {predictions[:, idx].mean():.4f}\n')
        stats_text.append(f'  Std: {predictions[:, idx].std():.4f}')
        
        ax.text(1.05, 0.5, ''.join(stats_text),
                transform=ax.transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'stock_predictions.png', dpi=300, bbox_inches='tight')
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
    gates = market_gate.gate(market_gate.market_projection(market_info))
    
    # Visualize the gating effect
    visualize_market_gating(x, output, gates)
    
    # Assertions
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    # Check that gating values are between 0 and 1
    assert torch.all((gates >= 0) & (gates <= 1)), "Gate values should be between 0 and 1"
    
    # Check that the output is modified from the input
    assert not torch.allclose(output, x), "Gating should modify the input features"
    
    # Check that the gating operation preserves the sign of the input
    assert torch.all(torch.sign(output) == torch.sign(x)), "Gating should preserve sign of input"
    
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

def analyze_model_components(model: ParallelFusion, x: torch.Tensor, market_info: torch.Tensor):
    """Analyze model components behavior"""
    analyzer = ComponentAnalysis(model)
    
    # Analyze attention patterns
    attention_patterns = analyzer.analyze_attention_patterns(x)
    logger.info("\nAttention Pattern Analysis:")
    logger.info(f"  Mean temporal feature value: {attention_patterns['temporal_mean'].mean().item():.4f}")
    logger.info(f"  Std of temporal features: {attention_patterns['temporal_std'].mean().item():.4f}")
    
    # Analyze market impact
    market_impact = analyzer.analyze_market_impact(x, market_info)
    logger.info("\nMarket Impact Analysis:")
    logger.info(f"  Average gate strength: {market_impact['gate_mean']:.4f}")
    logger.info(f"  Gate variation: {market_impact['gate_std']:.4f}")
    logger.info(f"  Number of high-impact features: {market_impact['high_impact_features']}")
    
    return attention_patterns, market_impact

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
    num_stocks = 5  # Number of stocks we're predicting
    
    x = torch.randn(batch_size, seq_len, d_feat)
    market_info = torch.randn(batch_size, d_feat)
    
    # Forward pass
    prediction, attention_weights = model(x, market_info)
    
    # Test loss computation
    dummy_targets = torch.randn_like(prediction)  # For testing
    losses = model.compute_losses(prediction, dummy_targets)
    
    logger.info(f"Computed losses: {losses}")
    
    # Analyze components
    attention_patterns, market_impact = analyze_model_components(model, x, market_info)
    
    # Assertions
    assert prediction.shape == (batch_size, num_stocks), \
        f"Expected prediction shape {(batch_size, num_stocks)}, got {prediction.shape}"
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
    
    # Create model
    model = ParallelFusion(config.config)
    
    # Test with one batch
    x, y = next(iter(train_loader))
    logger.info(f"Input data shape: {x.shape}")
    logger.info(f"Target data shape: {y.shape}")
    
    # Get stock names and return types
    stock_names = processor.stock_names if hasattr(processor, 'stock_names') else None
    return_types = processor.return_types if hasattr(processor, 'return_types') else None
    
    # Use last timestep as market info
    market_info = x[:, -1, :]
    
    # Forward pass
    prediction, attention_weights = model(x, market_info)
    
    # Log detailed information about each stock and return type
    logger.info("\nPer-Stock Statistics:")
    for idx, stock in enumerate(stock_names or range(y.shape[1] // 2)):
        logger.info(f"\n{stock}:")
        for j, return_type in enumerate(['returns', 'log_returns']):
            return_idx = idx * 2 + j
            logger.info(f"{return_type}:")
            logger.info(f"  Mean: {y[:, return_idx].mean().item():.4f}")
            logger.info(f"  Std: {y[:, return_idx].std().item():.4f}")
    
    # Visualize predictions
    visualize_predictions(y, prediction, stock_names, return_types)
    
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
