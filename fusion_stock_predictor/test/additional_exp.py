import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from fusion_stock_predictor.models.fusion_architectures.parallel_fusion import ParallelFusion
from fusion_stock_predictor.config.config import Config
from fusion_stock_predictor.data.processor import DataProcessor

# Set up logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_ablation_study(model: ParallelFusion, test_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict:
    """
    Run ablation studies by disabling different components of the model.
    
    Args:
        model: The parallel fusion model
        test_data: Tuple of (stock_features, market_features, targets)
    
    Returns:
        Dictionary containing performance metrics for each ablation
    """
    results = {}
    stock_features, market_features, targets = test_data
    
    # Baseline - full model
    with torch.no_grad():
        predictions, _ = model(stock_features, market_features)
        baseline_loss = model.compute_losses(predictions, targets)
        results['baseline'] = baseline_loss['total_loss']
    
    # Test without market gating
    model.market_gating = nn.Identity()
    with torch.no_grad():
        predictions, _ = model(stock_features, market_features)
        no_market_loss = model.compute_losses(predictions, targets)
        results['no_market_gating'] = no_market_loss['total_loss']
    
    # Test without cross-time attention
    model.cross_time_attention = nn.Identity()
    with torch.no_grad():
        predictions, _ = model(stock_features, market_features)
        no_cross_time_loss = model.compute_losses(predictions, targets)
        results['no_cross_time_attention'] = no_cross_time_loss['total_loss']
    
    return results

def test_market_conditions(model: ParallelFusion, 
                         data: pd.DataFrame,
                         volatility_threshold: float = 0.02) -> Dict:
    """
    Test model performance under different market conditions.
    
    Args:
        model: The parallel fusion model
        data: DataFrame containing market data
        volatility_threshold: Threshold to classify high/low volatility periods
    
    Returns:
        Dictionary containing performance metrics for different market conditions
    """
    results = {}
    
    # Calculate market volatility
    returns = data['market_return'].pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Split data into high/low volatility periods
    high_vol_mask = volatility > volatility_threshold
    low_vol_mask = volatility <= volatility_threshold
    
    # Test performance in different conditions
    for condition, mask in [('high_volatility', high_vol_mask), 
                          ('low_volatility', low_vol_mask)]:
        subset_data = data[mask]
        # Convert data to tensors and evaluate
        # Note: Actual implementation would need proper data preprocessing
        with torch.no_grad():
            stock_features = torch.tensor(subset_data[['feature1', 'feature2']].values)
            market_features = torch.tensor(subset_data['market_features'].values)
            targets = torch.tensor(subset_data['targets'].values)
            
            predictions, _ = model(stock_features, market_features)
            losses = model.compute_losses(predictions, targets)
            results[condition] = losses['total_loss']
    
    return results

def analyze_sequence_length_sensitivity(model: ParallelFusion,
                                     data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                     sequence_lengths: List[int]) -> Dict:
    """
    Analyze model sensitivity to different input sequence lengths.
    
    Args:
        model: The parallel fusion model
        data: Tuple of (stock_features, market_features, targets)
        sequence_lengths: List of sequence lengths to test
    
    Returns:
        Dictionary containing performance metrics for each sequence length
    """
    results = {}
    stock_features, market_features, targets = data
    
    for seq_len in sequence_lengths:
        # Truncate sequences to desired length
        truncated_stock = stock_features[:, -seq_len:, :]
        truncated_market = market_features[:, -seq_len:, :]
        
        with torch.no_grad():
            predictions, _ = model(truncated_stock, truncated_market)
            losses = model.compute_losses(predictions, targets)
            results[f'seq_len_{seq_len}'] = losses['total_loss']
    
    return results

def analyze_market_feature_importance(model: ParallelFusion,
                                   data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict:
    """
    Analyze the importance of different market features through perturbation analysis.
    
    Args:
        model: The parallel fusion model
        data: Tuple of (stock_features, market_features, targets)
    
    Returns:
        Dictionary containing feature importance scores
    """
    stock_features, market_features, targets = data
    feature_importance = {}
    
    # Get baseline performance
    with torch.no_grad():
        baseline_pred, _ = model(stock_features, market_features)
        baseline_loss = model.compute_losses(baseline_pred, targets)['total_loss']
    
    # Test importance of each market feature
    for feature_idx in range(market_features.shape[-1]):
        # Create copy with perturbed feature
        perturbed_market = market_features.clone()
        perturbed_market[..., feature_idx] = torch.randn_like(perturbed_market[..., feature_idx])
        
        with torch.no_grad():
            pred, _ = model(stock_features, perturbed_market)
            perturbed_loss = model.compute_losses(pred, targets)['total_loss']
            
        # Importance is the increase in loss when feature is perturbed
        feature_importance[f'market_feature_{feature_idx}'] = (perturbed_loss - baseline_loss).item()
    
    return feature_importance

def evaluate_time_horizons(model: ParallelFusion,
                         data: pd.DataFrame,
                         horizons: List[int]) -> Dict:
    """
    Evaluate model performance across different prediction time horizons.
    
    Args:
        model: The parallel fusion model
        data: DataFrame containing historical data
        horizons: List of time horizons (in days) to test
    
    Returns:
        Dictionary containing performance metrics for each horizon
    """
    results = {}
    
    for horizon in horizons:
        # Create targets for different horizons
        future_returns = data['close'].pct_change(horizon).shift(-horizon)
        
        # Prepare features and targets
        # Note: Actual implementation would need proper data preprocessing
        stock_features = torch.tensor(data[['feature1', 'feature2']].values)
        market_features = torch.tensor(data['market_features'].values)
        targets = torch.tensor(future_returns.values)
        
        with torch.no_grad():
            predictions, _ = model(stock_features, market_features)
            losses = model.compute_losses(predictions, targets)
            results[f'horizon_{horizon}'] = losses['total_loss']
    
    return results

def test_transfer_learning(model: ParallelFusion,
                         source_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         target_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         fine_tune_epochs: int = 5) -> Dict:
    """
    Test transfer learning capabilities of the model.
    
    Args:
        model: The parallel fusion model
        source_data: Training data from source domain
        target_data: Training data from target domain
        fine_tune_epochs: Number of epochs for fine-tuning
    
    Returns:
        Dictionary containing transfer learning performance metrics
    """
    results = {}
    
    # Evaluate on target domain before fine-tuning
    target_stock, target_market, target_labels = target_data
    with torch.no_grad():
        predictions, _ = model(target_stock, target_market)
        pre_transfer_loss = model.compute_losses(predictions, target_labels)
        results['pre_transfer'] = pre_transfer_loss['total_loss']
    
    # Fine-tune on target domain
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(fine_tune_epochs):
        predictions, _ = model(target_stock, target_market)
        loss = model.compute_losses(predictions, target_labels)['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        results[f'fine_tune_epoch_{epoch}'] = loss.item()
    
    # Evaluate after fine-tuning
    with torch.no_grad():
        predictions, _ = model(target_stock, target_market)
        post_transfer_loss = model.compute_losses(predictions, target_labels)
        results['post_transfer'] = post_transfer_loss['total_loss']
    
    return results

def run_all_experiments(model: ParallelFusion, data_processor: DataProcessor):
    """Run all experiments and log results"""
    logger.info("=" * 80)
    logger.info("Starting Parallel Fusion Model Experiments")
    logger.info("=" * 80)
    
    # Get test data
    _, _, test_loader = data_processor.prepare_data()
    test_batch = next(iter(test_loader))
    stock_features, market_features = test_batch[0], test_batch[0][:, -1, :]
    targets = test_batch[1]

    # 1. Ablation Study
    logger.info("\n1. Ablation Study Results")
    logger.info("-" * 40)
    ablation_results = run_ablation_study(model, (stock_features, market_features, targets))
    for variant, loss in ablation_results.items():
        logger.info(f"{variant}: {loss:.4f}")

    # 2. Market Conditions Test
    logger.info("\n2. Market Conditions Analysis")
    logger.info("-" * 40)
    market_results = test_market_conditions(model, test_loader)
    for condition, metrics in market_results.items():
        logger.info(f"{condition}: {metrics}")

    # 3. Sequence Length Sensitivity
    logger.info("\n3. Sequence Length Sensitivity")
    logger.info("-" * 40)
    seq_lengths = [30, 60, 90, 120]
    seq_results = analyze_sequence_length_sensitivity(
        model, 
        (stock_features, market_features, targets),
        seq_lengths
    )
    for seq_len, loss in seq_results.items():
        logger.info(f"{seq_len}: {loss:.4f}")

    # 4. Market Feature Importance
    logger.info("\n4. Market Feature Importance Analysis")
    logger.info("-" * 40)
    feature_importance = analyze_market_feature_importance(
        model,
        (stock_features, market_features, targets)
    )
    for feature, importance in feature_importance.items():
        logger.info(f"{feature}: {importance:.4f}")

    # 5. Time Horizon Analysis
    logger.info("\n5. Time Horizon Analysis")
    logger.info("-" * 40)
    horizons = [1, 3, 5, 10]
    horizon_results = evaluate_time_horizons(model, test_loader, horizons)
    for horizon, metrics in horizon_results.items():
        logger.info(f"{horizon}: {metrics}")

    # Save all results to CSV
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    all_results = {
        'ablation': pd.DataFrame.from_dict(ablation_results, orient='index'),
        'market_conditions': pd.DataFrame.from_dict(market_results, orient='index'),
        'sequence_length': pd.DataFrame.from_dict(seq_results, orient='index'),
        'feature_importance': pd.DataFrame.from_dict(feature_importance, orient='index'),
        'time_horizons': pd.DataFrame.from_dict(horizon_results, orient='index')
    }
    
    for name, df in all_results.items():
        df.to_csv(results_dir / f'{name}_results.csv')

    logger.info("\nAll results have been saved to CSV files in the results directory")
    logger.info(f"Complete log file available at: {log_file}")

def main():
    """Main function to run all experiments"""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
        config = Config(str(config_path)).config
        
        # Initialize model and data processor
        model = ParallelFusion(config)
        data_processor = DataProcessor(config)
        
        # Run all experiments
        run_all_experiments(model, data_processor)
        
    except Exception as e:
        logger.error(f"Error during experiments: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
