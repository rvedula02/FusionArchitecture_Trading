import torch
from ..models.fusion_architectures.parallel_fusion import ParallelFusion
from ..config.config import Config
from ..data.processor import DataProcessor

def test_parallel_fusion():
    # Load config
    config = Config('config/base_config.yaml').config
    
    # Create model
    model = ParallelFusion(config)
    
    # Create sample data
    batch_size = 32
    seq_len = 60
    d_feat = config['model']['d_feat']
    
    x = torch.randn(batch_size, seq_len, d_feat)
    market_info = torch.randn(batch_size, d_feat)
    
    # Forward pass
    prediction, attention_weights = model(x, market_info)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Attention weights: {attention_weights}")
    
if __name__ == "__main__":
    test_parallel_fusion()
