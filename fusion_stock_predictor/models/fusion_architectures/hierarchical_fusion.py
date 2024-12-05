import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Dict
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class TimeScaleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.mha(x, x, x)
        attended = self.dropout(attended)
        return self.norm(x + attended)

class HierarchicalProcessor(nn.Module):
    def __init__(self, d_feat: int, d_model: int, n_heads: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        
        # Input projection
        self.input_projection = nn.Linear(d_feat, d_model)
        
        # Hierarchical processing layers
        self.level_processors = nn.ModuleList([
            TimeScaleAttention(d_model, n_heads) 
            for _ in range(n_levels)
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x shape: (batch_size, seq_len, d_feat)
        x = self.input_projection(x)
        
        outputs = []
        current = x
        current_len = x.size(1)
        
        for level, processor in enumerate(self.level_processors):
            # Calculate target length for this level
            target_len = current_len if level == 0 else current_len // 2
            
            if level > 0:
                # Apply pooling for this level
                pooled = nn.functional.adaptive_avg_pool1d(
                    current.transpose(1, 2), 
                    target_len
                ).transpose(1, 2)
            else:
                pooled = current
                
            # Process at this time scale
            processed = processor(pooled)
            outputs.append(processed)
            
            current = pooled
            current_len = target_len
            
            logger.debug(f"Level {level}: input_len={current.size(1)}, target_len={target_len}")
            
        return outputs

class HierarchicalFusion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_feat = config['model']['d_feat']
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        self.n_levels = config['model'].get('n_levels', 3)
        
        # Hierarchical processing
        self.hierarchical = HierarchicalProcessor(
            self.d_feat,
            self.d_model,
            self.n_heads,
            self.n_levels
        )
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            self.d_model,
            self.n_heads,
            batch_first=True
        )
        
        # Final prediction layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * self.n_levels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 5)  # 5 stocks
        )
        
        # Loss functions
        self.returns_loss = nn.MSELoss()
        self.direction_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Process at different time scales
        hierarchical_outputs = self.hierarchical(x)
        
        # Align sequences to shortest length
        min_len = min(out.size(1) for out in hierarchical_outputs)
        aligned_outputs = [
            out[:, -min_len:, :] for out in hierarchical_outputs
        ]
        
        # Stack for cross-scale attention
        stacked = torch.stack(aligned_outputs, dim=1)  # (batch, n_levels, seq_len, d_model)
        B, L, S, D = stacked.shape
        stacked = stacked.view(B * L, S, D)
        
        # Apply cross-scale attention
        fused_features, _ = self.cross_scale_attention(stacked, stacked, stacked)
        fused_features = fused_features.view(B, L, S, D)
        
        # Pool across sequence dimension
        pooled = torch.mean(fused_features, dim=2)  # (batch, n_levels, d_model)
        
        # Final prediction
        flattened = pooled.reshape(B, -1)
        prediction = self.fusion_layer(flattened)
        
        return prediction, {}
    
    def compute_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        returns_loss = self.returns_loss(predictions, targets)
        direction_pred = torch.sign(predictions)
        direction_true = torch.sign(targets)
        direction_loss = self.direction_loss(direction_pred, direction_true)
        
        return {
            'returns_loss': returns_loss,
            'direction_loss': direction_loss,
            'total_loss': returns_loss + 0.5 * direction_loss
        }