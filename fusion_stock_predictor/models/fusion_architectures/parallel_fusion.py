import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class IntraStockAttention(nn.Module):
    def __init__(self, d_feat: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(d_feat, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # check the sequence length here !!!!!! optimize this
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model, # why 4 times??
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_feat)
        batch_size, seq_len, _ = x.shape
        
        # Project features to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.position_encoding[:, :seq_len, :]
        
        # Apply transformer for temporal encoding
        x = self.transformer(x)
        return x

class MarketGating(nn.Module):
    def __init__(self, d_feat: int, d_model: int):
        super().__init__()
        self.market_projection = nn.Linear(d_feat, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, market_info: torch.Tensor) -> torch.Tensor:
        # Project market info
        market_features = self.market_projection(market_info)
        # Generate dynamic gates
        gates = self.gate(market_features)
        # Apply gating mechanism
        return x * gates.unsqueeze(1)

class CrossTimeAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.mha(x, x, x)
        attended = self.dropout(attended)
        return self.norm(x + attended)

class ParallelFusion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_feat = config['model']['d_feat']
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        
        # Intra-stock temporal modeling
        self.intra_stock_attention = IntraStockAttention(
            self.d_feat, 
            self.d_model, 
            self.n_heads
        )
        
        # Market gating mechanism
        self.market_gating = MarketGating(self.d_feat, self.d_model)
        
        # Cross-time attention
        self.cross_time_attention = CrossTimeAttention(
            self.d_model, 
            self.n_heads
        )
        
        # Optional: Patch-based processing for efficiency
        self.use_patches = config['model'].get('use_patches', False)
        if self.use_patches:
            self.patch_size = config['model']['patch_size']
            self.patch_processor = nn.Sequential(
                nn.Linear(self.d_model * self.patch_size, self.d_model),
                nn.LayerNorm(self.d_model)
            )
        
        # Fusion layer - now outputs one prediction per stock
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 5)  # 5 stocks
        )
        
        # Add loss functions
        self.returns_loss = nn.MSELoss()
        self.direction_loss = nn.BCEWithLogitsLoss()
        
        # Prediction heads
        self.returns_head = nn.Linear(self.d_model, 5)  # Return values
        self.direction_head = nn.Linear(self.d_model, 5)  # Up/down prediction
    
    def forward(self, x: torch.Tensor, market_info: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Intra-stock temporal encoding
        temporal_features = self.intra_stock_attention(x)
        
        # Apply market-based gating
        gated_features = self.market_gating(temporal_features, market_info)
        
        # Cross-time attention
        cross_time_features = self.cross_time_attention(gated_features)
        
        # Optional: Patch-based processing
        if self.use_patches:
            B, L, D = cross_time_features.shape
            P = self.patch_size
            num_patches = L // P
            patched_features = cross_time_features.reshape(B, num_patches, P * D)
            patched_features = self.patch_processor(patched_features)
            final_features = patched_features
        else:
            final_features = cross_time_features
        
        # Adaptive fusion
        weights = torch.softmax(self.fusion_weights, dim=0)
        temporal_weight, market_weight = weights[0], weights[1]
        
        # Pool features
        temporal_pooled = torch.mean(final_features, dim=1)
        market_pooled = torch.mean(gated_features, dim=1)
        
        # Combine features
        combined = torch.cat([
            temporal_weight * temporal_pooled,
            market_weight * market_pooled
        ], dim=-1)
        
        # Final prediction
        prediction = self.fusion_layer(combined)
        
        attention_weights = {
            'temporal_weight': temporal_weight.item(),
            'market_weight': market_weight.item()
        }
        
        return prediction, attention_weights
    
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
