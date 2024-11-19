import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class PatchEncoder(nn.Module):
    def __init__(self, d_feat: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(d_feat * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_feat)
        batch_size, seq_len, in_features = x.shape
        # Reshape into patches
        patch_size = self.patch_size
        num_patches = seq_len // patch_size
        x = x.reshape(batch_size, num_patches, patch_size * in_features)
        
        # Encode patches
        x = self.linear(x)
        x = self.norm(x)
        return x

class MarketGuidedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, market_info: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # market_info shape: (batch_size, d_feat)
        
        # Create attention weights based on market info
        attn_weights = self._create_market_weights(x, market_info)
        
        # Apply attention 
        attended, _ = self.mha(x, x, x, attn_mask=None, key_padding_mask=None)
        attended = self.dropout(attended)
        
        # Apply market-based weighting
        attended = attended * attn_weights.unsqueeze(-1)
        
        return self.norm(x + attended)
    
    def _create_market_weights(self, x: torch.Tensor, market_info: torch.Tensor) -> torch.Tensor:
        # Create attention weights based on market conditions
        B, L, _ = x.shape
        
        # Project market info to sequence length
        market_weights = torch.sigmoid(market_info.mean(dim=-1))  # (batch_size,)
        
        # Expand to match sequence length
        weights = market_weights.unsqueeze(1).expand(-1, L)  # (batch_size, seq_len)
        
        return weights

class ParallelFusion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_feat = config['model']['d_feat']
        self.d_model = config['model']['d_model']
        self.patch_size = config['model']['patch_size']
        self.n_heads = config['model']['n_heads']
        
        # Patch-based branch
        self.patch_encoder = PatchEncoder(self.d_feat, self.d_model, self.patch_size)
        self.patch_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=4*self.d_model,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Market-guided branch
        self.market_encoder = nn.Linear(self.d_feat, self.d_model)
        self.market_attention = MarketGuidedAttention(self.d_model, self.n_heads)
        
        # Fusion layer
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )
        
    def forward(self, x: torch.Tensor, market_info: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Process through patch-based branch
        patch_encoded = self.patch_encoder(x)
        patch_output = self.patch_transformer(patch_encoded)
        
        # Process through market-guided branch
        market_encoded = self.market_encoder(x)
        market_output = self.market_attention(market_encoded, market_info)
        
        # Adaptive fusion
        weights = torch.softmax(self.fusion_weights, dim=0)
        patch_weight, market_weight = weights[0], weights[1]
        
        # Combine outputs
        patch_pooled = torch.mean(patch_output, dim=1)
        market_pooled = torch.mean(market_output, dim=1)
        
        combined = torch.cat([
            patch_weight * patch_pooled,
            market_weight * market_pooled
        ], dim=-1)
        
        # Final prediction
        prediction = self.fusion_layer(combined)
        
        # Return prediction and attention weights for analysis
        attention_weights = {
            'patch_weight': patch_weight.item(),
            'market_weight': market_weight.item()
        }
        
        return prediction, attention_weights
