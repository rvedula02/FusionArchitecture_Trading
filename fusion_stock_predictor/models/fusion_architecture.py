import torch
import torch.nn as nn

class SerialFusion(nn.Module):
    """Initial fusion architecture that processes patches through MASTER framework
    """
    def __init__(self, 
                 d_feat: int,
                 d_model: int,
                 patch_size: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Linear(d_feat * patch_size, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Market-guided gating
        self.market_gate = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.Sigmoid()
        )
        
        # Temporal attention for patches
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        
        # Output projection
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, 
                market_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patched input data (batch_size, num_patches, patch_size, features)
            market_features: Market condition data (batch_size, market_features)
        Returns:
            predictions: Shape (batch_size, 1)
        """
        B, N, P, F = x.shape
        
        # Reshape patches and embed
        x = x.view(B, N, -1)  # (B, N, P*F)
        x = self.patch_embedding(x)  # (B, N, d_model)
        
        # Apply market gating
        gate = self.market_gate(market_features)  # (B, d_model)
        x = x * gate.unsqueeze(1)
        
        # Apply temporal attention
        x = x.transpose(0, 1)  # (N, B, d_model)
        x, _ = self.temporal_attn(x, x, x)
        x = x.transpose(0, 1)  # (B, N, d_model)
        
        # Pool and predict
        x = x.mean(dim=1)  # (B, d_model)
        return self.output_layer(x)  # (B, 1)
