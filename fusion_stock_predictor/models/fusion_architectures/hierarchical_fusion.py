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

# class HierarchicalFusion(nn.Module):
#     def __init__(self, config: dict):
#         super().__init__()
#         self.d_feat = config['model']['d_feat']
#         self.d_model = config['model']['d_model']
#         self.n_heads = config['model']['n_heads']
#         self.n_levels = config['model'].get('n_levels', 3)
#         
#         # Hierarchical processing
#         self.hierarchical = HierarchicalProcessor(
#             self.d_feat,
#             self.d_model,
#             self.n_heads,
#             self.n_levels
#         )
#         
#         # Cross-scale attention
#         self.cross_scale_attention = nn.MultiheadAttention(
#             self.d_model,
#             self.n_heads,
#             batch_first=True
#         )
#         
#         # Final prediction layers
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(self.d_model * self.n_levels, self.d_model),
#             nn.ReLU(),
#             nn.Linear(self.d_model, 5)  # 5 stocks
#         )
#         
#         # Loss functions
#         self.returns_loss = nn.MSELoss()
#         self.direction_loss = nn.BCEWithLogitsLoss()
#         
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
#         # Process at different time scales
#         hierarchical_outputs = self.hierarchical(x)
#         
#         # Align sequences to shortest length
#         min_len = min(out.size(1) for out in hierarchical_outputs)
#         aligned_outputs = [
#             out[:, -min_len:, :] for out in hierarchical_outputs
#         ]
#         
#         # Stack for cross-scale attention
#         stacked = torch.stack(aligned_outputs, dim=1)  # (batch, n_levels, seq_len, d_model)
#         B, L, S, D = stacked.shape
#         stacked = stacked.view(B * L, S, D)
#         
#         # Apply cross-scale attention
#         fused_features, _ = self.cross_scale_attention(stacked, stacked, stacked)
#         fused_features = fused_features.view(B, L, S, D)
#         
#         # Pool across sequence dimension
#         pooled = torch.mean(fused_features, dim=2)  # (batch, n_levels, d_model)
#         
#         # Final prediction
#         flattened = pooled.reshape(B, -1)
#         prediction = self.fusion_layer(flattened)
#         
#         return prediction, {}
#     
#     def compute_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
#         returns_loss = self.returns_loss(predictions, targets)
#         direction_pred = torch.sign(predictions)
#         direction_true = torch.sign(targets)
#         direction_loss = self.direction_loss(direction_pred, direction_true)
#         
#         return {
#             'returns_loss': returns_loss,
#             'direction_loss': direction_loss,
#             'total_loss': returns_loss + 0.5 * direction_loss
#         }

class MarketAwareHierarchicalFusion(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.d_feat = config['model']['d_feat']
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        self.n_levels = config['model'].get('n_levels', 3)
        
        # Calculate expected input dimension for fusion layer
        self.fusion_input_dim = self.d_model * self.n_levels * 2  # *2 for temporal and market features
        logger.info(f"Initializing fusion layer with input dim: {self.fusion_input_dim}")
        
        # Market projection layer
        self.market_projection = nn.Linear(self.d_feat, self.d_model)
        
        # Market gating mechanism for each level
        self.market_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            ) for _ in range(self.n_levels)
        ])
        
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
        
        # Market-temporal fusion layer
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Final prediction layers with correct dimensions
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, 5)  # 5 stocks
        )
        
        # Loss functions
        self.returns_loss = nn.MSELoss()
        self.direction_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, x: torch.Tensor, market_info: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Add dimension logging
        logger.info(f"Input shapes - x: {x.shape}, market_info: {market_info.shape}")
        
        # Project market information
        market_features = self.market_projection(market_info)
        logger.info(f"Projected market features shape: {market_features.shape}")
        
        # Process at different time scales
        hierarchical_outputs = self.hierarchical(x)
        logger.info(f"Number of hierarchical outputs: {len(hierarchical_outputs)}")
        for i, out in enumerate(hierarchical_outputs):
            logger.info(f"Hierarchical output {i} shape: {out.shape}")
        
        # Apply market gating to each level
        gated_outputs = []
        for level_output, gate in zip(hierarchical_outputs, self.market_gates):
            market_gate = gate(market_features)
            gated_output = level_output * market_gate.unsqueeze(1)
            gated_outputs.append(gated_output)
            logger.info(f"Gated output shape: {gated_output.shape}")
        
        # Align sequences to shortest length
        min_len = min(out.size(1) for out in gated_outputs)
        aligned_outputs = [
            out[:, -min_len:, :] for out in gated_outputs
        ]
        
        # Stack for cross-scale attention
        stacked = torch.stack(aligned_outputs, dim=1)
        logger.info(f"Stacked shape: {stacked.shape}")
        
        B, L, S, D = stacked.shape
        stacked = stacked.view(B * L, S, D)
        
        # Apply cross-scale attention
        fused_features, _ = self.cross_scale_attention(stacked, stacked, stacked)
        fused_features = fused_features.view(B, L, S, D)
        logger.info(f"Fused features shape: {fused_features.shape}")
        
        # Pool across sequence dimension
        pooled = torch.mean(fused_features, dim=2)
        logger.info(f"Pooled shape: {pooled.shape}")
        
        # Adaptive fusion with market features
        weights = torch.softmax(self.fusion_weights, dim=0)
        temporal_weight, market_weight = weights[0], weights[1]
        
        # Combine hierarchical temporal and market features
        temporal_features = pooled.reshape(B, -1)
        logger.info(f"Temporal features shape: {temporal_features.shape}")
        
        market_features_expanded = market_features.unsqueeze(1).expand(-1, self.n_levels, -1)
        market_features_flat = market_features_expanded.reshape(B, -1)
        logger.info(f"Market features flat shape: {market_features_flat.shape}")
        
        # Calculate expected dimensions
        expected_temporal_dim = self.d_model * self.n_levels
        expected_market_dim = self.d_model * self.n_levels
        logger.info(f"Expected dimensions - temporal: {expected_temporal_dim}, market: {expected_market_dim}")
        
        # Ensure dimensions match before concatenation
        assert temporal_features.shape[1] == expected_temporal_dim, \
            f"Temporal features dimension mismatch. Expected {expected_temporal_dim}, got {temporal_features.shape[1]}"
        assert market_features_flat.shape[1] == expected_market_dim, \
            f"Market features dimension mismatch. Expected {expected_market_dim}, got {market_features_flat.shape[1]}"
        
        combined_features = torch.cat([
            temporal_weight * temporal_features,
            market_weight * market_features_flat
        ], dim=-1)
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        # Final prediction
        prediction = self.fusion_layer(combined_features)
        logger.info(f"Final prediction shape: {prediction.shape}")
        
        attention_info = {
            'temporal_weight': temporal_weight.item(),
            'market_weight': market_weight.item(),
            'market_gates': [gate(market_features).mean().item() for gate in self.market_gates]  
        }
        
        return prediction, attention_info

    def compute_losses(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Compute various losses for the model predictions
        
        Args:
            predictions: Model predictions (batch_size, n_stocks)
            targets: Target values (batch_size, n_stocks)
            
        Returns:
            Dictionary containing different loss components
        """
        # Basic MSE loss for returns
        returns_loss = self.returns_loss(predictions, targets)
        
        # Direction prediction loss
        direction_pred = torch.sign(predictions)
        direction_true = torch.sign(targets)
        direction_loss = self.direction_loss(direction_pred.float(), direction_true.float())
        
        # Combine losses
        total_loss = returns_loss + 0.5 * direction_loss
        
        return {
            'returns_loss': returns_loss.item(),
            'direction_loss': direction_loss.item(),
            'total_loss': total_loss
        }