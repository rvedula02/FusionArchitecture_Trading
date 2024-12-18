import torch
from typing import Dict
from ..models.fusion_architectures.parallel_fusion import ParallelFusion

class ComponentAnalysis:
    def __init__(self, model: ParallelFusion):
        self.model = model
    
    def analyze_attention_patterns(self, x: torch.Tensor) -> Dict:
        """Analyze intra-stock attention patterns"""
        with torch.no_grad():
            temporal_features = self.model.intra_stock_attention(x)
            patterns = {
                'temporal_mean': temporal_features.mean(dim=1),
                'temporal_std': temporal_features.std(dim=1)
            }
        return patterns
    
    def analyze_market_impact(self, x: torch.Tensor, market_info: torch.Tensor) -> Dict:
        """Analyze market gating impact"""
        with torch.no_grad():
            gates = self.model.market_gating.gate(
                self.model.market_gating.market_projection(market_info)
            )
            impact = {
                'gate_mean': gates.mean().item(),
                'gate_std': gates.std().item(),
                'high_impact_features': (gates > 0.8).sum().item()
            }
        return impact
