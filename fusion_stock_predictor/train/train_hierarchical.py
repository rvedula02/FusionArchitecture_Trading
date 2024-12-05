import torch
import logging
from pathlib import Path
from tqdm import tqdm
from ..models.fusion_architectures.hierarchical_fusion import HierarchicalFusion
from ..utils.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class HierarchicalTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HierarchicalFusion(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            prediction, _ = self.model(data)
            losses = self.model.compute_losses(prediction, target)
            loss = losses['total_loss']
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
