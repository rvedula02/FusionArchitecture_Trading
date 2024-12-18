import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm
from ..models.fusion_architectures.parallel_fusion import ParallelFusion
from ..utils.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class ParallelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ParallelFusion(config).to(self.device)
        self.criterion = nn.MSELoss()
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
            
            # Extract market info from data
            market_info = data[:, -1, :] # Use last timestep's features as market info
            
            self.optimizer.zero_grad()
            prediction, attention_weights = self.model(data, market_info)
            loss = self.criterion(prediction, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
                logger.info(f'Attention weights: {attention_weights}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                market_info = data[:, -1, :]
                
                prediction, _ = self.model(data, market_info)
                loss = self.criterion(prediction, target)
                
                total_loss += loss.item()
                predictions.extend(prediction.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Calculate evaluation metrics
        metrics = EvaluationMetrics.calculate_metrics(
            torch.tensor(targets),
            torch.tensor(predictions)
        )
        
        return total_loss / len(val_loader), metrics
