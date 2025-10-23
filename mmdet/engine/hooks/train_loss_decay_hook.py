import numpy as np
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class TrainLossDecayStoppingHook(Hook):
    """Early stop when train loss decay rate slows down."""
    
    def __init__(self, 
                 window_size=20,
                 decay_threshold=0.001,
                 patience=5):
        self.window_size = window_size
        self.decay_threshold = decay_threshold  
        self.patience = patience
        self.loss_history = []
        self.patience_counter = 0
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss_value = outputs['loss'].item()
        self.loss_history.append(loss_value)
        
        if len(self.loss_history) < self.window_size:
            return
            
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # Calculate decay rate over window
        recent_losses = self.loss_history[-self.window_size:]
        if len(recent_losses) >= self.window_size:
            decay_rate = self._calculate_decay_rate(recent_losses)
            
            if decay_rate < self.decay_threshold:
                self.patience_counter += 1
                runner.logger.info(f'Slow decay detected: {decay_rate:.6f}, patience: {self.patience_counter}/{self.patience}')
                
                if self.patience_counter >= self.patience:
                    checkpoint_name = f'early_stop_epoch_{runner.epoch}_iter_{runner.iter}.pth'
                    runner.logger.info(f'Saving early stopping checkpoint: {checkpoint_name}')
                    
                    runner.save_checkpoint(
                        out_dir=runner.work_dir,
                        filename=checkpoint_name,
                        meta=dict(
                            epoch=runner.epoch,
                            iter=runner.iter,
                            early_stop=True,
                            final_loss=self.loss_history[-1]
                        ),
                        save_optimizer=True,
                        save_param_scheduler=True
                    )
                    
                    runner.logger.info('Train loss decay rate too slow, stopping training!')
                    raise RuntimeError('Early stopping: train loss decay rate below threshold')
            else:
                self.patience_counter = 0
                
    def _calculate_decay_rate(self, losses):
        # Fit linear trend to log losses to detect decay rate
    
        x = np.arange(len(losses))
        slope = np.polyfit(x, losses, 1)[0]
        return -slope  # Negative slope = positive decay rate