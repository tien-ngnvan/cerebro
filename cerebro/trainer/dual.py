import logging
from typing import Any, Dict

import torch
from torch import nn
from .base import BaseTrainer
from transformers.utils import is_apex_available

    
if is_apex_available():
    from apex import amp
    

logger = logging.getLogger(__name__)



class DualTrainer(BaseTrainer):
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        pass
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor | Any]) -> torch.Tensor:
        """
        Perform training step 

        Args:
            model (Module): _description_
            inputs (Dict[str, Tensor  |  Any]): _description_

        Returns:
            Tensor: _description_
        """
        
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Custom loss function 
        with self.compute_loss_context_manager():            
            opt_losses = [
                self.compute_loss(model, inputs[idx], self.losses[idx]) for idx in range(len(self.losses))
            ]    
        loss = [l.weight * v for l, v in zip(self.losses, opt_losses)]
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    
        