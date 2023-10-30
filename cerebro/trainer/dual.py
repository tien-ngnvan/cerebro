import logging
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
from .base import BaseTrainer
from transformers.utils import is_apex_available

    
if is_apex_available():
    from apex import amp
    

logger = logging.getLogger(__name__)



class DualTrainer(BaseTrainer):    
    def training_step(self, model: nn.Module, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]]) -> torch.Tensor:
        """
        Perform training step 

        Args:
            model (Module): _description_
            inputs (Dict[str, Tensor  |  Any]): _description_

        Returns:
            Tensor: _description_
        """
        
        model.train()
        inputs = self._prepare_input(inputs) # inputs = (inputs[1], inputs[2])

        # Custom loss function 
        assert len(inputs) == len(self.losses)
        with self.compute_loss_context_manager():  
            if self.losses is not None:          
                opt_losses = [
                    self.compute_loss(model, inputs[idx], self.losses[idx]) for idx in range(len(self.losses))
                ]    
                loss = sum([l.weight * v for l, v in zip(self.losses, opt_losses)])
            else:
                loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def prediction_step(
        self, model, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]], *args, **kwargs) -> torch.Tensor:
        inputs = self._prepare_input(inputs)
        if len(inputs) > 0:
            inputs = inputs[0]

        loss = model(**inputs).loss.mean().detach().to(self.args.device) 
        
        return (loss, None, None)
    
        