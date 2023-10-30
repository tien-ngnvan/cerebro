from typing import Any
import torch.nn.functional as F

class BaseLoss:
    def __init__(self, name:str = 'loss', weight:float = 1.0) -> None:
        self.name = name
        self.weight = weight
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    

class CustomCE(BaseLoss):
    def __init__(self, name: str = 'loss', weight: float = 1.0) -> None:
        super().__init__(name, weight)
    
    def __call__(self, predicts, labels, *args: Any, **kwds: Any) -> Any:
        # Compute MLM loss.
        return F.cross_entropy(
            predicts.view(-1, predicts.shape[-1]), labels.view(-1)
        )