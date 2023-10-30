from typing import Any


class BaseLoss:
    def __init__(self, name:str = 'loss', weight:float = 1.0) -> None:
        self.name = name
        self.weight = weight
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass