import logging
from .base import BaseTrainer


logger = logging.getLogger(__name__)



class DualTrainer(BaseTrainer):
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        pass
        
        