import pytorch_lightning as pl
import torch.nn as nn
import ml_collections


class SignalTrainer(pl.LightningModule):

    def __init__(self, model: nn.Module, config: ml_collections.ConfigDict):
        super(SignalTrainer, self).__init__()
        self.model = model
        self.config = config
        
    def forward(self, signal):
        return self.model(signal)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location = None, hparams_file = None, strict = None, **kwargs):
        return super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()