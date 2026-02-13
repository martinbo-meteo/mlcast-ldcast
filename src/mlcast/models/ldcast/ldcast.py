# new file with respect to original code

import abc
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
import xarray as xr
from torch import nn

from ..base import NowcastingModelBase, NowcastingLightningModule

class LDCast(NowcastingModelBase):
    
    def __init__(self, autoencoder, latent_nowcaster):
        #super().__init__()
        self.autoencoder = autoencoder
        self.latent_nowcaster = latent_nowcaster
    
    def fit(self, da_rr: xr.DataArray, **kwargs: Any) -> None:
        pass
        
    
    def predict(self, inputs):
        '''inputs is of shape (batch_size, 1, 4) + spatial_shape'''
        latent_inputs = self.autoencoder.net.encode(inputs)
        latent_pred = self.latent_nowcaster(latent_inputs)
        return self.autoencoder.net.decode(latent_pred)
    
    def _train_autoencoder(self, da_rr: xr.DataArray, epochs: int, batch_size: int, **kwargs: Any) -> None:
        pass
    
    def _train_latent_nowcaster(self, da_rr: xr.DataArray, num_timesteps: int, epochs: int, batch_size: int, **kwargs: Any) -> None:
        pass
    