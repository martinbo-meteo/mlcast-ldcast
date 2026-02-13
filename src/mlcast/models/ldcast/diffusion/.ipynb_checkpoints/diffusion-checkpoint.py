import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Any
import contextlib

print('take care of ema scope, which was used as context manager each exactly when denoiser.forward was called, so it should be a taken care of in the code code about the denoiser or about the diffuser (nothing to do with samplers)')  

import pytorch_lightning as L
class LatentNowcaster(L.LightningModule):
    """Base class for PyTorch Lightning modules used in nowcasting models.

    This class provides a standard interface for training and validation
    steps, as well as optimizer configuration.
    """

    def __init__(
        self,
        conditioner: nn.Module,
        denoiser: nn.Module,
        loss: nn.Module,
        training_sampler: nn.Module,
        inference_sampler: nn.Module,
        optimizer_class: Any | None = None,
        optimizer_kwargs: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser", "conditioner", "training_sampler", "inference_sampler", "loss"])
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.loss = loss
        self.training_sampler = training_sampler
        self.inference_sampler = inference_sampler
        self.optimizer_class = torch.optim.Adam if optimizer_class is None else optimizer_class

        training_sampler.register_schedule(denoiser)

    def infer(self, latent_inputs, num_diffusion_iters = 50, verbose = True):

        condition = self.conditioner(latent_inputs)
        
        gen_shape = (32, 5, 256//4, 256//4)
        batch_size = len(list(condition.values())[0])
        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.inference_sampler.sample(
                num_diffusion_iters, 
                batch_size,
                gen_shape,
                condition,
                progbar=verbose
            )
        return s

    def model_step(self, latent_batch: Any, batch_idx: int, step_name: str = "train") -> torch.Tensor:
        """Generic model step for training or validation.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        latent_inputs, latent_targets = latent_batch
        
        condition = self.conditioner(latent_inputs)
        t, noise, latent_target_noisy = self.training_sampler.q_sample(self.denoiser, latent_targets)
        guessed_noise = self.denoiser(latent_target_noisy, t, context = condition)
        loss = self.loss(guessed_noise, noise)
        
        if isinstance(loss, dict):
            # append step name to loss keys for logging
            loss = {f"{step_name}/{k}": v for k, v in loss.items()}
            self.log_dict(loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss.get(f"{step_name}/total_loss", None)
            if loss is None:
                raise ValueError(f"Loss is None for step {step_name}. Ensure loss function returns a valid tensor.")
        else:
            self.log(f"{step_name}/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="val")
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            Optimizer instance to use for training
        """
        return self.optimizer_class(self.parameters(), **(self.hparams.optimizer_kwargs or {}))
    

    def on_train_start(self):
        self._current_sampler = self.training_sampler
        super().on_train_start()
    
    def on_validation_start(self):
        self._current_sampler_mode = self.training_sampler
        super().on_validation_start()
    
    def on_predict_start(self):
        self._current_sampler_mode = self.inference_sampler
        super().on_predict_start()
    
    def on_test_start(self):
        # training or inference sampler ???
        self._current_sampler_mode = self.training_sampler
        super().on_test_start()