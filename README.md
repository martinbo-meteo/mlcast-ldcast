# MLCast implementation of LDCast

see main branch ([https://github.com/mlcast-community/mlcast]) for details.

# Main LDCast class

The main class is LDCast and takes an autoencoder and a latent_nowcaster modules. Only the predict method is implemented, to show the encode-latent_nowcasting-decode pattern.

```
from src.mlcast.models.ldcast.ldcast import LDCast
ldcast = LDCast(autoencoder, latent_nowcaster)
```

# Autoencoder

```
from src.mlcast.models.ldcast.autoenc.autoenc import AutoencoderKLNet, autoenc_loss
from src.mlcast.models.base import NowcastingLightningModule
autoencoder = NowcastingLightningModule(AutoencoderKLNet(), autoenc_loss()).to('cuda')
```
The autoencoder is an instance of the NowcastingLightningModule. Training the autoencoder:
```
# create fake data
x = torch.randn(2, 1, 4, 256, 256, device = 'cuda', requires_grad = False)
y = autoencoder(x, 4)[0]
y = y.detach()
batch = (x, y)

import pytorch_lightning as L
trainer = L.Trainer()
trainer.fit(autoencoder, batch)
```

# Latent nowcaster (= conditioner + denoiser + samplers)
The latent nowcaster manages the conditioner, the denoiser and the samplers. There can be two different samplers for training and for inference.
```
# setup forecaster
conditioner = AFNONowcastNetCascade(
    32,
    train_autoenc=False,
    output_patches=future_timesteps//autoenc_time_ratio,
    cascade_depth=3,
    embed_dim=128,
    analysis_depth=4
).to('cuda')

# setup denoiser
from src.mlcast.models.ldcast.diffusion.unet import UNetModel
denoiser = UNetModel(in_channels=autoencoder.net.hidden_width,
    model_channels=256, out_channels=autoencoder.net.hidden_width,
    num_res_blocks=2, attention_resolutions=(1,2), 
    dims=3, channel_mult=(1, 2, 4), num_heads=8,
    num_timesteps=future_timesteps//autoenc_time_ratio,
    context_ch=[128, 256, 512] # context channels (= analysis_net.cascade_dims)
                    ).to('cuda')

# define the training and inference samplers
training_sampler = SimpleSampler()
inference_sampler = PLMSSampler(denoiser, 1000)

# define the latent_nowcaster
from torch.nn import L1Loss
from src.mlcast.models.ldcast.diffusion.diffusion import LatentNowcaster
latent_nowcaster = LatentNowcaster(conditioner, denoiser, L1Loss(), training_sampler, inference_sampler)
```
Create fake data for inference and training:
```
inputs = torch.randn(2, 1, 4, 256, 256, device = 'cuda')
target = torch.randn(2, 1, 20, 256, 256, device = 'cuda')
loss = nn.L1Loss()
autoencoder.eval()
latent_inputs = autoencoder.net.encode(inputs)[0].detach()
latent_target = autoencoder.net.encode(target)[0].detach()
```
Inference with the latent_nowcaster (PLMSSampler is used during inference)
```
latent_nowcaster.infer(latent_inputs)
```
Training the latent_nowcaster (SimpleSampler is used during training)
```
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(latent_inputs, latent_target)
dataloader = DataLoader(dataset, batch_size=2)

latent_batch = (latent_inputs, latent_target)
import pytorch_lightning as L
trainer = L.Trainer()
trainer.fit(latent_nowcaster, dataloader)
```

# Notes

I did not manage to make LatentNowcaster a sublcass of NowcastingLightningModule because I would basically have to overwrite everything... LatentNowcaster needs two nets (denoiser and conditioner) and the training logic is not as straightforward as it is for the moment in NowcastingLightningModule. One should also take into account the fact that two different samplers are used for training and inference, so that the forward method can not just be self.net(x)

It would be nice to have cleaner and consistent APIs for the samplers. For the moment, the PLMSSampler and the SimpleSampler are not totally consistent in their APIs, because the SimpleSampler (better/more common name for this one?) was only used during training, while the PLMSSampler was used during inference. The handling of the schedule of each sampler with respect to the schedule saved in the denoiser could also be clearer.

During training, an EMA scope was used for the weights of the denoiser, I removed this for the moment, but it should reincluded in some way.

The 'timesteps' variable sometimes refers to the timesteps of the diffusion process (= 1000 during training) and sometimes refers to the nowcasting timesteps (where each time step = 5 minutes). Better to have different names.

In /src/mlcast/models/ldcast/diffusion/diffusion.py, one has to choose which sampler to use for testing