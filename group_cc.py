from typing import Any
import torch
from torch import nn


#   NOTE:  copied from
#          https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/optim.py#L103
class L1Scheduler:
    def __init__(
        self,
        l1_warm_up_steps: float,
        total_steps: int,
        final_l1_coefficient: float,
    ):

        self.l1_warmup_steps = l1_warm_up_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_l1_coefficient = 0.0
        else:
            self.current_l1_coefficient = final_l1_coefficient

        self.final_l1_coefficient = final_l1_coefficient

        self.current_step = 0
        self.total_steps = total_steps
        # assert isinstance(self.final_l1_coefficient, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_l1_coefficient}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_l1_coefficient = self.final_l1_coefficient * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_l1_coefficient = self.final_l1_coefficient  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_l1_coefficient": self.current_l1_coefficient,
            "final_l1_coefficient": self.final_l1_coefficient,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])


class GCC(nn.Module):
    def __init__(self, input_dims, num_blocks, expansion=32, dtype=torch.float64, device="cuda", jumprelu=False, squeeze=10):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = input_dims * num_blocks
        self.gcc_dims = input_dims * expansion
        self.dtype = dtype
        self.device = device
        self.squeeze = squeeze
        self.jumprelu = jumprelu

        self.init_weights()

    def init_weights(self):
        self.b_enc = nn.Parameter(
            torch.zeros(self.gcc_dims, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.gcc_dims, self.output_dims, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.input_dims, self.gcc_dims, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.output_dims, dtype=self.dtype, device=self.device)
        )

        if self.jumprelu:
            self.threshold = nn.Parameter(
                torch.zeros(self.gcc_dims, dtype=self.dtype, device=self.device)
            )

    def encode(self, x):
        x = nn.ReLU()(x @ self.W_enc + self.b_enc)

        if self.jumprelu:
            x = x * self.squeeze_sigmoid(x - self.threshold)

        return nn.ReLU()(x)

    def squeeze_sigmoid(self, x):
        # return 1 / (1+torch.exp(-self.squeeze*x))
        return (1.01 / (1+torch.exp(-self.squeeze*x))) - 0.01

    def decode(self, x):
        return x @ self.W_dec + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        out = self.decode(features)

        return features, out


class ThresholdClipper(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'threshold'):
            module.threshold.data = torch.clamp(module.threshold.data, min=0, max=None)


#   Wrapper so that we have control over the forward pass.
class ModelWrapper(nn.Module):
    target_neuron = None
    all_acts = []

    def __init__(self, resnet, expansion, device, use_gcc=False):
        super().__init__()
        self.device = device

        self.model = nn.Sequential()
        self.model.append(resnet.conv1)
        self.model.append(resnet.bn1)
        self.model.append(resnet.relu)
        self.model.append(resnet.maxpool)
        self.model.append(resnet.layer1)
        self.model.append(resnet.layer2)
        self.model.append(resnet.layer3)
        self.model.append(resnet.layer4)
        # self.model.append(resnet.layer4[0])
        # self.model.append(resnet.layer4[1].conv1)
        # self.model.append(resnet.layer4[1].bn1)
        # self.model.append(resnet.layer4[1].relu)
        # self.model.append(resnet.layer4[1].conv2)
        # self.model.append(resnet.layer4[1].bn2)

        self.use_gcc = use_gcc
        if self.use_gcc:
            self.map = nn.Linear(512, 512*expansion, bias=False)

    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)       

        if self.use_gcc:
            center_coord = x.shape[-1] // 2
            x = x[:, :, center_coord, center_coord]

            # x = torch.mean(torch.flatten(x, start_dim=-2), dim=-1)

            x = self.map(x)
            x = x[:, :, None, None]

        return x


if __name__ == "__main__":
    gcc_states = torch.load('/media/andrelongon/DATA/gcc_ckpts/scale1_1.2_1.4_1.6_1.8/vanilla_32exp_gcc_weights_10ep.pth')['W_dec']
    w_dec = torch.reshape(gcc_states, (gcc_states.shape[0], 5, 256))

    #   How to best measure similarity?  I'd like for magnitudes to also be the same which means avoiding normalizing.
    #   Maybe could take euclid dist for now?  Also could measure variance in mags across the blocks and penalize for
    #   larger variance (1/var ?).
    all_sims = []
    for latent in w_dec:
        latent = torch.nn.functional.normalize(latent)

        # all_sims.append(torch.dot(latent[0], latent[-1]))
        sims = []
        for i in range(latent.shape[0]):
            for j in range(i, latent.shape[0]):
                sims.append(torch.dot(latent[i], latent[j]))

        all_sims.append(torch.tensor(sims).mean())

    print(torch.topk(torch.tensor(all_sims), 32, largest=True))