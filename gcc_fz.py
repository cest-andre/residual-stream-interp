from pathlib import Path
import os
import argparse
from thingsvision import get_extractor
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param

from group_cc import ModelWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str)
parser.add_argument('--gcc', type=str)
parser.add_argument('--module', type=str, default='0')
parser.add_argument('--neuron', type=int)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=16)
parser.add_argument('--direction', action='store_true', default=False)
args = parser.parse_args()

device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
transform.device = torch.device(device_str)
param.spatial.device = torch.device(device_str)
param.color.device = torch.device(device_str)

model = models.resnet18(True).to(device_str)
gcc_states = torch.load(f'/media/andrelongon/DATA/sae_ckpts/{args.gcc}/vanilla_64exp_8e-5L1_sae_weights_25ep.pth')
layer_dirs = torch.transpose(gcc_states['W_enc'], 0, 1)
# layer_dirs = gcc_states['W_dec'][:, :512]
if args.direction:
    print(f"DIRECTION FOR LATENT {args.neuron}")
    subdir = f'direction/{args.module}'
    obj = objectives.neuron_weight(args.module.replace('.', '_'), layer_dirs[args.neuron])
else:
    print(f"LATENT {args.neuron}")
    subdir = 'latent'
    model = ModelWrapper(model, 64, device_str, use_gcc=True)
    states = model.state_dict()
    states['map.weight'] = layer_dirs
    model.load_state_dict(states)
    model = nn.Sequential(model)
    obj = objectives.neuron(args.module, args.neuron)

savedir = os.path.join(args.basedir, args.gcc, subdir, f"unit{args.neuron}")
Path(savedir).mkdir(parents=True, exist_ok=True)
model.to(device_str).eval()

augs = None
if args.jitter < 4:
    augs = [
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
    ]
else:
    augs = [
        transform.pad(args.jitter),
        transform.jitter(args.jitter),
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        transform.jitter(int(args.jitter/2)),
        transforms.CenterCrop(224),
    ]
param_f = lambda: param.images.image(224, decorrelate=True)

imgs = render.render_vis(model, obj, param_f=param_f, transforms=augs, thresholds=(2560,), show_image=False)

img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
img.save(os.path.join(savedir, f"0_64exp_distill_center.png"))