import os
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from group_cc import GCC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acts_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--jumprelu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--expansion', type=int, default=32)
    args = parser.parse_args()

    acts_ds = [np.load(os.path.join(args.acts_dir, path)) for path in os.listdir(args.acts_dir)]
    input_dims = acts_ds[0].shape[-1]
    num_blocks = 1
    # acts_ds = np.concatenate(acts_ds, axis=1)
    # acts_ds = np.concatenate((acts_ds[0], acts_ds[2], acts_ds[4]), axis=1)
    acts_ds = acts_ds[0]  #  SAE
    acts_ds = np.clip(acts_ds, a_min=0, a_max=None)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    acts_ds = torch.tensor(acts_ds)
    acts_ds = TensorDataset(acts_ds)
    dataloader = DataLoader(acts_ds, batch_size=args.batch_size, shuffle=True)

    gcc = GCC(input_dims, num_blocks, args.expansion, jumprelu=args.jumprelu).to(device)
    if args.jumprelu:
        clipper = ThresholdClipper()
        gcc.apply(clipper)

    total_training_steps = args.epochs * len(dataloader)
    lr_warm_up_steps = total_training_steps // 5
    lr_decay_steps = total_training_steps // 5
    l1_warm_up_steps = total_training_steps // 20

    lr=0.0004
    adam_beta1=0.9 
    adam_beta2=0.999
    l1_coefficient = 8e-5
    lp_norm = 1

    optimizer = optim.Adam(
        gcc.parameters(),
        lr=lr,
        betas=(
            adam_beta1,
            adam_beta2,
        )
    )

    # schedulers = [optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)]
    # schedulers.append(optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1,
    #     end_factor=0,
    #     total_iters=lr_decay_steps,
    # ))
    # milestones = [total_training_steps - lr_decay_steps]
    # lr_scheduler = optim.lr_scheduler.SequentialLR(schedulers=schedulers, optimizer=optimizer, milestones=milestones)

    lr_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=(1/3),
        end_factor=1,
        total_iters=lr_warm_up_steps,
    )

    # l1_scheduler = L1Scheduler(
    #     l1_warm_up_steps=l1_warm_up_steps,
    #     total_steps=total_training_steps,
    #     final_l1_coefficient=l1_coefficient
    # )

    all_mses = []
    label = f'jumprelu_{args.expansion}exp' if args.jumprelu else f'vanilla_{args.expansion}exp'
    print(label)
    for ep in range(args.epochs):
        print(f"Epoch {ep}")
        total_mse = 0
        for i, acts in enumerate(dataloader, 0):
            optimizer.zero_grad()

            acts = acts[0].to(device)
            # acts = torch.clamp(acts, min=0)

            features, acts_hat = gcc(acts[:, :input_dims])

            mse_loss = torch.nn.functional.mse_loss(acts_hat, acts, reduction="none").sum(-1)
            total_mse += mse_loss.mean().cpu().detach().numpy()

            weighted_feature_acts = features * gcc.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(p=lp_norm, dim=-1)  # sum over the feature dimension
            # l1_loss = (l1_scheduler.current_l1_coefficient * sparsity).mean()
            l1_loss = l1_coefficient * sparsity

            loss = (mse_loss + l1_loss).mean()
            loss.backward()

            optimizer.step()
            lr_warmup.step()
            # l1_scheduler.step()

        print(f"Average mse:  {total_mse / i}")
        # if (ep+1) % 10 == 0:

        all_mses.append(total_mse / i)
        torch.save(gcc.state_dict(), f"{args.ckpt_dir}/{label}_8e-5L1_sae_weights_{ep+1}ep.pth")

        #   TODO:  validate by running on imnet val.  Record just the total reconstruction loss (omit sparsity loss).  Compare jumprelu vs vanilla.
        
    np.save(f"{args.ckpt_dir}/{label}_8e-5L1_sae_maes_{ep+1}.npy", np.array(all_mses))