#!/usr/bin/env python3
"""
train_autoencoder_210x160.py

Minimal, readable PyTorch autoencoder training script for Atari frames at 210x160.

Assumptions:
- You have already collected frames and saved them as a single .npy file:
    frames.npy  with shape (N, 210, 160)  OR (N, 210, 160, 1) OR (N, 1, 210, 160)
- Frames are grayscale-ish. If frames are uint8 [0,255], we normalize to [0,1].
- Output: saves encoder.pt, decoder.pt, and periodic reconstruction grids.

Example:
  python3 train_autoencoder_210x160.py \
      --data_path data/spaceinv_frames.npy \
      --outdir out/ae_spaceinv_210x160 \
      --latent_channels 38 \
      --epochs 20 \
      --batch_size 64
"""

import os
import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from PIL import Image
except ImportError:
    Image = None


# -----------------------------
# Utils
# -----------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_chw(frames: np.ndarray) -> np.ndarray:
    """
    Convert frames to (N, 1, 210, 160) float32 in [0,1].
    Accepts (N,H,W), (N,H,W,1), (N,1,H,W).
    """
    x = np.asarray(frames)

    if x.ndim == 3:  # (N,H,W)
        x = x[:, None, :, :]
    elif x.ndim == 4:
        if x.shape[1] == 1:  # (N,1,H,W)
            pass
        elif x.shape[-1] == 1:  # (N,H,W,1)
            x = np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"Expected grayscale frames; got shape {x.shape}")
    else:
        raise ValueError(f"Unexpected frames shape: {x.shape}")

    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    # Normalize if looks like uint8 scale
    if x.max() > 1.5:
        x = x / 255.0

    x = np.clip(x, 0.0, 1.0)
    if x.shape[2:] != (210, 160):
        raise ValueError(f"Expected (210,160); got {x.shape[2:]}")
    return x


def save_recon_grid(path: str, x: torch.Tensor, xhat: torch.Tensor, n: int = 16):
    """
    Save a simple 2-row grid: originals on top, reconstructions on bottom.
    x, xhat: (B,1,210,160) in [0,1]
    """
    if Image is None:
        return  # PIL not installed; silently skip

    x = x.detach().cpu().numpy()
    xhat = xhat.detach().cpu().numpy()

    n = min(n, x.shape[0])
    H, W = x.shape[2], x.shape[3]

    grid = np.zeros((2 * H, n * W), dtype=np.uint8)

    for i in range(n):
        a = (np.clip(x[i, 0], 0, 1) * 255.0).round().astype(np.uint8)
        b = (np.clip(xhat[i, 0], 0, 1) * 255.0).round().astype(np.uint8)
        grid[0:H, i * W:(i + 1) * W] = a
        grid[H:2 * H, i * W:(i + 1) * W] = b

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid, mode="L").save(path)


# -----------------------------
# Dataset
# -----------------------------
class NpyFramesDataset(Dataset):
    def __init__(self, npy_path: str):
        self.npy_path = npy_path
        frames = np.load(npy_path, mmap_mode="r")
        self.frames = ensure_chw(frames)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # return (1,210,160)
        return torch.from_numpy(self.frames[idx])


# -----------------------------
# Model: spatial-latent AE
# Latent elements ~ C * 27 * 20 if we do 3 downsamples by 2
# (210 -> 105 -> 53 -> 27) and (160 -> 80 -> 40 -> 20)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_channels: int = 38):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # -> (32,105,80)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> (64,53,40)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=5, stride=2, padding=2),  # -> (C,27,20)
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,C,27,20)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 38):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),  # keep output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xhat = self.net(z)

        # Due to stride/padding/output_padding, spatial size can be off by a bit.
        # Crop/pad to exactly (210,160).
        H, W = 210, 160
        xhat = xhat[..., :H, :W]
        if xhat.shape[-2] < H or xhat.shape[-1] < W:
            pad_h = H - xhat.shape[-2]
            pad_w = W - xhat.shape[-1]
            xhat = nn.functional.pad(xhat, (0, pad_w, 0, pad_h))
        return xhat


class AutoEncoder(nn.Module):
    def __init__(self, latent_channels: int = 38):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Path to frames .npy (N,210,160) or similar.")
    ap.add_argument("--outdir", required=True, help="Output dir for checkpoints and recon grids.")
    ap.add_argument("--latent_channels", type=int, default=38, help="Latent channels C (latent size ~ C*27*20).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1, help="Save recon grid every N epochs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = pick_device()
    print(f"[device] {device}")

    ds = NpyFramesDataset(args.data_path)
    n = len(ds)
    n_val = max(1, int(n * float(args.val_ratio)))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = AutoEncoder(latent_channels=int(args.latent_channels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    # fixed batch for visualization
    fixed_batch = next(iter(val_loader)).to(device)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            xhat, _ = model(x)
            loss = loss_fn(xhat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += float(loss.item()) * x.size(0)
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                xhat, _ = model(x)
                loss = loss_fn(xhat, x)
                val_loss += float(loss.item()) * x.size(0)
        val_loss /= max(1, n_val)

        print(f"[epoch {epoch:03d}] train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

        if (epoch % int(args.save_every)) == 0:
            with torch.no_grad():
                xhat, z = model(fixed_batch)
            grid_path = os.path.join(args.outdir, f"recon_epoch_{epoch:03d}.png")
            save_recon_grid(grid_path, fixed_batch, xhat, n=16)

        # Save checkpoints each epoch (small models; easy)
        torch.save(model.encoder.state_dict(), os.path.join(args.outdir, "encoder.pt"))
        torch.save(model.decoder.state_dict(), os.path.join(args.outdir, "decoder.pt"))

    print("Done.")
    print("Saved:")
    print(os.path.join(args.outdir, "encoder.pt"))
    print(os.path.join(args.outdir, "decoder.pt"))


if __name__ == "__main__":
    main()
