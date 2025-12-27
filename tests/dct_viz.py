"""
Simple script for the visualization of the DCT compression with, sparsification, quantization, and dropout.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from scipy.fftpack import dct

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import methods.compressionMethods as cm
import methods.invCompressionMethods as inv
import methods.nonLinearMethods as nm


def dct2(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return dct(dct(x.T, norm=norm).T, norm=norm)


def to_grayscale(img_arr: np.ndarray) -> np.ndarray:
    x = np.asarray(img_arr)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if x.max() > 1.0:
        x = x / 255.0
    g = cm.convert_to_grayscale(x)
    return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)


def save_img(path: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description="Save 3 inverse-DCT reconstructions for DCT(k) + sparsification, quantization, dropout.")
    parser.add_argument("--image", default=os.path.join(_REPO_ROOT, "test_image.jpg"), help="Path to input image.")
    parser.add_argument("--outdir", default=os.path.join(_REPO_ROOT, "out", "dct_viz"), help="Output directory.")
    parser.add_argument("--resize_h", type=int, default=210, help="Resize height")
    parser.add_argument("--resize_w", type=int, default=160, help="Resize width")
    parser.add_argument("--k", type=int, default=142, help="DCT crop size k")
    parser.add_argument("--percentile", type=float, default=91.0, help="Sparsification percentile.")
    parser.add_argument("--num_levels", type=int, default=125, help="Quantization num_levels.")
    parser.add_argument("--dropout_rate", type=float, default=0.18, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dropout rng.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.image)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if args.resize_h and args.resize_w:
        img = img.resize((int(args.resize_w), int(args.resize_h)), resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    gray = to_grayscale(arr)
    H, W = gray.shape

    k = int(args.k)
    k = max(1, min(k, H, W))

    # Forward DCT and truncate
    F = dct2(gray, norm="ortho")
    block = F[:k, :k].copy()

    block_sparse = nm.sparsification(block, {"percentile": float(args.percentile)})
    block_quant = nm.quantization(block, {"num_levels": int(args.num_levels)})
    rng = np.random.default_rng(int(args.seed))
    block_drop = nm.dropout_regularization(block, {"rate": float(args.dropout_rate), "rng": rng})

    recon_sparse = np.clip(inv.idct_k(block_sparse, (H, W), norm="ortho"), 0.0, 1.0)
    recon_quant = np.clip(inv.idct_k(block_quant, (H, W), norm="ortho"), 0.0, 1.0)
    recon_drop = np.clip(inv.idct_k(block_drop, (H, W), norm="ortho"), 0.0, 1.0)

    p_name = int(round(float(args.percentile)))
    q_name = int(args.num_levels)
    d_name = int(round(float(args.dropout_rate) * 100.0))

    out_sparse = os.path.join(args.outdir, f"dct_{k}_sparsification_{p_name}.png")
    out_quant = os.path.join(args.outdir, f"dct_{k}_quantization_{q_name}.png")
    out_drop = os.path.join(args.outdir, f"dct_{k}_dropout_{d_name}.png")

    save_img(out_sparse, recon_sparse)
    save_img(out_quant, recon_quant)
    save_img(out_drop, recon_drop)

    print("Saved:")
    print(out_sparse)
    print(out_quant)
    print(out_drop)


if __name__ == "__main__":
    main()


