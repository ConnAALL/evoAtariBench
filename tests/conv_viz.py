"""
Simple script for the visualization of the conv compression with, sparsification, quantization, and dropout.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import methods.compressionMethods as cm
import methods.invCompressionMethods as inv
import methods.nonLinearMethods as nm


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
    parser = argparse.ArgumentParser(description=("Save output for convolution + sparsification, quantization, dropout, using the built-in nonlinearity methods"))
    parser.add_argument("--image", default=os.path.join(_REPO_ROOT, "test_image.jpg"), help="Path to input image.")
    parser.add_argument("--outdir", default=os.path.join(_REPO_ROOT, "out", "conv_viz"), help="Output directory.")
    parser.add_argument("--resize_h", type=int, default=210, help="Resize height (default: 210). Set 0/negative to disable.")
    parser.add_argument("--resize_w", type=int, default=160, help="Resize width (default: 160). Set 0/negative to disable.")
    parser.add_argument("--percentile", type=float, default=41, help="Sparsification percentile (on |coeff|).")
    parser.add_argument("--num_levels", type=int, default=5, help="Quantization num_levels (applied to real+imag).")
    parser.add_argument("--dropout_rate", type=float, default=0.086, help="Dropout rate.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.image)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if int(args.resize_h) > 0 and int(args.resize_w) > 0:
        img = img.resize((int(args.resize_w), int(args.resize_h)), resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    gray = to_grayscale(arr)

    block_c = cm.conv2d(gray)    

    sparsified = nm.sparsification(block_c, {"percentile": float(args.percentile)})
    quantized = nm.quantization(block_c, {"num_levels": int(args.num_levels)})
    dropped = nm.dropout_regularization(block_c, {"rate": float(args.dropout_rate)})

    p_name = int(round(float(args.percentile)))
    q_name = int(args.num_levels)
    d_name = int(round(float(args.dropout_rate) * 100.0))

    out_sparsification = os.path.join(args.outdir, f"conv_sparsification_{p_name}.png")
    out_quantization = os.path.join(args.outdir, f"conv_quantization_{q_name}.png")
    out_dropout = os.path.join(args.outdir, f"conv_dropout_{d_name}.png")

    save_img(out_sparsification, sparsified)
    save_img(out_quantization, quantized)
    save_img(out_dropout, dropped)

    print("Saved:")
    print(out_sparsification)
    print(out_quantization)
    print(out_dropout)


if __name__ == "__main__":
    main()

