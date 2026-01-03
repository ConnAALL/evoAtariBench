"""
Simple script for the visualization of the DFT compression with sparsification, quantization, and dropout.
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

from methods.compressionMethods import convert_to_grayscale
from dct_viz import save_img


def main():
    parser = argparse.ArgumentParser(description=("Save inverse-DFT reconstructions for DFT(k) + sparsification, quantization, dropout, using the built-in nonlinearity methods. Sparsification is applied in the complex domain (magnitude thresholding, phase preserved); quantization/dropout are also applied in the complex domain."))
    parser.add_argument("--image", default=os.path.join(_REPO_ROOT, "test_image.jpg"), help="Path to input image.")
    parser.add_argument("--outdir", default=os.path.join(_REPO_ROOT, "out", "dft_viz"), help="Output directory.")
    parser.add_argument("--resize_h", type=int, default=210, help="Resize height (default: 210). Set 0/negative to disable.")
    parser.add_argument("--resize_w", type=int, default=160, help="Resize width (default: 160). Set 0/negative to disable.")
    parser.add_argument("--k", type=int, default=142, help="DFT crop size k (center crop)")
    parser.add_argument("--percentile", type=float, default=86, help="Sparsification percentile (on |coeff|).")
    parser.add_argument("--num_levels", type=int, default=136, help="Quantization num_levels (applied to real+imag).")
    parser.add_argument("--dropout_rate", type=float, default=0.20, help="Dropout rate.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.image)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if int(args.resize_h) > 0 and int(args.resize_w) > 0:
        img = img.resize((int(args.resize_w), int(args.resize_h)), resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    gray = convert_to_grayscale(arr)
    H, W = gray.shape

    k = int(args.k)
    k = max(1, min(k, H, W))

    dft_compressed = cm.dft_k(gray, {"k": k, "norm": "ortho"})

    sparsified = nm.sparsification_complex(dft_compressed, {"percentile": float(args.percentile)})
    quantized = nm.quantization_complex(dft_compressed, {"num_levels": int(args.num_levels)})
    dropped = nm.dropout_regularization_complex(dft_compressed, {"rate": float(args.dropout_rate)})

    reconstruct_sparsification = np.clip(inv.idft_k(sparsified, (H, W), output="real", norm="ortho"), 0.0, 1.0)
    reconstructed_quantization = np.clip(inv.idft_k(quantized.astype(np.complex64, copy=False), (H, W), output="real", norm="ortho"), 0.0, 1.0)
    reconstructed_dropout = np.clip(inv.idft_k(dropped.astype(np.complex64, copy=False), (H, W), output="real", norm="ortho"), 0.0, 1.0)

    p_name = int(round(float(args.percentile)))
    q_name = int(args.num_levels)
    d_name = int(round(float(args.dropout_rate) * 100.0))

    out_sparsification = os.path.join(args.outdir, f"dft_{k}_sparsification_{p_name}.png")
    out_quantization = os.path.join(args.outdir, f"dft_{k}_quantization_{q_name}.png")
    out_dropout = os.path.join(args.outdir, f"dft_{k}_dropout_{d_name}.png")

    save_img(out_sparsification, reconstruct_sparsification)
    save_img(out_quantization, reconstructed_quantization)
    save_img(out_dropout, reconstructed_dropout)

    print("Saved:")
    print(out_sparsification)
    print(out_quantization)
    print(out_dropout)


if __name__ == "__main__":
    main()


