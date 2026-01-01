"""
Simple script for the visualization of the DWT compression with sparsification, quantization, and dropout.
Mirrors `tests/dft_viz.py`, but uses a packed 2D DWT coefficient array and inverse via `inv.idwt(...)`.

Assumptions (as you requested):
- You already have a working forward DWT in `methods.compressionMethods`:
    coeff_arr, meta = cm.dwt(gray, {"wavelet":..., "levels":..., "mode":...})
  (i.e., returns a packed coefficient array + metadata needed for inverse)

- You already have a working inverse DWT in `methods.invCompressionMethods`:
    recon = inv.idwt(coeff_arr, meta)

- You already have working non-linearities for real arrays:
    nm.sparsification(...)
    nm.quantization(...)
    nm.dropout_regularization(...)
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
    parser = argparse.ArgumentParser(
        description=(
            "Save inverse-DWT reconstructions for DWT + sparsification, quantization, dropout.\n"
            "All nonlinearities are applied directly to the packed DWT coefficient array (real-valued)."
        )
    )
    parser.add_argument("--image", default=os.path.join(_REPO_ROOT, "test_image.jpg"), help="Path to input image.")
    parser.add_argument("--outdir", default=os.path.join(_REPO_ROOT, "out", "dwt_viz"), help="Output directory.")
    parser.add_argument("--resize_h", type=int, default=210, help="Resize height (default: 210). Set 0/negative to disable.")
    parser.add_argument("--resize_w", type=int, default=160, help="Resize width (default: 160). Set 0/negative to disable.")

    # DWT params
    parser.add_argument("--wavelet", type=str, default="bior4.4", help="PyWavelets wavelet name (default: bior4.4).")
    parser.add_argument("--levels", type=int, default=10, help="DWT decomposition levels (default: 4).")
    parser.add_argument("--mode", type=str, default="symmetric", help="Signal extension mode (default: symmetric).")

    # Nonlinearities
    parser.add_argument("--percentile", type=float, default=41, help="Sparsification percentile (on |coeff|).")
    parser.add_argument("--num_levels", type=int, default=5, help="Quantization num_levels.")
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

    # Forward DWT (packed array + meta)
    dwt_args = {"wavelet": args.wavelet, "levels": int(args.levels), "mode": args.mode}
    out = cm.dwt(gray, dwt_args)

    # Be robust: some implementations might return only coeff_arr.
    if isinstance(out, tuple) and len(out) == 2:
        coeff_arr, meta = out
    else:
        coeff_arr, meta = out, None

    coeff_arr = np.asarray(coeff_arr, dtype=np.float32)

    # Apply nonlinearities in wavelet domain (real packed coeff array)
    sparsified = nm.sparsification(coeff_arr, {"percentile": float(args.percentile)})
    quantized = nm.quantization(coeff_arr, {"num_levels": int(args.num_levels)})
    dropped = nm.dropout_regularization(coeff_arr, {"rate": float(args.dropout_rate)})

    # Inverse DWT back to image domain
    # Assumption: inv.idwt(coeff_arr, meta) exists.
    if meta is None:
        raise RuntimeError(
            "cm.dwt(...) did not return meta, but inv.idwt(...) typically needs it. "
            "Update cm.dwt to return (coeff_arr, meta) or modify this script to match your inverse API."
        )

    reconstruct_sparsification = np.clip(inv.idwt(sparsified, meta), 0.0, 1.0)
    reconstructed_quantization = np.clip(inv.idwt(quantized, meta), 0.0, 1.0)
    reconstructed_dropout = np.clip(inv.idwt(dropped, meta), 0.0, 1.0)

    # Filenames
    p_name = int(round(float(args.percentile)))
    q_name = int(args.num_levels)
    d_name = int(round(float(args.dropout_rate) * 100.0))
    w_name = str(args.wavelet).replace("/", "_")
    L = int(args.levels)

    out_sparsification = os.path.join(args.outdir, f"dwt_{w_name}_L{L}_sparsification_{p_name}.png")
    out_quantization = os.path.join(args.outdir, f"dwt_{w_name}_L{L}_quantization_{q_name}.png")
    out_dropout = os.path.join(args.outdir, f"dwt_{w_name}_L{L}_dropout_{d_name}.png")

    save_img(out_sparsification, reconstruct_sparsification)
    save_img(out_quantization, reconstructed_quantization)
    save_img(out_dropout, reconstructed_dropout)

    print("Saved:")
    print(out_sparsification)
    print(out_quantization)
    print(out_dropout)


if __name__ == "__main__":
    main()
