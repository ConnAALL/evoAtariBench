"""
Compression methods for using a known image compression algorithm to compress the input image.

Currently supported methods:
- Discrete Cosine Transform
- Discrete Fourier Transform
- Wavelet Transform
"""

import pywt
import numpy as np
from scipy.fftpack import dct as _dct
from scipy.signal import convolve2d as _convolve2d


def convert_to_grayscale(x):
    """Convert a given image matrix into grayscale"""
    x = np.asarray(x)
    if x.ndim == 2:  # If it is already grayscale, return it
        return x
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):  # If it is not grayscale, convert it
        return x[..., :3].mean(axis=-1)
    raise ValueError("expected (H,W) or (H,W,C)")


def direct_pass(x, args=None):
    """
    Direct pass without any compression.
    """
    x = convert_to_grayscale(x)
    if x.dtype != np.float32: x = x.astype(np.float32, copy=False)
    return x


def dct_k(x, args):
    """Apply 2-D Discrete Cosine Transform to x, and crop the top-left kxk corner"""
    k = args["k"]
    norm = args["norm"]

    if not k or not norm:
        raise ValueError("k and norm must be provided for dct_k")

    x = convert_to_grayscale(x).astype(np.float32, copy=False)
    c = _dct(_dct(x, axis=-1, norm=norm), axis=-2, norm=norm)
    return c[:k, :k].copy()


def dft_k(x, args):
    """Apply 2-D Discrete Fourier Transform to x, and crop the center kxk"""
    k = args["k"]
    norm = args["norm"]

    if not k or not norm:
        raise ValueError("k and norm must be provided for dft_k")

    x = convert_to_grayscale(x).astype(np.float32, copy=False)
    coeffs = np.fft.fftshift(np.fft.fft2(x, norm=norm))

    h, w = coeffs.shape
    h0, w0 = h // 2, w // 2
    half = k // 2

    low_freq = coeffs[
        h0 - half : h0 + half,
        w0 - half : w0 + half
    ]

    return low_freq.astype(np.complex64, copy=False)


def dwt(x, args):
    """
    Forward 2-D DWT returning (packed_coeff_array, meta) so inverse can reconstruct.

    meta contains the slices mapping and parameters needed for inv.idwt.
    """
    wavelet = args.get("wavelet", "bior4.4")
    levels = int(args.get("levels", 4))
    mode = args.get("mode", "symmetric")

    x = convert_to_grayscale(x).astype(np.float32, copy=False)

    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=levels, mode=mode)
    coeff_arr, slices = pywt.coeffs_to_array(coeffs)

    meta = {
        "slices": slices,
        "wavelet": wavelet,
        "levels": levels,
        "mode": mode,
        "shape": x.shape,  # original (H, W)
    }

    return coeff_arr.astype(np.float32, copy=False), meta


def get_compression_method(name):
    n = str(name).strip().lower()
    if n == "none":
        return direct_pass
    if n == "dct":
        return dct_k
    if n == "dft":
        return dft_k
    if n == "dwt":
        return dwt
    raise ValueError(f"unknown compression method: {name}")
