import numpy as np
import pywt
from scipy.fftpack import idct as _idct

### Discrete Cosine Transform ###
def idct2(x, norm="ortho"):
    x = np.asarray(x, dtype=np.float32)
    return _idct(_idct(x, axis=-1, norm=norm), axis=-2, norm=norm)

def idct_k(block, out_hw, norm="ortho"):
    block = np.asarray(block, dtype=np.float32)
    h, w = int(out_hw[0]), int(out_hw[1])
    shell = np.zeros((h, w), dtype=np.float32)
    kh = min(shell.shape[0], block.shape[0])
    kw = min(shell.shape[1], block.shape[1])
    shell[:kh, :kw] = block[:kh, :kw]
    return idct2(shell, norm=norm)

def idft2(x, norm=None):
    x = np.asarray(x)
    return np.fft.ifft2(x, norm=norm)

def idft_k(block, out_hw, output="real", norm=None, dtype=np.float32):
    block = np.asarray(block)
    H, W = int(out_hw[0]), int(out_hw[1])

    shell = np.zeros((H, W), dtype=np.complex64)

    kH, kW = block.shape

    if kH > H:
        start = (kH - H) // 2
        block = block[start:start + H, :]
        kH = H
    if kW > W:
        start = (kW - W) // 2
        block = block[:, start:start + W]
        kW = W

    cH, cW = H // 2, W // 2

    top = cH - (kH // 2)
    left = cW - (kW // 2)
    bottom = top + kH
    right = left + kW

    shell[top:bottom, left:right] = block.astype(np.complex64, copy=False)

    unshifted = np.fft.ifftshift(shell)
    x = idft2(unshifted, norm=norm)

    if output == "complex":
        return x
    if output == "real":
        return np.real(x).astype(dtype, copy=False)
    if output == "abs":
        return np.abs(x).astype(dtype, copy=False)

    raise ValueError

def idwt(arr, meta, output="real"):
    arr = np.asarray(arr)

    slices = meta["slices"]
    wavelet = meta["wavelet"]
    mode = meta["mode"]
    H, W = map(int, meta["shape"])

    coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedec2")
    x = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)

    x = x[:H, :W]

    if output == "complex":
        return x.astype(np.complex64, copy=False)
    if output == "real":
        return np.real(x).astype(np.float32, copy=False)
    if output == "abs":
        return np.abs(x).astype(np.float32, copy=False)

    raise ValueError

INVERSE_METHODS = {
    "idct_k": idct_k,
    "idft_k": idft_k,
    "dwt_keep_scales_inv": idwt,
}

def get_inverse(name):
    if name not in INVERSE_METHODS:
        raise KeyError(f"unknown inverse: {name}")
    return INVERSE_METHODS[name]


