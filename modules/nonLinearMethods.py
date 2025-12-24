"""
Methods that introduce non-linearity to the input data.

Currently supported methods:
- Quantization
- Sparsification
- Dropout Regularization
"""

import numpy as np

def quantization(x: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Function for uniform quantization. 

    Parameters:
        x: Input array of shape (H, W) or (H, W, C)
        num_levels: Number of quantization levels

    Returns:
        Quantized array of shape (H, W) or (H, W, C)
    """
    if int(num_levels) <= 1:  # If the number of quantization levels is less than or equal to 1, raise an error
        raise ValueError("num_levels must be greater than 1")
    
    x = np.asarray(x, dtype=np.float32)  # Convert the input array to a float32 array

    # Find the minimum and maximum values of the input array
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    scale = x_max - x_min

    # If the scale is 0, return the input array
    if scale == 0.0:
        return x
    
    # Normalize the input array to [0, 1]
    y = (x - x_min) / scale

    # Quantize the normalized array to the number of quantization levels
    y = np.round(y * (int(num_levels) - 1)) / (int(num_levels) - 1)

    # Scale the quantized array back to the original range
    return y * scale + x_min

def sparsification(x: np.ndarray, percentile: float) -> np.ndarray:
    """
    Function for sparsifying the input array based on a percentile.

    Parameters:
        x: Input array of shape (H, W) or (H, W, C)
        q: Percentile to sparsify the input array

    Returns:
        Sparsified array of shape (H, W) or (H, W, C)
    """
    # If the percentile is not in the range [0, 100], raise an error
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("q must be in [0, 100]")
    
    x = np.asarray(x, dtype=np.float32)  # Convert the input array to a float32 array

    # Find the percentile of the absolute values of the input array
    thr = np.percentile(np.abs(x), percentile)

    # Sparsify the input array based on the percentile
    return np.where(np.abs(x) >= thr, x, 0.0).astype(np.float32, copy=False)

def dropout_regularization(x: np.ndarray, rate: float, seed: int = 42) -> np.ndarray:
    """
    Function for dropout regularization.

    Parameters:
        x: Input array of any shape
        rate: Dropout probability (must be in [0, 1))
        seed: Random seed (default = 42)

    Returns:
        Array with dropout regularization applied
    """
    if not (0.0 <= rate < 1.0):
        raise ValueError("rate must be in [0.0, 1.0)")
    
    x = np.asarray(x, dtype=np.float32)

    # No dropout
    if rate == 0.0:
        return x

    keep_prob = 1.0 - rate
    rng = np.random.default_rng(seed)  # Seeded RNG

    # Bernoulli mask
    mask = rng.random(size=x.shape) < keep_prob

    # Apply mask and rescale to maintain expected value
    return np.where(mask, x / keep_prob, 0.0).astype(np.float32, copy=False)
