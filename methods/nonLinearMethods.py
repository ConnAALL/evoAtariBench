"""
Methods that introduce non-linearity to the input data.

Currently supported methods:
- Quantization
- Sparsification
- Dropout Regularization
"""

import numpy as np

def quantization(x: np.ndarray, args: dict) -> np.ndarray:
    """
    Function for uniform quantization. 

    Parameters:
        x: Input array of shape (H, W) or (H, W, C)
        args: Dictionary containing the number of quantization levels

    Returns:
        Quantized array of shape (H, W) or (H, W, C)
    """
    num_levels = args.get("num_levels", None)
    if num_levels is None:
        raise ValueError("num_levels must be provided for quantization")

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


def quantization_complex(x: np.ndarray, args: dict) -> np.ndarray:
    """
    Apply uniform quantization to a complex-valued array

    Parameters:
        x    : complex ndarray
        args : dict containing 'num_levels'

    Returns:
        Complex ndarray with quantized magnitude and preserved phase
    """
    if not np.iscomplexobj(x):
        raise ValueError("Input must be a complex-valued array")

    # Compute magnitude
    mag = np.abs(x).astype(np.float32)

    # Quantize magnitude using existing function
    mag_q = quantization(mag, args)

    # If magnitude is zero everywhere, just return zeros
    # (avoids division by zero)
    eps = 1e-8
    scale = mag_q / (mag + eps)

    # Apply scaling to preserve phase
    return (x * scale).astype(np.complex64, copy=False)


def sparsification(x: np.ndarray, args: dict) -> np.ndarray:
    """
    Function for sparsifying the input array based on a percentile.

    Parameters:
        x: Input array of shape (H, W) or (H, W, C)
        q: Percentile to sparsify the input array

    Returns:
        Sparsified array of shape (H, W) or (H, W, C)
    """
    percentile = args.get("percentile", None)
    if percentile is None:
        raise ValueError("percentile must be provided for sparsification")

    # If the percentile is not in the range [0, 100], raise an error
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100]")
    
    x = np.asarray(x, dtype=np.float32)  # Convert the input array to a float32 array

    # Find the percentile of the absolute values of the input array
    thr = np.percentile(np.abs(x), percentile)

    # Sparsify the input array based on the percentile
    return np.where(np.abs(x) >= thr, x, 0.0).astype(np.float32, copy=False)


def sparsification_complex(x: np.ndarray, args: dict) -> np.ndarray:
    """
    Sparsify a complex-valued array using magnitude-based thresholding, while preserving phase consistency.

    Parameters:
        x: Complex ndarray
        args: Dictionary containing 'percentile'

    Returns:
        Complex ndarray with both real and imaginary parts sparsified.
    """
    if not np.iscomplexobj(x):
        raise ValueError("Input must be a complex-valued array")

    percentile = args.get("percentile", None)
    if percentile is None:
        raise ValueError("percentile must be provided for sparsification")

    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100]")

    # Compute magnitude
    mag = np.abs(x)

    # Threshold on magnitude
    thr = np.percentile(mag, percentile)
    mask = mag >= thr

    # Apply mask to both real and imaginary parts
    real = np.where(mask, x.real, 0.0)
    imag = np.where(mask, x.imag, 0.0)

    return (real + 1j * imag).astype(np.complex64, copy=False)


def dropout_regularization(x: np.ndarray, args: dict) -> np.ndarray:
    """
    Function for dropout regularization.

    Parameters:
        x: Input array of any shape
        args: Dictionary containing the dropout rate and the random number generator (rng)

    Returns:
        Array with dropout regularization applied
    """
    rate = args.get("rate", None)
    rng = np.random.default_rng()
    if rate is None:
        raise ValueError("rate must be provided for dropout regularization")

    if not (0.0 <= rate < 1.0):
        raise ValueError("rate must be in [0.0, 1.0)")
    
    x = np.asarray(x, dtype=np.float32)

    # No dropout
    if rate == 0.0:
        return x

    keep_prob = 1.0 - rate

    # Bernoulli mask
    mask = rng.random(size=x.shape) < keep_prob

    # Apply mask and rescale to maintain expected value
    return np.where(mask, x / keep_prob, 0.0).astype(np.float32, copy=False)

def dropout_regularization_complex(x: np.ndarray, args: dict) -> np.ndarray:
    if not np.iscomplexobj(x):
        raise ValueError("Input must be a complex-valued array")

    rate = args.get("rate", None)
    if rate is None:
        raise ValueError("rate must be provided for dropout regularization")
    if not (0.0 <= rate < 1.0):
        raise ValueError("rate must be in [0.0, 1.0)")

    x = np.asarray(x, dtype=np.complex64)

    if rate == 0.0:
        return x

    keep_prob = 1.0 - rate
    rng = np.random.default_rng()
    mask = rng.random(size=x.shape) < keep_prob

    return np.where(mask, x / keep_prob, 0.0).astype(np.complex64, copy=False)

def get_nonlinearity_method(method_name):
    """Return the non-linearity function for the given method name."""
    if method_name is None:
        return None

    n = str(method_name).strip().lower()
    if n in {"", "none"}:
        return None
    if n == "quantization":
        return quantization
    if n == "quantization_complex":
        return quantization_complex
    if n == "sparsification":
        return sparsification
    if n == "sparsification_complex":
        return sparsification_complex
    if n == "dropout_regularization":
        return dropout_regularization
    if n == "dropout_regularization_complex":
        return dropout_regularization_complex
    raise ValueError(
        f"Unknown nonlinearity method: {method_name!r}. Expected one of "
        f"{{'sparsification','sparsification_complex','quantization','quantization_complex','dropout_regularization','dropout_regularization_complex'}} (or None/'none')."
    )