"""
Shaping methods for mapping the processed input into the final output.

Currently supported methods:
- Affine mapping
"""

import numpy as np

def affine_mapping(x: np.ndarray, weight1: np.ndarray, weight2: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Apply an affine mapping.

    Parameters:
      x       : (M, N) Grayscale image
      weight1 : (O, M) Weight matrix 1
      weight2 : (N, P) Weight matrix 2
      bias    : (O, P) Bias vector

    Returns:
      y       : (O, P) Output vector
    """
    x = np.asarray(x)
    W1 = np.asarray(weight1)
    W2 = np.asarray(weight2)
    b = np.asarray(bias)

    y = (W1 @ x @ W2) + b
    return y.astype(np.float32, copy=False)
