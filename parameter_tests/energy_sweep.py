import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.fftpack import dct

from methods.nonLinearMethods import quantization, sparsification, dropout_regularization

COMPRESSION_METHOD = "dct"
NONLINEARITY_METHOD = "sparsification"

FRAME_H = 20
FRAME_W = 110
N_FRAMES = 500
SEED = 0

ENERGY_THRESHOLD = 0.9

K_MIN = 1
K_MAX = 20
K_STEP = 1

PERCENTILE_VALUES = list(range(100, -1, -5))
NUM_LEVELS_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
DROPOUT_RATES = [0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0]

DEFAULT_PARAMS = {
    "dct": {"norm": "ortho"},
    "dft": {"norm": None},
    "sparsification": {"percentile": 90.0},
    "quantization": {"num_levels": 16},
    "dropout_regularization": {"rate": 0.1, "seed": 42},
}


def _energy(x):
    x = np.asarray(x)
    return float(np.sum(np.abs(x) ** 2))


def _apply_nonlinearity(x, method, params):
    if method == "sparsification":
        return sparsification(x, params["percentile"])
    if method == "dropout_regularization":
        return dropout_regularization(x, params["rate"], seed=params["seed"])
    if method == "quantization":
        if np.iscomplexobj(x):
            r = quantization(x.real, params["num_levels"])
            i = quantization(x.imag, params["num_levels"])
            return r + 1j * i
        return quantization(x, params["num_levels"])
    raise ValueError(method)


def _get_param_values(method):
    if method == "sparsification":
        return [{"percentile": float(p)} for p in PERCENTILE_VALUES]
    if method == "quantization":
        return [{"num_levels": int(n)} for n in sorted(NUM_LEVELS_VALUES)]
    if method == "dropout_regularization":
        return [{"rate": float(r), "seed": int(DEFAULT_PARAMS["dropout_regularization"]["seed"])} for r in DROPOUT_RATES]
    raise ValueError(method)


def _dct2(x, norm):
    x = np.asarray(x, dtype=np.float32)
    return dct(dct(x, axis=-1, norm=norm), axis=-2, norm=norm)


def _dft2(x, norm):
    x = np.asarray(x, dtype=np.float32)
    return np.fft.fftshift(np.fft.fft2(x, norm=norm))


def _crop(F, k):
    k = int(k)
    h, w = F.shape
    k = min(k, h, w)
    if COMPRESSION_METHOD == "dct":
        return F[:k, :k]
    if COMPRESSION_METHOD == "dft":
        h0, w0 = h // 2, w // 2
        hs = h0 - (k // 2)
        ws = w0 - (k // 2)
        return F[hs : hs + k, ws : ws + k]
    raise ValueError(COMPRESSION_METHOD)


def _transform(frame):
    if COMPRESSION_METHOD == "dct":
        return _dct2(frame, DEFAULT_PARAMS["dct"]["norm"])
    if COMPRESSION_METHOD == "dft":
        return _dft2(frame, DEFAULT_PARAMS["dft"]["norm"])
    raise ValueError(COMPRESSION_METHOD)


def _score(F_full, total_energy, k, nl_params, frame_idx):
    cropped = _crop(F_full, k)
    if NONLINEARITY_METHOD == "dropout_regularization":
        nl_params = dict(nl_params)
        nl_params["seed"] = int(SEED + frame_idx)
    out = _apply_nonlinearity(cropped, NONLINEARITY_METHOD, nl_params)
    return _energy(out) / (total_energy + 1e-12), out


def _nonzero_fraction(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) != 0))


def _find_k_and_params(F_full, tau, k_values, param_values, frame_idx):
    total_E = _energy(F_full)
    for k in k_values:
        block = _crop(F_full, k)
        for params in param_values:
            frac, out = _score(F_full, total_E, k, params, frame_idx)
            if frac >= tau:
                return int(k), params, float(frac), _nonzero_fraction(out)
    k = int(k_values[-1])
    params = param_values[-1]
    frac, out = _score(F_full, total_E, k, params, frame_idx)
    return int(k), params, float(frac), _nonzero_fraction(out)


def main():
    rng = np.random.default_rng(SEED)
    frames = rng.random((N_FRAMES, FRAME_H, FRAME_W), dtype=np.float32)

    k_values = list(range(int(K_MIN), int(K_MAX) + 1, int(K_STEP)))
    param_values = _get_param_values(NONLINEARITY_METHOD)

    F_full_list = [_transform(frames[i]) for i in range(N_FRAMES)]

    ks = []
    params_list = []
    taus = []
    nzs = []

    for i in range(N_FRAMES):
        k, params, tau_real, nz = _find_k_and_params(F_full_list[i], ENERGY_THRESHOLD, k_values, param_values, i)
        ks.append(k)
        params_list.append(params)
        taus.append(tau_real)
        nzs.append(nz)

    print("COMPRESSION_METHOD", COMPRESSION_METHOD)
    print("NONLINEARITY_METHOD", NONLINEARITY_METHOD)
    print("FRAME_H", FRAME_H)
    print("FRAME_W", FRAME_W)
    print("N_FRAMES", N_FRAMES)
    print("ENERGY_THRESHOLD", ENERGY_THRESHOLD)
    print("K_MEAN", float(np.mean(ks)))
    print("K_STD", float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0)
    print("TAU_MEAN", float(np.mean(taus)))
    print("TAU_STD", float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0)
    print("NONZERO_FRACTION_MEAN", float(np.mean(nzs)))
    if NONLINEARITY_METHOD == "sparsification":
        ps = [p["percentile"] for p in params_list]
        print("P_MEAN", float(np.mean(ps)))
        print("P_STD", float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0)
    if NONLINEARITY_METHOD == "quantization":
        ns = [p["num_levels"] for p in params_list]
        print("NUM_LEVELS_MEAN", float(np.mean(ns)))
        print("NUM_LEVELS_STD", float(np.std(ns, ddof=1)) if len(ns) > 1 else 0.0)
    if NONLINEARITY_METHOD == "dropout_regularization":
        rs = [p["rate"] for p in params_list]
        print("RATE_MEAN", float(np.mean(rs)))
        print("RATE_STD", float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0)


if __name__ == "__main__":
    main()


