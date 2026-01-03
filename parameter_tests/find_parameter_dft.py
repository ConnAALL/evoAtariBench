import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.nonLinearMethods import (
    quantization_complex,
    sparsification_complex,
    dropout_regularization_complex,
)

DFT_NORM = "ortho"
NONLINEARITY_METHODS = ("sparsification", "quantization", "dropout_regularization")

FRAME_H = 210
FRAME_W = 160
N_FRAMES = 1000

ENERGY_THRESHOLD = 0.9

SPARSIFICATION_PERCENTILES = list(range(100, -1, -1))
QUANTIZATION_NUM_LEVELS = list(range(2, 257, 2))
DROPOUT_REGULARIZATION_RATES = [i / 100.0 for i in range(99, -1, -1)]


def energy(x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return float(np.sum(np.abs(x) * np.abs(x)))
    return float(np.sum(x * x))


def dft2(x):
    x = np.asarray(x, dtype=np.float32)
    return np.fft.fftshift(np.fft.fft2(x, norm=DFT_NORM))


def center_crop(F, k):
    h, w = F.shape
    k = int(k)
    if k <= 0:
        return F[:0, :0]
    if k > h:
        k = h
    if k > w:
        k = w
    i0 = (h // 2) - (k // 2)
    j0 = (w // 2) - (k // 2)
    return F[i0 : i0 + k, j0 : j0 + k]


def find_k_by_energy(F, tau):
    total = energy(F)
    if total == 0.0:
        block = center_crop(F, 1)
        return 1, 0.0, block
    kmax = min(int(F.shape[0]), int(F.shape[1]))
    klast = 1
    tau_last = 0.0
    block_last = center_crop(F, 1)
    for k in range(1, kmax + 1):
        klast = k
        block = center_crop(F, k)
        t = energy(block) / total
        tau_last = float(t)
        block_last = block
        if t >= tau:
            return klast, float(t), block
    return int(klast), float(tau_last), block_last


def fidelity_score(x, y):
    denom = energy(x)
    if denom == 0.0:
        return 0.0
    return float(1.0 - (energy(y - x) / denom))


def apply_nl(x, method, params):
    if method == "sparsification":
        return sparsification_complex(x, params)
    if method == "quantization":
        return quantization_complex(x, params)
    if method == "dropout_regularization":
        return dropout_regularization_complex(x, params)
    raise ValueError(method)


def score_nl(x, y, method):
    if method == "sparsification":
        denom = energy(x)
        if denom == 0.0:
            return 0.0
        return float(energy(y) / denom)
    return fidelity_score(x, y)


def nl_params(method):
    if method == "sparsification":
        return [{"percentile": float(p)} for p in SPARSIFICATION_PERCENTILES]
    if method == "quantization":
        return [{"num_levels": int(n)} for n in sorted(int(n) for n in QUANTIZATION_NUM_LEVELS)]
    if method == "dropout_regularization":
        return [{"rate": float(r)} for r in DROPOUT_REGULARIZATION_RATES]
    raise ValueError(method)


def find_param(block, tau, method, params_list):
    if energy(block) == 0.0:
        p0 = params_list[0]
        y0 = apply_nl(block, method, p0)
        return p0, 0.0, y0
    for p in params_list:
        y = apply_nl(block, method, p)
        s = score_nl(block, y, method)
        if s >= tau:
            return p, float(s), y
    p = params_list[-1]
    y = apply_nl(block, method, p)
    return p, float(score_nl(block, y, method)), y


def nz(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) != 0))


def print_summary(method, ks, tau_k, params_list, tau_nl, tau_total, nzs):
    print("")
    print("=" * 72)
    print("COMPRESSION_METHOD", "dft")
    print("NONLINEARITY_METHOD", method)
    print("FRAME_H", FRAME_H)
    print("FRAME_W", FRAME_W)
    print("N_FRAMES", N_FRAMES)
    print("ENERGY_THRESHOLD", ENERGY_THRESHOLD)

    k_mean = float(np.mean(ks)) if ks else 0.0
    k_std = float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0
    print("Optimal Compressor Parameter", {"k_mean": k_mean, "k_std": k_std})

    if method == "sparsification":
        ps = [p["percentile"] for p in params_list]
        print("Optimal Non-Linearity Parameter (sparsification)", {"SPARSIFICATION_PERCENTILE_MEAN": float(np.mean(ps)), "SPARSIFICATION_PERCENTILE_STD": float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0})
    if method == "quantization":
        ns = [p["num_levels"] for p in params_list]
        print("Optimal Non-Linearity Parameter (quantization)", {"QUANTIZATION_NUM_LEVELS_MEAN": float(np.mean(ns)), "QUANTIZATION_NUM_LEVELS_STD": float(np.std(ns, ddof=1)) if len(ns) > 1 else 0.0})
    if method == "dropout_regularization":
        rs = [p["rate"] for p in params_list]
        print("Optimal Non-Linearity Parameter (dropout_regularization)", {"DROPOUT_REGULARIZATION_RATE_MEAN": float(np.mean(rs)), "DROPOUT_REGULARIZATION_RATE_STD": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0})

    print("K_MEAN", k_mean)
    print("K_STD", k_std)
    print("TAU_K_MEAN", float(np.mean(tau_k)) if tau_k else 0.0)
    print("TAU_K_STD", float(np.std(tau_k, ddof=1)) if len(tau_k) > 1 else 0.0)
    print("TAU_NL_MEAN", float(np.mean(tau_nl)) if tau_nl else 0.0)
    print("TAU_NL_STD", float(np.std(tau_nl, ddof=1)) if len(tau_nl) > 1 else 0.0)
    print("TAU_MEAN", float(np.mean(tau_total)) if tau_total else 0.0)
    print("TAU_STD", float(np.std(tau_total, ddof=1)) if len(tau_total) > 1 else 0.0)
    print("NONZERO_FRACTION_MEAN", float(np.mean(nzs)) if nzs else 0.0)


def main():
    frames = np.random.default_rng().random((N_FRAMES, FRAME_H, FRAME_W)).astype(np.float32, copy=False)

    ks = []
    tau_ks = []
    per_method = {}
    for m in NONLINEARITY_METHODS:
        per_method[m] = {"params": [], "tau_nl": [], "tau_total": [], "nz": []}

    for i in range(int(N_FRAMES)):
        F = dft2(frames[i])
        total = energy(F)
        k, tau_k, block = find_k_by_energy(F, ENERGY_THRESHOLD)
        ks.append(int(k))
        tau_ks.append(float(tau_k))

        for m in NONLINEARITY_METHODS:
            plist = nl_params(m)
            p, tau_nl, y = find_param(block, ENERGY_THRESHOLD, m, plist)
            tau_total = float(energy(y) / total) if total != 0.0 else 0.0
            per_method[m]["params"].append(p)
            per_method[m]["tau_nl"].append(float(tau_nl))
            per_method[m]["tau_total"].append(float(tau_total))
            per_method[m]["nz"].append(float(nz(y)))

    for m in NONLINEARITY_METHODS:
        print_summary(m, ks, tau_ks, per_method[m]["params"], per_method[m]["tau_nl"], per_method[m]["tau_total"], per_method[m]["nz"])


if __name__ == "__main__":
    main()
