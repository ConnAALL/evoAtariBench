import os
import sys
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.fftpack import dct

from methods.nonLinearMethods import quantization, sparsification, dropout_regularization

DCT_NORM = "ortho"
NONLINEARITY_METHODS = (
    "sparsification",
    "quantization",
    "dropout_regularization",
)

FRAME_H = 210
FRAME_W = 160
N_FRAMES = 10000
SEED = 0

ENERGY_THRESHOLD = 0.9

K_MIN = 1
K_MAX = None
K_STEP = 1

# Parameter grids (fine-grained)
SPARSIFICATION_PERCENTILES = list(range(100, -1, -1))
QUANTIZATION_NUM_LEVELS = list(range(2, 257, 2))
DROPOUT_REGULARIZATION_RATES = [i / 100.0 for i in range(99, -1, -1)]
CHUNKSIZE = 10

_WORKER = {}


def _energy(x):
    """
    Match temp/quickscope energy convention:
    - For real arrays: sum of squares (np.sum(F * F))
    - For complex arrays: sum of |F|^2
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return float(np.sum(np.abs(x) * np.abs(x)))
    return float(np.sum(x * x))


def _dct2_quickscope(x: np.ndarray) -> np.ndarray:
    """Match temp/quickscope `dct2`: dct(dct(x.T).T) with ortho norm."""
    x = np.asarray(x, dtype=np.float32)
    return dct(dct(x.T, norm=DCT_NORM).T, norm=DCT_NORM)


def _find_k_by_energy(F: np.ndarray, tau: float, k_values) -> tuple[int, float]:
    """
    Match temp/quickscope `find_k_by_energy`:
    minimal k such that sum(block^2)/sum(F^2) >= tau.
    """
    total_E = _energy(F)
    if total_E == 0.0:
        k0 = int(list(k_values)[0])
        return k0, 0.0
    k_last = None
    for k in k_values:
        k_last = int(k)
        block = F[:k_last, :k_last]
        tau_real = _energy(block) / total_E
        if tau_real >= tau:
            return k_last, float(tau_real)
    # fallback to max k
    k_last = int(k_last) if k_last is not None else 1
    block = F[:k_last, :k_last]
    return k_last, float(_energy(block) / total_E) if total_E != 0.0 else 0.0


def _sparse_quickscope(block: np.ndarray, p: float) -> np.ndarray:
    """Match temp/quickscope `sparse` (threshold on abs(block) percentile, < thr -> 0)."""
    frame_abs = np.abs(block)
    threshold = np.percentile(frame_abs.ravel(), p)
    out = np.array(block, copy=True)
    out[np.abs(out) < threshold] = 0
    return out


def _fidelity_nmse_score(x, y) -> float:
    """
    Fidelity score in [.., 1], where 1 is perfect reconstruction:
        score = 1 - (||y-x||^2 / ||x||^2)
    Uses the same energy convention as quickscope (sum of squares / |.|^2).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    denom = _energy(x)
    if denom == 0.0:
        return 0.0
    err = _energy(y - x)
    return float(1.0 - (err / denom))


def _nl_score(block: np.ndarray, out: np.ndarray, method: str) -> float:
    """
    Stage-2 selection metric:
    - sparsification: match quickscope => energy(out)/energy(block)
    - quantization/dropout_regularization: fidelity score => 1 - ||out-block||^2 / ||block||^2
    """
    method = str(method)
    if method == "sparsification":
        denom = _energy(block)
        if denom == 0.0:
            return 0.0
        return float(_energy(out) / denom)
    if method in {"quantization", "dropout_regularization"}:
        return _fidelity_nmse_score(block, out)
    raise ValueError(method)


def _nl_score_kind(method: str) -> str:
    method = str(method)
    if method == "sparsification":
        return "energy_ratio"
    if method in {"quantization", "dropout_regularization"}:
        return "fidelity_nmse"
    return "unknown"


def _find_param_by_energy(block: np.ndarray, tau: float, method: str, param_values):
    """
    Quickscope-style stage-2 search:
    - keep k fixed
    - iterate params in the configured order
    - choose the first params that satisfy the method-specific score >= tau
      (see `_nl_score`).
    Returns (best_params, realized_score, out_array).
    """
    if _energy(block) == 0.0:
        params0 = param_values[0]
        out0 = _apply_nonlinearity(block, method, params0)
        return params0, 0.0, out0

    for params in param_values:
        out = _apply_nonlinearity(block, method, params)
        score = _nl_score(block, out, method)
        if score >= tau:
            return params, float(score), out

    params = param_values[-1]
    out = _apply_nonlinearity(block, method, params)
    return params, float(_nl_score(block, out, method)), out


def _apply_nonlinearity(x, method, params):
    if method == "sparsification":
        # Use the exact quickscope sparsification implementation for comparability.
        return _sparse_quickscope(x, params["percentile"])
    if method == "quantization":
        if np.iscomplexobj(x):
            r = quantization(x.real, params)
            i = quantization(x.imag, params)
            return r + 1j * i
        return quantization(x, params)
    if method == "dropout_regularization":
        # Dropout is now pure random by default (no seeding). If complex, apply
        # dropout independently to real/imag to preserve complex structure.
        if np.iscomplexobj(x):
            r = dropout_regularization(x.real, params)
            i = dropout_regularization(x.imag, params)
            return r + 1j * i
        return dropout_regularization(x, params)
    raise ValueError(method)


def _get_param_values(method):
    if method == "sparsification":
        # Match quickscope search style: try higher percentiles first (more sparsity).
        ps = [float(p) for p in SPARSIFICATION_PERCENTILES]
        return [{"percentile": p} for p in ps]
    if method == "quantization":
        # Try fewer levels first (more compression).
        ns = sorted(int(n) for n in QUANTIZATION_NUM_LEVELS)
        return [{"num_levels": n} for n in ns]
    if method == "dropout_regularization":
        # Try higher dropout first (more compression).
        rs = [float(r) for r in DROPOUT_REGULARIZATION_RATES]
        return [{"rate": r} for r in rs]
    raise ValueError(method)


def _dct2(x, norm):
    # Keep signature, but implement to match quickscope ordering exactly.
    _ = norm
    return _dct2_quickscope(x)


def _nonzero_fraction(x):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) != 0))


def _init_worker(k_values, method_to_params, energy_threshold, dct_norm):
    _WORKER["k_values"] = list(k_values)
    _WORKER["method_to_params"] = dict(method_to_params)
    _WORKER["energy_threshold"] = float(energy_threshold)
    _WORKER["dct_norm"] = dct_norm


def _process_one(item):
    frame_idx, frame = item
    k_values = _WORKER["k_values"]
    method_to_params = _WORKER["method_to_params"]
    tau = _WORKER["energy_threshold"]
    dct_norm = _WORKER["dct_norm"]

    # Stage 1 (quickscope): choose k using full DCT energy.
    F_full = _dct2(frame, dct_norm)
    k, tau_k = _find_k_by_energy(F_full, tau, k_values)
    block = F_full[:k, :k]

    total_E_full = _energy(F_full)
    out = {
        "frame_idx": int(frame_idx),
        "k": int(k),
        "tau_k": float(tau_k),
        "methods": {},
    }

    # Stage 2 (quickscope-style): choose per-method parameter using block energy.
    for method, param_values in method_to_params.items():
        params, tau_nl, y = _find_param_by_energy(
            block,
            tau,
            method,
            param_values,
        )
        tau_total = float(_energy(y) / total_E_full) if total_E_full != 0.0 else 0.0
        out["methods"][method] = {
            "params": params,
            "tau_nl": float(tau_nl),
            "tau_total": float(tau_total),
            "nz": float(_nonzero_fraction(y)),
        }
    return out


def _print_summary(method, ks, params_list, tau_k_list, tau_nl_list, tau_total_list, nzs):
    print("")
    print("=" * 72)
    print("COMPRESSION_METHOD", "dct")
    print("NONLINEARITY_METHOD", method)
    print("FRAME_H", FRAME_H)
    print("FRAME_W", FRAME_W)
    print("N_FRAMES", N_FRAMES)
    print("ENERGY_THRESHOLD", ENERGY_THRESHOLD)

    k_mean = float(np.mean(ks)) if len(ks) else 0.0
    k_std = float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0
    print("Optimal Compressor Parameter", {"k_mean": k_mean, "k_std": k_std})

    if method == "sparsification":
        ps = [p["percentile"] for p in params_list]
        p_mean = float(np.mean(ps)) if len(ps) else 0.0
        p_std = float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0
        print("Optimal Non-Linearity Parameter (sparsification)", {"SPARSIFICATION_PERCENTILE_MEAN": p_mean, "SPARSIFICATION_PERCENTILE_STD": p_std})
    if method == "quantization":
        ns = [p["num_levels"] for p in params_list]
        n_mean = float(np.mean(ns)) if len(ns) else 0.0
        n_std = float(np.std(ns, ddof=1)) if len(ns) > 1 else 0.0
        print("Optimal Non-Linearity Parameter (quantization)", {"QUANTIZATION_NUM_LEVELS_MEAN": n_mean, "QUANTIZATION_NUM_LEVELS_STD": n_std})
    if method == "dropout_regularization":
        rs = [p["rate"] for p in params_list]
        r_mean = float(np.mean(rs)) if len(rs) else 0.0
        r_std = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
        print("Optimal Non-Linearity Parameter (dropout_regularization)", {"DROPOUT_REGULARIZATION_RATE_MEAN": r_mean, "DROPOUT_REGULARIZATION_RATE_STD": r_std})

    print("K_MEAN", k_mean)
    print("K_STD", k_std)
    print("TAU_K_MEAN", float(np.mean(tau_k_list)))
    print("TAU_K_STD", float(np.std(tau_k_list, ddof=1)) if len(tau_k_list) > 1 else 0.0)
    print("TAU_NL_KIND", _nl_score_kind(method))
    print("TAU_NL_MEAN", float(np.mean(tau_nl_list)))
    print("TAU_NL_STD", float(np.std(tau_nl_list, ddof=1)) if len(tau_nl_list) > 1 else 0.0)
    print("TAU_MEAN", float(np.mean(tau_total_list)))
    print("TAU_STD", float(np.std(tau_total_list, ddof=1)) if len(tau_total_list) > 1 else 0.0)
    print("NONZERO_FRACTION_MEAN", float(np.mean(nzs)))

    if method == "sparsification":
        ps = [p["percentile"] for p in params_list]
        print("SPARSIFICATION_PERCENTILE_MEAN", float(np.mean(ps)))
        print("SPARSIFICATION_PERCENTILE_STD", float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0)
    if method == "quantization":
        ns = [p["num_levels"] for p in params_list]
        print("QUANTIZATION_NUM_LEVELS_MEAN", float(np.mean(ns)))
        print("QUANTIZATION_NUM_LEVELS_STD", float(np.std(ns, ddof=1)) if len(ns) > 1 else 0.0)
    if method == "dropout_regularization":
        rs = [p["rate"] for p in params_list]
        print("DROPOUT_REGULARIZATION_RATE_MEAN", float(np.mean(rs)))
        print("DROPOUT_REGULARIZATION_RATE_STD", float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0)


def main():
    rng = np.random.default_rng(SEED)
    frames = rng.random((N_FRAMES, FRAME_H, FRAME_W), dtype=np.float32)

    k_max = min(int(FRAME_H), int(FRAME_W)) if K_MAX is None else min(int(K_MAX), int(FRAME_H), int(FRAME_W))
    k_values = list(range(int(K_MIN), int(k_max) + 1, int(K_STEP)))

    nproc = max(1, int((os.cpu_count() or 1) * 2))
    ctx = mp.get_context("spawn")

    method_to_params = {m: _get_param_values(m) for m in NONLINEARITY_METHODS}

    with ctx.Pool(
        processes=nproc,
        initializer=_init_worker,
        initargs=(k_values, method_to_params, ENERGY_THRESHOLD, DCT_NORM),
    ) as pool:
        items = ((i, frames[i]) for i in range(int(N_FRAMES)))
        results = list(pool.imap(_process_one, items, chunksize=int(CHUNKSIZE)))

    results.sort(key=lambda r: r["frame_idx"])
    ks_all = [r["k"] for r in results]
    tau_k_all = [r["tau_k"] for r in results]

    for method in NONLINEARITY_METHODS:
        params_list = [r["methods"][method]["params"] for r in results]
        tau_nl = [r["methods"][method]["tau_nl"] for r in results]
        tau_total = [r["methods"][method]["tau_total"] for r in results]
        nzs = [r["methods"][method]["nz"] for r in results]

        _print_summary(method, ks_all, params_list, tau_k_all, tau_nl, tau_total, nzs)


if __name__ == "__main__":
    main()


