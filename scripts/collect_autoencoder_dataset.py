"""
Collect Atari frames for autoencoder training
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
from numpy.lib.format import open_memmap
import yaml
import gymnasium as gym
import ale_py

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "config.yml")

# Add the repository root to the Python path
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.single_run import EvoAtariPipelinePolicy, normalize_frame, process_features
from methods import compressionMethods as cm

# Hyperparameters for the policy
DEFAULT_POLICY_COMPRESSION = "dct"
DEFAULT_POLICY_K = 142
DEFAULT_POLICY_NORM = "ortho"
DEFAULT_POLICY_NONLINEARITY = "sparsification"
DEFAULT_POLICY_PERCENTILE = 22.0
DEFAULT_POLICY_NPZ_PATH = os.path.join(_REPO_ROOT, "data", "sample_best_individual.npz")
FLUSH_EVERY = 10_000


def load_default_args(path: str) -> dict:
    """Load the default arguments from a YAML file"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_atari_env(env_name: str, obs_type: str, repeat_action_probability: float, frameskip: int):
    """Make an Atari environment"""
    return gym.make(id=env_name, obs_type=obs_type, repeat_action_probability=repeat_action_probability, frameskip=frameskip)


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_exists(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    defaults = load_default_args(_DEFAULT_CONFIG_PATH)
    p = argparse.ArgumentParser(description="Collect Atari frames for autoencoder training.")
    p.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="Path to a YAML config (defaults to scripts/config.yml).")
    p.add_argument("--env-name", default=str(defaults.get("ENV_NAME", "ALE/SpaceInvaders-v5")))
    p.add_argument("--obs-type", default=str(defaults.get("OBS_TYPE", "grayscale")))
    p.add_argument("--frameskip", type=int, default=int(defaults.get("FRAMESKIP", 4)))
    p.add_argument("--repeat-action-probability", type=float, default=0.25)
    p.add_argument("--compression", default=DEFAULT_POLICY_COMPRESSION, help="Compression method name for policy playback.")
    p.add_argument("--num-frames", type=int, default=1_000_000, help="Total number of frames to collect.")
    p.add_argument("--out-root", default=os.path.join(_REPO_ROOT, "data", "autoencoder_datasets"), help="Root output directory.")
    p.add_argument("--run-name", default=None, help="Optional run folder name. Default is auto timestamped (env + timestamp).")
    p.add_argument("--render", action="store_true", help="Render to a window (slow).")
    return p.parse_args()


def main():
    args = parse_args()  # Parse the arguments
    run_name = args.run_name
    if not run_name:
        safe_env = str(args.env_name).replace("/", "_")
        run_name = f"{safe_env}_{args.obs_type}_fs{int(args.frameskip)}_{generate_timestamp()}"

    out_dir = ensure_exists(os.path.join(os.path.abspath(args.out_root), run_name))
    env = make_atari_env(env_name=args.env_name, obs_type=args.obs_type, repeat_action_probability=args.repeat_action_probability, frameskip=args.frameskip)
    rng = np.random.default_rng()

    policy = None
    policy_meta = None
    policy_npz_path = os.path.abspath(DEFAULT_POLICY_NPZ_PATH)

    if not os.path.isfile(policy_npz_path):
        raise FileNotFoundError(f"Default policy file not found: {policy_npz_path}\nPut a best-individual .npz at data/sample_best_individual.npz (see temp/SCOPE-for-Atari/data/extract_best_individual.py).")
    
    d = np.load(policy_npz_path, allow_pickle=True)
    if "solution" not in d:
        raise ValueError(f"policy npz is missing key 'solution': {policy_npz_path}")

    # Allow the .npz to override K/P if present (and not NaN)
    k = int(DEFAULT_POLICY_K)
    pctl = float(DEFAULT_POLICY_PERCENTILE)
    if "K" in d:
        try:
            k0 = float(np.asarray(d["K"]))
            if not np.isnan(k0):
                k = int(k0)
        except Exception:
            pass
    if "P" in d:
        try:
            p0 = float(np.asarray(d["P"]))
            if not np.isnan(p0):
                pctl = float(p0)
        except Exception:
            pass

    policy_args = {
        "compression": str(args.compression),
        "k": int(k),
        "norm": str(DEFAULT_POLICY_NORM),
        "nonlinearity": str(DEFAULT_POLICY_NONLINEARITY),
        "percentile": float(pctl),
    }

    # Derive action/output size and feature shape exactly like scripts/single_run.py does.
    output_size = int(env.action_space.n) if hasattr(env.action_space, "n") else None
    if output_size is None:
        raise ValueError(f"Unsupported action space for policy playback: {env.action_space}")

    obs0, _ = env.reset()
    comp_fn = cm.get_compression_method(policy_args["compression"])
    feats0 = process_features(comp_fn(normalize_frame(obs0), policy_args))
    feature_shape = (int(feats0.shape[0]), int(feats0.shape[1]))

    chromosome = np.asarray(d["solution"], dtype=np.float32)
    policy = EvoAtariPipelinePolicy(
        chromosome=chromosome,
        output_size=output_size,
        feature_shape=feature_shape,
        args=policy_args,
    )

    policy_meta = {
        "policy_npz": policy_npz_path,
        "policy_fitness": (float(np.asarray(d["fitness"])) if "fitness" in d else None),
        "policy_K": int(k),
        "policy_P": float(pctl),
        "policy_args": policy_args,
        "chromosome_len": int(chromosome.size),
        "feature_shape": list(feature_shape),
        "output_size": int(output_size),
    }

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_name": str(args.env_name),
        "obs_type": str(args.obs_type),
        "frameskip": int(args.frameskip),
        "repeat_action_probability": float(args.repeat_action_probability),
        "num_frames_target": int(args.num_frames),
        "action_space": str(env.action_space),
        "policy": policy_meta,
        "flush_every": int(FLUSH_EVERY),
    }

    # Collect to a single .npy file (memmap) and flush every FLUSH_EVERY frames.
    obs, info = env.reset()
    meta["initial_obs_shape"] = list(np.asarray(obs).shape)
    meta["initial_obs_dtype"] = str(np.asarray(obs).dtype)

    obs_path = os.path.join(out_dir, "observations.npy")
    obs0 = np.asarray(obs)
    obs_mm = open_memmap(
        obs_path,
        mode="w+",
        dtype=obs0.dtype,
        shape=(int(args.num_frames),) + obs0.shape,
    )
    meta["observations_path"] = obs_path

    total = 0
    episode_id = 0
    step_id = 0
    episode_return = 0.0
    t0 = time.time()

    while total < int(args.num_frames):
        if args.render:
            env.render()

        obs_arr = np.asarray(obs)
        obs_mm[total] = obs_arr

        if policy is not None:
            logits = policy.forward(obs_arr)
            action = int(np.argmax(logits))
        else:
            # Random fallback (kept for debugging)
            if hasattr(env.action_space, "n"):
                action = int(rng.integers(0, int(env.action_space.n)))
            else:
                action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)

        episode_return += float(reward)

        total += 1
        step_id += 1

        if (total % int(FLUSH_EVERY)) == 0 or (total >= int(args.num_frames)):
            obs_mm.flush()
            dt = max(time.time() - t0, 1e-9)
            fps = float(total) / dt
            print(f"[collect] frames={total}/{int(args.num_frames)} fps={fps:.1f} saved={obs_path}")

        if done:
            print(
                f"[episode end] episode_id={episode_id} return={episode_return:.2f} steps={step_id + 1} total_frames={total}"
            )
            obs, info = env.reset()
            episode_id += 1
            step_id = 0
            episode_return = 0.0
        else:
            obs = next_obs

    env.close()
    obs_mm.flush()

    meta["num_frames_collected"] = int(total)
    meta["num_episodes"] = int(episode_id + 1)  # includes current episode

    write_json(os.path.join(out_dir, "meta.json"), meta)

    print(f"[done] wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()
