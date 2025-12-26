import argparse
import json
import os
import sys

import numpy as np
import cma
import gymnasium as gym
import ale_py
import ray

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods import compressionMethods as cm
from methods import nonLinearMethods as nm
from methods import shapingMethods as sm

def make_silent_env(env_name, obs_type, repeat_action_probability, frameskip):
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    os.dup2(devnull_fd, stdout_fd)
    os.dup2(devnull_fd, stderr_fd)
    try:
        env = gym.make(
            id=env_name,
            obs_type=obs_type,
            repeat_action_probability=repeat_action_probability,
            frameskip=frameskip,
        )
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)
    return env


def _normalize_obs(obs):
    x = np.asarray(obs)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x / 255.0


def _resolve_nonlinearity(name):
    if name is None:
        return None
    n = name.strip().lower()
    if n in {"none", "identity", ""}:
        return None
    if n in {"quant", "quantization"}:
        return nm.quantization
    if n in {"sparse", "sparsification"}:
        return nm.sparsification
    if n in {"dropout", "dropout_regularization"}:
        def _fn(x, args):
            seed = int(args.get("seed", 42))
            inner_args = dict(args)
            inner_args.pop("seed", None)
            return nm.dropout_regularization(x, inner_args, seed=seed)

        return _fn
    raise ValueError(f"Unknown nonlinearity method: {name!r}")


def _materialize_features(x):
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = np.asarray(x)
    if np.iscomplexobj(x):
        x = np.abs(x)
    if x.ndim != 2:
        raise ValueError(f"Compression output must be 2D (H,W); got shape={x.shape}")
    return x.astype(np.float32, copy=False)


def compute_chromosome_size(feature_shape, output_size):
    m, n = int(feature_shape[0]), int(feature_shape[1])
    return m + (n * int(output_size)) + int(output_size)


class EvoAtariPipelinePolicy:
    def __init__(self, chromosome, output_size, feature_shape, args):
        self.output_size = int(output_size)
        self.feature_shape = (int(feature_shape[0]), int(feature_shape[1]))
        self.args = args

        self._compression_fn = cm.get_compression_method(args["compression"])
        self._nonlin_fn = _resolve_nonlinearity(args.get("nonlinearity", None))

        self._process_chromosome(chromosome)

    def _process_chromosome(self, chromosome):
        m, n = self.feature_shape
        w1_len = m  # (1, M)
        w2_len = n * self.output_size  # (N, A)
        b_len = self.output_size  # (1, A)

        chromosome = np.asarray(chromosome, dtype=np.float64)
        expected = w1_len + w2_len + b_len
        if chromosome.size != expected:
            raise ValueError(f"chromosome length mismatch: got={chromosome.size}, expected={expected}")

        self.W1 = chromosome[:w1_len].reshape(1, m).astype(np.float32, copy=False)
        self.W2 = chromosome[w1_len : w1_len + w2_len].reshape(n, self.output_size).astype(np.float32, copy=False)
        self.b = chromosome[w1_len + w2_len :].reshape(1, self.output_size).astype(np.float32, copy=False)

    def forward(self, obs):
        x = _normalize_obs(obs)
        feats = self._compression_fn(x, self.args)
        feats = _materialize_features(feats)

        if self._nonlin_fn is not None:
            feats = self._nonlin_fn(feats, self.args)
            feats = np.asarray(feats, dtype=np.float32)

        logits = sm.affine_mapping(feats, self.W1, self.W2, self.b)
        return logits.flatten()


def _evaluate_individual(
    solution,
    individual_idx,
    gen_idx,
    env_name,
    obs_type,
    repeat_action_probability,
    frameskip,
    output_size,
    feature_shape,
    episodes_per_individual,
    max_steps_per_episode,
    args,
):
    policy = EvoAtariPipelinePolicy(
        chromosome=solution,
        output_size=output_size,
        feature_shape=feature_shape,
        args=args,
    )

    env = make_silent_env(
        env_name=env_name,
        obs_type=obs_type,
        repeat_action_probability=repeat_action_probability,
        frameskip=frameskip,
    )

    total_reward = 0.0
    rows = []

    for ep in range(int(episodes_per_individual)):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < int(max_steps_per_episode):
            prefs = policy.forward(obs)
            action = int(np.argmax(prefs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = bool(terminated) or bool(truncated)
            steps += 1

        total_reward += ep_reward
        rows.append([gen_idx + 1, individual_idx + 1, ep + 1, ep_reward])

    env.close()
    avg_reward = total_reward / float(episodes_per_individual)
    return individual_idx, float(avg_reward), rows


@ray.remote
def run_task(args, run_id):
    return _run_task_impl(args, run_id)


def _run_task_impl(args, run_id):
    env_name = args.get("ENV_NAME", "ALE/SpaceInvaders-v5")
    obs_type = args.get("OBS_TYPE", "grayscale")
    frameskip = int(args.get("FRAMESKIP", 4))
    repeat_action_probability = float(args.get("REPEAT_ACTION_PROBABILITY", 0.0))

    generations = int(args.get("GENERATIONS", 5000))
    cma_sigma = float(args.get("CMA_SIGMA", 0.5))
    population_size = args.get("POPULATION_SIZE", None)
    episodes_per_individual = int(args.get("EPISODES_PER_INDIVIDUAL", 1))
    max_steps_per_episode = int(args.get("MAX_STEPS_PER_EPISODE", 10000))
    verbosity = int(args.get("VERBOSITY_LEVEL", 1))
    seed = int(args.get("SEED", 0))

    if verbosity >= 1:
        print(f"[Run {run_id}] ENV={env_name} compression={args.get('compression')} nonlinearity={args.get('nonlinearity')}")

    probe_env = make_silent_env(
        env_name=env_name,
        obs_type=obs_type,
        repeat_action_probability=repeat_action_probability,
        frameskip=frameskip,
    )
    output_size = int(probe_env.action_space.n)
    obs0, _ = probe_env.reset()
    probe_env.close()

    comp_fn = cm.get_compression_method(args["compression"])
    feats0 = _materialize_features(comp_fn(_normalize_obs(obs0), args))
    feature_shape = (int(feats0.shape[0]), int(feats0.shape[1]))
    chromosome_size = compute_chromosome_size(feature_shape, output_size)

    inopts = {"seed": seed}
    if population_size is not None:
        inopts["popsize"] = int(population_size)
    es = cma.CMAEvolutionStrategy(np.zeros(chromosome_size), cma_sigma, inopts)

    fitness_log = []
    best_individuals = []
    plot_data = []

    best_fitness_so_far = float("-inf")
    best_solution_so_far = None

    for gen in range(generations):
        solutions = es.ask()
        results = [
            _evaluate_individual(
                solution=solutions[i],
                individual_idx=i,
                gen_idx=gen,
                env_name=env_name,
                obs_type=obs_type,
                repeat_action_probability=repeat_action_probability,
                frameskip=frameskip,
                output_size=output_size,
                feature_shape=feature_shape,
                episodes_per_individual=episodes_per_individual,
                max_steps_per_episode=max_steps_per_episode,
                args=args,
            )
            for i in range(len(solutions))
        ]

        costs = [None] * len(solutions)
        avg_rewards = [0.0] * len(solutions)
        for indiv_idx, avg_reward, rows in results:
            fitness_log.extend(rows)
            costs[indiv_idx] = -float(avg_reward)
            avg_rewards[indiv_idx] = float(avg_reward)

        es.tell(solutions, costs)

        best_idx = int(np.argmin(costs))
        best_val = float(avg_rewards[best_idx])
        avg_val = float(np.mean(avg_rewards))

        if best_val > best_fitness_so_far:
            best_fitness_so_far = best_val
            best_solution_so_far = np.asarray(solutions[best_idx]).copy()
            best_individuals.append(
                {
                    "generation": gen + 1,
                    "individual_index": best_idx + 1,
                    "fitness": best_val,
                    "solution": best_solution_so_far.tolist(),
                }
            )
            if verbosity >= 1:
                print(f"[Run {run_id}][GEN {gen+1}] NEW GLOBAL BEST: {best_val:.2f}")

        if verbosity >= 1:
            print(f"[Run {run_id}][GEN {gen+1}] BEST: {best_val:.2f}  AVG: {avg_val:.2f}")

        plot_data.append([float(gen + 1), float(best_val), float(avg_val)])

    return {
        "run_id": int(run_id),
        "args": args,
        "env_name": env_name,
        "output_size": output_size,
        "feature_shape": list(feature_shape),
        "chromosome_size": int(chromosome_size),
        "fitness_log": fitness_log,
        "best_individuals": best_individuals,
        "plot_data": plot_data,
        "best_fitness": float(best_fitness_so_far),
        "best_solution": (best_solution_so_far.tolist() if best_solution_so_far is not None else None),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a single EvoAtariBench pipeline task (local debug mode).")
    parser.add_argument(
        "--args-json",
        required=True,
        help="Either a JSON string or a path to a JSON file containing the args dict.",
    )
    args = parser.parse_args()

    raw = args.args_json
    if os.path.exists(raw):
        with open(raw, "r") as f:
            args_dict = json.load(f)
    else:
        args_dict = json.loads(raw)

    result = _run_task_impl(args_dict, run_id=1)
    print(json.dumps({"run_id": result["run_id"], "best_fitness": result["best_fitness"]}, indent=2))


if __name__ == "__main__":
    main()

