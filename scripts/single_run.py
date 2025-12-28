"""
Script for running a single task on a Ray worker
"""

import os
import sys
import concurrent.futures as cf
import multiprocessing as mp
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
    """Create an atari environment and redirect the stdout and stderr to /dev/null to avoid the clutter."""
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


def normalize_frame(frame):
    """Normalize the frame to be in the range [0, 1]."""
    x = np.asarray(frame)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x / 255.0


def process_features(x):
    """Process the output of the compression function"""
    if isinstance(x, tuple) and len(x) == 2:  # If the input is a tuple with two elements, use the first element
        x = x[0]
    x = np.asarray(x)  # Convert the input to a numpy array
    if x.ndim != 2:
        raise ValueError(f"Compression output must be 2D (H,W); got shape={x.shape}")
    return x.astype(np.float32, copy=False)


def compute_chromosome_size(feature_shape, output_size):
    """Compute the size of the chromosome."""
    m, n = int(feature_shape[0]), int(feature_shape[1])  # Get the number of rows and columns of the feature shape
    return m + (n * int(output_size)) + int(output_size)


class EvoAtariPipelinePolicy:
    """Class for the EvoAtariPipelinePolicy."""
    def __init__(self, chromosome, output_size, feature_shape, args):
        """Initialize the EvoAtariPipelinePolicy."""
        self.output_size = int(output_size)  # Set the output size
        self.feature_shape = (int(feature_shape[0]), int(feature_shape[1]))  # Set the feature shape
        self.args = args  # Set the arguments

        self._compression_fn = cm.get_compression_method(args["compression"])  # Set the compression function
        self._nonlin_fn = nm.get_nonlinearity_method(args.get("nonlinearity", None))  # Set the nonlinearity function
        self._process_chromosome(chromosome)  # Process the chromosome

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
        x = normalize_frame(obs)
        feats = self._compression_fn(x, self.args)
        feats = process_features(feats)

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
    """Evaluate the individual."""
    # Create the policy
    policy = EvoAtariPipelinePolicy(
        chromosome=solution,
        output_size=output_size,
        feature_shape=feature_shape,
        args=args,
    )

    # Create the environment
    env = make_silent_env(
        env_name=env_name,
        obs_type=obs_type,
        repeat_action_probability=repeat_action_probability,
        frameskip=frameskip,
    )

    # Evaluate the individual
    total_reward = 0.0
    rows = []

    for ep in range(int(episodes_per_individual)):  # For each episode
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < int(max_steps_per_episode):  # While the episode is not done and the steps are less than the maximum steps per episode
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


def _evaluate_generation_parallel(
    *,
    solutions,
    gen,
    env_name,
    obs_type,
    repeat_action_probability,
    frameskip,
    output_size,
    feature_shape,
    episodes_per_individual,
    max_steps_per_episode,
    args,
    max_workers,
):
    """
    Evaluate a CMA-ES generation in parallel (one individual per worker process).
    Returns the same shape as the sequential path: list[(indiv_idx, avg_score, rows)].
    """
    if max_workers <= 1 or len(solutions) <= 1:
        return [
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

    def _mp_context_for_platform():
        if sys.platform.startswith("win") or sys.platform == "darwin":
            return mp.get_context("spawn")
        return mp.get_context()

    ctx = _mp_context_for_platform()
    workers = int(min(max_workers, len(solutions)))
    results = [None] * len(solutions)

    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        fut_to_idx = {}
        for i in range(len(solutions)):
            fut = ex.submit(
                _evaluate_individual,
                solutions[i],
                i,
                gen,
                env_name,
                obs_type,
                repeat_action_probability,
                frameskip,
                output_size,
                feature_shape,
                episodes_per_individual,
                max_steps_per_episode,
                args,
            )
            fut_to_idx[fut] = i

        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    return results


@ray.remote
def run_task_remote(args, run_id):
    """Run the task on a remote Ray worker."""
    return run_task_local(args, run_id)


def run_task_local(args, run_id):
    """Run the task on a local machine."""
    def get_key(key):
        if key not in args or args[key] is None:
            raise ValueError(f"Missing required argument {key!r} (must be passed down; got None/missing).")
        return args[key]

    env_name = get_key("ENV_NAME")
    obs_type = get_key("OBS_TYPE")
    frameskip = int(get_key("FRAMESKIP"))
    repeat_action_probability = float(get_key("REPEAT_ACTION_PROBABILITY"))

    generations = int(get_key("GENERATIONS"))
    cma_sigma = float(get_key("CMA_SIGMA"))
    episodes_per_individual = int(get_key("EPISODES_PER_INDIVIDUAL"))
    max_steps_per_episode = int(get_key("MAX_STEPS_PER_EPISODE"))
    verbosity = int(get_key("VERBOSITY_LEVEL"))
    population_size = args.get("POPULATION_SIZE", None)
    cores_per_task = int(args.get("CORES_PER_TASK", 1))
    if verbosity >= 1:
        print(f"[Run {run_id}] ENV={env_name} compression={args.get('compression')} nonlinearity={args.get('nonlinearity')}")

    # Create a temporary environment to get the output size
    temp_env = make_silent_env(
        env_name=env_name,
        obs_type=obs_type,
        repeat_action_probability=repeat_action_probability,
        frameskip=frameskip,
    )
    output_size = int(temp_env.action_space.n)
    obs0, _ = temp_env.reset()
    temp_env.close()

    # Get the compression function and do a forward pass to get the feature shape
    comp_fn = cm.get_compression_method(args["compression"])
    feats0 = process_features(comp_fn(normalize_frame(obs0), args))
    feature_shape = (int(feats0.shape[0]), int(feats0.shape[1]))
    chromosome_size = compute_chromosome_size(feature_shape, output_size)

    # Pure-random init (no explicit seeding).
    x0 = np.random.default_rng().normal(size=chromosome_size)
    inopts = {}
    if population_size is not None:
        inopts["popsize"] = int(population_size)
    es = cma.CMAEvolutionStrategy(x0, cma_sigma, inopts)

    fitness_log = []
    best_individuals = []
    plot_data = []

    best_fitness_so_far = float("-inf")
    best_solution_so_far = None

    for gen in range(generations):
        solutions = es.ask()  # Ask the CMA-ES for new solutions in each generation
        results = _evaluate_generation_parallel(
            solutions=solutions,
            gen=gen,
            env_name=env_name,
            obs_type=obs_type,
            repeat_action_probability=repeat_action_probability,
            frameskip=frameskip,
            output_size=output_size,
            feature_shape=feature_shape,
            episodes_per_individual=episodes_per_individual,
            max_steps_per_episode=max_steps_per_episode,
            args=args,
            max_workers=cores_per_task,
        )

        fitness_vals = [None] * len(solutions)
        avg_scores = [0.0] * len(solutions)
        for indiv_idx, avg_score, rows in results:
            fitness_log.extend(rows)
            fitness_vals[indiv_idx] = -float(avg_score)  # Because CMA-ES minimizes the fitness function we need to negate the average score
            avg_scores[indiv_idx] = float(avg_score)

        es.tell(solutions, fitness_vals)

        best_idx = int(np.argmin(fitness_vals))
        best_val = float(avg_scores[best_idx])
        avg_val = float(np.mean(avg_scores))

        if best_val > best_fitness_so_far:  # If we have a new global best, update the best fitness and solution and log this
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

        plot_data.append([float(gen + 1), float(best_val), float(avg_val)])

        if verbosity >= 1:
            print(f"[Run {int(run_id):02d}] [GEN {int(gen + 1):04d}] [BEST {best_val:.2f}] [AVG {avg_val:.2f}]")

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
