"""
Main script for assigning tasks to a Ray cluster.
"""

import os
import sys
import json
import sqlite3
import argparse
import itertools
import ray

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Add the repository root to the Python path
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.evo_train_task import run_task  # This is the function to run a single task on a Ray cluster

# Default arguments for the tasks.
# These are the common arguments for all tasks.
DEFAULT_ARGS = {
    "ENV_NAME": "ALE/SpaceInvaders-v5",
    "OBS_TYPE": "grayscale",
    "FRAMESKIP": 4,
    "REPEAT_ACTION_PROBABILITY": 0.0,
    "MAX_STEPS_PER_EPISODE": 10000,
    "EPISODES_PER_INDIVIDUAL": 1,
    "CMA_SIGMA": 0.5,
    "POPULATION_SIZE": None,
    "GENERATIONS": 3,
    "VERBOSITY_LEVEL": 1,
    "CORES_PER_TASK": 1
}

RAY_HEAD_IP = "136.244.224.234"
RAY_HEAD_PORT = 6379
RAY_HEAD = f"{RAY_HEAD_IP}:{RAY_HEAD_PORT}"

# List for different compression methods to sweep through. 
COMPRESSION_SWEEP = [
    {"compression": "dct", "k": [142], "norm": ["ortho"]},
]

# List of the different non-linearity methods to sweep through.
NONLINEARITY_SWEEP = [
    {"nonlinearity": "sparsification", "percentile": [91.0]},
    {"nonlinearity": "quantization", "num_levels": [125]},
    {"nonlinearity": "dropout_regularization", "rate": [0.18], "seed": [42]},
]


def process_dict(d):
    """
    Process the dictionary of compression and non-linearity methods.
    In any of the dictionaries, if we have a constant value, it is directly added to the output.
    If we have a list, it is used to create a group of values for the same key.
    """
    fixed = {}
    grid_keys = []
    grid_vals = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):  # If it is a group of values, add it to grid_keys and grid_vals.
            grid_keys.append(k)  # Add the key to the list of keys to be used for the grid.
            grid_vals.append(list(v))  # Add the values to the list of values to be used for the grid.
        else:
            fixed[k] = v  # If it is a constant value, just add it to the fixed dictionary.

    if not grid_keys:  # If there are no keys to go over, just return the fixed dictionary
        return [fixed]

    tasks = []  # List of the tasks to go through.
    for combo in itertools.product(*grid_vals):  # For each  combination of values, create a new task
        task = dict(fixed)
        for k, val in zip(grid_keys, combo):  # Merge the keys to values
            task[k] = val
        tasks.append(task)
    return tasks


def build_tasks():
    """Function for building the tasks from the compression and nonlinearity informations."""
    comps = []  # List of the compression methods to go through. It unpacks the compression sweep into a list of dictionaries.
    for base in COMPRESSION_SWEEP:
        comps.extend(process_dict(base))

    nonlins = []  # List of the nonlinearity methods to go through. It unpacks the nonlinearity sweep into a list of dictionaries.
    for base in NONLINEARITY_SWEEP:
        nonlins.extend(process_dict(base))

    tasks = []
    for c in comps:
        for n in nonlins:
            # For each compression and nonlinearity method, create a new task.
            args = dict(DEFAULT_ARGS)  # Start with the default arguments.
            args.update(c)  # Update the arguments with the compression and nonlinearity methods.
            args.update(n)
            tasks.append(args)  # Add the new task to the list of tasks.
    return tasks


def check_db(db_path):
    """Check that the database and table exist."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER NOT NULL,
            env_name         TEXT    NOT NULL,
            task_json        TEXT    NOT NULL,
            best_fitness     REAL    NOT NULL,
            best_solution_json   TEXT,
            plot_data_json       TEXT NOT NULL,
            best_individuals_json TEXT NOT NULL,
            fitness_log_json     TEXT,
            inserted_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Dispatch EvoAtariBench training tasks to a Ray cluster.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded task list (preview) and exit without running Ray.")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to SQLite; only print progress.")
    return parser.parse_args()


def main():
    args = get_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of the repository
    data_dir = os.path.join(repo_root, "data")  # Data directory
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "evo_train_runs.db")  # Path to the SQLite database

    tasks = build_tasks()  # Build the tasks from the config dictionaries
    if args.dry_run:  # If we are in the dry-run mode, print the tasks and exit
        for i, t in enumerate(tasks, start=1):
            print(json.dumps({"run_id": i, "args": t}, sort_keys=True))
        return

    ray.init(  # Initialize the Ray cluster
        address=RAY_HEAD,  # The address of the Ray head
        ignore_reinit_error=True,  # Ignore the error if the Ray cluster is already initialized
        runtime_env={
            "working_dir": repo_root,  # The working directory of the ray cluster
            "excludes": [  # Remove all the extra folders
                "data/**",
                "temp/**",
                "__pycache__/**",
                "tests/**",
                ".gitignore",
                "environment.yml",
                "requirements.txt",
                "test_image.jpg",
            ],
        },
    )

    conn = None
    cursor = None
    if not args.no_save:  # If we are not in the no-save mode, connect to the database
        conn = check_db(db_path)
        cursor = conn.cursor()

    ray_tasks = []
    for i, args_dict in enumerate(tasks, start=1):
        cores = int(args_dict.get("CORES_PER_TASK", 1))
        ray_tasks.append(run_task.options(num_cpus=cores).remote(args_dict, run_id=i))

    remaining = set(ray_tasks)  # Set of the remaining tasks to be completed
    while remaining:  # While there are remaining tasks to be completed
        done, remaining = ray.wait(list(remaining), num_returns=1, timeout=None)
        finished_ref = done[0]
        result = ray.get(finished_ref)

        rid = int(result["run_id"])
        env_name = str(result["env_name"])
        best_fitness = float(result["best_fitness"])
        print(f"⇢ Finished run_id={rid} env={env_name} best_fitness={best_fitness:.2f}")

        if cursor is not None and conn is not None:
            task_json = json.dumps(result["args"])
            plot_data_json = json.dumps(result["plot_data"])
            best_individuals_json = json.dumps(result["best_individuals"])
            best_solution_json = json.dumps(result["best_solution"]) if result.get("best_solution") is not None else None
            fitness_log_json = json.dumps(result["fitness_log"])

            cursor.execute(
                """
                INSERT INTO runs
                  (run_id, env_name, task_json, best_fitness, best_solution_json, plot_data_json, best_individuals_json, fitness_log_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (rid, env_name, task_json, best_fitness, best_solution_json, plot_data_json, best_individuals_json, fitness_log_json),
            )
            conn.commit()

    # Close the connections to the database
    if not args.no_save:
        cursor.close()
        conn.close()

    ray.shutdown()


if __name__ == "__main__":
    main()
