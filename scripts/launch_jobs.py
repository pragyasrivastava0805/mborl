"""Job launcher for Unifloral policy-pool experiments.

Builds a deterministic job list (identical on every VM) and runs it on the local GPUs,
several jobs per GPU. Re-running the same command skips jobs that already finished.

Groups:
  dynamics     1 dynamics model per dataset (20 jobs) -- run before model_based / all
  model_free   8 algorithms x 20 datasets x 20 policies = 3200 jobs
  model_based  4 algorithms x 20 datasets x 20 policies = 1600 jobs
  all          model_free + model_based in one queue (4800 jobs)

By default each pool has 20 policies; use --pool-size 10 for the initial campaign.
Policy i uses seed i and hyperparameters
sampled uniformly from that algorithm's sweep config (values lists in the yaml).

Usage (see launcher.md):
  python scripts/launch_jobs.py --group all --shard 0 --num-shards 3 --dry-run
  python scripts/launch_jobs.py --group all --shard 0 --num-shards 3
"""

import argparse
import ast
import glob
import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _loader_datasets():
    """Dataset names from loader.SOURCES (same list as evaluate.ipynb), read without
    importing loader so the launcher doesn't need h5py/tqdm."""
    with open(os.path.join(ROOT, "loader.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError("SOURCES not found in loader.py")


DATASETS = _loader_datasets()
POOL_SIZE = 20

# algorithm -> sweep config. Unified configs where they sweep hyperparameters; standalone
# IQL because unifloral/iql.yaml sweeps nothing but the seed.
MODEL_FREE = {
    "bc": "configs/unifloral/bc.yaml",
    "iql": "configs/algorithms/iql.yaml",
    "td3_bc": "configs/unifloral/td3_bc.yaml",
    "cql": "configs/algorithms/cql.yaml",
    "sac_n": "configs/unifloral/sac_n.yaml",
    "edac": "configs/unifloral/edac.yaml",
    "rebrac": "configs/unifloral/rebrac.yaml",
    "td3_awr": "configs/unifloral/td3_awr.yaml",
}
MODEL_BASED = {
    "mopo": "configs/algorithms/mopo.yaml",
    "morel": "configs/algorithms/morel.yaml",
    "combo": "configs/algorithms/combo.yaml",
    "mobrac": "configs/unifloral/mobrac.yaml",
}
DYNAMICS_CONFIG = "configs/dynamics.yaml"

# Rough A100 hours per run, only used to start long jobs first and balance shards.
BASE_COST = {
    "bc": 0.3, "iql": 0.6, "td3_bc": 0.6, "rebrac": 0.8, "td3_awr": 0.9, "cql": 1.75,
    "sac_n": 3.0, "edac": 3.0, "mopo": 6.0, "morel": 6.0, "combo": 6.0, "mobrac": 2.0,
    "dynamics": 0.75,
}

# Keys the launcher sets itself rather than taking from the yaml.
OVERRIDDEN = {"seed", "dataset", "model_path", "log", "wandb_project", "wandb_team", "wandb_group"}


def stable_rng(*parts):
    """RNG seeded identically on every machine (Python's hash() is salted per process)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def load_sweep(path):
    with open(os.path.join(ROOT, path)) as f:
        cfg = yaml.safe_load(f)
    return cfg["program"], cfg["parameters"]


def sample_params(params, rng):
    """Fixed values pass through; `values` lists are sampled uniformly."""
    out = {}
    for key, spec in params.items():
        if key in OVERRIDDEN:
            continue
        if "values" in spec:
            choices = spec["values"]
        elif isinstance(spec.get("value"), list):
            # mobrac.yaml writes `step_penalty_coef: value: [...]`; treat it as a sweep list
            choices = spec["value"]
        else:
            out[key] = spec["value"]
            continue
        out[key] = choices[rng.integers(len(choices))]
    return out


def job_cost(algo, params):
    cost = BASE_COST[algo]
    if algo in ("sac_n", "edac"):
        cost *= max(1.0, float(params.get("num_critics", 10)) / 10)
    if algo == "combo":
        cost *= max(1.0, float(params.get("rollout_length", 5)) / 5)
    return cost


def build_jobs(group, pool_size=POOL_SIZE, seed_start=0):
    if pool_size < 1 or seed_start < 0:
        raise ValueError("pool_size must be positive and seed_start nonnegative")
    jobs = []
    if group == "dynamics":
        program, params = load_sweep(DYNAMICS_CONFIG)
        for ds in DATASETS:
            p = sample_params(params, stable_rng("dynamics", ds))
            p.update(seed=0, dataset=ds, model_path="dynamics_models")
            jobs.append(dict(id=f"dynamics__{ds}", algo="dynamics", dataset=ds,
                             program=program, params=p, cost=BASE_COST["dynamics"]))
        return jobs
    if group == "all":
        return (build_jobs("model_based", pool_size, seed_start)
                + build_jobs("model_free", pool_size, seed_start))

    algos = MODEL_FREE if group == "model_free" else MODEL_BASED
    for algo, cfg in algos.items():
        program, params = load_sweep(cfg)
        for ds in DATASETS:
            for i in range(seed_start, seed_start + pool_size):
                p = sample_params(params, stable_rng(algo, ds, i))
                p.update(seed=i, dataset=ds)
                if algos is MODEL_BASED:
                    p["model_path"] = None  # resolved when the job starts
                jobs.append(dict(id=f"{algo}__{ds}__{i:02d}", algo=algo, dataset=ds,
                                 program=program, params=p, cost=job_cost(algo, p)))
    return jobs


def select_shard(jobs, shard, num_shards):
    """Longest jobs first, dealt round-robin so every shard gets a similar mix."""
    jobs = sorted(jobs, key=lambda j: (-j["cost"], j["id"]))
    return [j for k, j in enumerate(jobs) if k % num_shards == shard]


def select_algorithms(jobs, algorithms):
    """Filter before sharding so separate hardware queues never overlap."""
    if algorithms is None:
        return jobs
    return [j for j in jobs if j["algo"] in algorithms]


def gpu_devices(count):
    """Preserve Slurm's assigned GPU indices or UUIDs in child processes."""
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = assigned.split(",") if assigned is not None else list(map(str, range(count)))
    devices = [d.strip() for d in devices if d.strip() and d.strip() != "-1"]
    if len(devices) < count:
        raise ValueError(f"Requested {count} GPUs but CUDA_VISIBLE_DEVICES exposes {devices}")
    return devices[:count]


def resolve_model_path(dataset, model_dir):
    matches = sorted(glob.glob(os.path.join(
        model_dir, f"ensemble_dynamics_model_{dataset}_*.pkl")))
    if not matches:
        raise FileNotFoundError(
            f"No dynamics model for {dataset} in {model_dir}; run --group dynamics first")
    return matches[-1]  # newest (timestamped filenames sort chronologically)


def to_cli(params):
    args = []
    for key, val in params.items():
        if isinstance(val, bool):
            args.append(f"--{key}" if val else f"--no-{key}")
        else:
            args += [f"--{key}", str(val)]
    return args


def make_command(job, cli):
    params = dict(job["params"])
    if params.get("model_path", "") is None:
        params["model_path"] = resolve_model_path(
            job["dataset"], os.path.join(cli.work_dir, cli.model_dir))
    if cli.eval_workers is not None and "eval_workers" in params:
        params["eval_workers"] = cli.eval_workers
    if cli.eval_interval is not None and job["algo"] != "dynamics":
        params["eval_interval"] = cli.eval_interval
    if cli.smoke:
        if job["algo"] == "dynamics":
            params["num_epochs"] = 2
        else:
            params.update(num_updates=5000, eval_interval=2500, eval_final_episodes=16)
    params["log"] = cli.wandb
    if cli.wandb:
        params.update(wandb_project=cli.wandb_project, wandb_team=cli.wandb_entity,
                      wandb_group=cli.wandb_group)
    return [sys.executable, os.path.join(ROOT, job["program"])] + to_cli(params)


def run(jobs, cli):
    devices = gpu_devices(cli.gpus)
    # Jobs write final_returns/ and dynamics_models/ into their working directory
    log_dir = os.path.join(cli.work_dir, cli.log_dir)
    done_dir, fail_dir = os.path.join(log_dir, "done"), os.path.join(log_dir, "failed")
    for d in (done_dir, fail_dir, os.path.join(log_dir, "jobs")):
        os.makedirs(d, exist_ok=True)

    todo = [j for j in jobs if not os.path.exists(os.path.join(done_dir, j["id"]))]
    print(f"group={cli.group} shard={cli.shard}/{cli.num_shards}: "
          f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run on {cli.gpus} GPUs x {cli.per_gpu} slots", flush=True)

    q = queue.Queue()
    for j in todo:
        q.put(j)
    lock = threading.Lock()
    stats = Counter()

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            log_path = os.path.join(log_dir, "jobs", f"{job['id']}.log")
            try:
                cmd = make_command(job, cli)
            except FileNotFoundError as e:
                with lock:
                    stats["failed"] += 1
                    print(f"[FAIL] {job['id']}: {e}", flush=True)
                open(os.path.join(fail_dir, job["id"]), "w").write(str(e))
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu],
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            start = time.time()
            with open(log_path, "w") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.flush()
                code = subprocess.call(cmd, cwd=cli.work_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            hours = (time.time() - start) / 3600
            with lock:
                if code == 0:
                    stats["done"] += 1
                    open(os.path.join(done_dir, job["id"]), "w").write(f"{hours:.3f}h\n")
                    fail_marker = os.path.join(fail_dir, job["id"])
                    if os.path.exists(fail_marker):
                        os.remove(fail_marker)
                else:
                    stats["failed"] += 1
                    open(os.path.join(fail_dir, job["id"]), "w").write(f"exit {code}\n")
                print(f"[{'OK' if code == 0 else 'FAIL'}] gpu{gpu} {job['id']} "
                      f"{hours:.2f}h | done {stats['done']} failed {stats['failed']} "
                      f"left {q.qsize()}", flush=True)

    threads = [threading.Thread(target=worker, args=(s % cli.gpus,))
               for s in range(cli.gpus * cli.per_gpu)]
    for t in threads:
        t.start()
        time.sleep(2)  # stagger start-up so dataset loading doesn't spike at once
    for t in threads:
        t.join()
    print(f"Finished: {stats['done']} ok, {stats['failed']} failed "
          f"(see {log_dir}/failed and {log_dir}/jobs)", flush=True)
    return stats["failed"]


def prefetch():
    """Download every D4RL dataset once, sequentially, before parallel jobs start."""
    import gym
    import d4rl

    for ds in DATASETS:
        print(f"Fetching {ds}", flush=True)
        d4rl.qlearning_dataset(gym.make(ds))
    print("All datasets cached.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["dynamics", "model_free", "model_based", "all"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE,
                    help="policies per method/dataset (default: 20; initial campaign: 10)")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first policy seed; use 10 with --pool-size 10 to add seeds 10-19")
    ap.add_argument("--algorithms", nargs="+", choices=sorted(MODEL_FREE | MODEL_BASED),
                    help="filter methods before sharding; use launcher method keys")
    ap.add_argument("--work-dir", default=ROOT,
                    help="output directory (separate directories for Slurm shards)")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="override eval_workers (CPU processes per run)")
    ap.add_argument("--eval-interval", type=int, default=None,
                    help="override intermediate evaluation frequency; changes training RNG sequence")
    ap.add_argument("--model-dir", default="dynamics_models")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--wandb", action="store_true", help="log runs to Weights & Biases")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-project", default="unifloral")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--smoke", action="store_true",
                    help="1 short job per algorithm on hopper-medium-v2 to test the setup; "
                         "outputs go to smoke_test/, not final_returns/")
    ap.add_argument("--prefetch", action="store_true", help="download all datasets and exit")
    cli = ap.parse_args()
    if cli.pool_size < 1 or cli.seed_start < 0:
        ap.error("--pool-size must be positive and --seed-start nonnegative")
    if cli.wandb_group is None:
        cli.wandb_group = f"pool{cli.pool_size}-seeds{cli.seed_start}-{cli.seed_start + cli.pool_size - 1}"
    if cli.num_shards < 1 or not 0 <= cli.shard < cli.num_shards:
        ap.error("require 0 <= --shard < --num-shards")
    if cli.gpus < 1 or cli.per_gpu < 1:
        ap.error("--gpus and --per-gpu must be positive")
    if cli.eval_workers is not None and cli.eval_workers < 1:
        ap.error("--eval-workers must be positive")
    if cli.eval_interval is not None and cli.eval_interval < 1:
        ap.error("--eval-interval must be positive")
    if cli.group == "dynamics" and cli.algorithms:
        ap.error("--algorithms applies to policy jobs, not dynamics")

    if cli.prefetch:
        return prefetch()
    if cli.group is None:
        ap.error("--group is required")
    if cli.wandb and not cli.wandb_entity:
        ap.error("--wandb needs --wandb-entity")

    cli.work_dir = os.path.abspath(cli.work_dir)
    if cli.smoke:
        cli.work_dir = os.path.join(cli.work_dir, "smoke_test")
    os.makedirs(cli.work_dir, exist_ok=True)
    jobs = select_algorithms(build_jobs(cli.group, cli.pool_size, cli.seed_start), cli.algorithms)
    if not jobs:
        ap.error("no jobs match --group and --algorithms")
    if cli.smoke:
        seen, smoke = set(), []
        for j in jobs:
            if j["dataset"] == "hopper-medium-v2" and j["algo"] not in seen:
                seen.add(j["algo"])
                smoke.append(j)
        jobs = smoke
    else:
        jobs = select_shard(jobs, cli.shard, cli.num_shards)

    if cli.dry_run:
        per_algo = Counter(j["algo"] for j in jobs)
        print(f"group={cli.group} shard={cli.shard}/{cli.num_shards}: {len(jobs)} jobs, "
              f"~{sum(j['cost'] for j in jobs):.0f} est. GPU-hours")
        for algo, n in sorted(per_algo.items()):
            print(f"  {algo:10s} {n}")
        for j in jobs[:3]:
            p = dict(j["params"])
            if p.get("model_path", "") is None:
                p["model_path"] = f"<{cli.model_dir}/ensemble_dynamics_model_{j['dataset']}_*.pkl>"
            print(" ", j["id"], "->", " ".join([j["program"]] + to_cli(p))[:300], "...")
        return 0

    return 1 if run(jobs, cli) else 0


if __name__ == "__main__":
    sys.exit(main())
