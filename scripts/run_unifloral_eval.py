#!/usr/bin/env python3
"""Usage: python3 run_unifloral_eval.py <repo_dir> <results_dir> <out_json>

Runs:
1. The launcher.md Section 5 completeness & metadata check across all 2,400 .npz files.
2. The launcher.md Section 6 / evaluation_plots.ipynb JAX Unifloral UCB policy selection
   (num_subsample=8, num_repeats=500, n_bootstraps=2000, max_pulls=200, ucb_alpha=2.0, seed=17)
   for every (algorithm, dataset) pool (12 methods x 20 D4RL datasets).
3. Saves structured results to <out_json> and renders evaluation_plots.pdf / .png.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def main():
    t0 = time.time()
    repo_dir = os.path.expanduser(sys.argv[1])
    results_dir = os.path.expanduser(sys.argv[2])
    out_json = os.path.expanduser(sys.argv[3])

    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "scripts"))
    import evaluation
    from launch_jobs import build_jobs

    # 1. Completeness check from launcher.md Section 5
    expected = Counter(
        (j["params"]["algorithm"], j["dataset"], j["params"]["seed"])
        for j in build_jobs("all", pool_size=10)
    )
    actual = Counter()
    npz_paths = sorted(Path(results_dir).glob("*.npz"))
    for path in npz_paths:
        with np.load(path, allow_pickle=True) as result:
            args = result["args"].item()
            assert result["final_scores"].size == 1000, f"Incomplete final evaluation: {path}"
            assert args["eval_final_episodes"] == 1000 and args["eval_interval"] == 2500, path
            assert args["eval_workers"] == 4, f"Mismatched evaluation workers: {path}"
            algo_key = args.get("algorithm")
            if algo_key is None:
                # Extract canonical algorithm name from filename
                fname = path.name
                for a in [
                    "td3_bc", "td3_awr", "sac_n", "rebrac", "mobrac",
                    "morel", "combo", "edac", "mopo", "cql", "iql", "bc"
                ]:
                    if fname.startswith(f"unifloral_{a}_") or fname.startswith(f"{a}_"):
                        algo_key = a
                        break
            actual[(algo_key, args["dataset"], args["seed"])] += 1

    missing = expected - actual
    extra = actual - expected
    print(f"Files checked: {len(npz_paths)}")
    print(f"Missing: {missing}")
    print(f"Extra or duplicate: {extra}")
    assert actual == expected, "Resolve incomplete or duplicate runs before evaluation"
    print("Complete: 2,400 policies in 240 pools")

    # 2. Load dataframe via evaluation.load_results_dataframe
    df = evaluation.load_results_dataframe(results_dir=results_dir)
    print(f"Loaded DataFrame with {len(df)} rows across {df['algorithm'].nunique()} algorithms and {df['dataset'].nunique()} datasets.")

    # Map evaluation.py's parsed algorithm column to canonical short names
    def canonical_algo(raw_algo: str) -> str:
        return raw_algo.replace("unifloral_", "")

    df["canonical_algo"] = df["algorithm"].apply(canonical_algo)

    # 3. Run Unifloral UCB bandit policy selection for each (algorithm, dataset)
    # Settings from launcher.md Section 6 & evaluation_plots.ipynb:
    # num_subsample=8, num_repeats=500, n_bootstraps=2000, max_pulls=200, ucb_alpha=2.0, seed=17
    records = []
    grouped = df.groupby(["canonical_algo", "dataset"], sort=True)
    total_groups = len(grouped)
    for idx, ((algo, dataset), grp) in enumerate(grouped, 1):
        # Sort by seed to ensure deterministic ordering of the 10 policies in the pool
        if "seed" in grp.columns:
            grp = grp.sort_values("seed")
        runs = np.stack(grp["final_scores"].values)  # shape (10, 1000)
        assert runs.shape == (10, 1000), f"Unexpected shape for {algo} {dataset}: {runs.shape}"

        res = evaluation.bootstrap_bandit_trials(
            runs,
            seed=17,
            num_subsample=8,
            num_repeats=500,
            max_pulls=200,
            ucb_alpha=2.0,
            n_bootstraps=2000,
            confidence=0.95,
        )
        pulls = np.asarray(res["pulls"])
        mean_curve = np.asarray(res["estimated_bests_mean"])
        ci_low_curve = np.asarray(res["estimated_bests_ci_low"])
        ci_high_curve = np.asarray(res["estimated_bests_ci_high"])

        per_seed_means = runs.mean(axis=1)  # shape (10,)
        budget_snapshots = {}
        for k in [1, 5, 10, 20, 50, 100, 200]:
            b_idx = int(np.where(pulls == k)[0][0])
            budget_snapshots[str(k)] = {
                "mean": float(mean_curve[b_idx]),
                "ci_low": float(ci_low_curve[b_idx]),
                "ci_high": float(ci_high_curve[b_idx]),
            }

        records.append({
            "algorithm": algo,
            "raw_algorithm": str(grp["algorithm"].iloc[0]),
            "dataset": dataset,
            "num_policies": int(runs.shape[0]),
            "ucb200_mean": float(mean_curve[-1]),
            "ucb200_ci_low": float(ci_low_curve[-1]),
            "ucb200_ci_high": float(ci_high_curve[-1]),
            "ucb200_ci_half": float((ci_high_curve[-1] - ci_low_curve[-1]) / 2.0),
            "ucb1_mean": float(mean_curve[0]),
            "pool_mean": float(per_seed_means.mean()),
            "pool_std": float(per_seed_means.std(ddof=1)),
            "pool_min": float(per_seed_means.min()),
            "pool_max_oracle": float(per_seed_means.max()),
            "per_seed_means": [float(x) for x in per_seed_means],
            "budgets": budget_snapshots,
        })
        if idx % 24 == 0 or idx == total_groups:
            print(f"  Evaluated {idx}/{total_groups} pools ({time.time() - t0:.1f}s elapsed)")

    out_data = {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "pool_size": 10,
            "num_subsample": 8,
            "num_repeats": 500,
            "n_bootstraps": 2000,
            "max_pulls": 200,
            "ucb_alpha": 2.0,
            "seed": 17,
            "confidence": 0.95,
            "eval_final_episodes": 1000,
            "eval_interval": 2500,
            "eval_workers": 4,
        },
        "num_pools": len(records),
        "results": records,
    }
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved {len(records)} pool results to {out_json} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
