#!/usr/bin/env python3
"""Build the complete 5-seed (seeds 0..4) x 25-task 1,000,000-step OGBench stitch-v0 queue (1,475 runs).

Ordering:
  - Phase 1 (Indices   0 ..  74,  75 runs): Seed 0, task1 (11 single-task baselines) + stitch_gc (4 GC baselines)
  - Phase 2 (Indices  75 .. 294, 220 runs): Seed 0, task2..task5 (11 single-task baselines, morel/reverse_morel* first)
  - Phase 3 (Indices 295 ..1474,1180 runs): Seeds 1, 2, 3, 4 across all tasks (295 runs/seed, morel/reverse_morel* first)
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERED_BASELINES = [
    "reverse_morel_anchor_bootstrap", "reverse_morel", "morel",
    "iql", "nstep_iql_n5", "nstep_iql_n15", "gciql", "qrl", "crl", "hiql",
    "rebrac", "mbpo", "mopo", "cql", "combo",
]


def generate_commands(suite, seeds, tmp_dir):
    cmd = [
        sys.executable,
        os.path.join(REPO, "scripts", "launch_baselines.py"),
        f"--suite={suite}",
        "--seeds", *[str(s) for s in seeds],
        "--offline_steps=1000000",
        "--eval_interval=100000",
        "--save_interval=100000",
        "--eval_episodes=15",
        "--job_prefix=base1m",
        "--save_dir=/home/rchitale_google_com/mborl-baselines/exp_baselines_1m",
        f"--queue_dir={tmp_dir}",
        "--baselines", *ORDERED_BASELINES,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(os.path.join(tmp_dir, "queue.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "artifacts", "runs", "baselines-stitch-1m-s0")
    os.makedirs(out_dir, exist_ok=True)
    tmp_root = "/tmp/mborl_queue_build"
    os.makedirs(tmp_root, exist_ok=True)

    # Seed 0: Phase 1 (task1 + GC = 75) + Phase 2 (task2..5 = 220)
    s0_p1 = generate_commands("stitch_task1", [0], os.path.join(tmp_root, "s0_p1"))
    s0_all = generate_commands("stitch_singletask", [0], os.path.join(tmp_root, "s0_all"))
    s0_p1_set = set(s0_p1)
    s0_p2 = [c for c in s0_all if c not in s0_p1_set]

    # Seeds 1..4: For each seed, run task1 + GC (75) then task2..5 (220) = 295 runs/seed x 4 seeds = 1,180 runs
    s1to4_cmds = []
    for s in [1, 2, 3, 4]:
        sp1 = generate_commands("stitch_task1", [s], os.path.join(tmp_root, f"s{s}_p1"))
        sall = generate_commands("stitch_singletask", [s], os.path.join(tmp_root, f"s{s}_all"))
        sp1_set = set(sp1)
        sp2 = [c for c in sall if c not in sp1_set]
        s1to4_cmds.extend(sp1 + sp2)

    full_queue = s0_p1 + s0_p2 + s1to4_cmds
    q_path = os.path.join(out_dir, "queue.txt")
    with open(q_path, "w") as f:
        f.write("\n".join(full_queue) + "\n")
    print(f"Wrote {len(full_queue)} total 1M-step commands (s0={len(s0_p1)+len(s0_p2)}, s1..4={len(s1to4_cmds)}) to {q_path}")


if __name__ == "__main__":
    main()
