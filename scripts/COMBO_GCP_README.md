# Running the COMBO ablation on a GCP VM (8x H100)

This guide runs the COMBO ablation sweep on one VM with 8 GPUs. You don't need Slurm. A small Python runner,
`scripts/run_combo_node.py`, starts one training run per GPU and starts the next run in the queue whenever a GPU
frees up. Every run saves checkpoints and picks up where it left off, so you can stop and restart it at any time.

## What the sweep is

There are 225 runs: 5 tasks × 15 configs × 3 seeds. Each run trains COMBO for 1M steps (`main_mbpo.py`).

- **Tasks:** task 1 of `cube-single`, `cube-double`, `scene`, `puzzle-3x3` and `puzzle-4x4` (OGBench single-task,
  1M-transition `play` datasets).
- **Configs:** each factor is varied on its own around a base config of beta=1, real-data fraction f=0.5 and rollout
  horizon H=5:
  - beta ∈ {0, 0.1, 0.5, 1, 2.5, 5}
  - f ∈ {0.25, 0.8, 1}
  - H ∈ {1, 10, 25}
  - CQL controls (f=1) with beta ∈ {0.5, 2.5, 5}
- **Seeds:** 0, 1, 2. The queue is seed-major: every config × task finishes seed 0 before any seed 1 run starts. If you
  stop early, you still have complete seeds.

A run takes about 11 hours on our cluster GPUs, so 225 runs on 8 GPUs take roughly 12 days. H100s should be faster.
You'll know after the first batch of 8 finishes.

## 1. Set up the environment

You need Python 3.10. Here are the versions we run with:

```bash
sudo apt-get install -y libegl1 libgl1 tmux     # MuJoCo renders headless through EGL

git clone -b combo-runs git@github.com:pragyasrivastava0805/mborl.git
cd mborl

python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "jax[cuda12]==0.6.2" flax==0.10.7 distrax==0.1.5 optax==0.2.8 chex==0.1.90 \
    tensorflow-probability==0.25.0 ogbench==1.2.1 mujoco==3.15.0 numpy==2.2.6 ml-collections==1.1.0 \
    wandb==0.19.11 matplotlib moviepy tqdm
```

Check that JAX can see all 8 GPUs:

```bash
.venv/bin/python -c "import jax; print(jax.devices())"    # should list 8 CudaDevice entries
```

## 2. Download the datasets (about 1.4 GB)

Each dataset goes in its own directory. `main_mbpo.py` loads every non-`-val` `.npz` file in `--dataset_dir`:

```bash
export DATA_ROOT=$HOME/ogbench/prepared
.venv/bin/python - <<'EOF'
import os, ogbench
root = os.environ['DATA_ROOT']
for name in ['cube-single-play-v0', 'cube-double-play-v0', 'scene-play-v0', 'puzzle-3x3-play-v0', 'puzzle-4x4-play-v0']:
    ogbench.download_datasets([name], dataset_dir=os.path.join(root, name))
EOF
ls $DATA_ROOT/*        # each directory holds <name>.npz and <name>-val.npz
```

## 3. Generate the run queue

Leave off `--submit`. Without it, the launcher only writes files: the queue (`queue.txt`, one shell command per run) and
`tasks.json`, which maps each queue index to its task, config and seed. Run it from the repo root:

```bash
.venv/bin/python scripts/launch_combo_ablation.py --suite mac_reward \
    --tasks cube-single-task1 cube-double-task1 scene-task1 puzzle-3x3-task1 puzzle-4x4-task1 \
    --one_at_a_time --betas 0 0.1 0.5 1 2.5 5 --fs 0.25 0.8 1 --horizons 1 10 25 \
    --add_configs 0.5,1,5 2.5,1,5 5,1,5 --seeds 0 1 2 \
    --node --data_root $DATA_ROOT --python .venv/bin/python
```

It should end with:

```
Node queue: 225 runs (5 tasks x 15 configs x 3 seeds), 1 chain(s) of 8 GPUs -> <repo>/artifacts/runs/slurm/combo-ablation-mac_reward-<timestamp>
Dry run. Submit with: sbatch ...
```

Ignore the `sbatch` line, since there's no Slurm here. Save the queue directory it printed:

```bash
export Q=$(ls -d $PWD/artifacts/runs/slurm/combo-ablation-mac_reward-* | tail -1)
```

## 4. Start the runs

Run the runner inside `tmux` so it keeps going after you disconnect:

```bash
tmux new -s combo
cd ~/mborl
export MUJOCO_GL=egl
.venv/bin/python scripts/run_combo_node.py $Q 2>&1 | tee -a $Q/logs/node.log
# detach with Ctrl-b d, reattach with: tmux attach -t combo
```

The runner uses GPUs 0–7 by default. To use only some of them, set `CUDA_VISIBLE_DEVICES=0,1,2,3` first. It splits
the VM's CPU cores evenly across the runs. When a run exits with an error, the runner retries it once, and gives up
after 2 failures (`--max_failures`).

## 5. Monitor progress

```bash
tail -f $Q/logs/node.log                  # START / DONE / FAIL lines from the runner
wc -l < $Q/done.txt                       # number of finished runs (out of 225)
cat $Q/failures.txt                       # queue indices that exited with an error (one line per failure)
tail -f $Q/logs/runs/00000.log            # full output of run 0 (the file name is the queue index)
nvidia-smi
```

To see which task, config and seed a queue index belongs to, look it up in `tasks.json`:

```bash
.venv/bin/python -c "import json,sys; t=json.load(open('$Q/tasks.json'))['tasks'][int(sys.argv[1])]; print({k: t[k] for k in ['task','beta','f','horizon','seed']})" 17
```

## 6. Stop and resume

- **Stop:** press Ctrl-C in the tmux window, or run `pkill -TERM -f run_combo_node.py`. The runner passes the signal on
  to every running job and does not count them as failures.
- **Resume:** run the command from step 4 again with the same `$Q`. Finished runs are skipped. Interrupted runs reload
  their latest checkpoint, which is saved every 100k steps, so at most 100k steps of progress are lost per run. Each
  queue line carries a fixed `--job_id`, and resuming relies on the files under `wandb_id/` in the repo root. Don't
  delete that directory while runs are still unfinished.
- **Retry failed runs** after fixing the cause: `rm $Q/failures.txt`, then resume.

If the VM is preemptible (Spot), the same steps apply after it comes back. Keep the repo on a persistent disk so
`artifacts/`, `exp/` and `wandb_id/` survive the restart.

## 7. Where results go

- **Metrics:** `exp/scalembrl/<run_group>/<exp_name>/` holds `eval.csv`, `train.csv`, `model.csv` (model pretraining
  and elite selection) and `flags.json`. `<run_group>` is `combo_ablation_<env_name>`. These are the files we need back.
- **Checkpoints:** the runner keeps only each run's latest checkpoint (about 0.5 GB) and deletes it when the run
  finishes. Peak disk use is a few GB.
- **W&B:** runs log in `offline` mode and need no login. To upload them, run `wandb login` once, then
  `.venv/bin/wandb sync $Q/tmp/*/*/wandb/offline-run-*`. The project is `scalembrl`.

When the sweep is done, send back the `exp/scalembrl/` directory and the queue directory `$Q`. `$Q` holds `tasks.json`
and the per-run logs:

```bash
tar czf combo_results.tgz exp/scalembrl $Q --exclude='*.pkl'
```

## Troubleshooting

- **`Missing dataset directories`** when generating the queue: `--data_root` must contain the five `*-play-v0`
  directories from step 2.
- **EGL / rendering errors:** make sure `MUJOCO_GL=egl` is exported and `libegl1` is installed.
- **Out of GPU memory:** one run per GPU should fit easily on an 80 GB H100. Runs don't preallocate GPU memory
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false` is set for each one).
- **A run fails twice:** look at the end of `$Q/logs/runs/<index>.log`.
