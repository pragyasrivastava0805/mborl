# Unifloral on 3 A100 VMs and 2 Slurm A40 nodes

Use your friend’s three VMs with **8 A100s each** and your two Slurm nodes with **8 A40s each**. Target training completion on **October 14** and final results on **October 15, 2026**; confirm feasibility from measured throughput. This guide first builds the policy pools and runs the JAX bandit evaluation. Dataset metrics (SACo and TQ) can be computed separately for later analysis.

**Your friend:** follow [VM setup and launch](#2-friend-set-up-and-launch-three-a100-vms) on each VM. **You:** follow [Slurm setup and launch](#3-you-set-up-and-launch-two-a40-nodes). Both can start independently after agreeing on the settings below.

## Before either of you starts

Agree on these once and send the values to each other:

| Setting | What to use |
| --- | --- |
| Repository / commit | Same repository and commit, including the updated `scripts/launch_jobs.py` |
| Python dependencies | Friend creates `environment.freeze.txt` on VM1; copy it to VM2, VM3, and the cluster |
| Policy pool | `--pool-size 10` everywhere; seeds 0–9 |
| Evaluation settings | Commands below use interval 2,500, four workers, and 1,000 final episodes |
| Result destination | One agreed GCS bucket and campaign prefix; grant both uploaders access |

Replace all `<...>`, `YOUR_...`, and `/shared/path/...` placeholders before running commands. Slurm partition/account, CPU count, and maximum wall time remain placeholders until you know your cluster settings. Both queues run concurrently; your A40 jobs do not wait for the VM runs.

## 1. Job plan

The existing `scripts/launch_jobs.py` handles job creation, GPU assignment, logs, and retries.

- **12 methods:** BC, IQL, TD3-BC, CQL, SAC-N, EDAC, ReBRAC, TD3-AWR, MOPO, MOReL, COMBO, MoBRAC.
- **20 datasets:** halfcheetah, walker2d, hopper, ant × random, medium, medium-replay, expert, medium-expert. The source of truth is `SOURCES` in `loader.py`.
- **10 policies per method/dataset:** seeds 0–9, with deterministic hyperparameter samples from the sweep configs.
- **2,400 policy training jobs total.** All training implementations use JAX; the launcher uses both unified and standalone implementations according to its config mapping.

| Owner / machine | Methods | Shard | Policy jobs |
| --- | --- | --- | --- |
| Friend: A100 VM1 | Heavy queue | 0 of 3 | 467 |
| Friend: A100 VM2 | Heavy queue | 1 of 3 | 467 |
| Friend: A100 VM3 | Heavy queue | 2 of 3 | 466 |
| You: A40 Slurm task 0 | Light queue | 0 of 2 | 500 |
| You: A40 Slurm task 1 | Light queue | 1 of 2 | 500 |

- **Heavy queue:** `cql sac_n edac mopo morel combo mobrac`.
- **Light queue:** `bc iql td3_bc rebrac td3_awr`.

This is an initial hardware allocation: A100s handle large critic ensembles and model-based methods; A40s start immediately with lighter methods and need no dynamics models. Each A100 VM trains 20 local dynamics models first. Within each queue, expensive jobs start first and are split round-robin. The queues cover all 2,400 policies without overlap.

Use the **same Git commit, resolved dependencies, configs, datasets, and evaluation settings** everywhere. Method filtering happens before sharding: keep the heavy list identical on all VMs and the light list identical on both Slurm tasks. Every training process uses one GPU; no distributed JAX setup is needed.

Use the new `pool10` output paths below. If you reuse older runs, reconcile their results and completion markers first; retain only seeds 0–9 with matching settings. Changing pool size also recomputes shard assignments. Markers are local, so a new split cannot detect completed work elsewhere.

## 2. Friend: set up and launch three A100 VMs

Repeat this section on VM1, VM2, and VM3. Use shard 0 on VM1, shard 1 on VM2, and shard 2 on VM3. Use an Ubuntu 22.04 GPU image with an NVIDIA driver, eight A100s, and at least a 100 GB disk. Check that `nvidia-smi -L` lists eight GPUs.

### 2.1 System dependencies

D4RL uses the legacy `mujoco-py` package and MuJoCo 2.1:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config git wget curl tmux \
  libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
mkdir -p ~/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar -xzf /tmp/mujoco.tar.gz -C ~/.mujoco
```

### 2.2 Code and Python environment

Use Python 3.10 for the legacy D4RL stack. Clone the repository and check out the same experiment commit on every VM:

```bash
git clone git@github.com:pragyasrivastava0805/mborl.git ~/mborl
cd ~/mborl
git checkout "<EXPERIMENT_COMMIT>"
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv .venv --python 3.10

cat > env.sh <<'ENV'
source "$HOME/mborl/.venv/bin/activate"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ENV
source env.sh
# On VM1, resolve dependencies and send the snapshot to the other machines:
uv pip install -r requirements.txt 'jax[cuda12]<0.6' pyyaml \
  jupyterlab ipykernel pandas seaborn matplotlib tqdm h5py
uv pip freeze > environment.freeze.txt
git rev-parse HEAD > experiment.commit.txt
```

The explicit CUDA 12 extra installs GPU support within this repo's JAX version constraint. CUDA 12 requires a Linux NVIDIA driver >= 525; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Avoid adding system CUDA directories to `LD_LIBRARY_PATH`, which can override pip's CUDA libraries. On VM2 and VM3, create the environment as above, then install the snapshot copied from VM1 using `uv pip sync environment.freeze.txt` instead of resolving packages again. Send the same snapshot to your collaborator for the Slurm environment. Install from the checked-out repository root so any relative paths resolve correctly.

**In every new shell:** `cd ~/mborl && source env.sh`.

### 2.3 Check, download, and smoke test

```bash
python - <<'PY'
import gym, d4rl, mujoco_py
import jax
print('JAX:', jax.__version__, 'devices:', jax.devices())
assert len(jax.devices()) == 8 and all(d.platform == 'gpu' for d in jax.devices())
PY

python scripts/launch_jobs.py --prefetch
python scripts/launch_jobs.py --group dynamics --smoke
python scripts/launch_jobs.py --group all --pool-size 10 --smoke --per-gpu 1
```

The first MuJoCo import compiles its extension. Prefetch downloads all datasets before concurrent jobs start. Both smoke commands must report zero failures; inspect `smoke_test/logs/jobs/` otherwise. Smoke outputs stay in `smoke_test/` and do not enter the experiment pools.

### 2.4 Launch your assigned shard

Use `tmux new -s unifloral` on each VM. Set `SHARD` to 0, 1, or 2:

```bash
cd ~/mborl
source env.sh
SHARD=0
WORK="$HOME/mborl/runs/pool10/a100-${SHARD}"
mkdir -p "$WORK/logs"
set -o pipefail

python scripts/launch_jobs.py --group all --pool-size 10 \
  --algorithms cql sac_n edac mopo morel combo mobrac \
  --shard "$SHARD" --num-shards 3 --dry-run

python scripts/launch_jobs.py --group dynamics --work-dir "$WORK" \
  --gpus 8 --per-gpu 1 2>&1 | tee -a "$WORK/logs/dynamics.log" && \
python scripts/launch_jobs.py --group all --pool-size 10 \
  --algorithms cql sac_n edac mopo morel combo mobrac \
  --shard "$SHARD" --num-shards 3 --work-dir "$WORK" \
  --gpus 8 --per-gpu 2 --eval-workers 4 \
  2>&1 | tee -a "$WORK/logs/launcher.log"
```

Start with two policy jobs per GPU; benchmark one for large ensembles or memory failures. Results, models, and logs go under `runs/pool10/a100-<SHARD>/`. Detach with `Ctrl-b d`; reconnect with `tmux attach -t unifloral`.

## 3. You: set up and launch two A40 nodes

Your assigned methods are **BC, IQL, TD3-BC, ReBRAC, and TD3-AWR**: 1,000 runs split between two nodes. You do not need dynamics models. Run setup once from a permitted cluster host, then submit one two-node job.

### 3.1 Prepare shared storage and dependencies

Choose a persistent shared path visible at the same location on both nodes. Avoid node-local `/tmp` for results or completion markers. The cluster needs a compatible NVIDIA driver and the MuJoCo build/OpenGL libraries listed in section 2.1. Use the cluster's modules/container or ask the administrator to provide those libraries; the Slurm script does not install system packages.

```bash
# Replace with your actual shared paths and the agreed commit.
UNIFLORAL_REPO="/shared/path/mborl"
UNIFLORAL_RUNS="/shared/path/unifloral-pool10"
mkdir -p "$UNIFLORAL_RUNS"
git clone git@github.com:pragyasrivastava0805/mborl.git "$UNIFLORAL_REPO"
cd "$UNIFLORAL_REPO"
git checkout "<EXPERIMENT_COMMIT>"

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv .venv --python 3.10

# Copy environment.freeze.txt from your friend to this directory first.
source .venv/bin/activate
uv pip sync environment.freeze.txt
git rev-parse HEAD > experiment.commit.txt

# MuJoCo and the dataset cache must also be available to both nodes.
mkdir -p "$UNIFLORAL_REPO/.mujoco" "$UNIFLORAL_REPO/d4rl-cache"
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco-a40.tar.gz
tar -xzf /tmp/mujoco-a40.tar.gz -C "$UNIFLORAL_REPO/.mujoco"

cat > env.sh <<'ENV'
export UNIFLORAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$UNIFLORAL_REPO/.venv/bin/activate"
export MUJOCO_PY_MUJOCO_PATH="$UNIFLORAL_REPO/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin"
export D4RL_DATASET_DIR="$UNIFLORAL_REPO/d4rl-cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ENV
source env.sh
python -c 'import gym, d4rl, mujoco_py'
python scripts/launch_jobs.py --prefetch

# Check both node assignments without needing an allocation.
for shard in 0 1; do
  python scripts/launch_jobs.py --group all --pool-size 10 \
    --algorithms bc iql td3_bc rebrac td3_awr \
    --shard "$shard" --num-shards 2 --dry-run
done
```

If the login host cannot build/import MuJoCo, perform the first import in a single compute allocation before launching both tasks. If compute nodes have no internet, download datasets and dependencies from a permitted host first. Load your required cluster modules in the batch script before sourcing `env.sh`; keep system CUDA directories out of `LD_LIBRARY_PATH` when using pip CUDA libraries.

### 3.2 Save and submit the two-node job

Save the following as `slurm-unifloral.sbatch` in the repository. Replace partition, account, repo path, and output path. Adjust CPU request and wall time to cluster limits. It starts one policy per GPU; tune concurrency after measuring throughput.

```bash
#!/bin/bash
#SBATCH --job-name=unifloral-a40
#SBATCH --partition=YOUR_A40_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
export UNIFLORAL_REPO="/shared/path/mborl"
export UNIFLORAL_RUNS="/shared/path/unifloral-pool10"
export UNIFLORAL_PER_GPU=1

srun --ntasks=2 --ntasks-per-node=1 --gpu-bind=none bash -c '
  set -euo pipefail
  cd "$UNIFLORAL_REPO"
  source env.sh
  WORK="$UNIFLORAL_RUNS/a40-${SLURM_PROCID}"
  mkdir -p "$WORK/logs"
  python -c "import jax; print(jax.devices()); assert len(jax.devices()) == 8 and all(d.platform == \"gpu\" for d in jax.devices())"
  python scripts/launch_jobs.py --group all --pool-size 10 --smoke \
    --algorithms bc iql td3_bc rebrac td3_awr \
    --work-dir "$WORK" --gpus 8 --per-gpu 1 --eval-workers 4
  python scripts/launch_jobs.py --group all --pool-size 10 \
    --algorithms bc iql td3_bc rebrac td3_awr \
    --shard "$SLURM_PROCID" --num-shards 2 --work-dir "$WORK" \
    --gpus 8 --per-gpu "$UNIFLORAL_PER_GPU" --eval-workers 4 \
    2>&1 | tee -a "$WORK/logs/launcher.log"
'
```

```bash
sbatch slurm-unifloral.sbatch
squeue -u "$USER"
```

This checks all eight allocated GPUs and smoke-tests your five methods on each node before starting production. Smoke results are isolated under each node’s `smoke_test/` directory. It launches one controller per node, with eight concurrent policy jobs per node. GPU type spelling and resource syntax depend on the cluster. In a short allocation, confirm each task sees eight GPUs. The launcher preserves Slurm’s assigned GPU indices or UUIDs; see [Slurm GPU allocation](https://slurm.schedmd.com/gres.html#GPU_Management).

Each task writes to its own persistent directory. Resubmit the same script at wall-time expiry: done markers skip completed policies, but interrupted policies restart from scratch. Use the longest allowed allocation to reduce lost work. Task ranks 0 and 1 can land on different physical nodes after resubmission.

## 4. Both: speed and deadline checks

### Tune concurrency from measurements

Measure completed policies/hour, CPU load, GPU utilization, and memory on both GPU types. Compare representative full configurations at one versus two jobs per GPU; try three only if aggregate throughput improves and memory permits. Smoke tests check correctness, not production runtime.

On A40s, two jobs/GPU with four evaluation workers can exceed 64 CPUs once parent processes are included. Increase allocated CPUs or reduce workers before increasing concurrency. Changing `--eval-workers` changes intermediate episode count as well as parallelism; final evaluation still collects 1,000 episodes. Use consistent evaluation settings across hardware.

### Keep the evaluation protocol close to the repository settings

For this initial campaign, use **10 policies per pool**, then sample **eight distinct policies** per bandit trial, keeping the notebook's **500 repeats, 2,000 bootstrap samples, 200-pull budget, and UCB alpha 2.0**. Keep the configured training updates and hyperparameter sweeps, intermediate evaluation interval **2,500**, and **1,000 final episodes per policy**. Do not add `--eval-interval 50000` for this campaign.

The main deviation from the previous plan is a smaller candidate pool (10 rather than 20), so label results **preliminary, pool size 10**. Sampling eight out of ten gives less candidate diversity than eight out of twenty. The existing uniform `--eval-workers 4` setting also differs from the configs' eight workers: it changes intermediate evaluation episode count and the RNG sequence, although final episode count stays at 1,000. Record that setting with the results. These commands preserve the repository's bandit evaluation settings; they do not claim an exact reproduction of the published paper.

Later, add the other ten policies with `--pool-size 10 --seed-start 10`, keeping training and evaluation settings identical. Their seeds and sampled hyperparameters match the second half of a 20-policy pool. Stop/reconcile existing launchers first: assignment is recomputed for the selected seed range, so hand off each original worker's outputs and markers consistently. Combine both seed ranges and update the completeness check to `build_jobs('all', pool_size=20)` before evaluating 20-policy pools.

### Rebalance when one queue finishes early

A40s may finish the lighter queue first. Transfer remaining **CQL** work first and benchmark it; transfer model-based work only with the dynamics models. If A100s finish first, move remaining light work there.

For a simple handoff without a cross-cloud queue service:

1. Finish/stop the affected source launcher and its child policies. Confirm it is no longer running.
2. Copy that shard’s valid results and `logs/done/` markers to a new destination work directory; copy dynamics models for model-based work. Reconcile any output saved before a done marker was written.
3. Retire the source assignment and launch the **same method list, pool size, seed range, shard, and number of shards** on the destination using `--work-dir`. Completed jobs are skipped. Use one destination launcher per transferred shard.

Never run the same shard twice concurrently or change shard counts mid-campaign. There is no global locking across machines. This initial split is not guaranteed optimal; use measured per-method times to decide transfers.

### October 15 target

Starting October 8 and finishing training October 14 gives about six days: the fleet needs roughly **17 completed policies/hour**, leaving October 15 for retries and bandit evaluation. Recompute from your actual start and chosen cutoff:

```text
required rate = remaining policies / hours until training cutoff
```

Check projected finish dates separately for each queue and method every day; a fleet average can hide slow COMBO or ensemble runs. If the deadline slips, rebalance, request extra A100 capacity, or explicitly reduce dataset scope. Extra A40s do not provide the same throughput as extra A100s.

Monitor each work directory and rerun the same VM commands or resubmit the same Slurm script to retry failures. In a new shell, set `WORK` first: your friend uses `$HOME/mborl/runs/pool10/a100-0` (or 1/2); you use `/shared/path/unifloral-pool10/a40-0` (or 1). Then:

```bash
tail -f "$WORK/logs/launcher.log"
ls "$WORK/logs/failed"
nvidia-smi
```

## 5. Both: collect and check results

Upload each worker to its own prefix. Grant bucket access to each uploader and repeat periodically for backups. On each A100 VM, set `SHARD` to that VM’s assigned value:

```bash
SHARD=0  # VM1: 0, VM2: 1, VM3: 2
WORK="$HOME/mborl/runs/pool10/a100-${SHARD}"
gsutil -m rsync -r "$WORK/final_returns" "gs://<BUCKET>/unifloral/pool10/a100-${SHARD}/final_returns"
```

From a permitted cluster transfer host with the Google Cloud CLI authenticated, upload the A40 outputs (or transfer them by SSH to the analysis VM):

```bash
UNIFLORAL_RUNS="/shared/path/unifloral-pool10"
for shard in 0 1; do
  gsutil -m rsync -r "$UNIFLORAL_RUNS/a40-${shard}/final_returns" \
    "gs://<BUCKET>/unifloral/pool10/a40-${shard}/final_returns"
done
```

Choose one A100 VM as the analysis machine. On that VM, combine all five prefixes into `~/mborl/collected/pool10`:

```bash
cd ~/mborl
mkdir -p collected/pool10
for worker in a100-0 a100-1 a100-2 a40-0 a40-1; do
  gsutil -m rsync -r "gs://<BUCKET>/unifloral/pool10/${worker}/final_returns" collected/pool10
done
```

Check **240 pools, each with exactly seeds 0–9**, using saved run metadata:

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.launch_jobs import build_jobs

expected = Counter((j['params']['algorithm'], j['dataset'], j['params']['seed'])
                   for j in build_jobs('all', pool_size=10))
actual = Counter()
for path in Path('collected/pool10').glob('*.npz'):
    with np.load(path, allow_pickle=True) as result:
        args = result['args'].item()
        assert result['final_scores'].size == 1000, f'Incomplete final evaluation: {path}'
        assert args['eval_final_episodes'] == 1000 and args['eval_interval'] == 2500, path
        assert args['eval_workers'] == 4, f'Mismatched evaluation workers: {path}'
    actual[(args['algorithm'], args['dataset'], args['seed'])] += 1
print('Missing:', expected - actual)
print('Extra or duplicate:', actual - expected)
assert actual == expected, 'Resolve incomplete or duplicate runs before evaluation'
print('Complete: 2,400 policies in 240 pools')
PY
```

A retry can produce duplicate result files if a process saved its output before its completion marker was written. Inspect the logs and retain one valid result for that method/dataset/seed before evaluation. Also archive `environment.freeze.txt`, `experiment.commit.txt`, launch settings, and job logs. Confirm matching evaluation settings across workers.

## 6. Analysis VM: run the JAX Unifloral evaluation

On the analysis VM with the complete pools:

```bash
cd ~/mborl
source env.sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  jupyter lab --ip 127.0.0.1 --port 8888 --no-browser
```

Forward the port from your laptop:

```bash
gcloud compute ssh "<VM_NAME>" --zone "<ZONE>" -- -L 8888:127.0.0.1:8888
```

Open Jupyter's printed token URL and select the `.venv` Python kernel. In the first cell of **`evaluation_plots.ipynb`**, set the input path:

```python
df = evaluation.load_results_dataframe(results_dir="collected/pool10")
```

Then run all cells. Leave `num_subsample=8`, `num_repeats=500`, and `n_bootstraps=2000` unchanged. It reads the validated 10-policy pools and uses JAX in `evaluation.py` to run UCB policy selection: eight sampled policies per pool, 500 repeats, and 2,000 bootstrap samples. It saves `evaluation_plots.pdf`. A single analysis GPU is sufficient to start; the notebook does not automatically shard across the 40 GPUs. Start evaluation on complete pools early to measure its runtime.

## 7. Later: compare performance against SACo/TQ

Run **`evaluate.ipynb`** to compute dataset TQ and SACo using the same D4RL cache. This step uses CPU and can run independently of training.

The notebook saves `dataset_metrics.json`, grouped by environment and full dataset name. TQ uses full ordered trajectories and dataset-derived random/expert reference returns. SACo averages five seeded samples of 100,000 transitions without replacement from each dataset, including the medium-replay normalization reference, using fixed bin boundaries per environment. Saved fields include `mean_return`, `tq`, `saco`, `saco_std`, `saco_sample_size`, `saco_seeds`, `saco_samples`, and `saco_normalized`.

Set `D4RL_DATASET_DIR` to the same cache used by training before importing `loader`; restart the notebook kernel if that setting changes. The notebook currently sets it to `/scratch/cluster/pragyas/d4rl/datasets`, so update that cell for your machine. Downloads use temporary `.part` files before being renamed.

The OGBench section provides an independent loading and inspection step. Install `ogbench` in the notebook kernel, choose `OGBENCH_DATASET_NAME`, and configure `OGBENCH_DATASET_DIR` if needed. It keeps training and validation separate; OGBench metric computation is still pending.

Use one metrics row per full dataset name from `loader.SOURCES`. Join these rows to the evaluation results on `dataset`. Compare methods at a fixed policy-evaluation budget and retain uncertainty estimates. The current plotting notebook shows performance against evaluation budget; plots against SACo/TQ are a separate analysis step.
