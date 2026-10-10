# Launching the Unifloral experiments on 3 GCP VMs

This guide covers setting up the environment, running the experiments on 3 VMs (8 A100s each), collecting results, and running the evaluation notebooks. The 3 VMs run **independently**: each one sets itself up, trains its own dynamics models, and works through its own third of the jobs. Nothing is shared between VMs until you collect the results at the end.

## What gets run

We evaluate 12 offline RL algorithms on 20 D4RL MuJoCo datasets, following the Unifloral protocol. For every (algorithm, dataset) pair we train a **pool of 20 policies**. Policy *i* uses seed *i* (0–19) and hyperparameters drawn at random from that algorithm's sweep config. The bandit evaluation later picks policies from each pool.

| Item | Count |
|---|---|
| Datasets | 4 envs (halfcheetah, walker2d, hopper, ant) × 5 types (random, medium, medium-replay, expert, medium-expert) = 20. The list is `SOURCES` in `loader.py`; the launcher and `evaluate.ipynb` both read it, so edit it there if it changes. |
| Model-free training runs | 8 algorithms (BC, IQL, TD3-BC, CQL, SAC-N, EDAC, ReBRAC, TD3-AWR) × 20 datasets × 20 = 3,200 |
| Model-based training runs | 4 algorithms (MOPO, MOReL, COMBO, MoBRAC) × 20 datasets × 20 = 1,600 |
| Training runs in total | 12 × 20 × 20 = **4,800** |
| Dynamics models (needed by the model-based runs) | 20 per VM (one per dataset), so 60 in total |

### Which VM runs what

| VM | Step 1 (~1 h) | Step 2 | Training jobs | Estimated GPU-hours |
|---|---|---|---|---|
| VM1 | 20 dynamics models | `--group all --shard 0 --num-shards 3` | 1,600 | ~7,800 |
| VM2 | 20 dynamics models | `--group all --shard 1 --num-shards 3` | 1,600 | ~7,800 |
| VM3 | 20 dynamics models | `--group all --shard 2 --num-shards 3` | 1,600 | ~7,800 |

All 4,800 jobs are sorted longest first and dealt round-robin to the 3 VMs. So each VM gets about 133 runs of every algorithm, an equal share of the slow model-based runs, and about the same total work. Each VM works out its own share from `--shard`, so no coordination is needed.

Each VM trains its own copy of the 20 dynamics models, using the same config and seed. That keeps the VMs independent and only costs about an hour per VM.

### Which config each algorithm uses
| Algorithm | Config | Algorithm | Config |
|---|---|---|---|
| BC | `configs/unifloral/bc.yaml` | ReBRAC | `configs/unifloral/rebrac.yaml` |
| IQL | `configs/algorithms/iql.yaml` | TD3-AWR | `configs/unifloral/td3_awr.yaml` |
| TD3-BC | `configs/unifloral/td3_bc.yaml` | MOPO | `configs/algorithms/mopo.yaml` |
| CQL | `configs/algorithms/cql.yaml` | MOReL | `configs/algorithms/morel.yaml` |
| SAC-N | `configs/unifloral/sac_n.yaml` | COMBO | `configs/algorithms/combo.yaml` |
| EDAC | `configs/unifloral/edac.yaml` | MoBRAC | `configs/unifloral/mobrac.yaml` |

All of this is handled by `scripts/launch_jobs.py`. It builds the same job list on every VM and runs it on local GPUs, 3 jobs per GPU by default. Re-running the same command skips jobs that already finished and retries failed ones.

### Where things are written (inside the repo clone, `~/mborl`)
| Path | Contents |
|---|---|
| `~/mborl/final_returns/` | one `.npz` per finished run, the results the evaluation uses. Named `<algorithm>_<dataset>_<timestamp>_s<seed>.npz`; the seed keeps runs from the same pool from overwriting each other. |
| `~/mborl/dynamics_models/` | this VM's 20 dynamics models |
| `~/mborl/logs/` | `launcher.log`, per-job logs in `jobs/`, and `done/` and `failed/` markers |
| `~/mborl/smoke_test/` | output of the smoke test only |
| `/scratch/cluster/pragyas/d4rl/datasets/` | the 20 datasets, shared by the training runs (via D4RL) and `evaluate.ipynb` (via `loader.py`), configured by `D4RL_DATASET_DIR` in `env.sh` |

---

## 1. Set up each VM (do this on all 3)

### 1.1 VM spec
- **Machine type:** `a2-highgpu-8g` (8 × A100 40 GB, 96 vCPUs) or `a2-ultragpu-8g` (8 × A100 80 GB).
- **Image:** Ubuntu 22.04 with the NVIDIA driver installed, for example a "Deep Learning VM with CUDA" image. You only need the driver: JAX installs its own CUDA libraries.
- **Boot disk:** at least 100 GB.

Check:
```bash
nvidia-smi -L     # should list 8 A100s
nproc             # CPU cores, see section 5
```

### 1.2 System packages and MuJoCo 2.1
D4RL uses `mujoco-py`, which needs the MuJoCo 2.1 binaries and a C compiler.
```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake pkg-config wget git unzip curl tmux htop \
  libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
# On Ubuntu 24.04, use libgl1 instead of libgl1-mesa-glx.

mkdir -p ~/.mujoco
wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz
tar -xzf /tmp/mujoco.tar.gz -C ~/.mujoco && rm /tmp/mujoco.tar.gz
```

### 1.3 Get the code and the environment variables
```bash
git clone git@github.com:pragyasrivastava0805/mborl.git ~/mborl
cd ~/mborl

cat > env.sh <<'EOF'
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR=/scratch/cluster/pragyas/d4rl/datasets
source $HOME/mborl/.venv/bin/activate
EOF
```
**Run `source ~/mborl/env.sh` in every new shell before any command below.** The training jobs need these variables.

### 1.4 Python environment (Python 3.10, because D4RL requires < 3.11)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd ~/mborl
uv venv .venv --python 3.10
source env.sh
uv pip install -r requirements.txt
```
Check the install. The first `import mujoco_py` compiles it, which takes a few minutes:
```bash
python -c "import gym, d4rl, mujoco_py; import jax; print(jax.devices())"
# expect 8 CudaDevice entries; warnings from d4rl about missing optional envs are fine
```
If you use conda instead of uv, the equivalent is `conda create -n mborl python=3.10 && conda activate mborl && pip install -r requirements.txt`. Change the last line of `env.sh` to `conda activate mborl`.

### 1.5 Download the datasets once
Download the datasets **before** launching. Otherwise 24 jobs start downloading the same file at once and leave corrupted files.
```bash
cd ~/mborl && source env.sh
python scripts/launch_jobs.py --prefetch     # ~5.2 GB into $D4RL_DATASET_DIR
```

### 1.6 Smoke test (about 15–30 min)
This trains one dynamics model, then runs one very short job for each of the 12 algorithms on `hopper-medium-v2`. Output goes to `~/mborl/smoke_test/`, so it doesn't mix with the real results.
```bash
cd ~/mborl && source env.sh
python scripts/launch_jobs.py --group dynamics --smoke && \
python scripts/launch_jobs.py --group all --smoke --per-gpu 2
```
Every line should end in `[OK]`, followed by `Finished: N ok, 0 failed`. For a failure, the full output is in `smoke_test/logs/jobs/<job>.log`. After a clean pass you can delete `smoke_test/`.

---

## 2. Launch the experiments

Run this on each VM, with `SHARD=0` on VM1, `SHARD=1` on VM2 and `SHARD=2` on VM3. Use `tmux` so the launcher keeps running after you disconnect: `tmux new -s runs`, paste the commands, then press `Ctrl-b d` to detach. `tmux attach -t runs` reattaches.

```bash
cd ~/mborl && source env.sh && mkdir -p logs
set -o pipefail   # so a failed dynamics job stops here despite the | tee
SHARD=0           # 0 on VM1, 1 on VM2, 2 on VM3

python scripts/launch_jobs.py --group dynamics 2>&1 | tee -a logs/dynamics.log && \
python scripts/launch_jobs.py --group all --shard $SHARD --num-shards 3 2>&1 | tee -a logs/launcher.log
```

- **Step 1** trains the 20 dynamics models (about 1 hour) into `dynamics_models/ensemble_dynamics_model_<dataset>_<timestamp>.pkl`. Check that all 20 exist: `ls ~/mborl/dynamics_models | wc -l`. If any dynamics job fails, step 2 doesn't start; run the same commands again to retry.
- **Step 2** runs this VM's 1,600 training jobs, longest first. `--group all` covers **both** model-free and model-based algorithms (about 1,067 model-free and 533 model-based jobs per VM). Only the model-based jobs use the dynamics models from step 1.

To see what a VM will run without running anything:
```bash
python scripts/launch_jobs.py --group all --shard 0 --num-shards 3 --dry-run
```

### Optional: Weights & Biases logging
Runs don't log to wandb by default; results are always written to `final_returns/`. To enable wandb, run `wandb login` once, then add `--wandb --wandb-entity <your-entity>` to the step 2 command.

---

## 3. Monitoring

```bash
tail -f ~/mborl/logs/launcher.log            # one line per finished job, with done / failed / left counts
ls ~/mborl/logs/done   | wc -l               # finished jobs on this VM (out of 1,600 + 20 dynamics)
ls ~/mborl/logs/failed                       # failed jobs; full output in logs/jobs/<job>.log
ls ~/mborl/final_returns | wc -l             # result files written
nvidia-smi                                   # expect 3 python processes per GPU
htop                                         # CPU load, see section 5
```

**Average hours per run, by algorithm** (each file in `logs/done/` records how long its run took). Check this after the first day to get a better estimate of the finish date:
```bash
cd ~/mborl/logs/done && for f in *; do echo "${f%%__*} $(cat $f)"; done | \
  awk '{gsub("h","",$2); s[$1]+=$2; n[$1]++} END {for (a in s) printf "%-10s %5d runs  %.2f h avg\n", a, n[a], s[a]/n[a]}'
```

**Retrying failures:** once the launcher has finished, run the same step 2 command again. Finished jobs are skipped and failed ones run again.

**Stopping and resuming:**
```bash
pkill -f launch_jobs.py; pkill -f "mborl/algorithms/"
```
To resume, run the same commands from section 2 again. Dynamics models and finished runs are skipped; runs that were in progress start over from the beginning.

---

## 4. Collecting results

Each VM writes one `.npz` file per finished run to `~/mborl/final_returns/`. Copy them all to one GCS bucket. This is safe to repeat; it only uploads new files.
```bash
gsutil -m rsync -r ~/mborl/final_returns gs://<BUCKET>/final_returns
```
Doing this every few hours (for example with `cron`) protects against losing results if a VM is lost.

**Check that every pool is complete** (240 pools × 20 runs) once all 3 VMs are done. Run this on any one VM:
```bash
cd ~/mborl && gsutil -m rsync -r gs://<BUCKET>/final_returns final_returns
python3 -c "
import glob, re, collections
c = collections.Counter(re.match(r'(.+)_(.+)_\d{4}-\d{2}-\d{2}_', f.split('/')[-1]).groups()
                        for f in glob.glob('final_returns/*.npz'))
print(len(c), 'pools (expect 240)'); print('not 20 runs:', {k: v for k, v in c.items() if v != 20})"
```
If a pool has more than 20 runs, a job finished but was run again. This happens when the VM was stopped between the run saving its result and the launcher recording it as done. Delete the extra file with the newest timestamp.

---

## 5. Tuning if needed

- **CPU-bound:** each run evaluates its policy with 8 CPU MuJoCo processes (`eval_workers`). With 24 runs per VM, check `htop`. If all cores are busy while `nvidia-smi` shows low GPU use, restart the launcher with `--eval-workers 4`. This doesn't change the number of evaluation episodes, only how many run in parallel.
- **GPU out of memory** (only possible on 40 GB cards, e.g. SAC-N with 200 critics): restart that VM's launcher with `--per-gpu 2`.
- Restarting with different settings is safe: finished jobs are skipped.

---

## 6. Evaluation notebooks

Both notebooks run on any VM, or on any machine with the Python environment from 1.4. Install the extra packages and start Jupyter from the repo:
```bash
cd ~/mborl && source env.sh
uv pip install jupyterlab seaborn matplotlib tqdm h5py
jupyter lab --ip 127.0.0.1 --port 8888 --no-browser
```
From your laptop, forward the port and open the `http://127.0.0.1:8888/lab?token=...` link that Jupyter prints:
```bash
gcloud compute ssh <VM_NAME> -- -L 8888:127.0.0.1:8888
```
VS Code Remote-SSH also works: open the notebook and pick the `.venv` kernel.

### 6.1 `evaluate.ipynb`: dataset metrics (TQ and SACo)
This notebook works out properties of the 20 datasets: trajectory quality (TQ, the dataset's mean return scaled between the random and expert datasets) and state-action coverage (SACo). It doesn't use any training results, so you can run it at any time, even while the experiments are running. It only uses the CPU.

- **It uses the same dataset files as the training runs.** Source `env.sh` before starting Jupyter so `loader.py` and D4RL both use `/scratch/cluster/pragyas/d4rl/datasets/`. After the prefetch in 1.5 nothing is downloaded again. On a machine without the cache, the notebook downloads any missing file (~5.2 GB for all 20) to the same place. Without `D4RL_DATASET_DIR`, both default to `~/.d4rl/datasets/`.
- Run all cells from the top. Cell 2 computes the random and expert reference returns, and cell 3 loops over all 20 datasets and prints `dstype`, `tqs`, `sacos` and the normalised `sacos_`. It also saves `dataset_metrics.json` in the working directory, grouped by environment and then full dataset name. Each dataset contains `mean_return`, `tq`, `saco` (mean unique bin count over five random samples of 100,000 transitions without replacement), `saco_std`, `saco_sample_size`, `saco_seeds`, `saco_samples`, and `saco_normalized` (relative to that environment’s medium-replay dataset).
- If you change `loader.py` or `D4RL_DATASET_DIR`, **restart the kernel**. The loader reads the setting only once, when it's first imported.
- Downloads by the loader are written to a `.part` file first and renamed when complete, so an interrupted download never leaves a broken `.hdf5`. Just run the cell again.

### 6.2 `evaluation_plots.ipynb`: bandit policy selection (the main result)
Run this after all experiments finish and the pool check in section 4 passes.

1. The notebook reads every `.npz` in `final_returns/` next to it, `~/mborl/final_returns`. On the VM where you run it, copy all 3 VMs' results into that folder, so it has all 4,800 files:
   ```bash
   gsutil -m rsync -r gs://<BUCKET>/final_returns ~/mborl/final_returns
   ls ~/mborl/final_returns | wc -l     # expect 4800
   ```
2. Run all cells. For each (algorithm, dataset) pool, it repeatedly picks 8 of the 20 policies at random (500 repeats, 2,000 bootstrap samples). It runs a UCB bandit over the policies' evaluation episodes and plots the score of the chosen policy against the number of online evaluations.

---

## Expected runtime (estimates)

These are planning estimates and haven't been measured yet. Update them with the per-algorithm averages from section 3 after the first day.

| | Estimate |
|---|---|
| Work per VM (`--dry-run` estimate) | ~7,800 GPU-hours |
| Throughput per VM (8 A100s × 3 runs per GPU, about 2–2.5× one run per GPU) | ~16–20 GPU-hours per hour |
| **Wall time (all 3 VMs run in parallel and finish at about the same time)** | **~11–20 days, most likely ~16** |

The upper end assumes SAC-N and EDAC take longer in proportion to their number of critics (the sweeps go up to 200 and 50). That is probably pessimistic, since JAX runs the critics in parallel. Ways to finish sooner: drop one dataset type, e.g. the medium-expert datasets (about 20% less; remove them from `SOURCES` in `loader.py`), cap SAC-N at 50 critics and EDAC at 20, or drop one of MOPO, MOReL and COMBO.
