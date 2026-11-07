# Scaling Offline Model-Based RL with Action chunking

This repository contains the official implementation of [Scalable Offline Model-Based RL with Action chunking](TODO).

If you use this code for your research, please consider citing our paper:
```
@article{park2025_MAC,
  title={Scalable Offline Model-Based RL with Action Chunking},
  author={Kwanyoung Park, Seohong Park, Youngwoon Lee, Sergey Levine},
  journal={arXiv Preprint},
  year={2025}
}
```

## Overview
This codebase contains implementations of 1) model-free methods, 2) model-based methods, 3) MAC, and ablated version of MAC for ablation studies. Specifically: 

```
# Baselines (Model-free)
from agents.gciql import GCIQLAgent         # GCIQL
from agents.ngcsacbc import NGCSACBCAgent   # n-step GCSAC+BC
from agents.sharsa import SHARSAAgent       # SHARSA

# Baselines (Model-based)
from agents.fmpc import FMPCAgent           # FMPC
from agents.leq import LEQAgent             # LEQ
from agents.mopo import MOPOAgent           # MOPO
from agents.mobile import MOBILEAgent       # MOBILE
from agents.combo import COMBOAgent         # COMBO

# Our method (MAC)
from agents.mac import MACAgent             # MAC
from agents.mbrs_ac import ACMBRSAgent      # MAC (Gau)
from agents.mbfql import MBFQLAgent         # MAC (FQL)
from agents.model_ac import ACModelAgent    # Model inaccuracy analysis
```

## Installation

Please install the libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Datasets

For downloading the datasets, please follow the instruction of [Horizon Reduction Makes RL Scalable](https://github.com/seohongpark/horizon-reduction).

## Example training scripts 

For MVE-based MBRL algorithms (MAC, LEQ, FMPC) and model-free RL algorithms (SHARSA, GCIQL, n-step GCSAC+BC):

```
# MAC in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mac.py

# SHARSA in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/sharsa.py

# MAC in cube-double-play-singletask-task2-v0 (default dataset)
python main.py --env_name=cube-double-play-singletask-task2-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/cube-double-play-singletask-v0 --agent=agents/mac.py
```

For MBPO-based algorithms (MOPO, MOBILE, COMBO):

```
# MOPO in puzzle-4x5-play-oraclerep-v0 (100M)
python main_mbpo.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mopo.py
```

### COMBO

COMBO pretrains the model (dynamics ensemble, success and reward heads) for `model_train_steps`, selects the 5 elites
with the lowest held-out next-state MSE, saves that state as `params_0.pkl`, and then trains the policy with the model
frozen. Restoring any COMBO checkpoint skips pretraining.

```
# Goal-conditioned OGBench, shared MOPO/MOBILE settings (default `ogbench` profile)
python main_mbpo.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/combo.py

# Single-task, settings from the paper (Appendix B)
python main_mbpo.py --env_name=cube-double-play-singletask-task2-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/cube-double-play-singletask-v0 --agent=agents/combo.py:paper
```

`--agent.real_ratio` is the real-data fraction f (it replaces `--mb_batch_ratio` for COMBO). As in the paper, each
batch holds round(f * B) real and B - round(f * B) model transitions, and every loss term is a plain mean over that
pooled batch; E_D[Q] in the regularizer uses the real part. With f = 1 no model is trained or rolled out and the update
is CQL(H) (uniform and pi(.|s) log-sum-exp proposals; CQL's extra pi(.|s') samples are not used). Use
`--agent.beta` equal to CQL's alpha as written in the CQL paper (both have 0.5 on the TD term), or half the
`min_q_weight` of CQL implementations that drop the 0.5. With f = 0 the TD and actor terms use only model data, and a
separate real batch supplies E_D[Q]. `conservative_states=model` requires f < 1. Main COMBO options: `--agent.beta` (paper candidates 0.5, 1, 5),
`--agent.conservative_states` (`mixture` or `model`), `--agent.rollout_policy` (`policy` or `uniform`),
`--agent.rollout_steps`, `--agent.actor_lr`, `--agent.critic_lr`. For offline hyperparameter selection as in the paper,
pick the setting with the lowest `validation/critic/regularizer`, and report evaluation returns separately.

Ablations over beta, f and the rollout horizon are launched as a Slurm array (dry run unless `--submit`):

```
python scripts/launch_combo_ablation.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 [--one_at_a_time] [--seeds 0 1 2] [--submit]
```

Repository adaptations to note:
- The log-partition term uses `num_samples` (10) uniform and 10 current-policy actions per state, each importance
  weighted by its proposal density and averaged over all 20 samples.
- Each critic gets its own `0.5 * TD + beta * regularizer` loss, and the losses are summed.
- The model uses this repository's separate reward and success heads instead of the paper's joint Gaussian over next state
  and reward. Goal-conditioned rewards and terminations come from the learned success classifier, with goal-conditioned
  target clipping, as in MOPO/MOBILE.
- Normalization statistics are saved as `scale.pkl` next to the checkpoints.

## Acknowledgement

This codebase is built on top of [Horizon Reduction makes RL scalable](https://github.com/seohongpark/horizon-reduction)'s codebase.
