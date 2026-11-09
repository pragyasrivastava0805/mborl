# OGBench `stitch-v0` 15-Baseline Suite Results (`baselines` Branch)

All 15 model-based, model-free, same-episode multi-step, goal-conditioned, and hierarchical baselines evaluated across all 5 OGBench `stitch-v0` environments × 5 evaluation tasks (`task1..task5` = 25 tasks) × 5 seeds (`0..4` = `1,875` total seed-task evaluations).

## 1. Multi-Task Summary Across All 25 Tasks (`Tasks 1–5` × 5 Environments × 5 Seeds)

| Baseline | Family | Module | `pointmaze-med` (5 Tasks) | `pointmaze-lrg` (5 Tasks) | `antmaze-med` (5 Tasks) | `antmaze-lrg` (5 Tasks) | `humanoidmaze-med` (5 Tasks) | Task 1 (5-Env Mean) | All 25 Tasks Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mbpo` | Model-Based Forward | `agents/mbpo.py` | 58.8 ± 8.4% | 15.2 ± 4.9% | 9.0 ± 1.1% | 0.0 ± 0.0% | 0.0 ± 0.0% | 16.8 ± 6.4% | **16.6 ± 2.7%** |
| `mopo` | Model-Based Forward | `agents/mopo.py` | 59.6 ± 5.0% | 15.0 ± 3.7% | 6.0 ± 2.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 20.0 ± 0.0% | **16.1 ± 1.0%** |
| `morel` | Model-Based Forward | `agents/morel.py` | 68.0 ± 4.9% | 16.6 ± 3.6% | 15.6 ± 2.4% | 0.0 ± 0.0% | 0.0 ± 0.0% | 20.4 ± 1.0% | **20.0 ± 0.8%** |
| `reverse_morel` | Model-Based Reverse (USAD-Truncated, Dataset-Anchored) | `agents/reverse_morel.py` | 100.0 ± 0.0% | 96.0 ± 0.6% | 49.6 ± 3.9% | 19.4 ± 3.3% | 11.6 ± 5.1% | 47.8 ± 4.5% | **55.3 ± 1.6%** |
| `reverse_morel_anchor_bootstrap` | model-based | `agents/reverse_morel_anchor_bootstrap.py` | 99.8 ± 0.4% | 99.8 ± 0.4% | 62.4 ± 2.9% | 32.0 ± 4.2% | 21.6 ± 4.5% | 58.4 ± 3.4% | **63.1 ± 1.7%** |
| `combo` | Model-Based Forward | `agents/combo.py` | 43.6 ± 6.4% | 7.4 ± 1.7% | 5.2 ± 1.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 13.0 ± 7.4% | **11.2 ± 1.3%** |
| `iql` | Model-Free 1-Step | `agents/iql.py` | 20.0 ± 0.0% | 0.0 ± 0.0% | 0.2 ± 0.4% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | **4.0 ± 0.1%** |
| `cql` | Model-Free 1-Step | `agents/cql.py` | 26.0 ± 3.2% | 0.0 ± 0.0% | 2.8 ± 1.2% | 0.0 ± 0.0% | 0.0 ± 0.0% | 1.6 ± 0.8% | **5.8 ± 0.7%** |
| `rebrac` | Model-Free 1-Step | `agents/rebrac.py` | 19.4 ± 0.8% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | **3.9 ± 0.2%** |
| `nstep_iql_n5` | Model-Free Multi-Step | `agents/nstep_iql.py (n=5)` | 55.4 ± 2.3% | 0.0 ± 0.0% | 10.2 ± 1.7% | 0.0 ± 0.0% | 0.2 ± 0.4% | 0.2 ± 0.4% | **13.2 ± 0.6%** |
| `nstep_iql_n15` | Model-Free Multi-Step | `agents/nstep_iql.py (n=15)` | 79.6 ± 0.5% | 0.0 ± 0.0% | 13.4 ± 4.8% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | **18.6 ± 1.0%** |
| `gciql` | Model-Free Goal-Conditioned | `agents/gciql.py` | 39.8 ± 0.4% | 0.0 ± 0.0% | 9.0 ± 2.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.4 ± 0.5% | **9.8 ± 0.6%** |
| `qrl` | Model-Free Goal-Conditioned | `agents/qrl.py` | 39.4 ± 1.2% | 0.0 ± 0.0% | 6.2 ± 1.7% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | **9.1 ± 0.4%** |
| `crl` | Model-Free Goal-Conditioned | `agents/crl.py` | 40.4 ± 0.8% | 0.0 ± 0.0% | 9.8 ± 2.1% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.6 ± 0.5% | **10.0 ± 0.3%** |
| `hiql` | Model-Free Hierarchical | `agents/hiql.py` | 100.0 ± 0.0% | 99.8 ± 0.4% | 41.4 ± 2.0% | 2.6 ± 1.6% | 0.0 ± 0.0% | 45.2 ± 2.9% | **48.8 ± 0.6%** |

## 2. Bootstrap Target Comparison: Forward `MoREL` vs. 1-Step `Reverse-MoREL` vs. `Reverse-MoREL-Anchor-Bootstrap`

| Method | Step `k = 1` Bootstrap Target | Steps `k = 2..H` Bootstrap Target | Synthetic States Bootstrapped on `D_real` | Critic Queries on Synthetic States | Task 1 (5-Env Mean) | All 25 Tasks Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `morel` (`agents/morel.py`) | Synthetic `s_tilde_(t+1)` (`not in D_real`) | Synthetic `s_tilde_(t+k)` (`not in D_real`) | 0% (`0/5` steps) | 100% (`5/5` steps) | 18.6 ± 1.1% | 20.0 ± 0.9% |
| `reverse_morel` (`agents/reverse_morel.py`) | Real anchor `s_anchor in D_real` | Intermediate synthetic `s_tilde_(t+H-k+1)` | 20% (`1/5` steps: `k=1` only) | 80% (`4/5` steps: `k=2..5`) | 49.0 ± 2.1% | 55.3 ± 1.6% |
| `reverse_morel_anchor_bootstrap` (`agents/reverse_morel_anchor_bootstrap.py`) | **Real anchor `s_anchor in D_real`** | **Real anchor `s_anchor in D_real` (via `k`-step `R^(k)`)** | **100% (`5/5` steps: all `k=1..5`)** | **0% (`0/5` steps)** | **58.4 ± 1.9%** | **63.1 ± 1.7% (`+7.8 pp`)** |

