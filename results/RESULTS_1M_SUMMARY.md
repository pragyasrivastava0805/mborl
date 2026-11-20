# 1M-Step OGBench Results Summary (`baselines` Branch)

- **Raw logs included:** `results/exp_baselines_1m/scalembrl/` (`1671` deduplicated runs: `1582` completed at 1M steps, `89` partial/paused checkpoints across `78` method groups).
- **Metrics:** `L3` = mean of last 3 evaluations (800k, 900k, 1M); `Peak` = max evaluation across 100k..1M (15 eval episodes per task, seed 0 unless noted).
- **Regenerate:** `python3 scripts/summarize_1m_results.py`

## 1. All-25 Stitch Tasks Leaderboard (Seed 0, 1M Steps, 25/25 Complete)

| Method | PM-M | PM-L | AM-M | AM-L | HM-M | All-25 L3 | All-25 Peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_uniform` | 83.1% | 80.0% | 70.7% | 16.4% | 0.0% | **50.0%** | 57.9% |
| `baselines_mac` | 100.0% | 80.0% | 45.3% | 12.9% | 4.4% | **48.5%** | 52.0% |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_exp1` | 82.2% | 71.1% | 65.8% | 20.4% | 0.0% | **47.9%** | 57.1% |
| `baselines_paper999_rcfm1s_revbc_multi_H5_exp1` | 93.8% | 20.0% | 76.4% | 42.2% | 0.0% | **46.5%** | 56.5% |
| `baselines_reverse_rollouts_argmax_nousad_H5` | 69.3% | 26.2% | 75.6% | 10.2% | 49.8% | **46.2%** | 61.6% |
| `baselines_reverse_rollouts_argmax_nousad_H1` | 73.8% | 16.9% | 83.1% | 16.0% | 35.6% | **45.1%** | 60.8% |
| `baselines_paper999_rcfm1s_revbc_multi_H5_uniform` | 80.0% | 46.7% | 69.3% | 24.0% | 0.0% | **44.0%** | 56.0% |
| `baselines_reverse_rollouts_pi_H5` | 44.5% | 21.3% | 79.6% | 15.6% | 48.5% | **41.9%** | 59.7% |
| `baselines_reverse_rollouts_exp` | 67.1% | 0.0% | 71.6% | 18.7% | 38.2% | **39.1%** | 55.7% |
| `baselines_reverse_rollouts_argmax` | 57.3% | 0.0% | 76.9% | 9.8% | 50.2% | **38.8%** | 53.6% |
| `baselines_reverse_rollouts_pi_H1` | 44.4% | 2.7% | 80.9% | 19.1% | 46.2% | **38.7%** | 52.3% |
| `baselines_reverse_rollouts_exp_nousad_H5` | 64.4% | 0.0% | 74.2% | 11.6% | 42.2% | **38.5%** | 48.8% |
| `baselines_reverse_rollouts_exp_nousad_H1` | 65.8% | 0.0% | 80.9% | 11.1% | 23.6% | **36.3%** | 58.1% |
| `baselines_reverse_untruncated_H1` | 87.6% | 0.0% | 64.0% | 9.8% | 9.3% | **34.1%** | 43.2% |
| `baselines_reverse_morel` | 90.7% | 0.0% | 42.7% | 11.6% | 4.9% | **29.9%** | 36.5% |
| `baselines_reverse_rollout_high_freq` | 70.7% | 7.6% | 47.6% | 11.1% | 7.5% | **28.9%** | 44.5% |
| `baselines_reverse_untruncated_H5` | 88.9% | 4.4% | 39.6% | 1.3% | 9.3% | **28.7%** | 38.1% |
| `baselines_reverse_rollout_cvae` | 58.2% | 0.0% | 53.8% | 6.7% | 13.8% | **26.5%** | 38.4% |
| `baselines_paper999_cql_f1` | 70.7% | 27.6% | 32.9% | 0.0% | 0.0% | **26.2%** | 35.7% |
| `baselines_reverse_rollout_cvae_nnproj` | 52.9% | 0.0% | 56.9% | 9.3% | 8.0% | **25.4%** | 36.5% |
| `baselines_paper999_rcfm1s_combo_fwdbon_H5` | 54.7% | 15.6% | 43.1% | 3.1% | 0.0% | **23.3%** | 33.3% |
| `baselines_iql_ckpt` | 56.0% | 8.4% | 20.9% | 9.8% | 14.7% | **22.0%** | 39.2% |
| `baselines_paper999_rcfm1s_combo_H5` | 46.7% | 40.0% | 18.7% | 2.7% | 0.0% | **21.6%** | 28.0% |
| `baselines_paper999_rcfm1s_combo_fwdsnis_H5` | 46.2% | 20.0% | 34.7% | 2.2% | 0.0% | **20.6%** | 26.9% |
| `baselines_paper999_combo_H5` | 60.0% | 40.0% | 0.4% | 2.7% | 0.0% | **20.6%** | 21.6% |
| `baselines_iql` | 39.6% | 8.4% | 22.7% | 11.6% | 20.0% | **20.4%** | 37.6% |
| `baselines_rebrac` | 28.4% | 0.0% | 21.3% | 32.4% | 14.7% | **19.4%** | 49.3% |
| `baselines_reverse_morel_anchor_bootstrap` | 28.9% | 0.0% | 47.5% | 9.8% | 7.5% | **18.8%** | 29.3% |
| `baselines_nstep_iql_n5` | 69.3% | 0.0% | 4.0% | 0.4% | 16.0% | **17.9%** | 27.5% |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_pdl1` | 33.3% | 6.7% | 2.7% | 0.0% | 0.0% | **8.5%** | 25.6% |
| `baselines_forward_morel_cql` | 26.7% | 0.0% | 11.1% | 0.0% | 0.9% | **7.7%** | 14.9% |
| `baselines_nstep_iql_n15` | 28.9% | 0.0% | 1.3% | 0.4% | 5.8% | **7.3%** | 20.0% |
| `baselines_reverse_morel_cql` | 0.0% | 0.0% | 10.2% | 1.8% | 0.9% | **2.6%** | 11.2% |
| `baselines_forward_morel_iql` | 0.0% | 0.0% | 6.7% | 0.4% | 0.4% | **1.5%** | 10.1% |
| `baselines_cql` | 0.0% | 0.0% | 3.1% | 0.0% | 4.0% | **1.4%** | 8.3% |
| `baselines_morel` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |

## 2. Task 1 Stitch Leaderboard (5 Envs, Seed 0, 1M Steps)

| Method | PM-M | PM-L | AM-M | AM-L | HM-M | Task-1 L3 | Task-1 Peak |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baselines_mac` | 100.0% | 100.0% | 88.9% | 8.9% | 0.0% | **59.6%** | 65.3% |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_exp1` | 66.7% | 66.7% | 84.4% | 48.9% | 0.0% | **53.3%** | 74.7% |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_uniform` | 100.0% | 100.0% | 60.0% | 0.0% | 0.0% | **52.0%** | 53.3% |
| `baselines_paper999_rcfm1s_revbc_multi_H5_exp1` | 100.0% | 0.0% | 57.8% | 77.8% | 0.0% | **47.1%** | 76.0% |
| `baselines_reverse_rollouts_exp` | 100.0% | 0.0% | 95.6% | 15.6% | 24.4% | **47.1%** | 54.7% |
| `baselines_paper999_fix_combo_H5` | 100.0% | 100.0% | 26.7% | 0.0% | 0.0% | **45.3%** | 60.0% |
| `baselines_paper999_rcfm1s_revbc_multi_H5_uniform` | 33.3% | 100.0% | 73.3% | 15.6% | 0.0% | **44.4%** | 61.3% |
| `baselines_paper999_fix_rsnis_multi_H5_exp1` | 100.0% | 66.7% | 31.1% | 15.6% | 0.0% | **42.7%** | 60.0% |
| `baselines_reverse_rollouts_qbeta` | 100.0% | 20.0% | 68.9% | 0.0% | 22.2% | **42.2%** | 56.0% |
| `baselines_paper999_combo_H5` | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | **40.0%** | 40.0% |
| `baselines_paper999_fix_rsnis_single_H5_uniform` | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | **40.0%** | 40.0% |
| `baselines_reverse_rollouts_argmax_nousad_H1` | 100.0% | 17.8% | 75.6% | 0.0% | 0.0% | **38.7%** | 54.7% |
| `baselines_reverse_rollouts_exp_nousad_H5` | 100.0% | 0.0% | 77.8% | 0.0% | 8.9% | **37.3%** | 42.7% |
| `baselines_paper999_rcfm1s_combo_fwdbon_H5` | 80.0% | 11.1% | 93.3% | 0.0% | 0.0% | **36.9%** | 53.3% |
| `baselines_paper999_rcfm1s_combo_fwdsnis_H5` | 97.8% | 0.0% | 84.4% | 0.0% | 0.0% | **36.4%** | 45.3% |
| `baselines_reverse_rollouts_pi_H5` | 66.7% | 6.7% | 64.4% | 0.0% | 35.6% | **34.7%** | 49.3% |
| `baselines_reverse_rollouts_exp_nousad_H1` | 60.0% | 0.0% | 88.9% | 4.4% | 13.3% | **33.3%** | 65.3% |
| `baselines_paper999_fix_rsnis_multi_H5_uniform` | 100.0% | 57.8% | 4.4% | 0.0% | 0.0% | **32.4%** | 42.7% |
| `baselines_reverse_rollouts_argmax_nousad_H5` | 37.8% | 20.0% | 55.6% | 0.0% | 44.4% | **31.6%** | 58.7% |
| `baselines_reverse_morel` | 100.0% | 0.0% | 53.3% | 0.0% | 0.0% | **30.7%** | 36.0% |
| `baselines_reverse_untruncated_H1` | 100.0% | 0.0% | 51.1% | 0.0% | 2.2% | **30.7%** | 36.0% |
| `baselines_reverse_rollout_cvae_nnproj` | 64.4% | 0.0% | 77.8% | 0.0% | 4.4% | **29.3%** | 37.3% |
| `baselines_paper999_rcfm1s_combo_H5` | 33.3% | 100.0% | 11.1% | 0.0% | 0.0% | **28.9%** | 50.7% |
| `baselines_paper999_fix_rsnis_single_H5_exp1` | 100.0% | 33.3% | 11.1% | 0.0% | 0.0% | **28.9%** | 46.7% |
| `baselines_reverse_rollouts_argmax` | 33.3% | 0.0% | 77.8% | 0.0% | 31.1% | **28.4%** | 38.7% |
| `baselines_reverse_rollout_cvae_nousad_H5` | 77.8% | 0.0% | 53.3% | 0.0% | 8.9% | **28.0%** | 38.7% |
| `baselines_reverse_untruncated_H10` | 100.0% | 0.0% | 33.3% | 0.0% | 2.2% | **27.1%** | 30.7% |
| `baselines_reverse_rollouts_pi_H1` | 62.2% | 13.3% | 57.8% | 0.0% | 0.0% | **26.7%** | 48.0% |
| `baselines_reverse_rollout_high_freq` | 97.8% | 0.0% | 35.6% | 0.0% | 0.0% | **26.7%** | 38.7% |
| `baselines_reverse_rollout_cvae` | 77.8% | 0.0% | 42.2% | 0.0% | 2.2% | **24.4%** | 36.0% |
| `baselines_reverse_rollout_cvae_nousad_H10` | 73.3% | 0.0% | 44.4% | 0.0% | 4.4% | **24.4%** | 34.7% |
| `baselines_reverse_untruncated_H5` | 100.0% | 0.0% | 22.2% | 0.0% | 0.0% | **24.4%** | 30.7% |
| `baselines_reverse_rollout_cvae_nousad_H1` | 46.7% | 0.0% | 71.1% | 0.0% | 0.0% | **23.6%** | 34.7% |
| `baselines_paper999_fix_rsnis_multi_H5_frontier` | 100.0% | 0.0% | 11.1% | 0.0% | 0.0% | **22.2%** | 24.0% |
| `baselines_reverse_rollout_cvae_nousad_H25` | 75.6% | 0.0% | 22.2% | 0.0% | 4.4% | **20.4%** | 33.3% |
| `baselines_paper999_combo_H1` | 33.3% | 66.7% | 0.0% | 0.0% | 0.0% | **20.0%** | 52.0% |
| `baselines_paper999_cql_f1` | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | **20.0%** | 20.0% |
| `baselines_reverse_morel_rebrac` | 0.0% | 0.0% | 55.6% | 40.0% | 2.2% | **19.6%** | 34.7% |
| `baselines_reverse_untruncated_H25` | 71.1% | 0.0% | 24.4% | 0.0% | 2.2% | **19.6%** | 34.7% |
| `baselines_nstep_iql_n5` | 62.2% | 0.0% | 0.0% | 0.0% | 2.2% | **12.9%** | 20.0% |
| `baselines_iql` | 20.0% | 0.0% | 24.4% | 0.0% | 15.6% | **12.0%** | 33.3% |
| `baselines_rebrac` | 0.0% | 0.0% | 0.0% | 15.6% | 35.6% | **10.2%** | 17.3% |
| `baselines_iql_ckpt` | 17.8% | 0.0% | 24.4% | 0.0% | 0.0% | **8.4%** | 32.0% |
| `baselines_forward_morel_rebrac` | 0.0% | 0.0% | 31.1% | 4.4% | 0.0% | **7.1%** | 20.0% |
| `baselines_reverse_rollouts_snis_H1` | 35.6% | 0.0% | 0.0% | 0.0% | 0.0% | **7.1%** | 20.0% |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_pdl1` | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | **6.7%** | 40.0% |
| `baselines_forward_morel_cql` | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | **6.7%** | 10.7% |
| `baselines_reverse_morel_anchor_bootstrap` | 0.0% | 0.0% | 17.8% | 0.0% | 2.2% | **4.0%** | 10.7% |
| `baselines_forward_morel_iql` | 0.0% | 0.0% | 6.7% | 0.0% | 0.0% | **1.3%** | 2.7% |
| `baselines_nstep_iql_n15` | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | **0.4%** | 14.7% |
| `baselines_cql` | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | **0.4%** | 1.3% |
| `baselines_mopo` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 20.0% |
| `baselines_reverse_morel_cql` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 2.7% |
| `baselines_combo` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |
| `baselines_mbpo` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |
| `baselines_morel` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |

## 3. Multi-Seed Completed Stitch Runs (Seeds > 0)

| Method | Completed 1M Stitch Runs by Seed |
| :--- | :--- |
| `baselines_combo` | s0: 5, s1: 20, s2: 20, s3: 5 |
| `baselines_cql` | s0: 25, s2: 5, s3: 25 |
| `baselines_forward_morel_cql` | s0: 25, s2: 3, s3: 25 |
| `baselines_forward_morel_iql` | s0: 25, s3: 25 |
| `baselines_forward_morel_rebrac` | s0: 5, s1: 8, s2: 17, s3: 5 |
| `baselines_iql` | s0: 25, s2: 2, s3: 5 |
| `baselines_mbpo` | s0: 5, s1: 17, s2: 25, s3: 5 |
| `baselines_mopo` | s0: 5, s1: 13, s2: 21, s3: 5 |
| `baselines_morel` | s0: 25, s3: 20 |
| `baselines_nstep_iql_n15` | s0: 25, s2: 2, s3: 5 |
| `baselines_nstep_iql_n5` | s0: 25, s2: 18, s3: 5 |
| `baselines_paper999_rcfm1s_rsnis_multi_H5_uniform` | s0: 25, s1: 12 |
| `baselines_rebrac` | s0: 25, s1: 20, s2: 25, s3: 5 |
| `baselines_reverse_morel` | s0: 25, s2: 1, s3: 25 |
| `baselines_reverse_morel_anchor_bootstrap` | s0: 25, s2: 5, s3: 25 |
| `baselines_reverse_morel_cql` | s0: 25, s3: 20 |
| `baselines_reverse_morel_rebrac` | s0: 6, s2: 5, s3: 18 |
