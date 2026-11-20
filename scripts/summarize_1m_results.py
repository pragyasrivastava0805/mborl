#!/usr/bin/env python3
"""Aggregate all 1M-step evaluation results under results/exp_baselines_1m/scalembrl."""

import csv
import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_ROOT = os.path.join(ROOT, 'results', 'exp_baselines_1m', 'scalembrl')
OUT_DIR = os.path.join(ROOT, 'results')

STITCH_ENVS = [
    ('pointmaze-medium', 'PM-M'),
    ('pointmaze-large', 'PM-L'),
    ('antmaze-medium', 'AM-M'),
    ('antmaze-large', 'AM-L'),
    ('humanoidmaze-medium', 'HM-M'),
]
STITCH_ENV_KEYS = [k for k, _ in STITCH_ENVS]
STITCH_TASKS = ['task1', 'task2', 'task3', 'task4', 'task5']

GROUP_RE = re.compile(
    r'^(.+?)_((?:pointmaze|antmaze|humanoidmaze|scene|puzzle|cube|powderworld|visual|antsoccer)[^/]+)$'
)
STITCH_RE = re.compile(r'^(pointmaze-medium|pointmaze-large|antmaze-medium|antmaze-large|humanoidmaze-medium)-stitch-singletask-(task[1-5])-v0$')
SEED_RE = re.compile(r'^sd(\d+)_')


def parse_eval_csv(path):
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    col = next((k for k in rows[0] if 'success' in k), None)
    if col is None:
        return None
    steps = [int(float(r['step'])) for r in rows if r.get('step', '') != '']
    vals = [float(r[col]) for r in rows if r.get(col, '') != '']
    if not vals:
        return None
    l3 = sum(vals[-3:]) / min(3, len(vals))
    return {
        'n_evals': len(vals),
        'max_step': max(steps) if steps else 0,
        'last_success': vals[-1],
        'l3_success': l3,
        'peak_success': max(vals),
    }


def main():
    records = []
    # Deduplicate if multiple timestamped dirs exist for the same (run_group, seed): keep highest max_step, then latest dir.
    best_by_key = {}

    for eval_path in sorted(glob.glob(os.path.join(EXP_ROOT, '*', 'sd*', 'eval.csv'))):
        run_dir = os.path.dirname(eval_path)
        seed_dir = os.path.basename(run_dir)
        run_group = os.path.basename(os.path.dirname(run_dir))

        gm = GROUP_RE.match(run_group)
        if not gm:
            continue
        method, env_full = gm.group(1), gm.group(2)

        sm = SEED_RE.match(seed_dir)
        seed = int(sm.group(1)) if sm else -1

        parsed = parse_eval_csv(eval_path)
        if parsed is None:
            continue

        stm = STITCH_RE.match(env_full)
        stitch_env = stm.group(1) if stm else ''
        stitch_task = stm.group(2) if stm else ''

        rec = {
            'method': method,
            'env_full': env_full,
            'stitch_env': stitch_env,
            'stitch_task': stitch_task,
            'seed': seed,
            'n_evals': parsed['n_evals'],
            'max_step': parsed['max_step'],
            'completed_1m': parsed['max_step'] >= 1000000,
            'last_success': round(parsed['last_success'] * 100.0, 2),
            'l3_success': round(parsed['l3_success'] * 100.0, 2),
            'peak_success': round(parsed['peak_success'] * 100.0, 2),
            'run_group': run_group,
            'seed_dir': seed_dir,
        }
        key = (run_group, seed)
        prev = best_by_key.get(key)
        if prev is None or (rec['max_step'], rec['seed_dir']) > (prev['max_step'], prev['seed_dir']):
            best_by_key[key] = rec

    records = sorted(best_by_key.values(), key=lambda r: (r['method'], r['env_full'], r['seed']))

    # 1. Write all-runs CSV
    all_runs_path = os.path.join(OUT_DIR, 'baselines_1m_all_runs.csv')
    fields = [
        'method', 'env_full', 'stitch_env', 'stitch_task', 'seed',
        'n_evals', 'max_step', 'completed_1m', 'last_success', 'l3_success', 'peak_success',
        'run_group', 'seed_dir',
    ]
    with open(all_runs_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # Index completed 1M Stitch seed-0 runs
    s0_stitch = defaultdict(dict)  # method -> (env, task) -> rec
    multiseed_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r['stitch_env'] and r['completed_1m']:
            multiseed_counts[r['method']][r['seed']] += 1
            if r['seed'] == 0:
                s0_stitch[r['method']][(r['stitch_env'], r['stitch_task'])] = r

    # 2. All-25 Tasks table (methods with all 25 Stitch tasks completed at 1M, seed 0)
    all25_rows = []
    for method, task_map in s0_stitch.items():
        if len(task_map) < 25:
            continue
        row = {'method': method, 'n_tasks': len(task_map)}
        all_l3, all_pk = [], []
        for env_key, short in STITCH_ENVS:
            el3 = [task_map[(env_key, t)]['l3_success'] for t in STITCH_TASKS]
            epk = [task_map[(env_key, t)]['peak_success'] for t in STITCH_TASKS]
            row[f'{short}_l3'] = round(sum(el3) / len(el3), 2)
            row[f'{short}_peak'] = round(sum(epk) / len(epk), 2)
            all_l3.extend(el3)
            all_pk.extend(epk)
        row['all25_l3'] = round(sum(all_l3) / len(all_l3), 2)
        row['all25_peak'] = round(sum(all_pk) / len(all_pk), 2)
        all25_rows.append(row)
    all25_rows.sort(key=lambda r: (-r['all25_l3'], -r['all25_peak'], r['method']))

    all25_path = os.path.join(OUT_DIR, 'baselines_1m_all25_table.csv')
    all25_fields = ['method', 'n_tasks'] + [
        f'{short}_{m}' for _, short in STITCH_ENVS for m in ('l3', 'peak')
    ] + ['all25_l3', 'all25_peak']
    with open(all25_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all25_fields)
        w.writeheader()
        w.writerows(all25_rows)

    # 3. Task 1 table (methods with all 5 Stitch envs on task1 completed at 1M, seed 0)
    t1_rows = []
    for method, task_map in s0_stitch.items():
        t1_recs = {e: task_map[(e, 'task1')] for e in STITCH_ENV_KEYS if (e, 'task1') in task_map}
        if len(t1_recs) < 5:
            continue
        row = {'method': method, 'n_envs': len(t1_recs)}
        l3s, pks = [], []
        for env_key, short in STITCH_ENVS:
            row[f'{short}_l3'] = t1_recs[env_key]['l3_success']
            row[f'{short}_peak'] = t1_recs[env_key]['peak_success']
            l3s.append(t1_recs[env_key]['l3_success'])
            pks.append(t1_recs[env_key]['peak_success'])
        row['t1_mean_l3'] = round(sum(l3s) / 5.0, 2)
        row['t1_mean_peak'] = round(sum(pks) / 5.0, 2)
        t1_rows.append(row)
    t1_rows.sort(key=lambda r: (-r['t1_mean_l3'], -r['t1_mean_peak'], r['method']))

    t1_path = os.path.join(OUT_DIR, 'baselines_1m_task1_table.csv')
    t1_fields = ['method', 'n_envs'] + [
        f'{short}_{m}' for _, short in STITCH_ENVS for m in ('l3', 'peak')
    ] + ['t1_mean_l3', 't1_mean_peak']
    with open(t1_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=t1_fields)
        w.writeheader()
        w.writerows(t1_rows)

    # 4. Summary JSON
    completed_total = sum(1 for r in records if r['completed_1m'])
    in_progress_total = len(records) - completed_total
    summary = {
        'total_deduped_runs': len(records),
        'completed_1m_runs': completed_total,
        'partial_runs': in_progress_total,
        'unique_methods': len({r['method'] for r in records}),
        'all25_complete_methods_s0': len(all25_rows),
        'task1_complete_methods_s0': len(t1_rows),
        'multiseed_completed_stitch_counts': {
            m: dict(sorted(sc.items())) for m, sc in sorted(multiseed_counts.items()) if any(s > 0 for s in sc)
        },
        'all25_table_s0': all25_rows,
        'task1_table_s0': t1_rows,
    }
    summary_path = os.path.join(OUT_DIR, 'baselines_1m_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # 5. Markdown report
    lines = [
        '# 1M-Step OGBench Results Summary (`baselines` Branch)',
        '',
        f'- **Raw logs included:** `results/exp_baselines_1m/scalembrl/` (`{len(records)}` deduplicated runs: `{completed_total}` completed at 1M steps, `{in_progress_total}` partial/paused checkpoints across `{summary["unique_methods"]}` method groups).',
        '- **Metrics:** `L3` = mean of last 3 evaluations (800k, 900k, 1M); `Peak` = max evaluation across 100k..1M (15 eval episodes per task, seed 0 unless noted).',
        '- **Regenerate:** `python3 scripts/summarize_1m_results.py`',
        '',
        '## 1. All-25 Stitch Tasks Leaderboard (Seed 0, 1M Steps, 25/25 Complete)',
        '',
        '| Method | PM-M | PM-L | AM-M | AM-L | HM-M | All-25 L3 | All-25 Peak |',
        '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in all25_rows:
        lines.append(
            f"| `{r['method']}` | {r['PM-M_l3']:.1f}% | {r['PM-L_l3']:.1f}% | {r['AM-M_l3']:.1f}% | "
            f"{r['AM-L_l3']:.1f}% | {r['HM-M_l3']:.1f}% | **{r['all25_l3']:.1f}%** | {r['all25_peak']:.1f}% |"
        )

    lines += [
        '',
        '## 2. Task 1 Stitch Leaderboard (5 Envs, Seed 0, 1M Steps)',
        '',
        '| Method | PM-M | PM-L | AM-M | AM-L | HM-M | Task-1 L3 | Task-1 Peak |',
        '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in t1_rows:
        lines.append(
            f"| `{r['method']}` | {r['PM-M_l3']:.1f}% | {r['PM-L_l3']:.1f}% | {r['AM-M_l3']:.1f}% | "
            f"{r['AM-L_l3']:.1f}% | {r['HM-M_l3']:.1f}% | **{r['t1_mean_l3']:.1f}%** | {r['t1_mean_peak']:.1f}% |"
        )

    lines += [
        '',
        '## 3. Multi-Seed Completed Stitch Runs (Seeds > 0)',
        '',
        '| Method | Completed 1M Stitch Runs by Seed |',
        '| :--- | :--- |',
    ]
    for m, sc in summary['multiseed_completed_stitch_counts'].items():
        sc_str = ', '.join(f's{k}: {v}' for k, v in sc.items())
        lines.append(f'| `{m}` | {sc_str} |')

    md_path = os.path.join(OUT_DIR, 'RESULTS_1M_SUMMARY.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(
        f'Wrote {len(records)} runs ({completed_total} complete 1M, {in_progress_total} partial); '
        f'{len(all25_rows)} all-25 methods, {len(t1_rows)} Task-1 methods.'
    )


if __name__ == '__main__':
    main()
