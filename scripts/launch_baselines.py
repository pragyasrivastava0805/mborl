"""Launch model-based and model-free baselines on OGBench environments.

Supported baselines:
  Model-based (via main_mbpo.py):
    - mbpo          : MBPO (Janner et al., 2019), unpenalized H-step rollouts (beta = 0.0)
    - mopo          : MOPO (Yu et al., 2020), uncertainty-penalized H-step rollouts (beta = 1.0)
    - morel                          : MOReL (Kidambi et al., 2020), USAD uncertainty-truncated forward P-MDP rollouts (morel_thresh = 0.5, halt_penalty = -10.0)
    - reverse_morel                  : Reverse-MoREL, USAD uncertainty-truncated 1-step reverse rollouts anchored on dataset states (morel_thresh = 0.5)
    - reverse_morel_anchor_bootstrap : Reverse-MoREL-Anchor-Bootstrap, USAD-truncated reverse rollouts where ALL k=1..H synthetic states bootstrap directly on s_anchor in D_real
    - combo                          : COMBO (Yu et al., 2021), conservative Q-learning on real + model rollouts (real_ratio = 0.5, beta = 1.0)
  Model-free single-task & multi-step (via main.py):
    - iql        : IQL (Kostrikov et al., 2022), 1-step expectile Q-learning
    - cql        : CQL (Kumar et al., 2020), conservative Q-learning on real data only
    - rebrac     : ReBRAC (Tarasov et al., 2023), TD3+BC with LayerNorm and target BC penalty
    - nstep_iql  : Same-Episode n-Step IQL (n = 5 by default), strictly in-trajectory n-step backup
  Model-free goal-conditioned & hierarchical (via main.py):
    - gciql      : Goal-Conditioned IQL (Park et al., 2025)
    - qrl        : Quasimetric RL (Wang et al., 2023; Park et al., 2025)
    - crl        : Contrastive RL (Eysenbach et al., 2022; Park et al., 2025)
    - hiql       : Hierarchical Implicit Q-Learning (Park et al., 2023)
"""
import argparse
import json
import os
import shlex
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = '/home/rchitale_google_com/ogbench/prepared'

MODEL_BASED_AGENTS = ('mbpo', 'mopo', 'morel', 'reverse_morel', 'reverse_morel_anchor_bootstrap', 'combo')
SINGLE_TASK_AGENTS = ('mbpo', 'mopo', 'morel', 'reverse_morel', 'reverse_morel_anchor_bootstrap', 'combo', 'iql', 'cql', 'rebrac', 'nstep_iql', 'nstep_iql_n5', 'nstep_iql_n15')
GOAL_COND_AGENTS = ('gciql', 'qrl', 'crl', 'hiql')
ALL_BASELINES = (
    'mbpo', 'mopo', 'morel', 'reverse_morel', 'reverse_morel_anchor_bootstrap', 'combo',
    'iql', 'cql', 'rebrac', 'nstep_iql_n5', 'nstep_iql_n15',
    'gciql', 'qrl', 'crl', 'hiql',
)

STITCH_ENVS = (
    'pointmaze-medium',
    'pointmaze-large',
    'antmaze-medium',
    'antmaze-large',
    'humanoidmaze-medium',
)

REWARD_ENVS = (
    'cube-single',
    'cube-double',
    'scene',
    'puzzle-3x3',
    'puzzle-4x4',
)

SUITES = {
    'stitch_task1': {
        f'{env}-task1': (f'{env}-stitch-singletask-task1-v0', f'{env}-stitch-v0')
        for env in STITCH_ENVS
    },
    'stitch_singletask': {
        f'{env}-task{k}': (f'{env}-stitch-singletask-task{k}-v0', f'{env}-stitch-v0')
        for env in STITCH_ENVS
        for k in range(1, 6)
    },
    'stitch_gc': {
        env: (f'{env}-stitch-v0', f'{env}-stitch-v0')
        for env in STITCH_ENVS
    },
    'mac_reward_task1': {
        f'{env}-task1': (f'{env}-play-singletask-task1-v0', f'{env}-play-v0')
        for env in REWARD_ENVS
    },
    'mac_reward': {
        f'{env}-task{k}': (f'{env}-play-singletask-task{k}-v0', f'{env}-play-v0')
        for env in REWARD_ENVS
        for k in range(1, 6)
    },
    'mac_long_horizon': {
        'humanoidmaze-medium': ('humanoidmaze-medium-navigate-oraclerep-v0', 'humanoidmaze-medium-navigate-v0'),
        'humanoidmaze-giant': ('humanoidmaze-giant-navigate-oraclerep-v0', 'humanoidmaze-giant-navigate-100m-v0'),
        'cube-double': ('cube-double-play-oraclerep-v0', 'cube-double-play-100m-v0'),
        'cube-octuple': ('cube-octuple-play-oraclerep-v0', 'cube-octuple-play-100m-v0'),
        'puzzle-3x3': ('puzzle-3x3-play-oraclerep-v0', 'puzzle-3x3-play-100m-v0'),
        'puzzle-4x5': ('puzzle-4x5-play-oraclerep-v0', 'puzzle-4x5-play-100m-v0'),
    },
}


def build_command(args, baseline, env_name, dataset_dir, seed, job_id):
    is_singletask = 'singletask' in env_name
    entrypoint = 'main_mbpo.py' if baseline in MODEL_BASED_AGENTS else 'main.py'
    agent_module = 'nstep_iql' if baseline.startswith('nstep_iql') else baseline

    flags = [
        f'--env_name={env_name}',
        f'--dataset_dir={dataset_dir}',
        f'--agent=agents/{agent_module}.py',
        f'--seed={seed}',
        f'--run_group={args.run_group or f"baselines_{baseline}_{env_name}"}',
        f'--job_id={job_id}',
        f'--save_dir={args.save_dir}',
        f'--offline_steps={args.offline_steps}',
        f'--eval_interval={args.eval_interval}',
        f'--save_interval={args.save_interval}',
        f'--eval_episodes={args.eval_episodes}',
        f'--video_episodes=0',
        f'--wandb_mode={args.wandb_mode}',
        '--keep_last_checkpoint_only=True',
        '--delete_checkpoints_on_finish=True',
    ]

    if baseline in MODEL_BASED_AGENTS:
        flags.append(f'--mb_batch_size={args.mb_batch_size}')
        if is_singletask:
            flags.append('--agent.dataset_class=CDataset')
        else:
            flags.append('--agent.dataset_class=GCDataset')
    elif baseline == 'nstep_iql_n5':
        flags.append('--agent.nstep=5')
        flags.append('--agent.dataset_class=NStepCDataset')
    elif baseline == 'nstep_iql_n15':
        flags.append('--agent.nstep=15')
        flags.append('--agent.dataset_class=NStepCDataset')
    elif baseline == 'nstep_iql':
        flags.append(f'--agent.nstep={args.nstep}')
        flags.append('--agent.dataset_class=NStepCDataset')
    elif baseline in ('iql', 'cql', 'rebrac'):
        flags.append(f'--agent.dataset_class={"CDataset" if is_singletask else "GCDataset"}')

    return 'cd {} && {} {} {}'.format(
        shlex.quote(args.repo),
        shlex.quote(args.python),
        entrypoint,
        ' '.join(shlex.quote(f) for f in flags),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', default='stitch_task1', choices=sorted(SUITES))
    parser.add_argument('--baselines', nargs='+', default=list(ALL_BASELINES), choices=list(ALL_BASELINES) + ['nstep_iql'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--nstep', type=int, default=5, help='Horizon n for nstep_iql.')
    parser.add_argument('--offline_steps', type=int, default=500000)
    parser.add_argument('--eval_interval', type=int, default=100000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--eval_episodes', type=int, default=15)
    parser.add_argument('--mb_batch_size', type=int, default=50000)
    parser.add_argument('--wandb_mode', default='offline', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--run_group', default=None)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--save_dir', default=os.path.join(REPO, 'exp_baselines'))
    parser.add_argument('--python', default=os.path.join(REPO, '.venv', 'bin', 'python'))
    parser.add_argument('--queue_dir', default=None)
    args = parser.parse_args()

    stamp = time.strftime('%Y%m%dT%H%M%S')
    queue_dir = args.queue_dir or os.path.join(args.repo, 'artifacts', 'runs', f'baselines-{args.suite}-{stamp}')
    os.makedirs(os.path.join(queue_dir, 'logs'), exist_ok=True)

    tasks = []
    for seed in args.seeds:
        for baseline in args.baselines:
            # Goal-conditioned baselines (gciql, qrl, crl, sharsa) evaluate all 5 tasks simultaneously on stitch_gc
            if baseline in GOAL_COND_AGENTS and args.suite in ('stitch_task1', 'stitch_singletask'):
                suite_map = SUITES['stitch_gc']
            else:
                suite_map = SUITES[args.suite]

            for task_name, (env_name, ds_subdir) in suite_map.items():
                dataset_dir = os.path.join(args.data_root, ds_subdir)
                job_id = f'base-{baseline}-{task_name}-s{seed}'
                cmd = build_command(args, baseline, env_name, dataset_dir, seed, job_id)
                tasks.append(dict(
                    index=len(tasks),
                    baseline=baseline,
                    task=task_name,
                    env_name=env_name,
                    seed=seed,
                    job_id=job_id,
                    command=cmd,
                ))

    with open(os.path.join(queue_dir, 'queue.txt'), 'w') as f:
        f.write('\n'.join(t['command'] for t in tasks) + '\n')
    with open(os.path.join(queue_dir, 'tasks.json'), 'w') as f:
        json.dump(dict(args=vars(args), tasks=tasks), f, indent=2)

    print(f'Wrote {len(tasks)} baseline commands to {queue_dir}/queue.txt')


if __name__ == '__main__':
    main()
