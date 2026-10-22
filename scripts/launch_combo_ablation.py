"""Launch COMBO ablations over beta, the real-data fraction f, and the rollout horizon as a Slurm array.

By default the full grid over beta x f x H is launched. With f = 1 there is no model data, so H has no effect and those
configs are launched once per beta (they are CQL(H) for beta > 0). Use --one_at_a_time to sweep each factor on its own
around the base config (beta=1, f=0.5, H=5) instead. Nothing is submitted without --submit.

Environments are given either as one --env_name/--dataset_dir pair or as a --suite: mac_reward (the reward-based OGBench
benchmark: cube-single, cube-double, scene, puzzle-3x3 and puzzle-4x4, tasks 1-5) or mac_long_horizon (the six
goal-conditioned long-horizon tasks). Each task gets its own Slurm array, plus a submit_all.sh to submit them together.
Single-task environments train with --agent.dataset_class=CDataset.

--node runs everything on one 8-GPU node instead of Slurm arrays: it writes a single queue (seed-major, then config, then
task, so every config is covered on every task before the next seed starts) and node.sbatch, which runs
scripts/run_combo_node.py with one run per GPU and resubmits itself until the queue is empty. Runs keep only their latest
checkpoint and delete it when they finish (about 0.5 GB each), since the disk cannot hold one per run.

Examples:
    # Dry run of the full grid on the reward-based benchmark (25 tasks).
    python scripts/launch_combo_ablation.py --suite mac_reward

    # Dry run: write the commands and sbatch script, print the summary.
    python scripts/launch_combo_ablation.py --env_name=puzzle-4x5-play-oraclerep-v0 \
        --dataset_dir=/scratch/cluster/pragyas/ogbench/prepared/puzzle-4x5-play-100m-v0

    # Submit only the beta sweep around the base config with 3 seeds.
    python scripts/launch_combo_ablation.py --env_name=puzzle-4x5-play-oraclerep-v0 \
        --dataset_dir=... --one_at_a_time --sweeps beta --seeds 0 1 2 --submit
"""
import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OGBENCH_PREPARED = '/scratch/cluster/pragyas/ogbench/prepared'

# MAC long-horizon benchmark (same tasks and datasets as the MAC / model-free runs in this repository).
SUITES = {
    'mac_long_horizon': {
        'humanoidmaze-medium': ('humanoidmaze-medium-navigate-oraclerep-v0', 'humanoidmaze-medium-navigate-v0'),
        'humanoidmaze-giant': ('humanoidmaze-giant-navigate-oraclerep-v0', 'humanoidmaze-giant-navigate-100m-v0'),
        'cube-double': ('cube-double-play-oraclerep-v0', 'cube-double-play-100m-v0'),
        'cube-octuple': ('cube-octuple-play-oraclerep-v0', 'cube-octuple-play-100m-v0'),
        'puzzle-3x3': ('puzzle-3x3-play-oraclerep-v0', 'puzzle-3x3-play-100m-v0'),
        'puzzle-4x5': ('puzzle-4x5-play-oraclerep-v0', 'puzzle-4x5-play-100m-v0'),
    },
    # Standard OGBench reward-based benchmark (1M datasets): five single tasks per environment.
    'mac_reward': {
        f'{env}-task{k}': (f'{env}-play-singletask-task{k}-v0', f'{env}-play-v0')
        for env in ('cube-single', 'cube-double', 'scene', 'puzzle-3x3', 'puzzle-4x4')
        for k in range(1, 6)
    },
}

DEFAULT_BETAS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
DEFAULT_FS = [0.25, 0.5, 0.8, 1.0]
DEFAULT_HORIZONS = [1, 5, 10, 25]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}%{max_parallel}
#SBATCH --output={run_dir}/logs/%A_%a.out
#SBATCH --error={run_dir}/logs/%A_%a.err
#SBATCH --chdir={repo}
#SBATCH --export=ALL
{requeue}
set -euo pipefail
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
# setup_wandb writes runs under mkdtemp(); keep them (and offline runs to sync later) on shared storage.
export TMPDIR={run_dir}/tmp/$SLURM_ARRAY_TASK_ID
mkdir -p "$TMPDIR" wandb_id
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {run_dir}/commands.txt)
echo "[$(date)] task $SLURM_ARRAY_TASK_ID on $(hostname): $CMD"
eval "$CMD"
"""


def fmt(x):
    return f'{x:g}'


def build_configs(args):
    base = dict(beta=args.base_beta, f=args.base_f, horizon=args.base_horizon)
    values = dict(beta=args.betas, f=args.fs, horizon=args.horizons)
    if args.one_at_a_time:
        configs = [{**base, k: v} for k in args.sweeps for v in values[k]]
    else:
        configs = [dict(zip(args.sweeps, combo)) for combo in itertools.product(*(values[k] for k in args.sweeps))]
        configs = [{**base, **c} for c in configs]
    for spec in args.add_configs:
        beta, f, horizon = spec.split(',')
        configs.append(dict(beta=float(beta), f=float(f), horizon=int(horizon)))
    # The rollout horizon is unused at f = 1 (no model data); collapse those configs to one per beta.
    configs = [{**c, 'horizon': args.base_horizon} if c['f'] == 1.0 else c for c in configs]
    unique = []
    for c in configs:
        if c not in unique:
            unique.append(c)
    return unique


NODE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=combo_node
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --signal=B:TERM@600
#SBATCH --output={queue_dir}/logs/node-%j.log
#SBATCH --chdir={repo}
#SBATCH --export=ALL
# Runs the queue with one process per GPU; resubmits itself (after this job ends) while runs are left.
# Stop the chain with: scancel $(tail -1 {queue_dir}/chain.txt)
set -uo pipefail
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
RUNNER="{python} scripts/run_combo_node.py {queue_dir} --max_failures={max_failures}"
LEFT=$($RUNNER --count)
echo "[$(date)] job $SLURM_JOB_ID on $(hostname): $LEFT runs left"
if [ "$LEFT" -gt 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID {queue_dir}/node.sbatch)
  echo "$NEXT" >> {queue_dir}/chain.txt
  echo "Queued follow-up job $NEXT"
fi
$RUNNER &
PID=$!
trap 'kill -TERM $PID' TERM
wait $PID
wait $PID
"""


def build_command(args, env_name, dataset_dir, config, seed, job_id):
    mb_batch_size = args.mb_batch_size
    if args.fixed_rollout_transitions:
        # Keep transitions per rollout round fixed (= mb_batch_size * base horizon) instead of start states.
        mb_batch_size = max(1, args.mb_batch_size * args.base_horizon // config['horizon'])
    flags = [
        f'--env_name={env_name}',
        f'--dataset_dir={dataset_dir}',
        f'--agent=agents/combo.py:{args.profile}',
        f'--agent.beta={fmt(config["beta"])}',
        f'--agent.real_ratio={fmt(config["f"])}',
        f'--agent.rollout_steps={config["horizon"]}',
        f'--agent.conservative_states={args.conservative_states}',
        f'--agent.rollout_policy={args.rollout_policy}',
        f'--agent.model_train_steps={args.model_train_steps}',
        f'--seed={seed}',
        f'--run_group={args.run_group or f"combo_ablation_{env_name}"}',
        f'--job_id={job_id}',
        f'--save_dir={args.save_dir}',
        f'--offline_steps={args.offline_steps}',
        f'--eval_interval={args.eval_interval}',
        f'--save_interval={args.save_interval}',
        f'--eval_episodes={args.eval_episodes}',
        f'--video_episodes={args.video_episodes}',
        f'--mb_batch_size={mb_batch_size}',
        f'--wandb_mode={args.wandb_mode}',
    ]
    if 'singletask' in env_name:
        flags.append('--agent.dataset_class=CDataset')
    if args.node:
        flags += ['--keep_last_checkpoint_only=True', '--delete_checkpoints_on_finish=True']
    return ' '.join([shlex.quote(args.python), 'main_mbpo.py'] + [shlex.quote(f) for f in flags] + args.extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=sorted(SUITES), help='Launch every task of a benchmark suite.')
    parser.add_argument('--tasks', nargs='+', help='Subset of the suite tasks (default: all).')
    parser.add_argument('--env_name', help='Single environment (instead of --suite).')
    parser.add_argument('--dataset_dir', help='Dataset directory for --env_name.')
    parser.add_argument('--data_root', default=OGBENCH_PREPARED,
                        help='Directory holding one subdirectory of .npz files per --suite dataset.')
    parser.add_argument('--profile', default='ogbench', choices=['ogbench', 'paper'])
    parser.add_argument('--sweeps', nargs='+', default=['beta', 'f', 'horizon'], choices=['beta', 'f', 'horizon'])
    parser.add_argument('--one_at_a_time', action='store_true',
                        help='Sweep each selected factor around the base config instead of the full grid.')
    parser.add_argument('--betas', nargs='+', type=float, default=DEFAULT_BETAS)
    parser.add_argument('--fs', nargs='+', type=float, default=DEFAULT_FS)
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--base_beta', type=float, default=1.0)
    parser.add_argument('--base_f', type=float, default=0.5)
    parser.add_argument('--base_horizon', type=int, default=5)
    parser.add_argument('--add_configs', nargs='+', default=[], metavar='BETA,F,H',
                        help='Extra configs appended to the sweep, e.g. 2.5,1,5 for CQL with beta=2.5.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--conservative_states', default='mixture', choices=['mixture', 'model'])
    parser.add_argument('--rollout_policy', default='policy', choices=['policy', 'uniform'])
    parser.add_argument('--fixed_rollout_transitions', action='store_true',
                        help='Scale mb_batch_size by base_horizon / H so every horizon adds the same number of transitions.')
    parser.add_argument('--offline_steps', type=int, default=1000000)
    parser.add_argument('--model_train_steps', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=100000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--eval_episodes', type=int, default=15)
    parser.add_argument('--video_episodes', type=int, default=0)
    parser.add_argument('--mb_batch_size', type=int, default=50000)
    parser.add_argument('--wandb_mode', default='offline', choices=['online', 'offline', 'disabled'],
                        help="'offline' runs are kept under each run directory's tmp/ and can be uploaded with `wandb sync`.")
    parser.add_argument('--run_group', default=None, help='Defaults to combo_ablation_<env_name>.')
    parser.add_argument('--save_dir', default=os.path.join(REPO, 'exp'))
    parser.add_argument('--python', default=os.path.join(REPO, '.venv-mac', 'bin', 'python'))
    parser.add_argument('--partition', default='allnodes')
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--mem', default='64G')
    parser.add_argument('--time', default='3-00:00:00')
    parser.add_argument('--max_parallel', type=int, default=8)
    parser.add_argument('--no_requeue', action='store_true')
    parser.add_argument('--node', action='store_true', help='Run all tasks on one multi-GPU node (see module docstring).')
    parser.add_argument('--node_gpus', type=int, default=8)
    parser.add_argument('--node_chains', type=int, default=1,
                        help='Split the --node queue round-robin into this many independent node chains.')
    parser.add_argument('--node_cpus', type=int, default=40)
    parser.add_argument('--node_mem', default='160G')
    parser.add_argument('--max_failures', type=int, default=2)
    parser.add_argument('--submit', action='store_true', help='Submit with sbatch (default: dry run).')
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Extra flags after `--`, passed to main_mbpo.py.')
    args = parser.parse_args()
    args.extra = [a for a in args.extra if a != '--']

    if args.suite:
        suite = SUITES[args.suite]
        names = args.tasks or list(suite)
        unknown = set(names) - set(suite)
        if unknown:
            sys.exit(f'Unknown tasks for {args.suite}: {sorted(unknown)}')
        envs = [(n, suite[n][0], os.path.join(args.data_root, suite[n][1])) for n in names]
    elif args.env_name and args.dataset_dir:
        envs = [(args.env_name, args.env_name, args.dataset_dir)]
    else:
        sys.exit('Give --suite, or both --env_name and --dataset_dir.')
    missing = [d for _, _, d in envs if not os.path.isdir(d)]
    if missing:
        sys.exit(f'Missing dataset directories: {missing}')
    if args.submit and args.wandb_mode == 'online' and not (
        os.environ.get('WANDB_API_KEY') or os.path.exists(os.path.expanduser('~/.netrc'))
    ):
        sys.exit("--wandb_mode=online needs wandb credentials (wandb login); use --wandb_mode=offline or disabled.")

    configs = build_configs(args)
    print(f'{len(configs)} configs x {len(args.seeds)} seeds = {len(configs) * len(args.seeds)} tasks per environment')
    for c in configs:
        h = '-' if c['f'] == 1.0 else c['horizon']
        tag = '  (CQL)' if c['f'] == 1.0 and c['beta'] > 0 else ('  (no conservatism)' if c['beta'] == 0 else '')
        print(f"  beta={fmt(c['beta']):>5}  f={fmt(c['f']):>4}  H={h:>2}{tag}")

    stamp = time.strftime('%Y%m%dT%H%M%S')
    top_dir = os.path.join(REPO, 'artifacts', 'runs', 'slurm', f'combo-ablation-{args.suite or envs[0][0]}-{stamp}')
    if args.node:
        return launch_node(args, envs, configs, stamp, top_dir)
    sbatch_paths = []
    for task_name, env_name, dataset_dir in envs:
        name = f'combo-{task_name}-{stamp}'
        run_dir = os.path.join(top_dir, task_name) if args.suite else top_dir
        os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)

        tasks = []
        for config in configs:
            for seed in args.seeds:
                job_id = f'{name}-{len(tasks)}'
                command = build_command(args, env_name, dataset_dir, config, seed, job_id)
                tasks.append(dict(**config, env_name=env_name, seed=seed, job_id=job_id, command=command))

        with open(os.path.join(run_dir, 'commands.txt'), 'w') as f:
            f.write('\n'.join(t['command'] for t in tasks) + '\n')
        with open(os.path.join(run_dir, 'tasks.json'), 'w') as f:
            json.dump(dict(args=vars(args), env_name=env_name, dataset_dir=dataset_dir, tasks=tasks), f, indent=2)
        sbatch_path = os.path.join(run_dir, 'combo_ablation.sbatch')
        with open(sbatch_path, 'w') as f:
            f.write(SBATCH_TEMPLATE.format(
                job_name=f'combo_{task_name}'[:32], partition=args.partition, cpus=args.cpus, mem=args.mem,
                time=args.time, last=len(tasks) - 1, max_parallel=args.max_parallel, run_dir=run_dir, repo=REPO,
                requeue='' if args.no_requeue else '#SBATCH --requeue',
            ))
        sbatch_paths.append(sbatch_path)
        print(f'{env_name}: {len(tasks)} tasks -> {run_dir}')

    submit_all = os.path.join(top_dir, 'submit_all.sh')
    with open(submit_all, 'w') as f:
        f.write('#!/bin/bash\nset -euo pipefail\n' + ''.join(f'sbatch {shlex.quote(p)}\n' for p in sbatch_paths))
    os.chmod(submit_all, 0o755)
    total = len(sbatch_paths) * len(configs) * len(args.seeds)
    print(f'Total: {total} tasks in {len(sbatch_paths)} arrays (at most {args.max_parallel} running per array).')

    if not args.submit:
        print(f'Dry run. Submit everything with: {submit_all}')
        return
    jobs = {}
    for path in sbatch_paths:
        result = subprocess.run(['sbatch', '--parsable', path], capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f'sbatch failed for {path}: {result.stderr.strip()} (submitted so far: {jobs})')
        jobs[path] = result.stdout.strip()
        print(f'Submitted {jobs[path]}: {path}')
    with open(os.path.join(top_dir, 'submission.json'), 'w') as f:
        json.dump(jobs, f, indent=2)


def launch_node(args, envs, configs, stamp, top_dir):
    tasks = []
    for seed in args.seeds:
        for c_idx, config in enumerate(configs):
            for task_name, env_name, dataset_dir in envs:
                job_id = f'combo-{task_name}-{stamp}-s{seed}-c{c_idx}'
                command = 'cd {} && {}'.format(
                    shlex.quote(REPO), build_command(args, env_name, dataset_dir, config, seed, job_id))
                tasks.append(dict(index=len(tasks), task=task_name, env_name=env_name, config_index=c_idx, **config,
                                  seed=seed, job_id=job_id, command=command))
    # Round-robin keeps every chain in the same seed-major priority order.
    chains = args.node_chains
    queue_dirs = [top_dir] if chains == 1 else [os.path.join(top_dir, f'chain{k}') for k in range(chains)]
    sbatch_paths = []
    for k, queue_dir in enumerate(queue_dirs):
        chain_tasks = tasks[k::chains]
        os.makedirs(os.path.join(queue_dir, 'logs'), exist_ok=True)
        with open(os.path.join(queue_dir, 'queue.txt'), 'w') as f:
            f.write('\n'.join(t['command'] for t in chain_tasks) + '\n')
        with open(os.path.join(queue_dir, 'tasks.json'), 'w') as f:
            json.dump(dict(args=vars(args), tasks=chain_tasks), f, indent=2)
        sbatch_path = os.path.join(queue_dir, 'node.sbatch')
        with open(sbatch_path, 'w') as f:
            f.write(NODE_TEMPLATE.format(
                partition=args.partition, gpus=args.node_gpus, cpus=args.node_cpus, mem=args.node_mem, time=args.time,
                queue_dir=queue_dir, repo=REPO, python=shlex.quote(args.python), max_failures=args.max_failures,
            ))
        sbatch_paths.append(sbatch_path)
    print(f'Node queue: {len(tasks)} runs ({len(envs)} tasks x {len(configs)} configs x {len(args.seeds)} seeds), '
          f'{chains} chain(s) of {args.node_gpus} GPUs -> {top_dir}')
    if not args.submit:
        print('Dry run. Submit with: ' + '; '.join(f'sbatch {p}' for p in sbatch_paths))
        return
    for queue_dir, sbatch_path in zip(queue_dirs, sbatch_paths):
        result = subprocess.run(['sbatch', '--parsable', sbatch_path], capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f'sbatch failed for {sbatch_path}: {result.stderr.strip()}')
        job = result.stdout.strip()
        with open(os.path.join(queue_dir, 'chain.txt'), 'a') as f:
            f.write(job + '\n')
        print(f'Submitted node job {job}: {sbatch_path}')

if __name__ == '__main__':
    main()
