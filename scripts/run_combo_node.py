"""Run a COMBO launch queue on one node: one main_mbpo.py process per GPU, refilled as runs finish.

The queue is the queue.txt written by launch_combo_ablation.py --node (one command per line, in priority order). A run is
finished when its line index is listed in done.txt. Interrupted runs resume from their latest checkpoint the next time
the queue is processed, because every command carries a stable --job_id. A run that exits with an error is recorded in
failures.txt and retried until it has failed --max_failures times. On SIGTERM (Slurm time limit) all children are stopped;
that does not count as a failure, and they resume later.
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time


def read_set(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {int(line) for line in f if line.strip()}


def read_counts(path):
    counts = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    counts[int(line)] = counts.get(int(line), 0) + 1
    return counts


def append(path, index):
    with open(path, 'a') as f:
        f.write(f'{index}\n')


def remaining(queue_dir, max_failures):
    with open(os.path.join(queue_dir, 'queue.txt')) as f:
        commands = [line.rstrip('\n') for line in f if line.strip()]
    done = read_set(os.path.join(queue_dir, 'done.txt'))
    failures = read_counts(os.path.join(queue_dir, 'failures.txt'))
    todo = [i for i in range(len(commands)) if i not in done and failures.get(i, 0) < max_failures]
    return commands, todo, done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('queue_dir')
    parser.add_argument('--max_failures', type=int, default=2, help='Give up on a run after this many failed exits.')
    parser.add_argument('--count', action='store_true', help='Print the number of runs left and exit.')
    args = parser.parse_args()

    commands, todo, done = remaining(args.queue_dir, args.max_failures)
    if args.count:
        print(len(todo))
        return

    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = visible.split(',') if visible else [str(i) for i in range(int(os.environ.get('NUM_GPUS', 8)))]
    cpus = sorted(os.sched_getaffinity(0))
    cpu_sets = [cpus[k * len(cpus) // len(gpus): (k + 1) * len(cpus) // len(gpus)] for k in range(len(gpus))]
    log_dir = os.path.join(args.queue_dir, 'logs', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    print(f'{len(done)} done, {len(todo)} to run on GPUs {gpus}', flush=True)

    active = {}  # slot -> (process, index, log file)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(f'Signal {signum}: stopping {len(active)} runs (they resume from their latest checkpoint).', flush=True)
        for process, _, _ in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)

    pending = iter(todo)
    exhausted = False
    while active or not (exhausted or stopping):
        for slot, gpu in enumerate(gpus):
            if slot in active or exhausted or stopping:
                continue
            index = next(pending, None)
            if index is None:
                exhausted = True
                continue
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            env['TMPDIR'] = os.path.join(args.queue_dir, 'tmp', str(index))
            os.makedirs(env['TMPDIR'], exist_ok=True)
            command = ['taskset', '--cpu-list', ','.join(map(str, cpu_sets[slot])), 'bash', '-c', commands[index]]
            log = open(os.path.join(log_dir, f'{index:05d}.log'), 'a')
            log.write(f'\n[{time.ctime()}] GPU {gpu}: {commands[index]}\n')
            log.flush()
            active[slot] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), index, log)
            print(f'START {index} GPU={gpu}', flush=True)
        for slot, (process, index, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[slot]
            if status == 0:
                append(os.path.join(args.queue_dir, 'done.txt'), index)
                print(f'DONE {index}', flush=True)
            elif stopping:
                print(f'STOPPED {index} (resumes later)', flush=True)
            else:
                append(os.path.join(args.queue_dir, 'failures.txt'), index)
                print(f'FAIL {index} exit={status} (log: {log_dir}/{index:05d}.log)', flush=True)
        if active:
            time.sleep(5)

    _, todo, _ = remaining(args.queue_dir, args.max_failures)
    print(f'Exiting with {len(todo)} runs left.', flush=True)


if __name__ == '__main__':
    sys.exit(main())
