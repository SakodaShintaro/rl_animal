import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl_animal_torch import arena
from rl_animal_torch.config import ENV_PATH, EnvConfig
from rl_animal_torch.env import AnimalEnv
from rl_animal_torch.network import AnimalAgent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--configs', default='configs/learning/competition_configurations',
                        help='directory of scenario yaml files')
    parser.add_argument('--num_envs', default=12, type=int)
    parser.add_argument('--episodes', default=1, type=int, help='episodes per scenario')
    parser.add_argument('--base_port', default=5900, type=int)
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stride', default=1, type=int,
                        help='take every Nth scenario, for a quick look')
    return parser.parse_args()


def load_agent(path, device):
    agent = AnimalAgent()
    state = torch.load(path, map_location=device)
    agent.load_state_dict(state['model'] if 'model' in state else state)
    return agent.to(device).eval()


def category_of(path):
    return os.path.basename(path).split('-')[0]


def build_tasks(paths, episodes, num_envs):
    '''
    Scenarios are dealt out up front, so a run is reproducible in which environment sees
    which scenario.
    '''
    tasks = [[] for _ in range(num_envs)]
    for index, path in enumerate(paths):
        for episode in range(episodes):
            tasks[index % num_envs].append((path, episode))

    return tasks


def report(rows):
    passed = [row['passed'] for row in rows]
    mean_reward = np.mean([row['reward'] for row in rows])
    print(f'pass rate: {np.mean(passed):.4f} ({sum(passed)} / {len(rows)} episodes)')
    print(f'mean reward: {mean_reward:.4f}')
    for category in sorted({row['category'] for row in rows}):
        in_category = [row['passed'] for row in rows if row['category'] == category]
        print(f'  category {category:<3} pass rate {np.mean(in_category):.3f} '
              f'({len(in_category)} episodes)')


@torch.no_grad()
def evaluate(envs, agent, tasks, writer, config, device, log_every):
    num_envs = len(envs)
    state = agent.initial_state(num_envs, device=device)
    cursor = [0] * num_envs
    pass_mark = [0.0] * num_envs
    max_steps = [0] * num_envs
    reward = [0.0] * num_envs
    steps = [0] * num_envs
    active = [False] * num_envs
    dones = np.ones(num_envs, dtype=np.float32)
    observations = [None] * num_envs
    rows = []
    total_steps = 0
    start = time.time()

    def start_task(index):
        path, _ = tasks[index][cursor[index]]
        raw = open(path).read()
        pass_mark[index] = arena.read_pass_mark(raw)
        max_steps[index] = (arena.read_arena_time(raw) * config.physics_steps_per_t
                            // config.decision_period + 100)
        reward[index] = 0.0
        steps[index] = 0
        active[index] = True
        dones[index] = 1.0
        observations[index] = envs[index].reset_to(path)

    for index in range(num_envs):
        start_task(index)

    while any(active):
        running = [index for index in range(num_envs) if active[index]]
        visual = torch.as_tensor(np.asarray([entry[0] for entry in observations]),
                                 device=device)
        vels = torch.as_tensor(np.asarray([entry[1] for entry in observations],
                                         dtype=np.float32), device=device)
        logits, _, state = agent(visual, vels,
                                 state, torch.as_tensor(dones, device=device), num_envs)
        actions = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1).cpu().numpy()
        dones = np.zeros(num_envs, dtype=np.float32)

        for index in running:
            envs[index].send(int(actions[index]))
        for index in running:
            observations[index], step_reward, done = envs[index].receive()
            reward[index] += step_reward
            steps[index] += 1
            total_steps += 1
            if not done and steps[index] < max_steps[index]:
                continue

            path, episode = tasks[index][cursor[index]]
            row = {'scenario': os.path.basename(path), 'category': category_of(path),
                   'pass_mark': pass_mark[index], 'episode': episode,
                   'reward': reward[index],
                   # a scenario is passed when the raw episode reward reaches its pass_mark
                   'passed': int(reward[index] >= pass_mark[index]),
                   'steps': steps[index]}
            rows.append(row)
            writer.writerow(row)
            cursor[index] += 1
            if cursor[index] < len(tasks[index]):
                start_task(index)
            else:
                active[index] = False

            if len(rows) % log_every == 0:
                elapsed = time.time() - start
                rate = np.mean([entry['passed'] for entry in rows])
                print(f'{len(rows)} episodes, {total_steps / elapsed:.0f} steps/s, '
                      f'{elapsed / 60.0:.1f} min elapsed, pass rate so far {rate:.3f}',
                      flush=True)

    return rows


def main():
    args = parse_args()
    config = EnvConfig()

    paths = arena.collect(args.configs, refuse_broken_colors=False)[::args.stride]
    print(f'scenarios: {len(paths)}, episodes each: {args.episodes}, '
          f'envs: {args.num_envs}')

    device = torch.device(args.device)
    agent = load_agent(args.checkpoint, device)
    stem = os.path.splitext(os.path.abspath(args.checkpoint))[0]
    output = f'{stem}_eval_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    print(f'loaded {args.checkpoint}, writing {output}')

    scratch = os.path.join(os.path.dirname(stem), '.arenas')
    os.makedirs(scratch, exist_ok=True)
    envs = [AnimalEnv(ENV_PATH, paths, index, args.base_port, args.seed + index,
                      config, shape_rewards=False, scratch_dir=scratch)
            for index in range(args.num_envs)]
    try:
        with open(output, 'w', buffering=1, newline='') as out_file:
            fields = ['scenario', 'category', 'pass_mark', 'episode', 'reward', 'passed',
                      'steps']
            writer = csv.DictWriter(out_file, fieldnames=fields)
            writer.writeheader()
            rows = evaluate(envs, agent, build_tasks(paths, args.episodes, args.num_envs),
                            writer, config, device, 25)
    finally:
        for env in envs:
            env.close()

    report(rows)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
