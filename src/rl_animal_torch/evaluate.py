"""Score a checkpoint on the Animal-AI Olympics competition scenarios, in PyTorch only.

    uv run evaluate --env-path /path/to/animalAI.x86_64 \
        --configs /path/to/animal-ai/configs/competition \
        --checkpoint nn/v4_torch_best.pt \
        --output results.csv

Each scenario carries a pass_mark and is passed when the episode's total environment
reward reaches it. The rewards used here are the raw ones, not the shaped ones training
uses, so the numbers stay comparable to the competition score.

Every environment keeps sending its last observation once it has run out of scenarios,
because the recurrent state is laid out one slot per environment and dropping a slot would
shift the rest.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl_animal_torch import arena
from rl_animal_torch.config import EnvConfig
from rl_animal_torch.env import AnimalEnv
from rl_animal_torch.network import AnimalAgent, load_from_reference


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-path', required=True, help='v4 animalAI.x86_64')
    parser.add_argument('--configs', required=True, help='directory of scenario yaml files')
    parser.add_argument('--checkpoint', required=True,
                        help='a .pt from train, or an npz of TensorFlow weights')
    parser.add_argument('--output', required=True, help='csv to write')
    parser.add_argument('--num-envs', default=12, type=int)
    parser.add_argument('--episodes', default=1, type=int, help='episodes per scenario')
    parser.add_argument('--base-port', default=5900, type=int)
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stride', default=1, type=int,
                        help='take every Nth scenario, for a quick look')
    return parser.parse_args()


def load_agent(path, device):
    if path.endswith('.npz'):
        return load_from_reference(path).to(device).eval()

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
    print('pass rate: %.4f (%d / %d episodes)' % (np.mean(passed), sum(passed), len(rows)))
    print('mean reward: %.4f' % np.mean([row['reward'] for row in rows]))
    for category in sorted({row['category'] for row in rows}, key=int):
        in_category = [row['passed'] for row in rows if row['category'] == category]
        print('  category %-3s pass rate %.3f (%d episodes)'
              % (category, np.mean(in_category), len(in_category)))


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
        '''
        A v4 episode ends on its own after t * physics_steps_per_t physics steps; the
        margin is only a guard against an instance that never reports terminal.
        '''
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
                print('%d episodes, %.0f steps/s, %.1f min elapsed, pass rate so far %.3f'
                      % (len(rows), total_steps / elapsed, elapsed / 60.0,
                         np.mean([entry['passed'] for entry in rows])), flush=True)

    return rows


def main():
    args = parse_args()
    config = EnvConfig()

    paths = arena.collect(args.configs, refuse_broken_colours=False)[::args.stride]
    print('scenarios: %d, episodes each: %d, envs: %d'
          % (len(paths), args.episodes, args.num_envs))

    device = torch.device(args.device)
    agent = load_agent(args.checkpoint, device)
    print('loaded ' + args.checkpoint)

    scratch = os.path.join(os.path.dirname(os.path.abspath(args.output)) or '.', '.arenas')
    os.makedirs(scratch, exist_ok=True)
    envs = [AnimalEnv(args.env_path, paths, index, args.base_port, args.seed + index,
                      config, shape_rewards=False, scratch_dir=scratch)
            for index in range(args.num_envs)]
    try:
        with open(args.output, 'w', buffering=1, newline='') as out_file:
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
