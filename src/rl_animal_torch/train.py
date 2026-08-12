"""Train against the Animal-AI v4 environment.

    uv run train --env-path /path/to/animalAI.x86_64
"""
import argparse
import os
import time

import numpy as np
import torch
import wandb

from rl_animal_torch import arena
from rl_animal_torch.config import TRAINING, EnvConfig
from rl_animal_torch.network import AnimalAgent
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv

ARENAS = 'configs/learning/stage3'


class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, values, step):
        self.run.log(values, step=step)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-path', required=True, help='v4 animalAI.x86_64')
    parser.add_argument('--checkpoint-dir', default='nn')
    parser.add_argument('--wandb-project', default='rl-animal-torch')
    parser.add_argument('--wandb-mode', default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--base-port', default=5005, type=int,
                        help='each instance serves on base-port + its index')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-actors', default=None, type=int,
                        help='override the config, for a short run')
    parser.add_argument('--max-epochs', default=None, type=int,
                        help='override the config, for a short run')
    parser.add_argument('--restore', default=None, help='checkpoint of this trainer')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TRAINING
    if args.num_actors is not None:
        config = type(config)(**dict(vars(config), num_actors=args.num_actors))
    if args.max_epochs is not None:
        config = type(config)(**dict(vars(config), max_epochs=args.max_epochs))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arena_paths = arena.collect(ARENAS, refuse_broken_colors=True)
    print(f'{len(arena_paths)} arena files in {ARENAS}, all accepted')

    run_name = time.strftime('%Y%m%d-%H%M%S')
    print(f'run {run_name}')

    env_config = EnvConfig()
    batch = config.num_actors * config.steps_num
    print(f'{config.num_actors} actors x {config.steps_num} steps = '
          f'{batch} per batch, {config.max_epochs} epochs, '
          f'{batch * config.max_epochs} steps total')

    agent = AnimalAgent()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vec_env = VecEnv(args.env_path, arena_paths, config.num_actors, args.base_port,
                     args.seed, env_config, shape_rewards=True)
    run = wandb.init(project=args.wandb_project, name=run_name, mode=args.wandb_mode,
                     config=dict(vars(config), arenas=ARENAS, seed=args.seed))
    trainer = PPOTrainer(agent, vec_env, config, torch.device(args.device),
                         WandbLogger(run))
    if args.restore is not None:
        trainer.restore(args.restore)
        print(f'restored {args.restore} at epoch {trainer.epoch}')

    checkpoint = os.path.join(args.checkpoint_dir, run_name + '.pt')
    best = os.path.join(args.checkpoint_dir, run_name + '_best.pt')
    try:
        trainer.train(checkpoint, best)
    finally:
        vec_env.close()
        run.finish()
    print(f'done, last checkpoint {checkpoint}')


if __name__ == '__main__':
    main()
