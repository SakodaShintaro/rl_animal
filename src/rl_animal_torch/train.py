"""Train the agent against the Animal-AI v4 environment, in PyTorch only.

    uv run train --env-path /path/to/animalAI.x86_64 \
        --arenas configs/learning/stage3 \
        --run-name v4_torch

Metrics go to wandb; --wandb-mode offline keeps them local, disabled drops them.

Training starts from a fresh network. Continuing an interrupted run takes --restore with
one of this trainer's own checkpoints.
"""
import argparse
import os

import numpy as np
import torch
import wandb

from rl_animal_torch import arena
from rl_animal_torch.config import CONFIGS, EnvConfig
from rl_animal_torch.network import AnimalAgent
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv


class WandbLogger:
    """The one method PPOTrainer asks for, over a wandb run.

    The step is the environment frame rather than the epoch, so runs with different batch
    sizes line up on the x axis.
    """
    def __init__(self, run):
        self.run = run

    def log(self, values, step):
        self.run.log(values, step=step)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-path', required=True, help='v4 animalAI.x86_64')
    parser.add_argument('--arenas', required=True, help='directory of arena yaml files')
    parser.add_argument('--run-name', required=True, help='names the checkpoints and the log')
    parser.add_argument('--config', default='stage1', choices=sorted(CONFIGS),
                        help='stage1 is the winning run; stage2 is what it switched to '
                             'after 50 million steps')
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
    config = CONFIGS[args.config]
    if args.num_actors is not None:
        config = type(config)(**dict(vars(config), num_actors=args.num_actors))
    if args.max_epochs is not None:
        config = type(config)(**dict(vars(config), max_epochs=args.max_epochs))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    '''
    Every arena is checked before a single instance is launched: a level the v4 player
    cannot build hangs it silently, and the workers draw levels at random on every reset,
    so one bad file stalls the whole run sooner or later.
    '''
    arena_paths = arena.collect(args.arenas, refuse_broken_colors=True)
    print('%d arena files, all accepted' % len(arena_paths))

    env_config = EnvConfig()
    print('config %s: %d actors x %d steps = %d per batch, %d epochs, %d steps total'
          % (args.config, config.num_actors, config.steps_num,
             config.num_actors * config.steps_num, config.max_epochs,
             config.num_actors * config.steps_num * config.max_epochs))

    agent = AnimalAgent()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vec_env = VecEnv(args.env_path, arena_paths, config.num_actors, args.base_port,
                     args.seed, env_config, shape_rewards=True)
    run = wandb.init(project=args.wandb_project, name=args.run_name, mode=args.wandb_mode,
                     config=dict(vars(config), arenas=args.arenas, seed=args.seed))
    trainer = PPOTrainer(agent, vec_env, config, torch.device(args.device),
                         WandbLogger(run))
    if args.restore is not None:
        trainer.restore(args.restore)
        print('restored %s at epoch %d' % (args.restore, trainer.epoch))

    checkpoint = os.path.join(args.checkpoint_dir, args.run_name + '.pt')
    best = os.path.join(args.checkpoint_dir, args.run_name + '_best.pt')
    try:
        trainer.train(checkpoint, best)
    finally:
        vec_env.close()
        run.finish()
    print('done, last checkpoint %s' % checkpoint)


if __name__ == '__main__':
    main()
