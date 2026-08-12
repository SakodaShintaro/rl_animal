"""Train the agent against the Animal-AI v4 environment, in PyTorch only.

    uv run train --env-path /path/to/animalAI.x86_64 \
        --arenas configs/learning/stage3 \
        --run-name v4_torch

Training starts from a fresh network. Continuing an interrupted run takes --restore with
one of this trainer's own checkpoints.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl_animal_torch import arena
from rl_animal_torch.config import CONFIGS, EnvConfig
from rl_animal_torch.network import AnimalAgent
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-path', required=True, help='v4 animalAI.x86_64')
    parser.add_argument('--arenas', required=True, help='directory of arena yaml files')
    parser.add_argument('--run-name', required=True, help='names the checkpoints and the log')
    parser.add_argument('--config', default='stage1', choices=sorted(CONFIGS),
                        help='stage1 is the winning run; stage2 is what it switched to '
                             'after 50 million steps')
    parser.add_argument('--checkpoint-dir', default='nn')
    parser.add_argument('--log-dir', default='runs')
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
    writer = SummaryWriter(os.path.join(args.log_dir, args.run_name))
    trainer = PPOTrainer(agent, vec_env, config, torch.device(args.device), writer)
    if args.restore is not None:
        trainer.restore(args.restore)
        print('restored %s at epoch %d' % (args.restore, trainer.epoch))

    checkpoint = os.path.join(args.checkpoint_dir, args.run_name + '.pt')
    best = os.path.join(args.checkpoint_dir, args.run_name + '_best.pt')
    try:
        trainer.train(checkpoint, best)
    finally:
        vec_env.close()
        writer.close()
    print('done, last checkpoint %s' % checkpoint)


if __name__ == '__main__':
    main()
