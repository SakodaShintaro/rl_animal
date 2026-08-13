import argparse
import os
import time

import numpy as np
import torch
import wandb

from rl_animal_torch import arena
from rl_animal_torch.config import ENV_PATH, TrainingConfig
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv

# every arena file under these roots is drawn from, so a curriculum stage such as
# external/animal-ai/configs/paper_curriculum_split_mirrored/stage03 can be added here
ARENAS = ['external/animal-ai/configs/rank1_training_data']
RESULT_DIR = 'results'
WANDB_PROJECT = 'rl-animal-torch'
# each instance serves on BASE_PORT + its index
BASE_PORT = 5005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', help='suffix of the wandb run and the result directory')
    parser.add_argument('--wandb_mode', default='online',
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--restore', default=None, help='checkpoint of this trainer')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arena_paths = arena.collect(ARENAS)
    run_name = f'{time.strftime("%Y%m%d_%H%M%S")}_{args.exp_name}'
    result_dir = os.path.join(RESULT_DIR, run_name)
    os.makedirs(result_dir, exist_ok=True)

    batch = config.num_actors * config.steps_num
    print(f'run {run_name}: {len(arena_paths)} arenas, {config.num_actors} actors x '
          f'{config.steps_num} steps = {batch} per batch, {config.max_epochs} epochs, '
          f'{batch * config.max_epochs} steps total')

    vec_env = VecEnv(ENV_PATH, arena_paths, config.num_actors, BASE_PORT, args.seed,
                     shape_rewards=True)
    run = wandb.init(project=WANDB_PROJECT, name=run_name, mode=args.wandb_mode,
                     dir=result_dir, config=dict(vars(config), arenas=ARENAS, seed=args.seed))
    trainer = PPOTrainer(vec_env, config, torch.device('cuda'), run)
    if args.restore is not None:
        trainer.restore(args.restore)
        print(f'restored {args.restore} at epoch {trainer.epoch}')

    try:
        trainer.train(os.path.join(result_dir, 'last.pt'),
                      os.path.join(result_dir, 'best.pt'))
    finally:
        vec_env.close()
        run.finish()


if __name__ == '__main__':
    main()
