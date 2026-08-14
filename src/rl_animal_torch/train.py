import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from rl_animal_torch import arena, evaluate
from rl_animal_torch.config import ENV_PATH, TrainingConfig
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv

# every arena file under these roots is drawn from, so a curriculum stage such as
# external/animal-ai/configs/paper_curriculum_split_mirrored/stage03 can be added here
ARENAS = [
    "external/animal-ai/configs/rank1_training_data",
    # "external/animal-ai/configs/paper_curriculum_split_mirrored/stage00",
    # "external/animal-ai/configs/paper_curriculum_split_mirrored/stage01",
    # "external/animal-ai/configs/paper_curriculum_split_mirrored/stage02",
    # "external/animal-ai/configs/paper_curriculum_split_mirrored/stage03",
]
# each instance serves on BASE_PORT + its index
BASE_PORT = 5005


def capture(*command):
    return subprocess.run(command, capture_output=True, text=True).stdout


def write_git_info(path):
    """
    The code a run came from. The diff is worth keeping because the arena roots and every
    hyperparameter are edited in place rather than passed on the command line.
    """
    path.write_text(
        f"branch:\n{capture('git', 'rev-parse', '--abbrev-ref', 'HEAD')}\n"
        f"git show -s:\n{capture('git', 'show', '-s')}\n"
        f"git diff:\n{capture('git', 'diff')}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="suffix of the wandb run and the result directory")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--restore", default=None, help="checkpoint of this trainer")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arena_paths = arena.collect(ARENAS)
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.exp_name}"
    result_dir = Path("results") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    write_git_info(result_dir / "git_info.txt")

    batch = config.num_actors * config.steps_num
    print(
        f"run {run_name}: {len(arena_paths)} arenas, {config.num_actors} actors x "
        f"{config.steps_num} steps = {batch} per batch, {config.max_epochs} epochs, "
        f"{batch * config.max_epochs} steps total"
    )

    vec_env = VecEnv(
        ENV_PATH, arena_paths, config.num_actors, BASE_PORT, args.seed, shape_rewards=True
    )
    run = wandb.init(
        project="rl-animal-torch",
        name=run_name,
        mode=args.wandb_mode,
        dir=str(result_dir),
        config=dict(vars(config), arenas=ARENAS, seed=args.seed),
    )
    trainer = PPOTrainer(vec_env, config, torch.device("cuda"), run)
    if args.restore is not None:
        trainer.restore(args.restore)
        print(f"restored {args.restore} at epoch {trainer.epoch}")

    best_checkpoint = result_dir / "best.pt"
    try:
        trainer.train(result_dir / "last.pt", best_checkpoint)
    finally:
        vec_env.close()
        run.finish()

    # the training instances are down by now, so the evaluation can take the machine
    evaluate.run(
        best_checkpoint, evaluate.CONFIGS, evaluate.NUM_ENVS, evaluate.BASE_PORT, args.seed, 1
    )


if __name__ == "__main__":
    main()
