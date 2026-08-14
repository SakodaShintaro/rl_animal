"""A light training run and a light evaluation against the real Animal-AI v4 player.

Skipped unless the player binary is installed.
"""

import csv
import os

import numpy as np
import pytest
import torch

from rl_animal_torch import arena
from rl_animal_torch.config import ENV_PATH, TrainingConfig
from rl_animal_torch.env import AnimalEnv, observation_shapes
from rl_animal_torch.evaluate import build_tasks, evaluate
from rl_animal_torch.network import AnimalAgent
from rl_animal_torch.ppo import PPOTrainer
from rl_animal_torch.vec_env import VecEnv

ARENAS = ["external/animal-ai/configs/rank1_training_data"]
BASE_PORT = 7100


@pytest.fixture(scope="module")
def env_path():
    if not os.path.exists(ENV_PATH):
        pytest.skip(f"{ENV_PATH} does not exist")
    return ENV_PATH


class SilentLogger:
    def log(self, values, step):
        pass


def test_one_instance_observes_and_steps(env_path, tmp_path):
    paths = arena.collect(ARENAS)
    env = AnimalEnv(env_path, paths, 0, BASE_PORT, 0, shape_rewards=True)
    try:
        visual, vels = env.reset()
        visual_shape, vels_shape = observation_shapes()
        assert visual.shape == visual_shape and visual.dtype == np.uint8
        assert vels.shape == vels_shape

        first_time = vels[-1]
        for _ in range(5):
            (_, vels), reward, done = env.step(3)
            assert np.isfinite(reward)
            assert isinstance(done, bool)
        assert vels[-1] < first_time
    finally:
        env.close()


def test_light_training(env_path, tmp_path):
    """
    Two instances and two epochs: enough to prove the worker processes, the rollout and the
    update all work against the real player.
    """
    config = TrainingConfig(
        num_actors=2, steps_num=16, minibatch_size=16, mini_epochs=1, seq_len=8, max_epochs=2
    )
    paths = arena.collect(ARENAS)
    vec_env = VecEnv(env_path, paths, config.num_actors, BASE_PORT + 10, 0, shape_rewards=True)
    try:
        trainer = PPOTrainer(vec_env, config, torch.device("cpu"), SilentLogger())
        before = torch.cat([p.detach().reshape(-1).clone() for p in trainer.agent.parameters()])
        trainer.train(str(tmp_path / "last.pt"), str(tmp_path / "best.pt"))
        after = torch.cat([p.detach().reshape(-1).clone() for p in trainer.agent.parameters()])
    finally:
        vec_env.close()

    assert trainer.epoch == config.max_epochs
    assert torch.max(torch.abs(after - before)).item() > 0
    assert os.path.exists(str(tmp_path / "last.pt"))


def test_light_evaluation(env_path, tmp_path):
    """
    Three scenarios through one instance, scored the way the competition scores them.
    """
    paths = arena.collect(ARENAS)[:3]
    envs = [AnimalEnv(env_path, paths, 0, BASE_PORT + 20, 0, shape_rewards=False)]
    output = tmp_path / "results.csv"
    try:
        agent = AnimalAgent().eval()
        with open(str(output), "w", newline="") as out_file:
            fields = ["scenario", "category", "pass_mark", "reward", "passed", "steps"]
            writer = csv.DictWriter(out_file, fieldnames=fields)
            writer.writeheader()
            rows = evaluate(
                envs, agent, build_tasks(paths, 1), writer, torch.device("cpu"), log_every=1000
            )
    finally:
        for env in envs:
            env.close()

    assert len(rows) == len(paths)
    for row in rows:
        assert row["steps"] > 0
        assert np.isfinite(row["reward"])
        assert row["passed"] in (0, 1)

    written = list(csv.DictReader(open(str(output))))
    assert len(written) == len(paths)
