"""A light training run against a stand-in environment."""

import numpy as np
import torch

from rl_animal_torch.config import EnvConfig, TrainingConfig
from rl_animal_torch.env import observation_shapes
from rl_animal_torch.network import LSTM_UNITS, AnimalAgent
from rl_animal_torch.ppo import PPO, swap_and_flatten
from rl_animal_torch.runner import Runner, format_duration

SHAPES = observation_shapes(EnvConfig())

LIGHT = TrainingConfig(
    num_actors=4, steps_num=16, minibatch_size=32, mini_epochs=2, seq_len=8, max_epochs=2
)
# what the port target runs: one environment, and the short window that goes with it
SINGLE = TrainingConfig(
    num_actors=1, steps_num=16, minibatch_size=8, mini_epochs=1, seq_len=2, max_epochs=1
)


class NoiseVecEnv:
    def __init__(self, num_actors):
        self.num_actors = num_actors
        self.visual_shape, self.vels_shape = SHAPES
        self.random = np.random.RandomState(0)
        self.steps = 0

    def observations(self):
        visual = self.random.randint(0, 256, (self.num_actors,) + self.visual_shape)
        vels = self.random.normal(size=(self.num_actors,) + self.vels_shape)
        return visual.astype(np.uint8), vels.astype(np.float32)

    def reset(self):
        return self.observations()

    def step(self, actions):
        assert actions.shape == (self.num_actors,)
        self.steps += 1
        rewards = self.random.normal(size=self.num_actors).astype(np.float32)
        dones = np.zeros(self.num_actors, dtype=bool)
        if self.steps % 5 == 0:
            dones[self.random.randint(0, self.num_actors)] = True
        return self.observations(), rewards, rewards + 0.5, dones

    def close(self):
        pass


class UnpaidVecEnv(NoiseVecEnv):
    """
    An arena that pays nothing while the shaping pays every step, so which of the two
    rewards reached which consumer is not a matter of interpretation.
    """

    def step(self, actions):
        self.steps += 1
        raw = np.zeros(self.num_actors, dtype=np.float32)
        shaped = np.ones(self.num_actors, dtype=np.float32)
        return self.observations(), raw, shaped, np.zeros(self.num_actors, dtype=bool)


class SilentLogger:
    def log(self, values, step):
        pass


def make_runner(config):
    device = torch.device("cpu")
    ppo = PPO(AnimalAgent(), config, device)
    return Runner(NoiseVecEnv(config.num_actors), ppo, config, SilentLogger(), show=False)


def test_swap_and_flatten_is_environment_major():
    """
    Environment e at step t has to land at e * steps + t, which is what the recurrent
    unrolling and the sequence slicing both assume.
    """
    steps, actors = 3, 2
    array = np.arange(steps * actors).reshape(steps, actors)
    assert swap_and_flatten(array).tolist() == [0, 2, 4, 1, 3, 5]


def test_rollout_shapes_and_one_state_per_sequence():
    trainer = make_runner(LIGHT)
    rollout = trainer.collect()
    batch = LIGHT.num_actors * LIGHT.steps_num

    assert rollout.visual.shape == (batch,) + SHAPES[0]
    assert rollout.vels.shape == (batch,) + SHAPES[1]
    assert rollout.actions.shape == (batch,)
    assert rollout.returns.shape == (batch,)
    assert rollout.states.shape == (batch // LIGHT.seq_len, 2 * LSTM_UNITS)
    assert rollout.actions.min() >= 0 and rollout.actions.max() < 9


def test_update_moves_the_weights_and_reports_finite_losses():
    trainer = make_runner(LIGHT)
    before = torch.cat([p.detach().reshape(-1).clone() for p in trainer.ppo.agent.parameters()])

    losses = trainer.ppo.update(trainer.collect())
    for name, value in losses.items():
        assert np.isfinite(value), (name, value)

    after = torch.cat([p.detach().reshape(-1).clone() for p in trainer.ppo.agent.parameters()])
    assert torch.max(torch.abs(after - before)).item() > 0


def test_train_runs_and_round_trips_a_checkpoint(tmp_path):
    trainer = make_runner(LIGHT)
    last = str(tmp_path / "last.pt")
    best = str(tmp_path / "best.pt")
    trainer.train(last, best)

    assert trainer.epoch == LIGHT.max_epochs
    assert trainer.frame == LIGHT.max_epochs * LIGHT.num_actors * LIGHT.steps_num

    restored = make_runner(LIGHT).restore(last)
    assert restored.epoch == trainer.epoch
    assert restored.frame == trainer.frame
    for saved, loaded in zip(trainer.ppo.agent.parameters(), restored.ppo.agent.parameters()):
        assert torch.equal(saved.detach(), loaded.detach())


def test_one_actor_collects_and_updates():
    """
    The port target drives a single environment, so the buffer and the update have to hold
    up at width one, with one recurrent state per seq_len window of the one stream.
    """
    runner = make_runner(SINGLE)
    rollout = runner.collect()

    assert rollout.visual.shape[0] == SINGLE.num_actors * SINGLE.steps_num
    assert rollout.states.shape == (SINGLE.steps_num // SINGLE.seq_len, 2 * LSTM_UNITS)
    for name, value in runner.ppo.update(rollout).items():
        assert np.isfinite(value), (name, value)


def test_the_score_is_the_arena_reward_and_the_update_sees_the_shaped_one():
    ppo = PPO(AnimalAgent(), SINGLE, torch.device("cpu"))
    runner = Runner(UnpaidVecEnv(SINGLE.num_actors), ppo, SINGLE, SilentLogger(), show=False)
    rollout = runner.collect()

    assert runner.current_rewards.tolist() == [0.0]
    assert rollout.returns.min().item() > 0.0


def test_format_duration():
    assert format_duration(0) == "0:00:00"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(90061) == "1d 1:01:01"
