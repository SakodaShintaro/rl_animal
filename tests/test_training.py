"""A light training run against a stand-in environment."""

import csv
from dataclasses import replace

import numpy as np
import torch

from rl_animal_torch.config import TrainingConfig
from rl_animal_torch.env import observation_shapes
from rl_animal_torch.network import LSTM_UNITS
from rl_animal_torch.ppo import PPOTrainer, format_duration, swap_and_flatten

LIGHT = TrainingConfig(
    num_actors=4, steps_num=16, minibatch_size=32, mini_epochs=2, seq_len=8, max_epochs=2
)


class NoiseVecEnv:
    def __init__(self, num_actors):
        self.num_actors = num_actors
        self.visual_shape, self.vels_shape = observation_shapes()
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
        return self.observations(), rewards, dones

    def close(self):
        pass


class SilentLogger:
    def log(self, values, step):
        pass


def make_trainer(config=LIGHT):
    return PPOTrainer(NoiseVecEnv(config.num_actors), config, torch.device("cpu"), SilentLogger())


def test_swap_and_flatten_is_environment_major():
    """
    Environment e at step t has to land at e * steps + t, which is what the recurrent
    unrolling and the sequence slicing both assume.
    """
    steps, actors = 3, 2
    array = np.arange(steps * actors).reshape(steps, actors)
    assert swap_and_flatten(array).tolist() == [0, 2, 4, 1, 3, 5]


def test_rollout_shapes_and_one_state_per_sequence():
    trainer = make_trainer()
    rollout = trainer.collect()
    batch = LIGHT.num_actors * LIGHT.steps_num

    assert rollout.visual.shape == (batch,) + observation_shapes()[0]
    assert rollout.vels.shape == (batch,) + observation_shapes()[1]
    assert rollout.actions.shape == (batch,)
    assert rollout.returns.shape == (batch,)
    assert rollout.states.shape == (batch // LIGHT.seq_len, 2 * LSTM_UNITS)
    assert rollout.actions.min() >= 0 and rollout.actions.max() < 9


def test_update_moves_the_weights_and_reports_finite_losses():
    trainer = make_trainer()
    before = torch.cat([p.detach().reshape(-1).clone() for p in trainer.agent.parameters()])

    losses = trainer.update(trainer.collect())
    for name, value in losses.items():
        assert np.isfinite(value), (name, value)

    after = torch.cat([p.detach().reshape(-1).clone() for p in trainer.agent.parameters()])
    assert torch.max(torch.abs(after - before)).item() > 0


def test_train_runs_and_round_trips_a_checkpoint(tmp_path):
    trainer = make_trainer()
    trainer.train(tmp_path)

    assert trainer.epoch == LIGHT.max_epochs
    assert trainer.frame == LIGHT.max_epochs * LIGHT.num_actors * LIGHT.steps_num

    restored = make_trainer().restore(tmp_path / "ckpt" / "trainer_last.pt")
    assert restored.epoch == trainer.epoch
    assert restored.frame == trainer.frame
    for saved, loaded in zip(trainer.agent.parameters(), restored.agent.parameters()):
        assert torch.equal(saved.detach(), loaded.detach())


def test_train_logs_every_epoch_and_appends_when_resumed(tmp_path):
    trainer = make_trainer()
    trainer.train(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))

    assert [int(row["epoch"]) for row in rows] == list(range(1, LIGHT.max_epochs + 1))
    assert [int(row["frame"]) for row in rows] == [
        epoch * LIGHT.num_actors * LIGHT.steps_num for epoch in range(1, LIGHT.max_epochs + 1)
    ]
    for row in rows:
        assert np.isfinite(float(row["actor"]))

    # a second session writes no second header and starts where the first stopped
    resumed = make_trainer(replace(LIGHT, max_epochs=2 * LIGHT.max_epochs))
    resumed.restore(tmp_path / "ckpt" / "trainer_last.pt")
    resumed.train(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert [int(row["epoch"]) for row in rows] == list(range(1, 2 * LIGHT.max_epochs + 1))


def test_checkpoints_are_written_every_division_of_the_run(tmp_path):
    """One model-only file per tenth (here: half) of the run, named after the frame it was
    written at, and none of them carrying optimizer state."""
    batch = LIGHT.num_actors * LIGHT.steps_num
    config = replace(LIGHT, max_epochs=4, checkpoint_divisions=2)
    trainer = make_trainer(config)
    trainer.train(tmp_path)

    written = sorted(path.name for path in (tmp_path / "ckpt").glob("model_*.pt"))
    assert written == sorted(
        ["model_best.pt", f"model_{2 * batch:09d}.pt", f"model_{4 * batch:09d}.pt"]
    )

    state = torch.load(tmp_path / "ckpt" / f"model_{4 * batch:09d}.pt", weights_only=False)
    assert "optimizer" not in state
    assert state["frame"] == 4 * batch


def test_format_duration():
    assert format_duration(0) == "0:00:00"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(90061) == "1d 1:01:01"
