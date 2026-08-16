"""
The training loop. It owns the environments, the recurrent state carried from one batch
to the next, and the reporting; PPO is called for the only two things PPO does.
"""

import time
from collections import deque

import numpy as np
import torch

from rl_animal_torch import display
from rl_animal_torch.ppo import RolloutBuffer


def format_duration(seconds):
    """
    A full run is tens of hours, so days are worth separating out.
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours}:{minutes:02d}:{seconds:02d}"

    return f"{hours}:{minutes:02d}:{seconds:02d}"


class Runner:
    def __init__(self, vec_env, ppo, config, logger, show):
        self.vec_env = vec_env
        self.ppo = ppo
        self.config = config
        self.logger = logger
        # a single environment means a human is watching rather than a cluster running
        self.show = show
        self.device = ppo.device

        self.state = ppo.initial_state(config.num_actors)
        self.dones = torch.zeros(config.num_actors, device=self.device)
        self.current_rewards = np.zeros(config.num_actors, dtype=np.float32)
        self.episode_rewards = deque([], maxlen=1000)
        self.batch_size = config.steps_num * config.num_actors
        self.frame = 0
        self.epoch = 0

    def to_device(self, visual, vels):
        return (
            torch.as_tensor(visual, device=self.device),
            torch.as_tensor(vels, device=self.device),
        )

    @torch.no_grad()
    def collect(self):
        config = self.config
        buffer = RolloutBuffer()

        visual, vels = self.to_device(*self.vec_env.observations())
        for _ in range(config.steps_num):
            state = self.state
            actions, values, neglogpacs, self.state = self.ppo.act(
                visual, vels, state, self.dones, config.num_actors
            )
            frames = visual.cpu().numpy()

            (next_visual, next_vels), rewards, shaped, dones = self.vec_env.step(
                actions.cpu().numpy()
            )
            buffer.add(
                visual=frames,
                vels=vels.cpu().numpy(),
                rewards=shaped,
                actions=actions.cpu().numpy(),
                values=values.cpu().numpy(),
                dones=self.dones.cpu().numpy(),
                neglogpacs=neglogpacs.cpu().numpy(),
                states=state.cpu().numpy(),
            )
            if self.show:
                display.show(
                    frames[0], f"epoch {self.epoch} action {actions[0]} reward {rewards[0]:+.3f}"
                )

            # the score that gets reported is the arena's own reward, never the shaped one
            self.current_rewards += rewards
            for reward, done in zip(self.current_rewards, dones):
                if done:
                    self.episode_rewards.append(float(reward))
            self.current_rewards = self.current_rewards * (1.0 - dones)

            self.dones = torch.as_tensor(dones.astype(np.float32), device=self.device)
            visual, vels = self.to_device(next_visual, next_vels)

        _, last_values, _, self.state = self.ppo.act(
            visual, vels, self.state, self.dones, config.num_actors
        )
        return buffer.finish(
            last_values.cpu().numpy(),
            self.dones.cpu().numpy(),
            config.gamma,
            config.lam,
            config.seq_len,
            self.device,
        )

    def train(self, checkpoint_path, best_checkpoint_path):
        config = self.config
        best_reward = -float("inf")
        self.vec_env.reset()
        started = time.time()
        epochs_this_session = 0
        while self.epoch < config.max_epochs:
            self.epoch += 1
            epochs_this_session += 1
            self.frame += self.batch_size

            collect_start = time.time()
            rollout = self.collect()
            collect_time = time.time() - collect_start

            update_start = time.time()
            losses = self.ppo.update(rollout)
            update_time = time.time() - update_start

            steps_per_second = self.batch_size / (collect_time + update_time)
            elapsed = time.time() - started
            remaining = (elapsed / epochs_this_session) * (config.max_epochs - self.epoch)

            values = {
                "epoch": self.epoch,
                "performance/steps_per_second": steps_per_second,
                "performance/collect_time": collect_time,
                "performance/update_time": update_time,
                "performance/elapsed_hours": elapsed / 3600.0,
                "performance/eta_hours": remaining / 3600.0,
            }
            for name, value in losses.items():
                values["losses/" + name] = value

            report = (
                f"epoch {self.epoch}/{config.max_epochs}  frame {self.frame}  "
                f"{steps_per_second:.0f} steps/s  "
                f"elapsed {format_duration(elapsed)}  "
                f"eta {format_duration(remaining)}  "
                f"actor {losses['actor']:.4f}  critic {losses['critic']:.4f}  "
                f"entropy {losses['entropy']:.4f}"
            )
            if len(self.episode_rewards) > 0:
                mean_reward = float(np.mean(self.episode_rewards))
                values["mean_reward"] = mean_reward
                report += f"  mean reward {mean_reward:.4f}"
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save(best_checkpoint_path)
                    report += " (best, saved)"
            print(report, flush=True)
            self.logger.log(values, step=self.frame)

            self.save(checkpoint_path)

        if self.show:
            display.close()

    def save(self, path):
        self.ppo.save(path, {"epoch": self.epoch, "frame": self.frame})

    def restore(self, path):
        progress = self.ppo.restore(path)
        self.epoch = progress["epoch"]
        self.frame = progress["frame"]
        return self
