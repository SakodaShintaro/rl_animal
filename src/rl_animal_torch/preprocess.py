"""
What sits between the raw player observation and the network: the frame / velocity
stacking, the episode clock, and the reward the update is trained on. None of it needs
the player, so it is the part an agent driving some other Animal-AI environment reuses.
"""

from collections import deque

import numpy as np


class Stacker:
    """
    The frame and velocity stacking of animalai_wrapper.AnimalStack. SKIP_FRAMES was 1, so
    there is no action repeat to reproduce.
    """

    def __init__(self, config):
        self.config = config
        self.frames = deque([], maxlen=config.visual_frames)
        self.vels = deque([], maxlen=config.velocity_frames)
        self.time = 0.0
        self.velocity_scale = np.asarray(config.velocity_scale, dtype=np.float32)
        self.time_decrement = config.decision_period / (
            config.physics_steps_per_t * config.time_unit
        )

    @staticmethod
    def to_raw_velocity(vector):
        # v4's vector observation is [health, vx, vy, vz, px, py, pz]
        return np.asarray(vector[1:4], dtype=np.float32)

    def to_frame(self, camera):
        # v4 sends CHW float in [0, 1]; the network takes uint8 HWC
        return np.asarray(np.transpose(camera, (1, 2, 0)) * 255.0, dtype=np.uint8)

    def reset(self, camera, vector, arena_time):
        self.time = arena_time / self.config.time_unit
        frame = self.to_frame(camera)
        self.frames.clear()
        self.vels.clear()
        for _ in range(self.config.visual_frames):
            self.frames.append(frame)
        for _ in range(self.config.velocity_frames - 1):
            self.vels.append(np.array([0.0, 0.0, 0.0, self.time], dtype=np.float32))
        self.vels.append(self.velocity_entry(vector))

    def step(self, camera, vector):
        self.time -= self.time_decrement
        self.frames.append(self.to_frame(camera))
        self.vels.append(self.velocity_entry(vector))

    def velocity_entry(self, vector):
        scaled = self.to_raw_velocity(vector) / self.velocity_scale
        return np.append(scaled, self.time).astype(np.float32)

    def observation(self):
        return (np.concatenate(self.frames, axis=-1), np.concatenate(self.vels))


def shape_reward(reward, velocity, config):
    """
    The reward the PPO update is trained on: reaching a goal is worth more than the arena
    says, climbing is encouraged and walking backwards is discouraged. The episode return
    that gets reported is the arena's own reward, so this never moves the reported score.
    """
    if reward > 0.1:
        reward += config.reward_bonus
    if velocity[1] > 0.01:
        reward += velocity[1] * config.ramps_coef
    if velocity[2] < 0:
        reward += velocity[2] * config.back_move_coef

    return reward
