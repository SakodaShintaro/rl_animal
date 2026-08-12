import os
import random
from collections import deque

import numpy as np
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.base_env import ActionTuple

from rl_animal_torch import arena
from rl_animal_torch.config import EnvConfig

ACTIONS_NUM = 9


class Stacker:
    '''
    The frame and velocity stacking of animalai_wrapper.AnimalStack. SKIP_FRAMES was 1, so
    there is no action repeat to reproduce.
    '''
    def __init__(self, config):
        self.config = config
        self.frames = deque([], maxlen=config.visual_frames)
        self.vels = deque([], maxlen=config.velocity_frames)
        self.time = 0.0
        self.velocity_scale = np.asarray(config.velocity_scale, dtype=np.float32)
        self.time_decrement = config.decision_period / (
            config.physics_steps_per_t * config.time_unit)

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


class AnimalEnv:
    '''
    reset() picks one of `arena_paths` at random with a randomized episode length, the way
    the original training did, and returns (visual uint8 HWC, velocity float32).
    '''
    def __init__(self, env_path, arena_paths, worker_id, base_port, seed, shape_rewards,
                 scratch_dir):
        config = EnvConfig()
        self.config = config
        self.arena_paths = arena_paths
        self.shape_rewards = shape_rewards
        self.scratch_dir = scratch_dir
        self.random = random.Random(seed)
        self.stacker = Stacker(config)
        self.arena_time = arena.read_arena_time(open(arena_paths[0]).read())

        self.env = AnimalAIEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            play=False,
            arenas_configurations=arena_paths[0],
            useCamera=True,
            resolution=config.resolution,
            grayscale=False,
            useRayCasts=False,
            no_graphics=False,
            decisionPeriod=config.decision_period,
            timescale=config.timescale,
            targetFrameRate=config.target_frame_rate,
        )
        self.behavior = next(iter(self.env.behavior_specs.keys()))

    def _observe(self):
        decision, terminal = self.env.get_steps(self.behavior)
        if len(terminal) > 0:
            return terminal.obs[0][0], terminal.obs[1][0], float(terminal.reward[0]), True
        return decision.obs[0][0], decision.obs[1][0], float(decision.reward[0]), False

    def reset(self):
        raw = open(self.random.choice(self.arena_paths)).read()
        self.arena_time = self.random.randrange(self.config.min_time, self.config.max_time)
        path = arena.write_with_time(raw, self.arena_time, self.scratch_dir)
        try:
            self.env.reset(path)
        finally:
            os.remove(path)

        camera, vector, _, _ = self._observe()
        self.stacker.reset(camera, vector, self.arena_time)
        return self.stacker.observation()

    def reset_to(self, path):
        '''
        Reset onto one specific arena, keeping its own episode length. Used by evaluation,
        which has to run the released scenarios as written.
        '''
        self.arena_time = arena.read_arena_time(open(path).read())
        self.env.reset(path)
        camera, vector, _, _ = self._observe()
        self.stacker.reset(camera, vector, self.arena_time)
        return self.stacker.observation()

    def send(self, action):
        '''
        Split from receive so that a driver holding several instances can hand the action
        to all of them before waiting on any.
        '''
        self.env.set_actions(self.behavior, ActionTuple(
            continuous=np.zeros((1, 0), dtype=np.float32),
            # v4's two 3-way branches, in the order the flattened Discrete(9) assumed
            discrete=np.array([[action // 3, action % 3]], dtype=np.int32)))
        self.env.step()

    def receive(self):
        camera, vector, reward, done = self._observe()
        self.stacker.step(camera, vector)
        if self.shape_rewards:
            reward = self.shape(reward, self.stacker.to_raw_velocity(vector))

        return self.stacker.observation(), reward, done

    def shape(self, reward, velocity):
        if reward > 0.1:
            reward += self.config.reward_bonus
        if velocity[1] > 0.01:
            reward += velocity[1] * self.config.ramps_coef
        if velocity[2] < 0:
            reward += velocity[2] * self.config.back_move_coef

        return reward

    def step(self, action):
        self.send(action)
        return self.receive()

    def close(self):
        self.env.close()


def observation_shapes(config=EnvConfig()):
    return ((config.resolution, config.resolution, 3 * config.visual_frames),
            (4 * config.velocity_frames,))
