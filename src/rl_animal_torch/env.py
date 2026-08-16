import random

import numpy as np
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.base_env import ActionTuple

from rl_animal_torch import arena
from rl_animal_torch.config import EnvConfig
from rl_animal_torch.preprocess import Stacker, shape_reward

ACTIONS_NUM = 9


class AnimalEnv:
    """
    reset() picks one of `arena_paths` at random and returns
    (visual uint8 HWC, velocity float32).
    """

    def __init__(self, env_path, arena_paths, worker_id, base_port, seed):
        config = EnvConfig()
        self.config = config
        self.arena_paths = arena_paths
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
        return self.reset_to(self.random.choice(self.arena_paths))

    def reset_to(self, path):
        """
        Reset onto one specific arena, keeping the episode length it was written with.
        """
        self.arena_time = arena.read_arena_time(open(path).read())
        self.env.reset(path)
        camera, vector, _, _ = self._observe()
        self.stacker.reset(camera, vector, self.arena_time)
        return self.stacker.observation()

    def send(self, action):
        """
        Split from receive so that a driver holding several instances can hand the action
        to all of them before waiting on any.
        """
        self.env.set_actions(
            self.behavior,
            ActionTuple(
                continuous=np.zeros((1, 0), dtype=np.float32),
                # v4's two 3-way branches, in the order the flattened Discrete(9) assumed
                discrete=np.array([[action // 3, action % 3]], dtype=np.int32),
            ),
        )
        self.env.step()

    def receive(self):
        """
        (observation, the arena's own reward, the reward to train on, done). Both rewards
        come out so the caller reports the score the competition scores while the update
        sees the shaped one, instead of the environment deciding which of the two exists.
        """
        camera, vector, reward, done = self._observe()
        self.stacker.step(camera, vector)
        shaped = shape_reward(reward, self.stacker.to_raw_velocity(vector), self.config)

        return self.stacker.observation(), reward, shaped, done

    def step(self, action):
        self.send(action)
        return self.receive()

    def close(self):
        self.env.close()


def observation_shapes(config):
    return (
        (config.resolution, config.resolution, 3 * config.visual_frames),
        (4 * config.velocity_frames,),
    )
