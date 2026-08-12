"""Evaluate the trained checkpoint against the Animal-AI v4 (Unity 6) binary.

evaluate_competition.py drives the v1 binary that ships with this repository. This
runs the same checkpoint against the current Animal-AI environment, whose Unity
build, gRPC protocol and Python API are all different, to see how much of the
44.56% pass rate survives the move.

    python aai4_eval.py \
        --checkpoint nn/last84_10_5 \
        --env-path /aai4/animalAI.x86_64 \
        --configs 'configs/learning/competition_configurations/*.yml' \
        --episodes 1 \
        --num-envs 8 \
        --output competition_results_aai4.csv \
        --seed 32 \
        --base-port 5300 \
        --gpu-memory-fraction 0.3 \
        --timescale 1 \
        --target-frame-rate 60 \
        --decision-period 3

What had to be bridged, and what did not:

- The arena YAML is unchanged. v4 still reads `!ArenaConfig / !Arena / !Item /
  !Vector3 / !RGB` and still understands `pass_mark`, and all 19 item names used
  by the 900 scenarios exist in the v4 build.
- The action space matches exactly. v1 flattened two 3-way branches into
  Discrete(9) as [a // 3, a % 3]; v4's MultiDiscrete([3, 3]) uses the same
  branch order (0 noop / 1 forward / 2 back, 0 noop / 1 right / 2 left).
- The camera observation is CHW float in [0, 1] in v4 and was HWC in v1, so it is
  transposed here.
- The vector observation grew from the 3 velocity components to
  [health, vx, vy, vz, px, py, pz]. The network only ever saw velocity, so the
  extra fields are dropped rather than fed in.
- Rewards are the raw environment rewards in both, including the per-step time
  penalty, so episode totals stay comparable to the pass marks.
"""
import argparse
import csv
import glob
import os
import sys
import time
from collections import deque

import numpy as np
import tensorflow as tf

import competition_common
import games_configurations
import hyperparams as hps
from evaluate_competition import BatchedPlayer


def import_animalai_v4():
    '''
    PYTHONPATH=/workspace puts this repository first on sys.path, and it contains
    the v1 `animalai` package, which shadows the installed animalai 5.0.1 that
    speaks to the v4 binary. Drop the repository entries for the duration of the
    import; the v1 package is only needed by evaluate_competition.py.
    '''
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    saved = list(sys.path)
    sys.path = [p for p in sys.path
                if p not in ('', '.', repo_dir) and os.path.abspath(p or '.') != repo_dir]
    try:
        from animalai.environment import AnimalAIEnvironment
        from mlagents_envs.base_env import ActionTuple
    finally:
        sys.path = saved

    if 'animalai' in sys.modules and 'v1' in getattr(sys.modules['animalai'], '__file__', ''):
        raise ImportError('the v1 animalai package was imported instead of 5.0.1')

    return AnimalAIEnvironment, ActionTuple


AnimalAIEnvironment, ActionTuple = import_animalai_v4()

'''
The visual input is VISUAL_FRAMES_COUNT stacked RGB frames and the vector input is
VEL_FRAMES_COUNT stacked [vx, vy, vz / 16, time] entries, which is what
animalai_wrapper.AnimalStack produced for the v1 environment.
'''
RESOLUTION = 84
ACTIONS_NUM = 9
'''
AnimalStack expressed the remaining episode time as a fraction of 250 steps and
counted it down by one step at a time, so the same constant is needed here.
'''
TIME_UNIT = 250.0
'''
Measured, by holding the no-op action until the episode ended: a v4 episode lasts
`t` * 5 physics steps regardless of decisionPeriod, and the accumulated time penalty
over it is exactly -1.0.

    t=250 dp=3 -> 417 decisions (1.67 x t)   t=250 dp=5 -> 250 decisions (1.00 x t)
    t=500 dp=3 -> 834 decisions (1.67 x t)   t=500 dp=5 -> 500 decisions (1.00 x t)

v1 instead ended hard at `t` decisions: over the 900 v1 episodes, steps/t never
exceeded 1.00 and 44.2% of them ended exactly at t. So v4 grants 5/3 as much
simulated time per episode as v1 did for the same arena, and no decisionPeriod
setting removes that difference: dp=3 reproduces v1's motion per decision while
stretching the episode, dp=5 reproduces v1's episode length in decisions while
making each decision travel 5/3 further.
'''
PHYSICS_STEPS_PER_T = 5


class ObsSpace:
    def __init__(self, shape):
        self.shape = shape


class ActionSpace:
    def __init__(self, n):
        self.n = n


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='e.g. nn/last84_10_5')
    parser.add_argument('--env-path', required=True, help='v4 animalAI.x86_64')
    parser.add_argument('--configs', required=True, help='glob of competition yml files')
    parser.add_argument('--episodes', required=True, type=int, help='episodes per scenario')
    parser.add_argument('--num-envs', required=True, type=int, help='parallel Unity instances')
    parser.add_argument('--output', required=True, help='csv file to write')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--base-port', required=True, type=int)
    parser.add_argument('--gpu-memory-fraction', required=True, type=float)
    '''
    animalai 5.0.1 defaults to timescale 1 with targetFrameRate 60, i.e. real time,
    which with decisionPeriod 3 caps each instance at about 20 decisions per second.
    The v1 API never sent an engine configuration at all, so the v1 player ran
    unthrottled and reached 851 env-steps/s over 12 instances against 142 here.
    Raising the timescale is what makes a full run practical, at the risk the
    upstream docstring warns about: the physics can change under a large multiplier.
    '''
    parser.add_argument('--timescale', required=True, type=int)
    parser.add_argument('--target-frame-rate', required=True, type=int)
    parser.add_argument('--decision-period', required=True, type=int)
    '''
    Which countdown to feed the network as the last element of each velocity entry.
    `v1` is literally what AnimalStack did, one 250th per decision, which reaches 0
    at decision t and then goes negative for the rest of a v4 episode. `episode`
    keeps the same starting value but reaches 0 exactly when the episode ends, so the
    network never sees a value it could not have seen in training; at
    decisionPeriod 5 the two are the same thing.
    '''
    parser.add_argument('--time-countdown', required=True, choices=['v1', 'episode'])
    return parser.parse_args()


class Stacker:
    '''
    The same frame and velocity stacking animalai_wrapper.AnimalSkip/AnimalWrapper/
    AnimalStack applied in the v1 pipeline, over the v4 observations. SKIP_FRAMES is
    1 in hyperparams, so there is no action repeat to reproduce.
    '''
    def __init__(self, decrement):
        self.frames = deque([], maxlen=hps.VISUAL_FRAMES_COUNT)
        self.vels = deque([], maxlen=hps.VEL_FRAMES_COUNT)
        self.time = 0.0
        self.decrement = decrement

    @staticmethod
    def to_frame(camera):
        '''
        v4 hands over (3, 84, 84) in [0, 1]; the network wants uint8 HWC.
        '''
        return np.asarray(np.transpose(camera, (1, 2, 0)) * 255.0, dtype=np.uint8)

    @staticmethod
    def to_velocity(vector):
        return np.asarray(vector[1:4], dtype=np.float32) / hps.VEC_SCALE

    def reset(self, camera, vector, arena_time):
        self.time = arena_time / TIME_UNIT
        frame = self.to_frame(camera)
        for _ in range(hps.VISUAL_FRAMES_COUNT):
            self.frames.append(frame)
        for _ in range(hps.VEL_FRAMES_COUNT - 1):
            self.vels.append(np.array([0.0, 0.0, 0.0, self.time], dtype=np.float32))
        self.vels.append(np.append(self.to_velocity(vector), self.time))

    def step(self, camera, vector):
        self.time -= self.decrement
        self.frames.append(self.to_frame(camera))
        self.vels.append(np.append(self.to_velocity(vector), self.time))

    def visual(self):
        return np.concatenate(self.frames, axis=-1)

    def vector(self):
        return np.concatenate(self.vels)


class Env:
    '''
    One Unity instance plus the stacker for it. v4 takes the arena YAML as raw text
    over a side channel, so switching scenario is a reset with a different path.
    '''
    def __init__(self, env_path, worker_id, seed, first_config, engine):
        self.env = AnimalAIEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            base_port=self.base_port,
            seed=seed,
            play=False,
            arenas_configurations=first_config,
            useCamera=True,
            resolution=RESOLUTION,
            grayscale=False,
            useRayCasts=False,
            no_graphics=False,
            timescale=engine['timescale'],
            targetFrameRate=engine['target_frame_rate'],
            decisionPeriod=engine['decision_period'],
        )
        self.behavior = list(self.env.behavior_specs.keys())[0]
        self.stacker = Stacker(engine['time_decrement'])

    def reset(self, path, arena_time):
        self.env.reset(path)
        camera, vector = self.observe()
        self.stacker.reset(camera, vector, arena_time)

    def observe(self):
        decision, terminal = self.env.get_steps(self.behavior)
        if len(terminal) > 0:
            return terminal.obs[0][0], terminal.obs[1][0]
        return decision.obs[0][0], decision.obs[1][0]

    def send(self, action):
        self.env.set_actions(self.behavior, ActionTuple(
            continuous=np.zeros((1, 0), dtype=np.float32),
            discrete=np.array([[action // 3, action % 3]], dtype=np.int32)))

    def advance(self):
        '''
        Returns (reward, done). The reward on the terminal step is the one that
        carries the goal, so it has to be read from terminal_steps.
        '''
        self.env.step()
        decision, terminal = self.env.get_steps(self.behavior)
        if len(terminal) > 0:
            self.stacker.step(terminal.obs[0][0], terminal.obs[1][0])
            return float(terminal.reward[0]), True

        self.stacker.step(decision.obs[0][0], decision.obs[1][0])
        return float(decision.reward[0]), False

    def close(self):
        self.env.close()


def make_envs(env_path, num_envs, seed, base_port, first_config, engine):
    Env.base_port = base_port
    envs = []
    for i in range(num_envs):
        '''
        Every instance gets its own seed so that the scenarios which randomise item
        placement are not replayed identically, matching the v1 driver.
        '''
        envs.append(Env(env_path, i, seed + i, first_config, engine))
    return envs


def evaluate(envs, player, tasks, writer, decision_period, log_every):
    num_envs = len(envs)
    cursor = [0] * num_envs
    pass_mark = [0.0] * num_envs
    max_steps = [0] * num_envs
    reward = [0.0] * num_envs
    steps = [0] * num_envs
    active = [False] * num_envs
    resets = np.ones(num_envs, dtype=bool)
    rows = []
    total_steps = 0
    start = time.time()

    def start_task(index):
        path, episode = tasks[index][cursor[index]]
        raw = open(path).read()
        arena_time = competition_common.read_arena_time(raw)
        pass_mark[index] = competition_common.read_pass_mark(raw)
        '''
        A v4 episode ends on its own after t * PHYSICS_STEPS_PER_T physics steps; the
        margin is only a guard against an instance that never reports terminal.
        '''
        max_steps[index] = arena_time * PHYSICS_STEPS_PER_T // decision_period + 100
        reward[index] = 0.0
        steps[index] = 0
        active[index] = True
        resets[index] = True
        envs[index].reset(path, arena_time)

    for index in range(num_envs):
        start_task(index)

    while any(active):
        running = [i for i in range(num_envs) if active[i]]
        '''
        The graph is built for num_envs observations and carries one recurrent state
        per slot, so finished slots keep sending their last observation instead of
        being dropped from the batch.
        '''
        visual = np.asarray([env.stacker.visual() for env in envs])
        vector = np.asarray([env.stacker.vector() for env in envs])
        actions = player.act([visual, vector], resets)
        resets = np.zeros(num_envs, dtype=bool)

        for index in running:
            envs[index].send(actions[index])
        for index in running:
            step_reward, done = envs[index].advance()
            reward[index] += step_reward
            steps[index] += 1
            total_steps += 1
            if not done and steps[index] < max_steps[index]:
                continue

            path, episode = tasks[index][cursor[index]]
            passed = int(reward[index] >= pass_mark[index])
            rows.append((path, pass_mark[index], reward[index], passed))
            writer.writerow([path.split('/')[-1], competition_common.category_of(path),
                             pass_mark[index], episode, '%.4f' % reward[index], passed,
                             steps[index]])
            cursor[index] += 1
            if cursor[index] < len(tasks[index]):
                start_task(index)
            else:
                active[index] = False

            if len(rows) % log_every == 0:
                elapsed = time.time() - start
                print('%d episodes, %.0f steps/s, %.1f min elapsed, pass rate so far %.3f' % (
                    len(rows), total_steps / elapsed, elapsed / 60.0,
                    np.mean([row[3] for row in rows])))
                sys.stdout.flush()

    return rows


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.configs))
    if len(paths) == 0:
        raise ValueError('no configs matched ' + args.configs)
    print('scenarios: %d, episodes each: %d, envs: %d' % (len(paths), args.episodes, args.num_envs))

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True, per_process_gpu_memory_fraction=args.gpu_memory_fraction)))
    visual_channels = 3 * hps.VISUAL_FRAMES_COUNT
    player = BatchedPlayer(sess, games_configurations.animal_ai_ray_times1, args.num_envs,
                           ObsSpace((RESOLUTION, RESOLUTION, visual_channels)),
                           ActionSpace(ACTIONS_NUM))
    player.restore(args.checkpoint)
    print('restored ' + args.checkpoint)

    if args.time_countdown == 'v1':
        time_decrement = 1.0 / TIME_UNIT
    else:
        time_decrement = args.decision_period / (PHYSICS_STEPS_PER_T * TIME_UNIT)
    engine = {'timescale': args.timescale,
              'target_frame_rate': args.target_frame_rate,
              'decision_period': args.decision_period,
              'time_decrement': time_decrement}
    print('engine: %s' % engine)
    envs = make_envs(args.env_path, args.num_envs, args.seed, args.base_port, paths[0], engine)
    try:
        '''
        A full run takes minutes and both the csv and stdout are normally read
        through a pipe, so keep them line buffered instead of letting the block
        buffer hide the progress.
        '''
        with open(args.output, 'w', buffering=1) as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['scenario', 'category', 'pass_mark', 'episode', 'reward',
                             'passed', 'steps'])
            rows = evaluate(envs, player,
                            competition_common.build_tasks(paths, args.episodes, args.num_envs),
                            writer, args.decision_period, 25)
    finally:
        for env in envs:
            env.close()

    competition_common.report(rows)


if __name__ == '__main__':
    main()
