"""Evaluate a checkpoint against the Animal-AI v4 (Unity 6) binary.

evaluate_competition.py drives the v1 binary that ships with this repository. This
runs a checkpoint against the current Animal-AI environment, whose Unity build, gRPC
protocol and Python API are all different. The observation and action bridging lives
in aai4_common, which training shares.

    python aai4_eval.py \
        --checkpoint nn/last84_10_5 \
        --env-path /aai4/animalAI.x86_64 \
        --configs '/competition_configs/*.yaml' \
        --episodes 1 \
        --num-envs 12 \
        --output competition_results_aai4.csv \
        --seed 32 \
        --base-port 5300 \
        --gpu-memory-fraction 0.3 \
        --timescale 300 \
        --target-frame-rate -1 \
        --decision-period 5 \
        --time-countdown episode

Rewards here are the raw environment rewards, not the shaped ones training uses, so
that the numbers are comparable to the competition score and to the v1 run.
"""
import argparse
import csv
import glob
import sys
import time

import numpy as np
import tensorflow as tf

import aai4_common
import competition_common
import games_configurations
from evaluate_competition import BatchedPlayer


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
    unthrottled and reached 830 env-steps/s over 12 instances against 142 here. The
    timescale is what makes a full run practical and it does not change what the agent
    experiences: the per-decision forward velocity is identical at timescale 1, 20,
    100 and 300 (0.85, 3.06, 5.26, 7.25, 8.96).
    '''
    parser.add_argument('--timescale', required=True, type=int)
    parser.add_argument('--target-frame-rate', required=True, type=int)
    parser.add_argument('--decision-period', required=True, type=int)
    '''
    Which countdown to feed the network as the last element of each velocity entry.
    `v1` is literally what AnimalStack did, one 250th per decision, which reaches 0 at
    decision t and then goes negative for the rest of a v4 episode. `episode` keeps
    the same starting value but reaches 0 exactly when the episode ends, so the network
    never sees a value it could not have seen in training; at decisionPeriod 5 the two
    are the same thing.
    '''
    parser.add_argument('--time-countdown', required=True, choices=['v1', 'episode'])
    return parser.parse_args()


def make_envs(args, first_config, time_decrement):
    envs = []
    for i in range(args.num_envs):
        '''
        Every instance gets its own seed so that the scenarios which randomise item
        placement are not replayed identically, matching the v1 driver.
        '''
        envs.append(aai4_common.AnimalV4Env(
            env_path=args.env_path,
            worker_id=i,
            base_port=args.base_port,
            seed=args.seed + i,
            first_config=first_config,
            decision_period=args.decision_period,
            timescale=args.timescale,
            target_frame_rate=args.target_frame_rate,
            time_decrement=time_decrement,
            shape_rewards=False))
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
    observation = [None] * num_envs
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
        max_steps[index] = arena_time * aai4_common.PHYSICS_STEPS_PER_T // decision_period + 100
        reward[index] = 0.0
        steps[index] = 0
        active[index] = True
        resets[index] = True
        envs[index].set_arena_time(arena_time)
        observation[index] = envs[index].reset(path)

    for index in range(num_envs):
        start_task(index)

    while any(active):
        running = [i for i in range(num_envs) if active[i]]
        '''
        The graph is built for num_envs observations and carries one recurrent state
        per slot, so finished slots keep sending their last observation instead of
        being dropped from the batch.
        '''
        visual = np.asarray([observation[i][0] for i in range(num_envs)])
        vector = np.asarray([observation[i][1] for i in range(num_envs)])
        actions = player.act([visual, vector], resets)
        resets = np.zeros(num_envs, dtype=bool)

        for index in running:
            envs[index].send(actions[index])
        for index in running:
            observation[index], step_reward, done, _ = envs[index].receive()
            reward[index] += float(step_reward)
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

    if args.time_countdown == 'v1':
        time_decrement = 1.0 / aai4_common.TIME_UNIT
    else:
        time_decrement = args.decision_period / (
            aai4_common.PHYSICS_STEPS_PER_T * aai4_common.TIME_UNIT)
    print('time decrement per decision: %.6f' % time_decrement)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True, per_process_gpu_memory_fraction=args.gpu_memory_fraction)))
    player = BatchedPlayer(sess, games_configurations.animal_ai_ray_times1, args.num_envs,
                           ObsSpace(aai4_common.observation_shape()),
                           ActionSpace(aai4_common.ACTIONS_NUM))
    player.restore(args.checkpoint)
    print('restored ' + args.checkpoint)

    envs = make_envs(args, paths[0], time_decrement)
    try:
        '''
        A full run takes minutes and both the csv and stdout are normally read through
        a pipe, so keep them line buffered instead of letting the block buffer hide
        the progress.
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
