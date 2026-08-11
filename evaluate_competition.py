"""Evaluate a checkpoint on the 900 Animal AI Olympics competition scenarios.

The scenarios come from the AnimalAI-Olympics repository (competition_configurations),
which is what hyperparams.LEARNING_DIR originally pointed at. Each scenario carries a
pass_mark: the agent passes it when the episode reward reaches that value.

    python evaluate_competition.py \
        --checkpoint nn/last84_10_5 \
        --configs 'configs/learning/competition_configurations/*.yml' \
        --episodes 1 \
        --num-envs 6 \
        --output competition_results.csv \
        --seed 32

Rewards here are the raw environment rewards, not the shaped ones the training
wrappers produce, so that the numbers are comparable to the competition score.

--num-envs 1 runs everything in this process; larger values spread the scenarios
over that many ray workers, each with its own Unity instance, and evaluate them
with a single batched forward pass per step. Note that the policy samples its
actions, so two runs of the same set never match exactly.
"""
import argparse
import csv
import glob
import re
import time

import numpy as np
import ray
import tensorflow as tf
import yaml

import animalai_wrapper
import env_configurations
import games_configurations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='e.g. nn/last84_10_5')
    parser.add_argument('--configs', required=True, help='glob of competition yml files')
    parser.add_argument('--episodes', required=True, type=int, help='episodes per scenario')
    parser.add_argument('--num-envs', required=True, type=int, help='parallel Unity instances')
    parser.add_argument('--output', required=True, help='csv file to write')
    parser.add_argument('--seed', required=True, type=int)
    return parser.parse_args()

def load_competition_config(path):
    '''
    The competition configs were released with AnimalAI v2 and carry a pass_mark
    field, which the v1 Arena used by this repository does not accept.
    '''
    raw = open(path).read()
    config = yaml.load(re.sub(r'^[ \t]*pass_mark:.*\n', '', raw, flags=re.M), Loader=yaml.Loader)

    match = re.search(r'pass_mark:\s*([-\d.]+)', raw)
    if match is None:
        '''
        10-22-2.yml is the only released scenario without a pass_mark. The other
        two variants of the same test (10-22-1, 10-22-3) both use 0.
        '''
        return config, 0.0

    return config, float(match.group(1))

def category_of(path):
    return path.split('/')[-1].split('-')[0]

def use_raw_rewards():
    '''
    Evaluation scores the environment reward itself; the training reward shaping
    (bonus for picking up food, penalty for moving backwards) would not be
    comparable to the competition score.
    '''
    animalai_wrapper.calc_rewards_v2 = lambda reward, vel: reward


class BatchedPlayer:
    '''
    players.PpoPlayerDiscrete acts on one observation at a time. This is the same
    network, built for a batch of num_envs observations so that all environments
    can be advanced with a single session run.
    '''
    def __init__(self, sess, config, num_envs, obs_space, action_space):
        self.sess = sess
        self.num_envs = num_envs
        self.network = config['NETWORK']
        assert self.network.is_rnn()

        self.obs_ph = tf.placeholder('uint8', (None,) + obs_space.shape, name='obs')
        self.vels_ph = tf.placeholder(tf.float32, [num_envs, 8], name='vels')
        self.epoch_num = tf.Variable(tf.constant(0, shape=(), dtype=tf.float32), trainable=False)

        run_dict = {
            'name': 'agent',
            'inputs': tf.to_float(self.obs_ph) / 255.0,
            'batch_num': num_envs,
            'games_num': num_envs,
            'actions_num': action_space.n,
            'prev_actions_ph': None,
            'vels_ph': self.vels_ph,
        }
        _, _, self.action, _, self.states_ph, self.masks_ph, self.lstm_state, self.initial_state = \
            self.network(run_dict, reuse=False)
        self.states = self.initial_state

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def restore(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)

    def act(self, obs, resets):
        '''
        resets[i] tells the recurrent state to start over, which is what the
        training loop does with the done flags of the previous step.
        '''
        actions, self.states = self.sess.run([self.action, self.lstm_state], {
            self.obs_ph: obs[0],
            self.vels_ph: obs[1],
            self.states_ph: self.states,
            self.masks_ph: resets,
        })

        return np.reshape(actions, (self.num_envs,))


class LocalEnvs:
    def __init__(self, num_envs, seed):
        assert num_envs == 1
        use_raw_rewards()
        self.env = env_configurations.create_animal(1, False, None, seed)

    def reset(self, index, path):
        config, _ = load_competition_config(path)
        return self.env.reset(config)

    def step(self, indices, actions):
        return [self.env.step(actions[i]) for i in indices]

    def close(self):
        self.env.close()


@ray.remote
class RayEvalWorker:
    def __init__(self, seed):
        use_raw_rewards()
        self.env = env_configurations.create_animal(1, False, None, seed)

    def reset(self, path):
        config, _ = load_competition_config(path)
        return self.env.reset(config)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class RayEnvs:
    def __init__(self, num_envs, seed):
        '''
        Every worker gets its own seed so that the scenarios which randomise item
        placement are not replayed identically in each worker.
        '''
        self.workers = [RayEvalWorker.remote(seed + i) for i in range(num_envs)]

    def reset(self, index, path):
        return ray.get(self.workers[index].reset.remote(path))

    def step(self, indices, actions):
        pending = [self.workers[i].step.remote(actions[i]) for i in indices]
        return ray.get(pending)

    def close(self):
        ray.get([worker.close.remote() for worker in self.workers])


def build_tasks(paths, episodes, num_envs):
    '''
    Scenarios are dealt out to the environments up front, so a run is reproducible
    in terms of which environment sees which scenario.
    '''
    tasks = [[] for _ in range(num_envs)]
    for i, path in enumerate(paths):
        for episode in range(episodes):
            tasks[i % num_envs].append((path, episode))

    return tasks


def evaluate(envs, player, tasks, writer, log_every):
    num_envs = len(tasks)
    cursor = [0] * num_envs
    config = [None] * num_envs
    pass_mark = [0.0] * num_envs
    max_steps = [0] * num_envs
    reward = [0.0] * num_envs
    steps = [0] * num_envs
    active = [False] * num_envs
    resets = np.ones(num_envs, dtype=bool)
    obs0, obs1 = [None] * num_envs, [None] * num_envs
    rows = []
    total_steps = 0
    start = time.time()

    def start_task(index):
        path, episode = tasks[index][cursor[index]]
        config[index], pass_mark[index] = load_competition_config(path)
        max_steps[index] = config[index].arenas[0].t + 100
        reward[index] = 0.0
        steps[index] = 0
        active[index] = True
        resets[index] = True
        observation = envs.reset(index, path)
        obs0[index], obs1[index] = observation[0], observation[1]

    for index in range(num_envs):
        start_task(index)

    while any(active):
        running = [i for i in range(num_envs) if active[i]]
        actions = player.act([np.asarray(obs0), np.asarray(obs1)], resets)
        resets = np.zeros(num_envs, dtype=bool)

        for index, result in zip(running, envs.step(running, actions)):
            observation, r, done, _ = result
            obs0[index], obs1[index] = observation[0], observation[1]
            reward[index] += r
            steps[index] += 1
            total_steps += 1
            if not done and steps[index] < max_steps[index]:
                continue

            path, episode = tasks[index][cursor[index]]
            passed = int(reward[index] >= pass_mark[index])
            rows.append((path, pass_mark[index], reward[index], passed))
            writer.writerow([path.split('/')[-1], category_of(path), pass_mark[index],
                             episode, '%.4f' % reward[index], passed, steps[index]])
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

    return rows

def report(rows):
    print('pass rate: %.4f (%d / %d episodes)' % (
        np.mean([row[3] for row in rows]), sum(row[3] for row in rows), len(rows)))
    print('mean reward: %.4f' % np.mean([row[2] for row in rows]))
    for category in sorted(set(category_of(row[0]) for row in rows), key=int):
        in_category = [row[3] for row in rows if category_of(row[0]) == category]
        print('  category %-3s pass rate %.3f (%d episodes)' % (
            category, np.mean(in_category), len(in_category)))

def main():
    args = parse_args()

    paths = sorted(glob.glob(args.configs))
    if len(paths) == 0:
        raise ValueError('no configs matched ' + args.configs)
    print('scenarios: %d, episodes each: %d, envs: %d' % (len(paths), args.episodes, args.num_envs))

    sess = tf.InteractiveSession(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)))
    obs_space, action_space = env_configurations.get_obs_and_action_spaces('AnimalAI')
    player = BatchedPlayer(sess, games_configurations.animal_ai_ray_times1,
                           args.num_envs, obs_space, action_space)
    player.restore(args.checkpoint)

    if args.num_envs == 1:
        envs = LocalEnvs(1, args.seed)
    else:
        ray.init(num_gpus=1)
        envs = RayEnvs(args.num_envs, args.seed)

    try:
        with open(args.output, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(['scenario', 'category', 'pass_mark', 'episode', 'reward', 'passed', 'steps'])
            rows = evaluate(envs, player, build_tasks(paths, args.episodes, args.num_envs),
                            writer, 25)
    finally:
        # the Unity processes outlive the interpreter otherwise
        envs.close()

    report(rows)


if __name__ == '__main__':
    main()
