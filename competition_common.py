"""Pieces shared by the v1 and the Animal-AI v4 evaluation drivers.

Kept free of TensorFlow and of any animalai import so that both interpreters can
use it: the v1 driver runs under TensorFlow 1.15 with the bundled animalai, the
v4 driver under python 3.10 with animalai 5.0.1.
"""
import re

import numpy as np


def read_pass_mark(raw):
    '''
    10-22-2.yml is the only released scenario without a pass_mark. The other two
    variants of the same test (10-22-1, 10-22-3) both use 0.
    '''
    match = re.search(r'pass_mark:\s*([-\d.]+)', raw)
    if match is None:
        return 0.0

    return float(match.group(1))


def read_arena_time(raw):
    match = re.search(r'^[ \t]*t:\s*([\d.]+)', raw, flags=re.M)
    if match is None:
        raise ValueError('no t: field in the arena config')

    return int(float(match.group(1)))


def category_of(path):
    return path.split('/')[-1].split('-')[0]


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


def report(rows):
    '''
    rows are (path, pass_mark, reward, passed) tuples.
    '''
    print('pass rate: %.4f (%d / %d episodes)' % (
        np.mean([row[3] for row in rows]), sum(row[3] for row in rows), len(rows)))
    print('mean reward: %.4f' % np.mean([row[2] for row in rows]))
    for category in sorted(set(category_of(row[0]) for row in rows), key=int):
        in_category = [row[3] for row in rows if category_of(row[0]) == category]
        print('  category %-3s pass rate %.3f (%d episodes)' % (
            category, np.mean(in_category), len(in_category)))
