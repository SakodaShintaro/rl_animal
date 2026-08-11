"""Check that the container can run TensorFlow on the GPU and drive the Unity env.

    docker compose run --rm animal python docker/smoke_test.py
"""
import numpy as np
import tensorflow as tf

from env_configurations import create_animal


def check_tensorflow():
    print('tensorflow', tf.__version__)
    print('built with cuda:', tf.test.is_built_with_cuda())
    gpus = [d.name for d in tf.Session().list_devices() if '/device:GPU:' in d.name]
    print('visible gpus:', gpus)
    if len(gpus) == 0:
        print('WARNING: no GPU visible, training will fall back to the CPU')


def check_environment(steps):
    env = create_animal(num_actors=1, inference=False, config=None, seed=0)
    obs = env.reset()
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    total_reward = 0.0
    for _ in range(steps):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            obs = env.reset()
    print('visual obs shape:', np.shape(obs[0]), 'vector obs shape:', np.shape(obs[1]))
    print('reward over {} steps: {}'.format(steps, total_reward))
    env.close()


if __name__ == '__main__':
    check_tensorflow()
    check_environment(50)
    print('smoke test OK')
