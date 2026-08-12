"""Several Animal-AI instances in worker processes, stepped in lockstep.

The original used ray for this; a pool of processes over pipes is enough and keeps the
dependency list to torch plus animalai. Each worker owns one Unity instance, and the
worker index is what decides its port: an instance serves gRPC on base_port + worker_id,
and two instances landing on the same port leave the client waiting for a server that
never answers.

An episode that ends is reset inside the worker, so the observation returned with a done
flag already belongs to the next episode, which is what the training loop expects.
"""
import multiprocessing as mp
import os
import tempfile

import numpy as np

from rl_animal_torch.env import AnimalEnv


def worker_main(connection, env_path, arena_paths, worker_id, base_port, seed, config,
                shape_rewards):
    scratch_dir = tempfile.mkdtemp(prefix='aai_arenas_')
    env = AnimalEnv(env_path, arena_paths, worker_id, base_port, seed, config,
                    shape_rewards, scratch_dir)
    try:
        connection.send(env.reset())
        while True:
            command, payload = connection.recv()
            if command == 'step':
                observation, reward, done = env.step(payload)
                if done:
                    observation = env.reset()
                connection.send((observation, reward, done))
            elif command == 'reset':
                connection.send(env.reset())
            elif command == 'close':
                return
            else:
                raise ValueError('unknown command ' + command)
    finally:
        env.close()
        for name in os.listdir(scratch_dir):
            os.remove(os.path.join(scratch_dir, name))
        os.rmdir(scratch_dir)
        connection.close()


class VecEnv:
    def __init__(self, env_path, arena_paths, num_actors, base_port, seed, config,
                 shape_rewards):
        '''
        spawn rather than fork: the parent holds CUDA context and file descriptors that a
        forked child must not inherit.
        '''
        context = mp.get_context('spawn')
        self.num_actors = num_actors
        self.connections = []
        self.processes = []
        for index in range(num_actors):
            parent, child = context.Pipe()
            process = context.Process(
                target=worker_main,
                args=(child, env_path, arena_paths, index, base_port, seed + index,
                      config, shape_rewards),
                daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

        '''
        Each worker sends its first observation as soon as its instance is up, so this
        also waits out the startup of every player before the first step is asked for.
        '''
        self.last = [connection.recv() for connection in self.connections]

    def observations(self):
        visual = np.asarray([entry[0] for entry in self.last])
        vels = np.asarray([entry[1] for entry in self.last], dtype=np.float32)
        return visual, vels

    def reset(self):
        for connection in self.connections:
            connection.send(('reset', None))
        self.last = [connection.recv() for connection in self.connections]
        return self.observations()

    def step(self, actions):
        for connection, action in zip(self.connections, actions):
            connection.send(('step', int(action)))
        results = [connection.recv() for connection in self.connections]
        self.last = [observation for observation, _, _ in results]
        rewards = np.asarray([reward for _, reward, _ in results], dtype=np.float32)
        dones = np.asarray([done for _, _, done in results], dtype=bool)
        return self.observations(), rewards, dones

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
