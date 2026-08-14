import multiprocessing as mp

import numpy as np

from rl_animal_torch.env import AnimalEnv


def worker_main(connection, env_path, arena_paths, worker_id, base_port, seed, shape_rewards):
    env = AnimalEnv(env_path, arena_paths, worker_id, base_port, seed, shape_rewards)
    try:
        connection.send(env.reset())
        while True:
            command, payload = connection.recv()
            if command == "step":
                observation, reward, done = env.step(payload)
                if done:
                    observation = env.reset()
                connection.send((observation, reward, done))
            elif command == "reset":
                connection.send(env.reset())
            elif command == "close":
                return
    finally:
        env.close()
        connection.close()


class VecEnv:
    def __init__(self, env_path, arena_paths, num_actors, base_port, seed, shape_rewards):
        """
        spawn rather than fork: the parent holds CUDA context and file descriptors that a
        forked child must not inherit.
        """
        context = mp.get_context("spawn")
        self.num_actors = num_actors
        self.connections = []
        self.processes = []
        for index in range(num_actors):
            parent, child = context.Pipe()
            process = context.Process(
                target=worker_main,
                args=(child, env_path, arena_paths, index, base_port, seed + index, shape_rewards),
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

        self.last = [connection.recv() for connection in self.connections]

    def observations(self):
        visual = np.asarray([entry[0] for entry in self.last])
        vels = np.asarray([entry[1] for entry in self.last], dtype=np.float32)
        return visual, vels

    def reset(self):
        for connection in self.connections:
            connection.send(("reset", None))
        self.last = [connection.recv() for connection in self.connections]
        return self.observations()

    def step(self, actions):
        for connection, action in zip(self.connections, actions):
            connection.send(("step", int(action)))
        results = [connection.recv() for connection in self.connections]
        self.last = [observation for observation, _, _ in results]
        rewards = np.asarray([reward for _, reward, _ in results], dtype=np.float32)
        dones = np.asarray([done for _, _, done in results], dtype=bool)
        return self.observations(), rewards, dones

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
