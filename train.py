"""Train an agent, either against the v1 binary bundled here or the Animal-AI v4 one.

    # v1, as before
    python train.py --config animal_ai_ray_times2 --name ampere_run

    # v4 (Unity 6), from scratch; needs the v4 environment variables, see
    # docker-compose.yml's animal-aai4 service
    python train.py --config animal_ai_v4 --name v4_scratch

Checkpoints are written to nn/<name><ENV_NAME> by the training loop itself, and the
final one to nn/<name>.
"""
import argparse

import tensorflow as tf
import ray

import env_configurations
import games_configurations
from a2c_discrete import A2CAgent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='a name in games_configurations')
    parser.add_argument('--name', required=True, help='run name, used for the checkpoints')
    parser.add_argument('--restore', required=False, default=None,
                        help='checkpoint to start from instead of a fresh network')
    parser.add_argument('--max-epochs', required=False, default=None, type=int,
                        help='override the config, for a short smoke run')
    parser.add_argument('--num-actors', required=False, default=None, type=int,
                        help='override the config, for a short smoke run')
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(getattr(games_configurations, args.config))
    if args.max_epochs is not None:
        config['MAX_EPOCHS'] = args.max_epochs
    if args.num_actors is not None:
        config['NUM_ACTORS'] = args.num_actors
    print('config %s: %d actors, %d epochs, env %s' % (
        args.config, config['NUM_ACTORS'], config['MAX_EPOCHS'], config['ENV_NAME']))

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    if config['ENV_NAME'] == 'AnimalAIRayV4':
        '''
        The v4 shapes are fixed by the stacking in aai4_common, so they are taken from
        there rather than by launching an extra Unity instance just to read them.
        '''
        import aai4_common
        from gym import spaces

        obs_space = spaces.Box(low=0, high=255, shape=aai4_common.observation_shape(),
                               dtype='uint8')
        action_space = spaces.Discrete(aai4_common.ACTIONS_NUM)
    else:
        obs_space, action_space = env_configurations.get_obs_and_action_spaces(
            config['ENV_NAME'])

    ray.init(num_gpus=1)
    agent = A2CAgent(sess, args.name, obs_space, True, action_space, config)
    if args.restore is not None:
        agent.restore(args.restore)
        print('restored ' + args.restore)
    agent.train()
    agent.save('nn/' + args.name)


if __name__ == '__main__':
    main()
