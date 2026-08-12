import os

import networks 
import tr_helpers
import gym
import numpy as np
from hyperparams import USE_GREYSCALE_OBSES, VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, SKIP_FRAMES, BASE_DIR



def create_animal(num_actors=1, inference = True, config=None, seed=None):
    from animalai.envs.gym.environment import AnimalAIEnv
    from animalai.envs.arena_config import ArenaConfig
    import random
    from animalai_wrapper import AnimalWrapper, AnimalStack, AnimalSkip
    env_path = 'AnimalAI'
    worker_id = random.randint(1, 60000)
    arena_config_in = ArenaConfig(BASE_DIR + '/configs/learning/stage4/3-Food Moving.yaml')

    if config is None:
        config = arena_config_in
    else: 
        config = ArenaConfig(config)
    if seed is None:
        seed = 0#random.randint(0, 100500)
        
    env = AnimalAIEnv(environment_filename=env_path,
                      worker_id=worker_id,
                      n_arenas=num_actors,
                      seed = seed,
                      arenas_configurations=config,
                      greyscale = False,
                      docker_training=False,
                      inference = inference,
                      retro=False,
                      resolution=84
                      )
    env = AnimalSkip(env, skip=SKIP_FRAMES)                  
    env = AnimalWrapper(env)
    env = AnimalStack(env,VISUAL_FRAMES_COUNT, VEL_FRAMES_COUNT, greyscale=USE_GREYSCALE_OBSES)
    return env


def create_animal_v4(inference, config, worker_id):
    '''
    The Animal-AI v4 (Unity 6) build, behind the same interface create_animal exposes.
    decisionPeriod 5 is the setting under which a v4 episode lasts `t` decisions as
    v1's did, which is also what makes the countdown AnimalStack feeds the network
    reach 0 at the end of the episode; see aai4_common for the measurements.

    worker_id has to be unique and is therefore passed in rather than drawn here: each
    instance serves gRPC on base_port + worker_id, and two instances landing on the
    same port leave the client waiting for a server that never answers.
    '''
    import random

    import aai4_common

    env = aai4_common.AnimalV4Env(
        env_path=os.environ['AAI4_ENV_PATH'],
        worker_id=worker_id,
        base_port=int(os.environ['AAI4_BASE_PORT']),
        seed=random.randint(0, 100500),
        first_config=config,
        decision_period=5,
        timescale=int(os.environ['AAI4_TIMESCALE']),
        target_frame_rate=-1,
        time_decrement=1.0 / aai4_common.TIME_UNIT,
        shape_rewards=not inference,
    )
    env.set_arena_time(aai4_common.read_arena_time(open(config).read()))
    return env


configurations = {
    'AnimalAI' : {
        'ENV_CREATOR' : lambda : create_animal(),
        'VECENV_TYPE' : 'ANIMAL'
    },
    'AnimalAIRay' : {
        'ENV_CREATOR' : lambda inference=False, config=None: create_animal(1, inference, config=config),
        'VECENV_TYPE' : 'RAY'
    },
    'AnimalAIRayV4' : {
        'ENV_CREATOR' : create_animal_v4,
        'VECENV_TYPE' : 'RAY_V4'
    },

}


def get_obs_and_action_spaces(name):
    env = configurations[name]['ENV_CREATOR']()
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()
    return observation_space, action_space

def register(name, config):
    configurations[name] = config
