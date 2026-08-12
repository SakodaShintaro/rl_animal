import networks
import models
import tr_helpers


animal_ai = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm2),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 5e-5,
    'NAME' : 'pong',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.004,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAI',
    'PPO' : True,
    'E_CLIP' : 0.1,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 256,
    'MINIBATCH_SIZE' : 2048,
    'MINI_EPOCHS' : 2,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 12000
}

animal_ai_ray = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm2),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 5e-5,
    'NAME' : 'pong',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.005,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.1,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 4,
    'MINIBATCH_SIZE' : 384 * 4,
    'MINI_EPOCHS' : 2,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 48000
}

animal_ai_ray_orig = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 1e-4,
    'NAME' : 'animal',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.005,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.2,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 4,
    'MINIBATCH_SIZE' : 384,
    'MINI_EPOCHS' : 5,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 48000
}

animal_ai_ray_poses = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm2),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 1e-4,
    'NAME' : 'animal',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.005,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.1,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 4,
    'MINIBATCH_SIZE' : 384 * 4,
    'MINI_EPOCHS' : 4,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 48000
}

animal_ai_ray_times1 = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm6),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 1e-4,
    'NAME' : 'animal',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.01,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.2,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 8,
    'MINIBATCH_SIZE' : 384 * 4,
    'MINI_EPOCHS' : 4,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',#'POLYNOM_DECAY',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 48000
}

animal_ai_ray_times2 = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm6),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 5e-5,
    'NAME' : 'pong',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.005,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.1,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 8,
    'MINIBATCH_SIZE' : 384 * 4,
    'MINI_EPOCHS' : 4,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 32000
}

animal_ai_ray_times3 = {
    'GAMMA' : 0.99,
    'LAMBDA' : 0.9,
    'NETWORK' : models.LSTMModelA2C(networks.animal_a2c_network_lstm6),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(scale_value = 1.0),
    'NORMALIZE_ADVANTAGE' : True,
    'LEARNING_RATE' : 1e-5,
    'NAME' : 'pong',
    'SCORE_TO_WIN' : 100500,
    'GRAD_NORM' : 0.5,
    'ENTROPY_COEF' : 0.002,
    'TRUNCATE_GRADS' : True,
    'ENV_NAME' : 'AnimalAIRay',
    'PPO' : True,
    'E_CLIP' : 0.05,
    'NUM_ACTORS' : 24,
    'STEPS_NUM' : 32 * 8,
    'MINIBATCH_SIZE' : 384 * 4,
    'MINI_EPOCHS' : 4,
    'CRITIC_COEF' : 1.0,
    'CLIP_VALUE' : True,
    'LR_SCHEDULE' : 'NONE',
    'NORMALIZE_INPUT' : False,
    'SEQ_LEN' : 8,
    'MAX_EPOCHS' : 32000
}


"""
Scratch training against the Animal-AI v4 (Unity 6) build. Same network and same
hyperparameters as animal_ai_ray_times1, the configuration the released
nn/last84_10_5 checkpoint was trained with and the one the evaluation drivers build
their player from; only the environment differs.
"""
"""
MAX_EPOCHS is set to the 70 million steps the presentation reports for the winning
run ("Total 70 million of steps", 600 FPS, 40 hours of training), not to the 48000
the times1 config carries, which is only an upper bound. A batch is
NUM_ACTORS * STEPS_NUM = 24 * 256 = 6144 steps, so 70e6 / 6144 is about 11400.
The presentation also describes dropping the entropy coefficient, learning rate and
epsilon clip after the first 50 million steps, which is what animal_ai_ray_times2 is;
this configuration stays on the times1 values for the whole run.
"""
animal_ai_v4 = dict(animal_ai_ray_times1, ENV_NAME='AnimalAIRayV4', NAME='animal_v4',
                    MAX_EPOCHS=11400)
