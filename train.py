import tensorflow as tf
import ray

import env_configurations
import games_configurations
from a2c_discrete import A2CAgent

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

obs_space, action_space = env_configurations.get_obs_and_action_spaces('AnimalAI')
config = games_configurations.animal_ai_ray_times2

ray.init(num_gpus=1)
agent = A2CAgent(sess, 'ampere_run', obs_space, True, action_space, config)
agent.train()
agent.save('nn/ampere_run')
