"""The hyperparameters of the winning run, and the environment settings the port needs.

The training values come from games_configurations.animal_ai_ray_times1, which is the
configuration the released checkpoints were trained with, and from the presentation:
"Total 70 million of steps", 600 frames per second, 40 hours. A batch is
num_actors * steps_num = 6144 steps, so 70e6 steps is about 11400 epochs, not the 48000
the original config carried as an upper bound.

The presentation also describes dropping the entropy coefficient to 0.001, the learning
rate to 5e-5 and the clipping to 0.1 after the first 50 million steps. That is a second
stage rather than a schedule; run it by restoring the first stage's checkpoint with the
second stage's values.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float
    lam: float
    learning_rate: float
    grad_norm: float
    entropy_coef: float
    critic_coef: float
    e_clip: float
    clip_value: bool
    normalize_advantage: bool
    num_actors: int
    steps_num: int
    minibatch_size: int
    mini_epochs: int
    seq_len: int
    max_epochs: int


STAGE1 = TrainingConfig(
    gamma=0.99,
    lam=0.9,
    learning_rate=1e-4,
    grad_norm=0.5,
    entropy_coef=0.01,
    critic_coef=1.0,
    e_clip=0.2,
    clip_value=True,
    normalize_advantage=True,
    num_actors=24,
    steps_num=256,
    minibatch_size=1536,
    mini_epochs=4,
    seq_len=8,
    max_epochs=11400,
)

'''
The values the presentation reports switching to after 50 million steps.
'''
STAGE2 = TrainingConfig(
    gamma=0.99,
    lam=0.9,
    learning_rate=5e-5,
    grad_norm=0.5,
    entropy_coef=0.001,
    critic_coef=1.0,
    e_clip=0.1,
    clip_value=True,
    normalize_advantage=True,
    num_actors=24,
    steps_num=256,
    minibatch_size=1536,
    mini_epochs=4,
    seq_len=8,
    max_epochs=11400,
)

CONFIGS = {'stage1': STAGE1, 'stage2': STAGE2}


@dataclass(frozen=True)
class EnvConfig:
    '''
    decision_period 5 is the setting under which a v4 episode lasts `t` decisions as the
    v1 episodes did: a v4 episode runs for t * 5 physics steps whatever the decision
    period, measured by holding the no-op action until the episode ended. It is also what
    makes the countdown fed to the network reach zero at the end of the episode, as it did
    in the original training.

    timescale only buys wall-clock speed. The per-decision forward velocity is identical
    at 1, 20, 100 and 300, because the physics step is fixed and decisions are counted in
    Academy steps rather than frames.
    '''
    resolution: int = 84
    decision_period: int = 5
    timescale: int = 300
    target_frame_rate: int = -1
    '''
    The episode length randomization the original training applied to every new level.
    '''
    min_time: int = 200
    max_time: int = 1100
    '''
    animalai_wrapper divided the velocity by this before the network saw it.
    '''
    velocity_scale: tuple = (1.0, 1.0, 16.0)
    visual_frames: int = 2
    velocity_frames: int = 2
    '''
    AnimalStack expressed the remaining time as a fraction of 250 decisions.
    '''
    time_unit: float = 250.0
    physics_steps_per_t: int = 5
    '''
    animalai_wrapper.calc_rewards_v2: a bonus on any positive reward, a bonus for upward
    velocity so that ramps get explored, and a penalty for reversing.
    '''
    reward_bonus: float = 0.5
    ramps_coef: float = 1.0 / 100.0
    back_move_coef: float = 1.0 / 1000.0
