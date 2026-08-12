"""The hyperparameters of the winning run, and the environment settings."""
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


TRAINING = TrainingConfig(
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
    min_time: int = 200
    max_time: int = 1100
    velocity_scale: tuple = (1.0, 1.0, 16.0)
    visual_frames: int = 2
    velocity_frames: int = 2
    time_unit: float = 250.0
    physics_steps_per_t: int = 5
    reward_bonus: float = 0.5
    ramps_coef: float = 1.0 / 100.0
    back_move_coef: float = 1.0 / 1000.0
