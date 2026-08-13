import os
from dataclasses import dataclass

ENV_PATH = os.path.expanduser("~/animalai_env/Linux/animalAI.x86_64")


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.99
    lam: float = 0.9
    learning_rate: float = 1e-4
    grad_norm: float = 0.5
    entropy_coef: float = 0.01
    critic_coef: float = 1.0
    e_clip: float = 0.2
    clip_value: bool = True
    normalize_advantage: bool = True
    num_actors: int = 24
    steps_num: int = 256
    minibatch_size: int = 1536
    mini_epochs: int = 4
    seq_len: int = 8
    max_epochs: int = 2500


@dataclass(frozen=True)
class EnvConfig:
    resolution: int = 84
    # a v4 episode lasts t * physics_steps_per_t physics steps whatever the decision period,
    # so 5 is the value under which it lasts t decisions, as a v1 episode did
    decision_period: int = 5
    # only buys wall-clock speed: the physics step is fixed and decisions are counted in
    # Academy steps, so the per-decision motion is the same at 1 and at 300
    timescale: int = 300
    target_frame_rate: int = -1
    min_time: int = 200
    max_time: int = 1100
    # the network was trained on the velocity divided by this
    velocity_scale: tuple = (1.0, 1.0, 16.0)
    visual_frames: int = 2
    velocity_frames: int = 2
    time_unit: float = 250.0
    physics_steps_per_t: int = 5
    reward_bonus: float = 0.5
    ramps_coef: float = 1.0 / 100.0
    back_move_coef: float = 1.0 / 1000.0
