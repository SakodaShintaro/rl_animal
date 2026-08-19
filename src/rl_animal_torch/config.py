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
    # Frames at which a model-only checkpoint is written, so the evaluation sweep can draw
    # a steps-vs-pass-rate curve instead of scoring one end point. The single-environment
    # runs this is the reference for are compared at 2,000,000 frames, so that value has to
    # be one of these. The last one is `num_actors * steps_num * max_epochs`, i.e. the end
    # of the run. A frame count is only crossed, never hit exactly -- the counter advances
    # by a whole batch per epoch -- so the file is named after the frame it was written at.
    checkpoint_frames: tuple = (
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
        8_000_000,
        15_360_000,
    )
    # How often the resume checkpoint (model and optimizer, ~114 MB) is rewritten. Writing
    # it every epoch was 2500 rewrites of that file over a run.
    trainer_save_interval_epochs: int = 25


@dataclass(frozen=True)
class EnvConfig:
    # 96 rather than the v1 wrappers' 84 to match the Animal-AI environment the
    # single-environment runs are trained in. The convolution tower pools by 2 four times
    # and rounds up, so 84 and 96 both leave a 6x6 map and the network is unchanged: only
    # what the camera renders differs.
    resolution: int = 96
    # a v4 episode lasts t * physics_steps_per_t physics steps whatever the decision period,
    # so 5 is the value under which it lasts t decisions, as a v1 episode did
    decision_period: int = 5
    # only buys wall-clock speed: the physics step is fixed and decisions are counted in
    # Academy steps, so the per-decision motion is the same at 1 and at 300
    timescale: int = 300
    target_frame_rate: int = -1
    # the network was trained on the velocity divided by this
    velocity_scale: tuple = (1.0, 1.0, 16.0)
    visual_frames: int = 1
    velocity_frames: int = 1
    time_unit: float = 250.0
    physics_steps_per_t: int = 5
    reward_bonus: float = 0.5
    ramps_coef: float = 1.0 / 100.0
    back_move_coef: float = 1.0 / 1000.0
