"""The Animal-AI v4 environment, adapted to the interface the v1 code expects.

Both the evaluation driver (aai4_eval.py) and training (vecenv.py via
env_configurations) go through here, so the observation and action bridging is
stated once.

What differs between the two Animal-AI generations, and what this does about it:

- The camera observation is CHW float in [0, 1] in v4 and was HWC in v1.
- The vector observation grew from the 3 velocity components to
  [health, vx, vy, vz, px, py, pz]. The network only ever saw velocity, so the
  extra fields are dropped rather than fed in.
- The action space matches exactly: v1 flattened two 3-way branches into
  Discrete(9) as [a // 3, a % 3], and v4's MultiDiscrete([3, 3]) uses the same
  branch order (0 noop / 1 forward / 2 back, 0 noop / 1 right / 2 left).
- The arena YAML is unchanged, but v4 takes it as raw text over a side channel
  instead of as an ArenaConfig object, so overriding `t` means rewriting the text.
- Episode length differs. Measured by holding the no-op action until the episode
  ended, a v4 episode lasts `t` * PHYSICS_STEPS_PER_T physics steps whatever the
  decisionPeriod, and the time penalty accumulated over it is exactly -1.0:

      t=250 dp=3 -> 417 decisions (1.67 x t)   t=250 dp=5 -> 250 decisions (1.00 x t)
      t=500 dp=3 -> 834 decisions (1.67 x t)   t=500 dp=5 -> 500 decisions (1.00 x t)

  v1 ended hard at `t` decisions: over 900 v1 episodes steps/t never exceeded 1.00
  and 44.2% ended exactly at t. So decisionPeriod 5 is the setting under which a v4
  episode is `t` decisions like v1's, which also makes AnimalStack's countdown of
  one 250th per decision reach 0 exactly at the end, as it did in training.
  decisionPeriod 3 instead matches v1's motion per decision (the forward velocity
  after each of the first decisions is 0.85, 3.06, 5.26, 7.25 against v1's 0.85,
  3.00, 5.10, 6.95) at the cost of stretching the episode by 5/3.
"""
import os
import re
import sys
import tempfile
from collections import deque

import numpy as np
from gym import spaces

import animalai_wrapper
import hyperparams as hps

RESOLUTION = 84
ACTIONS_NUM = 9
'''
Every item name the v4 build accepts, taken from the union of the names used by the 900
competition scenarios and by configs/learning/stage3, all of which were checked against
the strings in the v4 build.

This is worth validating up front because of how v4 fails on an unknown name: the arena
ends up with no arenas in it and TrainingArena.SetNextArenaID divides by zero on every
FixedUpdate forever, so the player never answers and the only symptom on this side is a
connection timeout. One arena spun 4.75 million exceptions into a 3.8 GB player log
before it was caught. `GoodMulti` in stage3/lava_mid2 (copy).yaml was such a name -- a
typo for GoodGoalMulti that is absent from the v1 build too, where it was silently
skipped rather than fatal.
'''
SPAWNABLE_NAMES = frozenset([
    'Agent', 'BadGoal', 'BadGoalBounce', 'Cardbox1', 'Cardbox2', 'Cylinder',
    'CylinderTunnel', 'CylinderTunnelTransparent', 'DeathZone', 'GoodGoal',
    'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce', 'HotZone', 'LObject',
    'LObject2', 'Ramp', 'UObject', 'Wall', 'WallTransparent',
])
'''
AnimalStack expressed the remaining episode time as a fraction of 250 steps and
counted it down one step at a time.
'''
TIME_UNIT = 250.0
PHYSICS_STEPS_PER_T = 5


def import_animalai_v4():
    '''
    PYTHONPATH=/workspace puts this repository first on sys.path, and it contains the
    v1 `animalai` package. It carries an __init__.py precisely so that it wins over
    the installed animalai 5.0.1; drop the repository entries for the duration of
    this import to get the 5.0.1 one, which is what speaks to the v4 binary.
    '''
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    saved = list(sys.path)
    sys.path = [p for p in sys.path
                if p not in ('', '.', repo_dir) and os.path.abspath(p or '.') != repo_dir]
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name == 'animalai' or name.startswith('animalai.')}
    for name in saved_modules:
        del sys.modules[name]
    try:
        from animalai.environment import AnimalAIEnvironment
        from mlagents_envs.base_env import ActionTuple
    finally:
        sys.path = saved
        for name, module in saved_modules.items():
            sys.modules[name] = module

    return AnimalAIEnvironment, ActionTuple


def observation_shape():
    return (RESOLUTION, RESOLUTION, 3 * hps.VISUAL_FRAMES_COUNT)


def read_arena_time(raw):
    match = re.search(r'^[ \t]*t:\s*([\d.]+)', raw, flags=re.M)
    if match is None:
        raise ValueError('no t: field in the arena config')

    return int(float(match.group(1)))


LIST_KEYS = ('positions', 'rotations', 'sizes', 'colors', 'skins', 'spawnColors')


def null_list_keys(lines):
    """Yields (key, line index) for every list key written with no value.

    A sequence entry belongs to the key only if it is indented at least as far as the
    key: `colors:` at six spaces followed by `- !Item` at four is the next item of the
    enclosing `items:` list, so that `colors:` is null.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.endswith(':') or stripped[:-1] not in LIST_KEYS:
            continue
        indent = len(line) - len(line.lstrip())
        for following in lines[i + 1:]:
            if len(following.strip()) == 0:
                continue
            following_indent = len(following) - len(following.lstrip())
            if following.lstrip().startswith('-') and following_indent >= indent:
                break
            yield stripped[:-1], i
            break
        else:
            yield stripped[:-1], i


def validate_arena(raw, path):
    '''
    Both of these make the v4 player hang instead of reporting anything, so they are
    worth refusing up front. The player deserialises the arena itself and
    ArenasConfigurations.UpdateWithConfigurationsReceived neither null-checks nor
    catches, so anything the deserialiser trips over leaves the configuration dictionary
    empty; TrainingArena.SetNextArenaID then divides by GetTotalArenas() == 0 on every
    FixedUpdate, forever. One such arena wrote 4.75 million exceptions into a 3.8 GB
    player log before it was noticed.
    '''
    unknown = sorted(set(re.findall(r'^[ \t]*name:[ \t]*([A-Za-z0-9_]+)', raw, flags=re.M))
                     - SPAWNABLE_NAMES)
    if len(unknown) > 0:
        raise ValueError('%s uses item names the v4 build does not have: %s. It would '
                         'hang the player rather than report an error.'
                         % (path, ', '.join(unknown)))

    '''
    A list key with no value deserialises to null and overwrites the C# side's default
    empty list, and Spawnable's constructor iterates it (initVec3sFromRGBs). The v1
    python parser normalised null to [] instead, which is why such files worked there.
    '''
    lines = raw.split('\n')
    empty = [key for key, _ in null_list_keys(lines)]
    if len(empty) > 0:
        raise ValueError('%s has list keys with no value: %s. They deserialise to null '
                         'in the v4 build and hang the player; delete the keys instead.'
                         % (path, ', '.join(sorted(set(empty)))))

    '''
    ArenaBuilders.InstantiateSpawnables reads agentSpawnablesFromUser[0].positions[0]
    without checking the list, so an Agent item carrying no positions throws
    ArgumentOutOfRangeException on every arena reset instead of being placed at random
    the way it was in v1. Asking for a random coordinate explicitly, with -1, is the way
    to express the same thing to v4.
    '''
    starts = [i for i, line in enumerate(lines) if re.match(r'^[ \t]*-[ \t]+!Item[ \t]*$', line)]
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = lines[start:end]
        is_agent = any(re.match(r'^[ \t]*name:[ \t]*Agent[ \t]*$', l) for l in block)
        has_positions = any(re.match(r'^[ \t]*positions:', l) for l in block)
        if is_agent and not has_positions:
            raise ValueError('%s has an Agent with no positions. The v4 build indexes '
                             'positions[0] unconditionally and hangs; give it '
                             '!Vector3 {x: -1, y: 0, z: -1} to keep it random.' % path)


def write_config_with_time(raw, arena_time, directory):
    '''
    v4 reads the arena as text, so a randomised episode length has to be written out
    as a file rather than set on a parsed config object the way training did in v1.
    '''
    patched = re.sub(r'^([ \t]*)t:\s*[\d.]+', r'\g<1>t: %d' % arena_time, raw,
                     count=1, flags=re.M)
    handle, path = tempfile.mkstemp(suffix='.yaml', dir=directory)
    with os.fdopen(handle, 'w') as out_file:
        out_file.write(patched)
    return path


class Stacker:
    '''
    The frame and velocity stacking that animalai_wrapper.AnimalSkip/AnimalWrapper/
    AnimalStack applied in the v1 pipeline. SKIP_FRAMES is 1 in hyperparams, so there
    is no action repeat to reproduce.
    '''
    def __init__(self, time_decrement):
        self.frames = deque([], maxlen=hps.VISUAL_FRAMES_COUNT)
        self.vels = deque([], maxlen=hps.VEL_FRAMES_COUNT)
        self.time = 0.0
        self.time_decrement = time_decrement

    @staticmethod
    def to_frame(camera):
        return np.asarray(np.transpose(camera, (1, 2, 0)) * 255.0, dtype=np.uint8)

    @staticmethod
    def to_raw_velocity(vector):
        return np.asarray(vector[1:4], dtype=np.float32)

    def to_velocity(self, vector):
        '''
        AnimalWrapper divided the velocity by VEC_SCALE before AnimalStack saw it.
        '''
        return self.to_raw_velocity(vector) / hps.VEC_SCALE

    def reset(self, camera, vector, arena_time):
        self.time = arena_time / TIME_UNIT
        frame = self.to_frame(camera)
        for _ in range(hps.VISUAL_FRAMES_COUNT):
            self.frames.append(frame)
        for _ in range(hps.VEL_FRAMES_COUNT - 1):
            self.vels.append(np.array([0.0, 0.0, 0.0, self.time], dtype=np.float32))
        self.vels.append(np.append(self.to_velocity(vector), self.time))

    def step(self, camera, vector):
        self.time -= self.time_decrement
        self.frames.append(self.to_frame(camera))
        self.vels.append(np.append(self.to_velocity(vector), self.time))

    def observation(self):
        return [np.concatenate(self.frames, axis=-1), np.concatenate(self.vels)]


class AnimalV4Env:
    '''
    One Unity instance behind the reset/step interface the v1 wrappers exposed:
    reset returns [visual uint8 (84, 84, 3 * VISUAL_FRAMES_COUNT), vels float32 (4 *
    VEL_FRAMES_COUNT,)] and step takes the flattened Discrete(ACTIONS_NUM) action.
    '''
    def __init__(self, env_path, worker_id, base_port, seed, first_config,
                 decision_period, timescale, target_frame_rate, time_decrement,
                 shape_rewards):
        environment_class, self.action_tuple_class = import_animalai_v4()
        '''
        When the player fails to come up, the only account of why is Unity's own log,
        which otherwise lands in the container's throwaway home directory. Set
        AAI4_LOG_DIR to an absolute path to keep it; animalai names the file after the
        worker id.
        '''
        if 'AAI4_LOG_DIR' in os.environ:
            log_folder = os.environ['AAI4_LOG_DIR']
        else:
            log_folder = ''
        self.env = environment_class(
            file_name=env_path,
            log_folder=log_folder,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            play=False,
            arenas_configurations=first_config,
            useCamera=True,
            resolution=RESOLUTION,
            grayscale=False,
            useRayCasts=False,
            no_graphics=False,
            decisionPeriod=decision_period,
            timescale=timescale,
            targetFrameRate=target_frame_rate,
        )
        self.behavior = list(self.env.behavior_specs.keys())[0]
        self.stacker = Stacker(time_decrement)
        '''
        Training shaped the reward (a bonus for picking food up, a penalty for moving
        backwards, a bonus for moving up ramps); evaluation scores the environment
        reward itself so that the numbers stay comparable to the competition score.
        '''
        self.shape_rewards = shape_rewards

        visual_channels = 3 * hps.VISUAL_FRAMES_COUNT
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(RESOLUTION, RESOLUTION, visual_channels),
            dtype=np.uint8)
        self.action_space = spaces.Discrete(ACTIONS_NUM)

    def _current(self):
        decision, terminal = self.env.get_steps(self.behavior)
        if len(terminal) > 0:
            return terminal.obs[0][0], terminal.obs[1][0], float(terminal.reward[0]), True
        return decision.obs[0][0], decision.obs[1][0], float(decision.reward[0]), False

    def reset(self, config=None):
        if config is not None:
            self.env.reset(config)
        else:
            self.env.reset()
        camera, vector, _, _ = self._current()
        self.stacker.reset(camera, vector, self.arena_time)
        return self.stacker.observation()

    def set_arena_time(self, arena_time):
        '''
        Set before reset: the countdown fed to the network starts from it.
        '''
        self.arena_time = arena_time

    def send(self, action):
        '''
        Split from receive so that a driver holding several instances can hand the
        action to all of them before waiting on any, which is what keeps the Unity
        processes working in parallel.
        '''
        self.env.set_actions(self.behavior, self.action_tuple_class(
            continuous=np.zeros((1, 0), dtype=np.float32),
            discrete=np.array([[action // 3, action % 3]], dtype=np.int32)))
        self.env.step()

    def receive(self):
        camera, vector, reward, done = self._current()
        self.stacker.step(camera, vector)
        if self.shape_rewards:
            reward = animalai_wrapper.calc_rewards_v2(
                reward, self.stacker.to_raw_velocity(vector))

        return self.stacker.observation(), np.asarray(reward), np.asarray(done, dtype=bool), {}

    def step(self, action):
        self.send(action)
        return self.receive()

    def close(self):
        self.env.close()
