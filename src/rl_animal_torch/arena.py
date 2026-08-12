"""Reading, checking and rewriting the Animal-AI arena configuration files.

v4 takes the arena as raw YAML text over a side channel and parses it itself, so this
never builds an object model: it reads the two fields the driver needs and refuses the
files that would hang the player.

Two ways of writing an arena make the v4 player spin instead of reporting anything, and
neither is visible from python: the symptoms are a connection timeout and a player log
growing by gigabytes. One arena wrote 4.75 million exceptions into 3.8 GB before it was
caught, which is why they are refused up front.

- An item name the build does not have. The deserialized configuration ends up empty
  because ArenasConfigurations.UpdateWithConfigurationsReceived neither null-checks nor
  catches, and TrainingArena.SetNextArenaID then divides by a zero arena count on every
  FixedUpdate. `GoodMulti`, a typo for GoodGoalMulti in one training level, did this; the
  name is absent from the v1 build too, where it was silently skipped.
- An Agent item with no positions. ArenaBuilders.InstantiateSpawnables reads
  agentSpawnablesFromUser[0].positions[0] unconditionally, so it throws
  ArgumentOutOfRangeException on every arena reset. In v1 an item with no positions was
  placed at random; the way to ask v4 for that is an explicit -1 coordinate.

`null_list_keys` finds a third suspicious shape, a list key written with no value, which
deserializes to null rather than to the C# side's default empty list. It is reported by
the tooling but not refused: the 900 released competition scenarios contain nine of them
and run, so on its own it is survivable.
"""
import os
import re
import tempfile

'''
Every item name the v4 build accepts, from the union of the names used by the 900
competition scenarios and by configs/learning/stage3.
'''
SPAWNABLE_NAMES = frozenset([
    'Agent', 'BadGoal', 'BadGoalBounce', 'Cardbox1', 'Cardbox2', 'Cylinder',
    'CylinderTunnel', 'CylinderTunnelTransparent', 'DeathZone', 'GoodGoal',
    'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce', 'HotZone', 'LObject',
    'LObject2', 'Ramp', 'UObject', 'Wall', 'WallTransparent',
])

LIST_KEYS = ('positions', 'rotations', 'sizes', 'colors', 'skins', 'spawnColors')

ITEM_PATTERN = re.compile(r'^[ \t]*-[ \t]+!Item[ \t]*$')
AGENT_PATTERN = re.compile(r'^[ \t]*name:[ \t]*Agent[ \t]*$')
POSITIONS_PATTERN = re.compile(r'^[ \t]*positions:')
NAME_PATTERN = re.compile(r'^[ \t]*name:[ \t]*([A-Za-z0-9_]+)', re.M)
TIME_PATTERN = re.compile(r'^[ \t]*t:[ \t]*([\d.]+)', re.M)


def read_arena_time(raw):
    match = TIME_PATTERN.search(raw)
    if match is None:
        raise ValueError('no t: field in the arena config')

    return int(float(match.group(1)))


def read_pass_mark(raw):
    '''
    10-22-2 is the only released competition scenario without a pass_mark; the other two
    variants of the same test both use 0.
    '''
    match = re.search(r'pass_mark:\s*([-\d.]+)', raw)
    if match is None:
        return 0.0

    return float(match.group(1))


def null_list_keys(lines):
    '''
    Yields (key, line index) for every list key written with no value. A sequence entry
    belongs to the key only if it is indented at least as far as the key: `colors:` at six
    spaces followed by `- !Item` at four is the next entry of the enclosing `items:` list,
    which leaves `colors:` null.
    '''
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.endswith(':') or stripped[:-1] not in LIST_KEYS:
            continue
        indent = len(line) - len(line.lstrip())
        for following in lines[index + 1:]:
            content = following.strip()
            '''
            Blank lines and comments say nothing about whether the key has a value. Nine of
            the released competition scenarios write a long run of commented-out !RGB
            entries between `colors:` and the entries that are still live, and reading the
            first comment as the answer calls a perfectly good list null.
            '''
            if len(content) == 0 or content.startswith('#'):
                continue
            following_indent = len(following) - len(following.lstrip())
            if content.startswith('-') and following_indent >= indent:
                break
            yield stripped[:-1], index
            break
        else:
            yield stripped[:-1], index


def item_blocks(lines):
    starts = [i for i, line in enumerate(lines) if ITEM_PATTERN.match(line)]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        yield lines[start:end]


def broken_colors(raw):
    '''
    A `colors:` written with no value deserializes to null instead of leaving the C# side's
    default empty list, and Spawnable's constructor iterates it. Verified by restoring the
    single line into two training levels: with it the player dies, without it both run.

        Error processing SideChannel message: System.NullReferenceException
          at ArenasParameters.Spawnable.initVec3sFromRGBs
          at ArenasParameters.Spawnable..ctor
          at ArenasParameters.ArenasConfigurations.Add
        KeyNotFoundException: Tried to load arena 0 but it did not exist
        DivideByZeroException x 774678

    How bad it is depends on whether an arena was added before the exception.
    ArenasConfigurations.Add throwing on arena 0 leaves no arenas at all, and
    TrainingArena.SetNextArenaID then divides by zero forever. Failing later leaves the
    earlier arenas in place, and the player keeps answering while running an arena that was
    never fully built: nine of the released competition scenarios are like this, and every
    one of them scores exactly the step cap with only the time penalty.

    A null `positions` or `sizes` on a non-Agent item is survivable; 101 of those appear
    across the released scenarios, which all complete.
    '''
    return sorted({key for key, _ in null_list_keys(raw.split('\n'))}
                  & {'colors', 'spawnColors'})


def validate(raw, path):
    unknown = sorted(set(NAME_PATTERN.findall(raw)) - SPAWNABLE_NAMES)
    if len(unknown) > 0:
        raise ValueError('%s uses item names the v4 build does not have: %s. It would '
                         'hang the player rather than report an error.'
                         % (path, ', '.join(unknown)))

    lines = raw.split('\n')
    for block in item_blocks(lines):
        is_agent = any(AGENT_PATTERN.match(line) for line in block)
        has_positions = any(POSITIONS_PATTERN.match(line) for line in block)
        if is_agent and not has_positions:
            raise ValueError('%s has an Agent with no positions. The v4 build indexes '
                             'positions[0] unconditionally and hangs; give it '
                             '!Vector3 {x: -1, y: 0, z: -1} to keep it random.' % path)


def write_with_time(raw, arena_time, directory):
    '''
    v4 reads the arena as text, so a randomized episode length has to be written out as a
    file rather than set on a parsed config object the way the v1 training did.
    '''
    patched = re.sub(r'^([ \t]*)t:[ \t]*[\d.]+', r'\g<1>t: %d' % arena_time, raw,
                     count=1, flags=re.M)
    handle, path = tempfile.mkstemp(suffix='.yaml', dir=directory)
    with os.fdopen(handle, 'w') as out_file:
        out_file.write(patched)
    return path


def collect(directory, refuse_broken_colors):
    '''
    Every arena file in a directory, checked.

    What hangs the player is always refused: one bad draw stalls training for good, and the
    workers draw levels at random on every reset. A null color list is only refused when
    asked for, because it does not always hang and because the released competition
    scenarios have to be runnable as written; scoring them means accepting that nine of the
    900 are built wrong. Training levels are ours to fix, so there it is an error.
    '''
    paths = sorted(os.path.join(directory, name) for name in os.listdir(directory)
                   if name.endswith('.yml') or name.endswith('.yaml'))
    if len(paths) == 0:
        raise ValueError('no arena files in ' + directory)

    broken = []
    for path in paths:
        raw = open(path).read()
        validate(raw, path)
        keys = broken_colors(raw)
        if len(keys) > 0:
            broken.append((path, keys))

    if len(broken) > 0:
        if refuse_broken_colors:
            path, keys = broken[0]
            raise ValueError('%s writes %s with no value, which deserializes to null and '
                             'kills the v4 player inside initVec3sFromRGBs; delete the key. '
                             '%d file(s) in this directory do it.'
                             % (path, ', '.join(keys), len(broken)))
        print('warning: %d arena(s) write a color list with no value, so the v4 player '
              'builds them wrong: %s'
              % (len(broken), ', '.join(os.path.basename(path) for path, _ in broken)))

    return paths
