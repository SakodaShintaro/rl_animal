"""Reading, checking and rewriting the arena configuration files.

An arena the v4 player cannot build makes it stop answering instead of reporting anything,
so the shapes that do that are refused before any instance is launched.
"""
import os
import re
import tempfile

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
    assert match is not None, 'no t: field in the arena config'

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
    assert len(unknown) == 0, (
        f'{path} uses item names the v4 build does not have: {", ".join(unknown)}. It would '
        f'hang the player rather than report an error.')

    for block in item_blocks(raw.split('\n')):
        is_agent = any(AGENT_PATTERN.match(line) for line in block)
        has_positions = any(POSITIONS_PATTERN.match(line) for line in block)
        assert not (is_agent and not has_positions), (
            f'{path} has an Agent with no positions. The v4 build indexes positions[0] '
            f'unconditionally and hangs; give it !Vector3 {{x: -1, y: 0, z: -1}} to keep it '
            f'random.')


def write_with_time(raw, arena_time, directory):
    '''
    v4 reads the arena as text, so a randomized episode length has to be written out as a
    file rather than set on a parsed config object the way the v1 training did.
    '''
    patched = re.sub(r'^([ \t]*)t:[ \t]*[\d.]+', rf'\g<1>t: {arena_time}', raw,
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
    assert len(paths) > 0, f'no arena files in {directory}'

    broken = []
    for path in paths:
        raw = open(path).read()
        validate(raw, path)
        keys = broken_colors(raw)
        if len(keys) > 0:
            broken.append((path, keys))

    names = ', '.join(os.path.basename(path) for path, _ in broken)
    assert not (refuse_broken_colors and len(broken) > 0), (
        f'{names} write a color list with no value, which deserializes to null and kills the '
        f'v4 player inside initVec3sFromRGBs; delete the key.')
    if len(broken) > 0:
        print(f'warning: {len(broken)} arena(s) write a color list with no value, so the v4 '
              f'player builds them wrong: {names}')

    return paths
