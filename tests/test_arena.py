"""The arena checks, against inputs whose verdict is known.

These exist because the null-list-key detector was once written and trusted without being
tested, and it read a comment as a value: an arena whose `colors:` is followed by
commented-out entries and then by live ones was called null, and files that were perfectly
good got edited on the strength of it.
"""
import pytest

from rl_animal_torch import arena

GOOD = """!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 1, y: 0, z: 1}
    - !Item
      name: GoodGoal
"""

'''
The shape that the released competition scenarios use: the key, a run of commented-out
entries, and then the entries that are still live. The list is not null.
'''
COMMENTED_THEN_LIVE = """!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 1, y: 0, z: 1}
    - !Item
      name: Wall
      positions:
      - !Vector3 {x: 10, y: 0, z: 10}
      colors:
      # - !RGB {r: 0, g: 0, b: 255}
      # - !RGB {r: 0, g: 0, b: 255}
      - !RGB {r: 0, g: 0, b: 255}
"""

'''
Genuinely null: the next thing at this level is another key.
'''
NULL_COLOURS = """!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 1, y: 0, z: 1}
    - !Item
      name: Wall
      positions:
      - !Vector3 {x: 10, y: 0, z: 10}
      colors:
      rotations: [0]
"""

'''
Also null: the key is the last thing in its item, and the next line dedents into the
enclosing items list.
'''
NULL_COLOURS_AT_END = """!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Wall
      colors:
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 1, y: 0, z: 1}
"""

UNKNOWN_NAME = GOOD.replace('GoodGoal', 'GoodMulti')

AGENT_WITHOUT_POSITIONS = """!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Agent
    - !Item
      name: GoodGoal
"""


def test_reads_time_and_pass_mark():
    assert arena.read_arena_time(GOOD) == 250
    assert arena.read_pass_mark(GOOD) == 0.0
    assert arena.read_pass_mark('pass_mark: 1.5\nt: 250\n') == 1.5


def test_commented_entries_are_not_a_value():
    assert arena.broken_colors(COMMENTED_THEN_LIVE) == []
    arena.validate(COMMENTED_THEN_LIVE, 'fixture')


def test_null_colors_are_found():
    assert arena.broken_colors(NULL_COLOURS) == ['colors']
    assert arena.broken_colors(NULL_COLOURS_AT_END) == ['colors']


def test_good_arena_passes():
    assert arena.broken_colors(GOOD) == []
    arena.validate(GOOD, 'fixture')


def test_unknown_item_name_is_refused():
    with pytest.raises(ValueError, match='GoodMulti'):
        arena.validate(UNKNOWN_NAME, 'fixture')


def test_agent_without_positions_is_refused():
    with pytest.raises(ValueError, match='Agent with no positions'):
        arena.validate(AGENT_WITHOUT_POSITIONS, 'fixture')


def test_write_with_time_replaces_only_the_arena_time(tmp_path):
    path = arena.write_with_time(GOOD, 999, str(tmp_path))
    written = open(path).read()
    assert arena.read_arena_time(written) == 999
    assert written.count('t: ') == 1
    assert 'name: GoodGoal' in written


def test_collect_refuses_or_warns(tmp_path, capsys):
    (tmp_path / 'good.yaml').write_text(GOOD)
    (tmp_path / 'null.yaml').write_text(NULL_COLOURS)

    with pytest.raises(ValueError, match='colors'):
        arena.collect(str(tmp_path), refuse_broken_colors=True)

    assert len(arena.collect(str(tmp_path), refuse_broken_colors=False)) == 2
    assert 'color list with no value' in capsys.readouterr().out


def test_the_shipped_arenas_are_accepted():
    assert len(arena.collect('configs/learning/stage3', refuse_broken_colors=True)) == 59
