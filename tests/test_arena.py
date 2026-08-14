"""The arena checks, against inputs whose verdict is known."""

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

UNKNOWN_NAME = GOOD.replace("GoodGoal", "GoodMulti")

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
    assert arena.read_pass_mark("pass_mark: 1.5\nt: 250\n") == 1.5


def test_commented_entries_are_not_a_value():
    assert arena.null_color_lists(COMMENTED_THEN_LIVE) == []
    arena.validate(COMMENTED_THEN_LIVE, "fixture")


def test_null_color_lists_are_found():
    assert arena.null_color_lists(NULL_COLOURS) == ["colors"]
    assert arena.null_color_lists(NULL_COLOURS_AT_END) == ["colors"]


def test_good_arena_passes():
    assert arena.null_color_lists(GOOD) == []
    arena.validate(GOOD, "fixture")


def test_unknown_item_name_is_refused():
    with pytest.raises(AssertionError, match="GoodMulti"):
        arena.validate(UNKNOWN_NAME, "fixture")


def test_agent_without_positions_is_refused():
    with pytest.raises(AssertionError, match="Agent with no positions"):
        arena.validate(AGENT_WITHOUT_POSITIONS, "fixture")


def test_collect_refuses_a_broken_arena(tmp_path):
    (tmp_path / "good.yaml").write_text(GOOD)
    assert len(arena.collect([str(tmp_path)])) == 1

    (tmp_path / "null.yaml").write_text(NULL_COLOURS)
    with pytest.raises(AssertionError, match="with no value"):
        arena.collect([str(tmp_path)])


def test_the_shipped_arenas_are_accepted():
    assert len(arena.collect(["external/animal-ai/configs/rank1_training_data"])) == 59
    assert len(arena.collect(["external/animal-ai/configs/competition"])) == 900
