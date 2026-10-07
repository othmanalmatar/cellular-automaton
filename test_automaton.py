import json
import os

import numpy as np
import pytest

from automaton import Automaton
from main import App

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIFE = os.path.join(ROOT, "rules", "life.json")
WIRE = os.path.join(ROOT, "rules", "wireworld.json")
BRAIN = os.path.join(ROOT, "rules", "brians_brain.json")


def make_life(rows=10, cols=10):
    return Automaton(LIFE, rows, cols)


class FakeKey:
    def __init__(self, key):
        self.key = key


def test_rules_loaded_from_json():
    a = Automaton(LIFE)
    assert a.name == "Conway's Game of Life"
    assert len(a.states) == 2
    assert len(a.rules) == 3


def test_rules_can_be_changed_without_code(tmp_path):
    rules = {
        "name": "Everything lives",
        "rows": 5, "cols": 5,
        "states": [{"name": "off", "color": "black"},
                   {"name": "on", "color": "white"}],
        "rules": [{"from": 0, "to": 1}],
    }
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(rules))
    a = Automaton(str(path))
    a.step()
    assert a.grid.sum() == 25


def test_bad_rules_are_rejected(tmp_path):
    rules = {
        "states": [{"name": "off", "color": "black"}],
        "rules": [{"from": 0, "to": 5}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rules))
    with pytest.raises(ValueError):
        Automaton(str(path))


def test_board_size():
    a = make_life(12, 20)
    assert a.grid.shape == (12, 20)


def test_set_cell_ignores_outside_board():
    a = make_life()
    a.set_cell(-1, 3, 1)
    a.set_cell(3, 10, 1)
    assert a.grid.sum() == 0
    a.set_cell(2, 3, 1)
    assert a.grid[2, 3] == 1


def test_block_is_still_life():
    a = make_life()
    for r, c in [(4, 4), (4, 5), (5, 4), (5, 5)]:
        a.set_cell(r, c, 1)
    before = a.grid.copy()
    a.step()
    assert np.array_equal(a.grid, before)


def test_blinker_oscillates():
    a = make_life()
    for c in (3, 4, 5):
        a.set_cell(4, c, 1)
    start = a.grid.copy()
    a.step()
    assert not np.array_equal(a.grid, start)
    assert a.grid[3, 4] == a.grid[4, 4] == a.grid[5, 4] == 1
    a.step()
    assert np.array_equal(a.grid, start)


def test_underpopulation_and_overcrowding():
    a = make_life()
    a.set_cell(5, 5, 1)
    a.step()
    assert a.grid.sum() == 0
    for r, c in [(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)]:
        a.set_cell(r, c, 1)
    a.step()
    assert a.grid[5, 5] == 0


def test_dead_edge_treats_outside_as_dead():
    a = make_life()
    for c in (0, 1, 2):
        a.set_cell(0, c, 1)
    a.step()
    assert a.grid[0, 1] == 1
    assert a.grid[1, 1] == 1
    assert a.grid[9, 1] == 0


def test_wrap_edge_connects_sides():
    a = make_life()
    a.edge = "wrap"
    for c in (0, 1, 2):
        a.set_cell(0, c, 1)
    a.step()
    assert a.grid[9, 1] == 1


def test_glider_moves_diagonally():
    a = make_life(12, 12)
    a.load_state(os.path.join(ROOT, "states", "glider.txt"))
    start = np.argwhere(a.grid == 1).min(axis=0)
    for _ in range(4):
        a.step()
    end = np.argwhere(a.grid == 1).min(axis=0)
    assert tuple(end - start) == (1, 1)


def test_wireworld_has_four_states():
    a = Automaton(WIRE)
    assert len(a.states) == 4
    a.set_cell(1, 1, 1)
    a.step()
    assert a.grid[1, 1] == 2
    a.step()
    assert a.grid[1, 1] == 3


def test_wireworld_electron_moves_along_wire():
    a = Automaton(WIRE, 5, 10)
    for c in range(10):
        a.set_cell(2, c, 3)
    a.set_cell(2, 0, 2)
    a.set_cell(2, 1, 1)
    a.step()
    assert a.grid[2, 2] == 1
    assert a.grid[2, 1] == 2


def test_wireworld_loop_keeps_one_electron():
    a = Automaton(WIRE)
    a.load_state(os.path.join(ROOT, "states", "wire_loop.txt"))
    for _ in range(50):
        a.step()
        assert (a.grid == 1).sum() == 1


def test_brians_brain_three_states():
    a = Automaton(BRAIN, 10, 10)
    a.set_cell(5, 5, 1)
    a.set_cell(5, 6, 1)
    a.step()
    assert a.grid[5, 5] == 2
    assert a.grid.max() <= 2


def test_larger_neighborhood_counts_more_cells():
    a = make_life(9, 9)
    a.radius = 2
    a.offsets = a.make_offsets()
    assert len(a.offsets) == 24
    a.set_cell(4, 6, 1)
    assert a.count_neighbors(1)[4, 4] == 1
    a.radius = 1
    a.offsets = a.make_offsets()
    assert a.count_neighbors(1)[4, 4] == 0


def test_von_neumann_neighborhood():
    a = make_life()
    a.neighborhood = "neumann"
    a.offsets = a.make_offsets()
    assert len(a.offsets) == 4


def test_save_and_load_state(tmp_path):
    a = make_life()
    a.set_cell(2, 3, 1)
    a.set_cell(7, 8, 1)
    path = tmp_path / "state.txt"
    a.save_state(str(path))
    b = make_life()
    b.load_state(str(path))
    assert np.array_equal(a.grid, b.grid)


def test_small_state_file_is_centered(tmp_path):
    path = tmp_path / "small.txt"
    path.write_text("11\n11\n")
    a = make_life(10, 10)
    a.load_state(str(path))
    assert a.grid.sum() == 4
    assert a.grid[4, 4] == 1


def test_state_file_too_big_fails(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("1" * 20 + "\n")
    a = make_life()
    with pytest.raises(ValueError):
        a.load_state(str(path))


def test_state_file_with_unknown_state_fails(tmp_path):
    path = tmp_path / "wrong.txt"
    path.write_text("3\n")
    a = make_life()
    with pytest.raises(ValueError):
        a.load_state(str(path))


def test_pause_and_resume_keeps_state():
    app = App(make_life())
    app.auto.set_cell(4, 3, 1)
    app.auto.set_cell(4, 4, 1)
    app.auto.set_cell(4, 5, 1)
    assert not app.running
    app.on_key(FakeKey(" "))
    assert app.running
    app.update(0)
    gen = app.auto.generation
    app.on_key(FakeKey(" "))
    saved = app.auto.grid.copy()
    app.update(1)
    app.update(2)
    assert app.auto.generation == gen
    assert np.array_equal(app.auto.grid, saved)
    app.on_key(FakeKey(" "))
    app.update(3)
    assert app.auto.generation == gen + 1


def test_starts_paused_so_user_can_draw():
    app = App(make_life())
    app.update(0)
    assert app.auto.generation == 0


def test_all_states_have_a_legend_entry():
    app = App(Automaton(WIRE))
    legend = app.ax.get_legend()
    assert len(legend.get_texts()) == 4


def test_speed_limited_to_ten_steps_per_second():
    app = App(make_life())
    for _ in range(10):
        app.on_key(FakeKey("+"))
    assert app.interval >= 100
