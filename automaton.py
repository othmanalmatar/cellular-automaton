import json

import numpy as np


class Automaton:
    def __init__(self, rules_file, rows=None, cols=None):
        self.load_rules(rules_file)
        if rows is not None:
            self.rows = rows
        if cols is not None:
            self.cols = cols
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.generation = 0

    def load_rules(self, rules_file):
        with open(rules_file) as f:
            data = json.load(f)

        self.name = data.get("name", "Unnamed")
        self.rows = data.get("rows", 40)
        self.cols = data.get("cols", 40)
        self.edge = data.get("edge", "dead")
        self.neighborhood = data.get("neighborhood", "moore")
        self.radius = data.get("radius", 1)
        self.states = data["states"]
        self.rules = data["rules"]
        self.check_rules()
        self.offsets = self.make_offsets()

    def check_rules(self):
        if self.edge not in ("dead", "wrap"):
            raise ValueError("edge must be 'dead' or 'wrap'")
        if self.neighborhood not in ("moore", "neumann"):
            raise ValueError("neighborhood must be 'moore' or 'neumann'")
        if self.radius < 1:
            raise ValueError("radius must be at least 1")
        n = len(self.states)
        for rule in self.rules:
            for key in ("from", "to", "neighbor_state"):
                if key in rule and not 0 <= rule[key] < n:
                    raise ValueError(f"{key} state {rule[key]} does not exist")

    def make_offsets(self):
        offsets = []
        r = self.radius
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                if self.neighborhood == "neumann" and abs(dr) + abs(dc) > r:
                    continue
                offsets.append((dr, dc))
        return offsets

    def shift(self, mask, dr, dc):
        if self.edge == "wrap":
            return np.roll(mask, (-dr, -dc), axis=(0, 1))
        out = np.zeros_like(mask)
        rows, cols = mask.shape
        src_r = slice(max(dr, 0), rows + min(dr, 0))
        dst_r = slice(max(-dr, 0), rows + min(-dr, 0))
        src_c = slice(max(dc, 0), cols + min(dc, 0))
        dst_c = slice(max(-dc, 0), cols + min(-dc, 0))
        out[dst_r, dst_c] = mask[src_r, src_c]
        return out

    def count_neighbors(self, state):
        mask = (self.grid == state).astype(int)
        total = np.zeros_like(mask)
        for dr, dc in self.offsets:
            total += self.shift(mask, dr, dc)
        return total

    def step(self):
        new_grid = self.grid.copy()
        done = np.zeros(self.grid.shape, dtype=bool)
        for rule in self.rules:
            hit = (self.grid == rule["from"]) & ~done
            if "neighbor_state" in rule:
                count = self.count_neighbors(rule["neighbor_state"])
                hit &= count >= rule.get("min", 0)
                hit &= count <= rule.get("max", 10 ** 6)
                if "counts" in rule:
                    hit &= np.isin(count, rule["counts"])
            new_grid[hit] = rule["to"]
            done |= hit
        self.grid = new_grid
        self.generation += 1

    def set_cell(self, row, col, state):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row, col] = state

    def clear(self):
        self.grid[:] = 0
        self.generation = 0

    def randomize(self, density=0.3):
        n = len(self.states)
        alive = np.random.rand(self.rows, self.cols) < density
        values = np.random.randint(1, n, size=self.grid.shape)
        self.grid = np.where(alive, values, 0)
        self.generation = 0

    def load_state(self, path):
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        data = np.array([[int(ch) for ch in line] for line in lines])
        if data.max() >= len(self.states):
            raise ValueError("state file uses a state the rules do not have")
        h, w = data.shape
        if h > self.rows or w > self.cols:
            raise ValueError("state file is bigger than the board")
        self.clear()
        top = (self.rows - h) // 2
        left = (self.cols - w) // 2
        self.grid[top:top + h, left:left + w] = data

    def save_state(self, path):
        with open(path, "w") as f:
            for row in self.grid:
                f.write("".join(str(v) for v in row) + "\n")
