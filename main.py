import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from automaton import Automaton

SAVE_FILE = "saved_state.txt"
MIN_INTERVAL = 100  # ms, so at most 10 steps per second


class App:
    def __init__(self, auto):
        self.auto = auto
        self.running = False
        self.brush = 1
        self.interval = 200
        self.setup_keys()
        self.fig, self.ax = plt.subplots(figsize=(8, 7))

        colors = [s["color"] for s in auto.states]
        self.img = self.ax.imshow(
            auto.grid, cmap=ListedColormap(colors),
            vmin=0, vmax=len(colors) - 1, interpolation="nearest")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        handles = [Patch(facecolor=s["color"], edgecolor="gray",
                         label=f"{i}: {s['name']}")
                   for i, s in enumerate(auto.states)]
        self.ax.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(1.01, 1), title="States")
        self.fig.subplots_adjust(right=0.75)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.anim = FuncAnimation(
            self.fig, self.update, interval=self.interval,
            cache_frame_data=False)
        self.draw_title()

    def setup_keys(self):
        for key in list(plt.rcParams):
            if key.startswith("keymap.") and key != "keymap.quit":
                plt.rcParams[key] = []

    def draw_title(self):
        status = "RUNNING" if self.running else "PAUSED"
        brush = self.auto.states[self.brush]["name"]
        self.ax.set_title(
            f"{self.auto.name} | gen {self.auto.generation} | {status}\n"
            f"brush: {brush} | edge: {self.auto.edge}")
        self.img.set_data(self.auto.grid)

    def update(self, frame):
        if self.running:
            self.auto.step()
        self.draw_title()
        return [self.img]

    def on_click(self, event):
        if event.inaxes != self.ax or event.button not in (1, 3):
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        state = self.brush if event.button == 1 else 0
        self.auto.set_cell(row, col, state)
        self.draw_title()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == " ":
            self.running = not self.running
        elif event.key == "n" and not self.running:
            self.auto.step()
        elif event.key == "c":
            self.running = False
            self.auto.clear()
        elif event.key == "r":
            self.running = False
            self.auto.randomize()
        elif event.key == "s":
            self.auto.save_state(SAVE_FILE)
            print("saved to", SAVE_FILE)
        elif event.key == "l":
            self.load(SAVE_FILE)
        elif event.key == "+":
            self.set_speed(self.interval - 50)
        elif event.key == "-":
            self.set_speed(self.interval + 50)
        elif event.key in [str(i) for i in range(len(self.auto.states))]:
            self.brush = int(event.key)
        self.draw_title()
        self.fig.canvas.draw_idle()

    def set_speed(self, interval):
        self.interval = max(MIN_INTERVAL, min(interval, 1000))
        self.anim.event_source.interval = self.interval

    def load(self, path):
        self.running = False
        try:
            self.auto.load_state(path)
        except (OSError, ValueError) as err:
            print("could not load", path, "-", err)


def main():
    parser = argparse.ArgumentParser(description="Cellular automaton")
    parser.add_argument("rules", help="path to a rules JSON file")
    parser.add_argument("--state", help="state file to load at start")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    args = parser.parse_args()

    auto = Automaton(args.rules, args.rows, args.cols)
    app = App(auto)
    if args.state:
        app.load(args.state)
        app.draw_title()
    print("Controls: click = paint, right click = erase, 0-9 = pick brush,")
    print("space = start/pause, n = one step, c = clear, r = random,")
    print("s = save, l = load saved, + / - = speed")
    plt.show()


if __name__ == "__main__":
    main()
