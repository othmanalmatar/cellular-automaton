# Cellular Automaton

Run with a rules file, and optionally a starting state:

    python main.py rules/life.json --state states/glider.txt
    python main.py rules/wireworld.json --state states/wire_loop.txt
    python main.py rules/cyclic.json

The program starts paused so you can draw the starting pattern.

Controls: left click paints, right click erases, number keys choose the
brush state, space starts or pauses, n steps once while paused, c clears,
r fills randomly, s saves to saved_state.txt, l loads it back,
plus and minus change the speed.

Rules files are JSON. See the files in rules/ for examples. Each rule says
a cell in state "from" becomes "to" when the number of neighbors in
"neighbor_state" is in "counts" or between "min" and "max". The first
matching rule wins.

State files are plain text, one digit per cell. If the pattern is smaller
than the board it is placed in the middle.

Tests: pytest
