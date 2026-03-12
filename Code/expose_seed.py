import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from MazeGenerator import generate_maze

SEED = 2127825239

# ACTION: Force validate=False to extract the corrupted geometry
raw_matrix = generate_maze(seed=SEED, validate=False)

print(f"RAW MATRIX FOR SEED {SEED}:")
for row in raw_matrix:
    # 1 = Wall (██), 0 = Open (  ), 2 = Door (DD)
    print("".join(["██" if c == 1 else ("DD" if c == 2 else "  ") for c in row]))