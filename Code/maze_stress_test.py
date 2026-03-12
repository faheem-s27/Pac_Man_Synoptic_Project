import os
import sys
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Code.Maze import Maze
from Code.Pathfinding import validate_maze_connectivity


def stress_test(num_trials: int = 1000,
                algorithm: str = "recursive_backtracking"):
    """Generate random mazes until an invalid seed is found, then show it."""
    print(f"[StressTest] Starting maze stress test: up to {num_trials} trials")

    for i in range(1, num_trials + 1):
        seed = random.randint(0, 2**31 - 1)
        print(f"[StressTest] Trial {i}/{num_trials} — testing seed {seed}")
        try:
            maze = Maze(tile_size=40,
                        width=20,
                        height=21,
                        algorithm=algorithm,
                        seed=seed)

            if not validate_maze_connectivity(maze):
                print("[StressTest] FOUND INVALID SEED:", seed)
                print("[StressTest] Launching maze viewer for this seed...")

                from Code import maze_viewer
                maze_viewer.run_viewer(
                    seed=seed,
                    algorithm=algorithm,
                    tile_size=40,
                    show_pellets=True,
                    show_power_pellets=True,
                )
                print("[StressTest] Viewer closed. Stopping stress test.")
                return

        except Exception as e:
            print(f"[StressTest] ERROR on seed {seed}: {e}")

    print("[StressTest] No invalid seeds found in this run.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maze generation stress-test utility.")
    parser.add_argument("--trials", type=int, default=100000,
                        help="Maximum number of random mazes to attempt.")
    parser.add_argument("--algorithm", type=str, default="recursive_backtracking",
                        choices=["recursive_backtracking", "prims", "random_walk"],
                        help="Maze generation algorithm for random mazes.")

    args = parser.parse_args()

    stress_test(num_trials=args.trials,
                algorithm=args.algorithm)
