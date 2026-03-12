import os
import sys
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Code.Maze import Maze
from Code.Pathfinding import validate_maze_connectivity


def stress_test(num_trials: int = 1000,
                use_classic: bool = False,
                algorithm: str = "recursive_backtracking"):
    """Generate random mazes until an invalid seed is found, then show it.

    Behaviour:
      - For each trial, pick a random seed and build a Maze for it.
      - Wrap the Maze and run `validate_maze_connectivity`.
      - When a seed fails validation, print it and immediately invoke the
        maze viewer for visual inspection, then stop further trials.
    """
    print(f"[StressTest] Starting maze stress test: up to {num_trials} trials")

    for i in range(1, num_trials + 1):
        seed = random.randint(0, 2**31 - 1)
        print(f"[StressTest] Trial {i}/{num_trials} — testing seed {seed}")
        try:
            maze = Maze(tile_size=40,
                        width=20,
                        height=21,
                        use_classic=use_classic,
                        algorithm=algorithm,
                        seed=seed)

            # Only makes sense for non-classic generated mazes
            if use_classic:
                continue

            # Use the same validator as the main code path
            if not validate_maze_connectivity(maze):
                print("[StressTest] FOUND INVALID SEED:", seed)
                print("[StressTest] Launching maze viewer for this seed...")

                # Lazy import of viewer entry point to avoid circular deps
                from Code import maze_viewer
                maze_viewer.run_viewer(
                    seed=seed,
                    use_classic=False,
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
    parser.add_argument("--trials", type=int, default=10000,
                        help="Maximum number of random mazes to attempt.")
    parser.add_argument("--use-classic", action="store_true",
                        help="Use the classic maze layout (validation skipped).")
    parser.add_argument("--algorithm", type=str, default="recursive_backtracking",
                        choices=["recursive_backtracking", "prims", "random_walk"],
                        help="Maze generation algorithm for random mazes.")

    args = parser.parse_args()

    stress_test(num_trials=args.trials,
                use_classic=args.use_classic,
                algorithm=args.algorithm)

