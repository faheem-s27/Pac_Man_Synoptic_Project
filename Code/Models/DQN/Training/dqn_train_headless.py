import os
import sys
import torch
import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
_DQN_ROOT  = os.path.dirname(_HERE)                          # Code/Models/DQN/
_TESTING   = os.path.join(_DQN_ROOT, "Testing")
_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _DQN_ROOT not in sys.path:
    sys.path.insert(0, _DQN_ROOT)   # dqn_agent, checkpoint_utils, action_masking_wrapper
if _TESTING not in sys.path:
    sys.path.insert(0, _TESTING)    # eval_dqn

from Code.Environment.PacManEnv import PacManEnv
from Code.Environment.CurriculumManager import CurriculumManager
from dqn_agent import DQNAgent
from eval_dqn import evaluate_model
from checkpoint_utils import save_checkpoint, load_checkpoint
from action_masking_wrapper import DQNActionMaskingWrapper

# ── Configuration ──
TARGET_UPDATE_FREQUENCY = 1000
BATCH_SIZE = 128
SAVE_EVERY_EPISODES = 50
SAVE_PATH = os.path.join(_DQN_ROOT, "Checkpoints", "dqn_pacman.pth")
CHECKPOINT_PATH = os.path.join(_DQN_ROOT, "Checkpoints", "dqn_checkpoint.pt")
INCLUDE_CURRICULUM_STATE = True
# For DQN runs we prefer starvation/game-over as terminal causes over max-step truncation.
DQN_MAX_EPISODE_STEPS = None


def train():
    curriculum = CurriculumManager()

    if DQN_MAX_EPISODE_STEPS is None:
        print("[DQN][Headless] max_episode_steps=None -> no max-step truncation; episodes end via win/death/starvation.")
    else:
        print(f"[DQN][Headless] max_episode_steps={DQN_MAX_EPISODE_STEPS} -> max-step truncation enabled.")

    # Egocentric 31-dim observation, 4 relative actions (F,L,R,B).
    # Obs layout: 4 dirs × 6 channels (wall,food,power,lethal_ghost,edible_ghost,visit_sat)
    # + 3 BFS + 2 power state + 2 progress (pellets_remaining_ratio, explore_rate) = 31.
    agent = DQNAgent(input_dim=31, output_dim=4)
    print(f"Agent initialized on: {agent.device}")

    total_steps = 0
    episode = 0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            load_meta = load_checkpoint(CHECKPOINT_PATH, agent, curriculum=curriculum, map_location=agent.device)
            if load_meta.get("loaded"):
                episode = int(load_meta.get("episode", 0))
                total_steps = int(agent.step_count)
                loaded_keys = int(load_meta.get("loaded_keys", 0))
                total_keys = int(load_meta.get("total_keys", 0))
                load_mode = str(load_meta.get("load_mode", "full"))
                print(
                    f"Resumed checkpoint {CHECKPOINT_PATH} | "
                    f"episode={episode} epsilon={agent.epsilon:.4f} step_count={agent.step_count} "
                    f"load={load_mode} ({loaded_keys}/{total_keys} tensors)"
                )
            else:
                reason = load_meta.get("reason", "unknown")
                print(f"Checkpoint skipped ({reason}); continuing fresh.")
        except Exception as e:
            print(f"Checkpoint load failed, continuing fresh. Error: {e}")
    elif os.path.exists(SAVE_PATH):
        try:
            load_meta = load_checkpoint(SAVE_PATH, agent, curriculum=None, map_location=agent.device)
            if load_meta.get("loaded"):
                loaded_keys = int(load_meta.get("loaded_keys", 0))
                total_keys = int(load_meta.get("total_keys", 0))
                load_mode = str(load_meta.get("load_mode", "full"))
                print(f"Loaded legacy weights from {SAVE_PATH} ({load_mode}, {loaded_keys}/{total_keys} tensors).")
            else:
                reason = load_meta.get("reason", "unknown")
                print(f"Legacy weights skipped ({reason}); continuing fresh.")
        except Exception as e:
            print(f"Legacy weights load failed, continuing fresh. Error: {e}")

    while True:
        episode += 1

        # Get current curriculum stage settings and randomize maze seed
        current_settings = curriculum.get_settings()
        dynamic_seed = int(torch.randint(0, 10_000_000, (1,)).item())
        current_settings['maze_seed'] = dynamic_seed
        current_settings['max_episode_steps'] = DQN_MAX_EPISODE_STEPS

        env = DQNActionMaskingWrapper(PacManEnv(render_mode=None, **current_settings))
        state, _ = env.reset(seed=dynamic_seed)

        total_reward = 0.0
        loss_tracker = []
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            policy_action, exploring = agent.select_action(
                state,
                valid_actions=valid_actions,
                return_exploration=True,
            )
            action = env.pick_action(policy_action, exploring=exploring)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            internal_ticks_step = int(info.get("internal_ticks", 0)) if isinstance(info, dict) else 0
            step_ticks = max(1, internal_ticks_step)
            next_valid_actions = env.get_valid_actions()
            next_valid_mask = np.zeros(agent.action_dim, dtype=np.float32)
            for a in next_valid_actions:
                if 0 <= int(a) < agent.action_dim:
                    next_valid_mask[int(a)] = 1.0
            discount_pow = float(agent.gamma ** step_ticks)

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_mask=next_valid_mask,
                discount_pow=discount_pow,
            )

            loss = agent.optimize_model(batch_size=BATCH_SIZE)
            if loss is not None:
                loss_tracker.append(loss)

            state = next_state
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

        # Episode finished
        avg_loss = sum(loss_tracker) / len(loss_tracker) if loss_tracker else 0.0
        won = env.engine.won
        max_steps = env.max_episode_steps
        print(
            f"Episode {episode} | Stage: {curriculum.current_stage} | MaxSteps: {max_steps} | "
            f"Reward: {total_reward:.1f} | Won: {won} | Epsilon: {agent.epsilon:.3f} | "
            f"Steps: {total_steps} | Avg Loss: {avg_loss:.4f}"
        )

        # Update curriculum performance and possibly promote stage
        curriculum.update_performance(won)
        promoted = curriculum.check_promotion()
        if promoted:
            agent.apply_exploration_jolt(min_epsilon=0.2, duration_steps=50_000)

        env.close()

        # Periodically save
        if episode % SAVE_EVERY_EPISODES == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                agent,
                episode,
                curriculum=curriculum,
                include_curriculum=INCLUDE_CURRICULUM_STATE,
            )
            torch.save(agent.policy_net.state_dict(), SAVE_PATH)
            print(f"Saved checkpoint to {CHECKPOINT_PATH}")
            print(f"Saved weights to {SAVE_PATH}")

            # Launch a short visual evaluation run using the freshly saved model
            # This will open a pygame window and play a few episodes.
            try:
                evaluate_model(SAVE_PATH, episodes=2)
            except SystemExit:
                # Allow user to close the eval window without killing training loop
                pass
            except Exception as e:
                print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    train()