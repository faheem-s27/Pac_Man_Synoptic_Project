import os
from typing import Any

import torch


def _serialize_curriculum(curriculum) -> dict[str, Any]:
    return {
        "current_stage": int(getattr(curriculum, "current_stage", 0)),
        "recent_results": list(getattr(curriculum, "recent_results", [])),
    }


def _restore_curriculum(curriculum, state: dict[str, Any]):
    if curriculum is None or not isinstance(state, dict):
        return

    curriculum.current_stage = int(state.get("current_stage", curriculum.current_stage))
    restored_results = state.get("recent_results", [])
    curriculum.recent_results.clear()
    for value in restored_results:
        curriculum.recent_results.append(bool(value))


def save_checkpoint(path: str, agent, episode: int, curriculum=None, include_curriculum: bool = True):
    payload: dict[str, Any] = {
        "format_version": 1,
        "episode": int(episode),
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": float(agent.epsilon),
        "step_count": int(agent.step_count),
    }

    if include_curriculum and curriculum is not None:
        payload["curriculum_state"] = _serialize_curriculum(curriculum)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, agent, curriculum=None, map_location=None) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"loaded": False, "reason": "missing"}

    data = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(data, dict) and "policy_state_dict" in data:
        agent.policy_net.load_state_dict(data["policy_state_dict"])

        target_state = data.get("target_state_dict")
        if target_state:
            agent.target_net.load_state_dict(target_state)
        else:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        optimizer_state = data.get("optimizer_state_dict")
        if optimizer_state:
            agent.optimizer.load_state_dict(optimizer_state)

        agent.epsilon = float(data.get("epsilon", agent.epsilon))
        agent.step_count = int(data.get("step_count", agent.step_count))

        if curriculum is not None and "curriculum_state" in data:
            _restore_curriculum(curriculum, data["curriculum_state"])

        return {
            "loaded": True,
            "episode": int(data.get("episode", 0)),
            "is_full_checkpoint": True,
        }

    # Legacy fallback: weights-only .pth files
    if isinstance(data, dict):
        agent.policy_net.load_state_dict(data)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        return {
            "loaded": True,
            "episode": 0,
            "is_full_checkpoint": False,
        }

    return {"loaded": False, "reason": "unsupported_format"}

