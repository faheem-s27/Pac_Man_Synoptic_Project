import os
from typing import Any

import torch


def _load_state_dict_compatible(module, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Load only keys that exist and match shape; return load stats."""
    if not isinstance(state_dict, dict):
        return {"loaded_keys": 0, "total_keys": 0, "missing": [], "unexpected": []}

    target_state = module.state_dict()
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key in target_state and getattr(target_state[key], "shape", None) == getattr(value, "shape", None):
            filtered[key] = value

    if not filtered:
        return {
            "loaded_keys": 0,
            "total_keys": len(state_dict),
            "missing": list(target_state.keys()),
            "unexpected": list(state_dict.keys()),
        }

    load_result = module.load_state_dict(filtered, strict=False)
    return {
        "loaded_keys": len(filtered),
        "total_keys": len(state_dict),
        "missing": list(load_result.missing_keys),
        "unexpected": list(load_result.unexpected_keys),
    }


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
        "amp_enabled": bool(getattr(agent, "amp_enabled", False)),
    }

    grad_scaler = getattr(agent, "grad_scaler", None)
    if grad_scaler is not None and hasattr(grad_scaler, "state_dict"):
        payload["grad_scaler_state_dict"] = grad_scaler.state_dict()

    if include_curriculum and curriculum is not None:
        payload["curriculum_state"] = _serialize_curriculum(curriculum)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, agent, curriculum=None, map_location=None) -> dict[str, Any]:
    if not os.path.exists(path):
        return {"loaded": False, "reason": "missing"}

    data = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(data, dict) and "policy_state_dict" in data:
        policy_stats = _load_state_dict_compatible(agent.policy_net, data["policy_state_dict"])
        if policy_stats["loaded_keys"] <= 0:
            return {
                "loaded": False,
                "reason": "incompatible_policy_state",
                "episode": int(data.get("episode", 0)),
                "is_full_checkpoint": True,
                "loaded_keys": 0,
                "total_keys": int(policy_stats["total_keys"]),
            }

        target_state = data.get("target_state_dict")
        if isinstance(target_state, dict):
            target_stats = _load_state_dict_compatible(agent.target_net, target_state)
            if target_stats["loaded_keys"] <= 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        else:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        optimizer_state = data.get("optimizer_state_dict")
        if optimizer_state:
            try:
                agent.optimizer.load_state_dict(optimizer_state)
            except Exception:
                # Optimizer parameter groups can differ across model revisions.
                pass

        agent.epsilon = float(data.get("epsilon", agent.epsilon))
        agent.step_count = int(data.get("step_count", agent.step_count))

        # AMP restore is best-effort and remains backward-compatible with old checkpoints.
        if hasattr(agent, "amp_enabled") and "amp_enabled" in data:
            ckpt_amp_enabled = bool(data.get("amp_enabled", False))
            agent.amp_enabled = bool(ckpt_amp_enabled and getattr(agent.device, "type", "cpu") == "cuda")

        scaler_state = data.get("grad_scaler_state_dict")
        if scaler_state and hasattr(agent, "grad_scaler") and agent.grad_scaler is not None:
            try:
                agent.grad_scaler.load_state_dict(scaler_state)
            except Exception:
                pass

        if curriculum is not None and "curriculum_state" in data:
            _restore_curriculum(curriculum, data["curriculum_state"])

        load_mode = "full" if policy_stats["loaded_keys"] == policy_stats["total_keys"] else "partial"

        return {
            "loaded": True,
            "episode": int(data.get("episode", 0)),
            "is_full_checkpoint": True,
            "load_mode": load_mode,
            "loaded_keys": int(policy_stats["loaded_keys"]),
            "total_keys": int(policy_stats["total_keys"]),
        }

    # Legacy fallback: weights-only .pth files
    if isinstance(data, dict):
        policy_stats = _load_state_dict_compatible(agent.policy_net, data)
        if policy_stats["loaded_keys"] <= 0:
            return {
                "loaded": False,
                "reason": "incompatible_legacy_weights",
                "episode": 0,
                "is_full_checkpoint": False,
                "loaded_keys": 0,
                "total_keys": int(policy_stats["total_keys"]),
            }

        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        load_mode = "full" if policy_stats["loaded_keys"] == policy_stats["total_keys"] else "partial"
        return {
            "loaded": True,
            "episode": 0,
            "is_full_checkpoint": False,
            "load_mode": load_mode,
            "loaded_keys": int(policy_stats["loaded_keys"]),
            "total_keys": int(policy_stats["total_keys"]),
        }

    return {"loaded": False, "reason": "unsupported_format"}

