"""Headless closed-loop validation for BlueROVHeavy control tasks."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf
import torch

import active_adaptation as aa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "heavy-pose": "BlueROVHeavyExplicit",
    "heavy-lemniscate": "BlueROVHeavyLemniscate",
    "arm-pose": "BlueROVHeavyArmExplicit",
    "arm-lemniscate": "BlueROVHeavyArmLemniscate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    return parser.parse_args()


def _make_config(task: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "cfg"), version_base=None):
        task_cfg = compose(config_name=f"task/UW/{TASKS[task]}").task
    task_cfg.max_episode_length = 2_000_000_000
    return OmegaConf.create(
        {
            "backend": "isaac",
            "device": "cuda",
            "headless": True,
            "app": {"headless": True, "enable_cameras": False},
            "task": task_cfg,
        }
    )


def _step(env, td):
    td["action"] = torch.zeros(
        env.num_envs,
        env.action_manager.action_dim,
        device=env.device,
    )
    return env.step(td)["next"]


def _arm_joint_ids(env) -> torch.Tensor | None:
    if "Arm" not in env.cfg.name:
        return None
    ids, names = env.robot.find_joints("arm_joint[1-8]")
    assert names == [f"arm_joint{index}" for index in range(1, 9)]
    return torch.as_tensor(ids, device=env.device, dtype=torch.long)


def _validate_common(env, max_arm_error: torch.Tensor) -> None:
    action = env.action_manager
    assert env.robot_wrapper.num_rotors == 8
    assert action.action_dim == 8
    assert action.controller.allocator.rank == 6
    assert action.latest_output is not None
    assert torch.isfinite(action.latest_output.throttle).all()
    assert torch.isfinite(env.robot.data.root_link_pos_w).all()
    assert float(max_arm_error) < 0.10


def _run_pose(env, td) -> dict[str, float]:
    command = env.command_manager
    arm_ids = _arm_joint_ids(env)
    max_arm_error = torch.zeros((), device=env.device)
    duration = command.trajectory.duration + 4.0
    for _ in range(int(duration / env.step_dt)):
        td = _step(env, td)
        if arm_ids is not None:
            max_arm_error = torch.maximum(
                max_arm_error, env.robot.data.joint_pos[:, arm_ids].abs().max()
            )

    _validate_common(env, max_arm_error)
    position_error = torch.linalg.vector_norm(
        command.goal_position_w - env.robot.data.root_link_pos_w,
        dim=-1,
    )
    attitude_error_deg = torch.rad2deg(
        torch.linalg.vector_norm(command.attitude_error_b, dim=-1)
    )
    assert position_error.max() < 0.25, float(position_error.max())
    assert attitude_error_deg.max() < 6.0, float(attitude_error_deg.max())
    return {
        "position_error_m": float(position_error.max()),
        "attitude_error_deg": float(attitude_error_deg.max()),
        "max_arm_joint_error": float(max_arm_error),
    }


def _run_lemniscate(env, td) -> dict[str, float]:
    command = env.command_manager
    arm_ids = _arm_joint_ids(env)
    max_arm_error = torch.zeros((), device=env.device)
    reference_positions = []
    actual_positions = []
    attitude_errors = []
    peak_throttles = []
    total_steps = int((command.entry_duration + command.period) / env.step_dt)
    for step in range(total_steps):
        td = _step(env, td)
        if arm_ids is not None:
            max_arm_error = torch.maximum(
                max_arm_error, env.robot.data.joint_pos[:, arm_ids].abs().max()
            )
        peak_throttles.append(
            env.action_manager.latest_output.throttle.abs().max().detach()
        )
        if (step + 1) * env.step_dt >= command.entry_duration:
            reference_positions.append(command.target_position_w.detach().clone())
            actual_positions.append(env.robot.data.root_link_pos_w.detach().clone())
            attitude_errors.append(command.attitude_error_b.detach().clone())

    _validate_common(env, max_arm_error)
    reference = torch.stack(reference_positions)
    actual = torch.stack(actual_positions)
    error = torch.linalg.vector_norm(reference - actual, dim=-1)
    rmse = torch.sqrt(error.square().mean())
    span_ratio = (actual.amax(dim=0) - actual.amin(dim=0)) / (
        reference.amax(dim=0) - reference.amin(dim=0)
    ).clamp_min(1.0e-6)
    attitude_error_deg = torch.rad2deg(
        torch.linalg.vector_norm(torch.stack(attitude_errors), dim=-1)
    )
    peak_throttle = torch.stack(peak_throttles)
    saturation_fraction = (peak_throttle >= 0.99).float().mean()

    assert torch.isfinite(error).all()
    assert rmse < 0.35
    assert torch.all(span_ratio > 0.70)
    assert torch.all(span_ratio < 1.30)
    assert attitude_error_deg.mean() < 12.0
    assert attitude_error_deg.max() < 30.0
    assert saturation_fraction < 0.15
    return {
        "position_rmse_m": float(rmse),
        "position_max_m": float(error.max()),
        "attitude_error_mean_deg": float(attitude_error_deg.mean()),
        "attitude_error_max_deg": float(attitude_error_deg.max()),
        "saturation_fraction": float(saturation_fraction),
        "peak_abs_throttle": float(peak_throttle.max()),
        "max_arm_joint_error": float(max_arm_error),
    }


def main() -> None:
    args = _parse_args()
    env = None
    try:
        cfg = _make_config(args.task)
        aa.init(cfg, auto_rank=False)
        import active_adaptation.envs.backends.isaac  # noqa: F401
        from active_adaptation.envs import _EnvBase

        env_cls = _EnvBase.registry["IsaacBackendEnv"]
        env = env_cls(cfg.task, cfg.device, headless=True)
        env.set_seed(42)
        env.eval()
        td = env.reset()
        metrics = (
            _run_lemniscate(env, td)
            if "lemniscate" in args.task
            else _run_pose(env, td)
        )
        print(f"{TASKS[args.task]} validation passed", flush=True)
        for name, value in metrics.items():
            print(f"  {name}={value:.6f}", flush=True)
    except BaseException:
        if env is not None:
            env.close(raise_if_closed=False)
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        if env is not None:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    main()
