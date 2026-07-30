"""Headless closed-loop validation for BlueROV Lemniscate 3D tracking."""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, List

import hydra
import torch
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import active_adaptation as aa
from active_adaptation.control.contracts import POSE_REFERENCE_FIELDS


DEFAULTS = [{"task": "UW/BlueROVLemniscate"}, "_self_"]


@dataclass
class IsaacAppConfig:
    headless: bool = True
    enable_cameras: bool = False


@dataclass
class ValidationTaskOverride:
    num_envs: int = 1
    max_episode_length: int = 2_000_000_000
    record_video: bool = False


@dataclass
class ValidationConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)
    headless: bool = True
    backend: str = "isaac"
    device: str = "cuda"
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    seed: int = 42
    task: ValidationTaskOverride = field(default_factory=ValidationTaskOverride)


ConfigStore.instance().store(
    name="validate_bluerov_lemniscate",
    node=ValidationConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_validation/${now:%Y-%m-%d}/"
                "${now:%H-%M-%S}-BlueROVLemniscate"
            )
        )
    ),
)


@hydra.main(
    config_path="../cfg",
    config_name="validate_bluerov_lemniscate",
    version_base=None,
)
def main(cfg: ValidationConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = 2_000_000_000
    cfg.task.record_video = False

    env = None
    try:
        aa.init(cfg, auto_rank=False)

        import active_adaptation.envs.backends.isaac  # noqa: F401
        from active_adaptation.envs import _EnvBase

        env_cls = _EnvBase.registry["IsaacBackendEnv"]
        env = env_cls(cfg.task, cfg.device, headless=True)
        env.set_seed(cfg.seed)
        env.eval()
        td = env.reset()

        command = env.command_manager
        action = env.action_manager
        reference_positions: list[torch.Tensor] = []
        actual_positions: list[torch.Tensor] = []
        yaw_errors: list[torch.Tensor] = []
        max_abs_throttles: list[torch.Tensor] = []
        finite_references = True
        total_steps = int((command.entry_duration + command.period) / env.step_dt)

        for step in range(total_steps):
            td["action"] = torch.zeros(
                env.num_envs,
                action.action_dim,
                device=env.device,
            )
            td = env.step(td)["next"]
            max_abs_throttles.append(
                action.latest_output.throttle.abs().max().detach().clone()
            )
            finite_references = finite_references and all(
                bool(torch.isfinite(getattr(command, field)).all())
                for field in POSE_REFERENCE_FIELDS
            )
            if (step + 1) * env.step_dt >= command.entry_duration:
                reference_positions.append(command.target_position_w.detach().clone())
                actual_positions.append(
                    env.robot.data.root_link_pos_w.detach().clone()
                )
                yaw_errors.append(command.attitude_error_b[:, 2].detach().clone())

        reference_history = torch.stack(reference_positions)
        actual_history = torch.stack(actual_positions)
        position_error = torch.linalg.vector_norm(
            reference_history - actual_history, dim=-1
        )
        position_rmse = torch.sqrt(position_error.square().mean())
        reference_span = reference_history.amax(dim=0) - reference_history.amin(dim=0)
        actual_span = actual_history.amax(dim=0) - actual_history.amin(dim=0)
        span_ratio = actual_span / reference_span.clamp_min(1.0e-6)
        yaw_error_deg = torch.rad2deg(torch.stack(yaw_errors).abs())
        throttle_history = torch.stack(max_abs_throttles)
        saturation_fraction = (throttle_history >= 0.99).float().mean()

        assert type(command).__name__ == "Lemniscate3DCommand"
        assert type(command).__module__ == (
            "active_adaptation.envs.mdp.commands.lemniscate"
        )
        from active_adaptation.envs.mdp.actions.underwater import (
            Lemniscate3DTrackingAction,
        )

        assert isinstance(action, Lemniscate3DTrackingAction)
        assert action.latest_output is not None
        expected_rank = 6 if env.robot_wrapper.num_rotors == 8 else 5
        assert action.controller.allocator.rank == expected_rank
        assert finite_references
        assert torch.isfinite(actual_history).all()
        assert torch.isfinite(action.latest_output.throttle).all()
        assert torch.any(throttle_history > 0.0)
        assert position_rmse < 0.25
        assert torch.all(span_ratio > 0.80)
        assert torch.all(span_ratio < 1.20)
        assert yaw_error_deg.mean() < 10.0
        assert yaw_error_deg.max() < 25.0
        assert saturation_fraction < 0.05

        print(
            "BlueROV Lemniscate validation passed\n"
            f"  command={type(command).__module__}.{type(command).__name__}\n"
            f"  action={type(action).__module__}.{type(action).__name__}\n"
            f"  position_rmse={float(position_rmse):.4f} m, "
            f"position_max={float(position_error.max()):.4f} m\n"
            f"  reference_span={reference_span[0].tolist()}\n"
            f"  actual_span={actual_span[0].tolist()}\n"
            f"  span_ratio={span_ratio[0].tolist()}\n"
            f"  yaw_error_mean={float(yaw_error_deg.mean()):.3f} deg, "
            f"yaw_error_max={float(yaw_error_deg.max()):.3f} deg\n"
            f"  peak_abs_throttle={float(throttle_history.max()):.6f}, "
            f"saturation_fraction={float(saturation_fraction):.6f}",
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    main()
