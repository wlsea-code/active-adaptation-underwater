"""Visualize BlueROV tracking a 3D lemniscate in the Isaac GUI."""

from __future__ import annotations

import itertools
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import hydra
import torch
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import active_adaptation as aa


DEFAULTS = [{"task": "UW/BlueROVLemniscate"}, "_self_"]


@dataclass
class IsaacAppConfig:
    headless: bool = False
    enable_cameras: bool = "${..record_video}"


@dataclass
class VisualizationTaskOverride:
    num_envs: int = 1
    max_episode_length: int = 2_000_000_000
    record_video: bool = "${..record_video}"


@dataclass
class VisualizationConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)
    headless: bool = False
    backend: str = "isaac"
    device: str = "cuda"
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    seed: int = 42
    realtime: bool = True
    max_duration: float = 0.0
    print_interval: int = 100
    curve_samples: int = 241
    trail_length: int = 1500
    record_video: bool = False
    output_dir: str = ""
    task: VisualizationTaskOverride = field(
        default_factory=VisualizationTaskOverride
    )


ConfigStore.instance().store(
    name="visualize_bluerov_lemniscate",
    node=VisualizationConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_validation/${now:%Y-%m-%d}/"
                "${now:%H-%M-%S}-BlueROVLemniscate-GUI"
            )
        )
    ),
)


def print_summary(
    reference_positions: list[torch.Tensor],
    actual_positions: list[torch.Tensor],
    heading_errors: list[torch.Tensor],
) -> None:
    if not reference_positions:
        return
    reference = torch.stack(reference_positions)
    actual = torch.stack(actual_positions)
    error = torch.linalg.vector_norm(reference - actual, dim=-1)
    reference_span = reference.amax(dim=0) - reference.amin(dim=0)
    actual_span = actual.amax(dim=0) - actual.amin(dim=0)
    yaw_error_deg = torch.rad2deg(torch.stack(heading_errors).abs())
    print(
        "Tracking summary after startup ramp\n"
        f"  position_rmse={float(torch.sqrt(error.square().mean())):.4f} m, "
        f"position_max={float(error.max()):.4f} m\n"
        f"  reference_span={reference_span.tolist()}\n"
        f"  actual_span={actual_span.tolist()}\n"
        f"  span_ratio={(actual_span / reference_span.clamp_min(1e-6)).tolist()}\n"
        f"  yaw_error_mean={float(yaw_error_deg.mean()):.3f} deg, "
        f"yaw_error_max={float(yaw_error_deg.max()):.3f} deg",
        flush=True,
    )


@hydra.main(
    config_path="../cfg",
    config_name="visualize_bluerov_lemniscate",
    version_base=None,
)
def main(cfg: VisualizationConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.headless or cfg.backend != "isaac":
        raise ValueError("visualization requires headless=false and backend=isaac")
    if cfg.max_duration < 0.0 or cfg.print_interval <= 0:
        raise ValueError(
            "max_duration must be non-negative and print_interval positive"
        )
    if cfg.curve_samples < 2 or cfg.trail_length < 2:
        raise ValueError("curve_samples and trail_length must be greater than one")
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = 2_000_000_000
    cfg.task.record_video = cfg.record_video

    env = None
    try:
        aa.init(cfg, auto_rank=False)

        import active_adaptation.envs.backends.isaac  # noqa: F401
        from active_adaptation.envs import _EnvBase
        from active_adaptation.utils.math import quat_rotate

        env_cls = _EnvBase.registry["IsaacBackendEnv"]
        env = env_cls(cfg.task, cfg.device, headless=False)
        env.set_seed(cfg.seed)
        env.eval()
        td = env.reset()

        robot = env.robot
        command = env.command_manager
        action = env.action_manager
        import omni.kit.app

        app = omni.kit.app.get_app()
        planned_curve_w = command.trajectory.curve_points(cfg.curve_samples)[0]
        actual_trail_w: deque[torch.Tensor] = deque(maxlen=cfg.trail_length)
        reference_history: list[torch.Tensor] = []
        actual_history: list[torch.Tensor] = []
        heading_errors: list[torch.Tensor] = []

        camera_target = command.trajectory.center_w[0].detach().cpu()
        camera_scale = max(
            command.semi_axis_x,
            command.semi_axis_y,
            command.vertical_amplitude,
            1.0,
        )
        eye = camera_target + torch.tensor(
            [5.0, -5.0, 4.0], dtype=camera_target.dtype
        ) * camera_scale
        env.sim.set_camera_view(
            eye=tuple(float(value) for value in eye),
            target=tuple(float(value) for value in camera_target),
        )

        reference_marker = env.scene.create_sphere_marker(
            "/Visuals/Lemniscate/reference",
            color=(1.0, 0.35, 0.05),
            radius=0.065,
        )
        center_marker = env.scene.create_sphere_marker(
            "/Visuals/Lemniscate/center",
            color=(0.85, 0.15, 0.85),
            radius=0.035,
        )

        def draw_tracking() -> None:
            position_w = robot.data.root_link_pos_w
            forward_b = torch.zeros_like(position_w)
            forward_b[:, 0] = 0.6
            forward_w = quat_rotate(robot.data.root_link_quat_w, forward_b)
            reference_marker.visualize(command.target_position_w)
            center_marker.visualize(command.trajectory.center_w)
            env.debug_draw.plot(
                planned_curve_w, size=2.5, color=(1.0, 0.72, 0.08, 1.0)
            )
            if len(actual_trail_w) > 1:
                env.debug_draw.plot(
                    torch.stack(tuple(actual_trail_w)),
                    size=3.0,
                    color=(0.05, 0.85, 1.0, 1.0),
                )
            env.debug_draw.vector(
                position_w,
                command.target_position_w - position_w,
                size=2.5,
                color=(0.2, 1.0, 0.25, 1.0),
            )
            env.debug_draw.vector(
                command.target_position_w,
                command.target_linear_velocity_w,
                size=2.0,
                color=(0.15, 0.35, 1.0, 1.0),
            )
            env.debug_draw.vector(
                position_w,
                forward_w,
                size=3.0,
                color=(1.0, 0.15, 0.65, 1.0),
            )

        env._debug_draw_callbacks = [
            callback
            for callback in env._debug_draw_callbacks
            if getattr(callback, "__self__", None) is not env.robot_wrapper
        ]
        env._debug_draw_callbacks.append(draw_tracking)

        output_dir = (
            Path(cfg.output_dir)
            if cfg.output_dir
            else Path(HydraConfig.get().runtime.output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "bluerov_lemniscate.mp4"
        max_steps = (
            None if cfg.max_duration == 0.0 else int(cfg.max_duration / env.step_dt)
        )
        wall_start = time.perf_counter()
        print(
            "BlueROV Lemniscate 3D visualization\n"
            "  planned curve: gold; reference: orange; actual trail: cyan\n"
            "  tracking error: green; reference velocity: blue; nose: magenta",
            flush=True,
        )

        with env.get_recorder(video_path, enabled=cfg.record_video) as recorder:
            for step in itertools.count():
                if max_steps is not None and step >= max_steps:
                    break
                if app is not None and not app.is_running():
                    break
                td["action"] = torch.zeros(
                    env.num_envs, action.action_dim, device=env.device
                )
                td = env.step(td)["next"]
                actual_trail_w.append(
                    robot.data.root_link_pos_w[0].detach().cpu().clone()
                )
                if (step + 1) * env.step_dt >= command.entry_duration:
                    reference_history.append(
                        command.target_position_w[0].detach().cpu().clone()
                    )
                    actual_history.append(
                        robot.data.root_link_pos_w[0].detach().cpu().clone()
                    )
                    heading_errors.append(
                        command.attitude_error_b[0, 2].detach().cpu().clone()
                    )
                if cfg.record_video:
                    recorder.add_frame()
                if step % cfg.print_interval == 0:
                    error = torch.linalg.vector_norm(
                        command.target_position_w - robot.data.root_link_pos_w,
                        dim=-1,
                    )[0]
                    throttle = (
                        0.0
                        if action.latest_output is None
                        else float(action.latest_output.throttle[0].abs().max())
                    )
                    print(
                        f"  t={step * env.step_dt:7.2f}s "
                        f"phase={float(command.phase[0]):7.3f} "
                        f"error={float(error):.3f}m throttle={throttle:.3f}",
                        flush=True,
                    )
                if cfg.realtime:
                    deadline = wall_start + (step + 1) * env.step_dt
                    time.sleep(max(0.0, deadline - time.perf_counter()))

        print_summary(reference_history, actual_history, heading_errors)
        if cfg.record_video:
            print(f"Saved video to {video_path}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    main()
