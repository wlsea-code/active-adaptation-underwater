"""Interactively validate the explicit BlueROV controller in IsaacLab."""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
import time
import traceback
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


DEFAULTS = [{"task": "UW/BlueROVExplicit"}, "_self_"]


@dataclass
class IsaacAppConfig:
    headless: bool = "${..headless}"
    enable_cameras: bool = "${..record_video}"


@dataclass
class ValidationTaskOverride:
    num_envs: int = 1
    max_episode_length: int = 2_000_000_000
    record_video: bool = "${..record_video}"


@dataclass
class ControllerConfig:
    position_kp: tuple[float, float, float] = (0.7, 0.7, 0.8)
    position_kd: tuple[float, float, float] = (2.0, 2.0, 2.2)
    attitude_kp: tuple[float, float, float] = (8.0, 1.0, 12.0)
    attitude_kd: tuple[float, float, float] = (3.0, 1.8, 3.0)
    max_linear_acceleration: tuple[float, float, float] = (0.20, 0.20, 0.20)
    max_angular_acceleration: tuple[float, float, float] = (1.5, 0.4, 2.0)
    wrench_weights: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        5.0,
        0.5,
        5.0,
    )
    control_pitch: bool = False
    mass: float = 0.0
    # Box-model estimate for the 11.2 kg, approximately 0.46 x 0.34 x 0.25 m hull.
    # An empty list preserves the inertia inferred by PhysX from the USD.
    inertia_diag: List[float] = field(default_factory=lambda: [0.16, 0.25, 0.30])
    use_added_mass: bool = True
    # Instantaneous hydrodynamic feed-forward contains a differentiated
    # added-mass term and can amplify simulation noise. Static restoring-force
    # compensation remains active regardless of this diagnostic switch.
    compensate_hydrodynamics: bool = False


@dataclass
class TargetConfig:
    mode: str = "interactive"
    prim_path: str = "/World/BlueROVControlTarget"
    initial_offset: tuple[float, float, float] = (1.0, 0.0, 0.0)
    trajectory_duration: float = 4.0
    position_change_threshold: float = 0.002
    orientation_change_threshold_deg: float = 0.5
    max_duration: float = 0.0
    scripted_segment_duration: float = 6.0
    scripted_heave_offset: float = 0.5
    scripted_yaw_deg: float = 45.0


@dataclass
class FlowConfig:
    enabled: bool = True
    max_velocity: tuple[float, float, float, float, float, float] = (
        0.15,
        0.08,
        0.03,
        0.0,
        0.0,
        0.0,
    )
    noise_std: tuple[float, float, float, float, float, float] = (
        0.02,
        0.02,
        0.01,
        0.0,
        0.0,
        0.0,
    )


@dataclass
class StabilityConfig:
    center_of_buoyancy_offset: float = 0.06
    roll_linear_damping: float = 1.8
    pitch_linear_damping: float = 2.0


@dataclass
class ValidationConfig:
    defaults: List[Any] = field(default_factory=lambda: DEFAULTS)
    hydra: HydraConf = field(default_factory=HydraConf)
    headless: bool = False
    backend: str = "isaac"
    device: str = "cuda"
    app: IsaacAppConfig = field(default_factory=IsaacAppConfig)
    seed: int = 42
    record_video: bool = False
    realtime: bool = True
    print_interval: int = 100
    output_dir: str = ""
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    task: ValidationTaskOverride = field(default_factory=ValidationTaskOverride)


cs = ConfigStore.instance()
cs.store(
    name="validate_bluerov_controller",
    node=ValidationConfig(
        hydra=HydraConf(
            run=RunDir(
                dir="./outputs_validation/${now:%Y-%m-%d}/${now:%H-%M-%S}-BlueROV"
            )
        )
    ),
)


class InteractiveTargetPrim:
    """A selectable USD target whose world pose is read every control step."""

    def __init__(
        self,
        stage,
        path: str,
        position_w: torch.Tensor,
        orientation_wb: torch.Tensor,
        device: torch.device,
    ):
        from pxr import Gf, UsdGeom

        self.device = device
        self.path = path
        target = UsdGeom.Xform.Define(stage, path)
        target.ClearXformOpOrder()
        target.AddTranslateOp().Set(
            Gf.Vec3d(*position_w[0].detach().cpu().double().tolist())
        )
        quat = orientation_wb[0].detach().cpu().double().tolist()
        target.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:]))
        )
        sphere = UsdGeom.Sphere.Define(stage, f"{path}/TargetSphere")
        sphere.GetRadiusAttr().Set(0.10)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 1.0, 0.2)])
        self.prim = target.GetPrim()

        import omni.usd

        omni.usd.get_context().get_selection().set_selected_prim_paths([path], True)

    def get_world_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        from pxr import Usd, UsdGeom

        transform = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        transform.Orthonormalize()
        translation = transform.ExtractTranslation()
        rotation = transform.ExtractRotationQuat()
        position = torch.tensor(
            [[*translation]], dtype=torch.float32, device=self.device
        )
        orientation = torch.tensor(
            [[rotation.real, *rotation.imaginary]],
            dtype=torch.float32,
            device=self.device,
        )
        return position, orientation


def _apply_simulation_inertia(robot, inertia_diag: List[float]) -> None:
    if not inertia_diag:
        return
    if len(inertia_diag) != 3 or any(value <= 0.0 for value in inertia_diag):
        raise ValueError("controller.inertia_diag must contain three positive values")
    base_id = robot.body_names.index("base_link")
    inertias = robot.root_physx_view.get_inertias().clone()
    diagonal = torch.tensor(
        inertia_diag, device=inertias.device, dtype=inertias.dtype
    )
    inertias[:, base_id] = torch.diag(diagonal).reshape(1, 9)
    indices = torch.arange(robot.num_instances, device=inertias.device)
    robot.root_physx_view.set_inertias(inertias, indices)


def _quaternion_distance(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(q0 * q1, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _scripted_target(
    phase: int,
    initial_position_w: torch.Tensor,
    initial_orientation_wb: torch.Tensor,
    cfg: TargetConfig,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    from active_adaptation.utils.math import quat_from_euler_xyz, quat_mul

    position = initial_position_w.clone()
    rpy_offset = torch.zeros_like(position)
    if phase == 0:
        name = "hover"
    elif phase == 1:
        name = "surge"
        position[:, 0] += cfg.initial_offset[0]
    else:
        name = "heave_yaw"
        position[:, 0] += cfg.initial_offset[0]
        position[:, 2] += cfg.scripted_heave_offset
        rpy_offset[:, 2] = math.radians(cfg.scripted_yaw_deg)
    orientation = quat_mul(initial_orientation_wb, quat_from_euler_xyz(rpy_offset))
    return name, position, orientation


def _summarize(rows: list[dict[str, float | str]]) -> dict[str, float]:
    if not rows:
        return {}
    tail = rows[max(0, len(rows) * 3 // 4) :]

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in tail) / len(tail)

    return {
        "tail_target_position_error_mean_m": mean("target_position_error_m"),
        "tail_reference_position_error_mean_m": mean("reference_position_error_m"),
        "tail_roll_error_mean_deg": mean("roll_error_deg"),
        "tail_pitch_error_mean_deg": mean("pitch_error_deg"),
        "tail_yaw_error_mean_deg": mean("yaw_error_deg"),
        "tail_actuation_residual_mean": mean("actuation_residual_norm"),
        "peak_pitch_error_deg": max(float(row["pitch_error_deg"]) for row in rows),
        "peak_angular_speed_rad_s": max(
            float(row["angular_speed_rad_s"]) for row in rows
        ),
        "peak_abs_throttle": max(float(row["max_abs_throttle"]) for row in rows),
    }


@hydra.main(
    config_path="../cfg", config_name="validate_bluerov_controller", version_base=None
)
def main(cfg: ValidationConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    if cfg.backend != "isaac":
        raise ValueError("BlueROV validation currently requires backend=isaac")
    if cfg.target.mode not in {"interactive", "scripted"}:
        raise ValueError("target.mode must be 'interactive' or 'scripted'")
    if cfg.headless and cfg.target.mode == "interactive":
        raise ValueError("headless mode requires target.mode=scripted")
    if not 2.0 <= cfg.target.trajectory_duration <= 4.0:
        raise ValueError("target.trajectory_duration must be between 2 and 4 seconds")
    if cfg.controller.inertia_diag and (
        len(cfg.controller.inertia_diag) != 3
        or any(value <= 0.0 for value in cfg.controller.inertia_diag)
    ):
        raise ValueError("controller.inertia_diag must contain three positive values")

    cfg.task.num_envs = 1
    cfg.task.record_video = cfg.record_video
    attitude_kp = list(cfg.controller.attitude_kp)
    attitude_kd = list(cfg.controller.attitude_kd)
    angular_limits = list(cfg.controller.max_angular_acceleration)
    wrench_command_mask = [1.0] * 6
    if not cfg.controller.control_pitch:
        attitude_kp[1] = 0.0
        attitude_kd[1] = 0.0
        angular_limits[1] = 0.0
        wrench_command_mask[4] = 0.0
    cfg.task.input.action.pose_controller = {
        "position_kp": list(cfg.controller.position_kp),
        "position_kd": list(cfg.controller.position_kd),
        "attitude_kp": attitude_kp,
        "attitude_kd": attitude_kd,
        "max_linear_acceleration": list(
            cfg.controller.max_linear_acceleration
        ),
        "max_angular_acceleration": angular_limits,
    }
    cfg.task.input.action.wrench_weights = list(cfg.controller.wrench_weights)
    cfg.task.input.action.wrench_command_mask = wrench_command_mask
    cfg.task.input.action.use_added_mass = cfg.controller.use_added_mass
    cfg.task.input.action.compensate_hydrodynamics = (
        cfg.controller.compensate_hydrodynamics
    )
    cfg.task.input.action.mass = (
        cfg.controller.mass if cfg.controller.mass > 0.0 else None
    )
    cfg.task.input.action.inertia_b = (
        [
            [cfg.controller.inertia_diag[0], 0.0, 0.0],
            [0.0, cfg.controller.inertia_diag[1], 0.0],
            [0.0, 0.0, cfg.controller.inertia_diag[2]],
        ]
        if cfg.controller.inertia_diag
        else None
    )
    cfg.task.command.position_offset_b = list(cfg.target.initial_offset)
    cfg.task.command.trajectory_duration = cfg.target.trajectory_duration
    # Validation owns its target lifecycle. Do not let the training task reset
    # the articulation when a waypoint happens to cross an episode boundary.
    cfg.task.max_episode_length = 2_000_000_000

    env = None
    try:
        aa.init(cfg, auto_rank=False)

        from tensordict import TensorDictBase

        from active_adaptation.envs import _EnvBase
        from active_adaptation.utils.math import (
            axis_angle_from_quat,
            quat_conjugate,
            quat_mul,
            quat_rotate,
        )

        import active_adaptation.envs.backends.isaac  # noqa: F401

        env_cls = _EnvBase.registry[cfg.task.get("env_class", "IsaacBackendEnv")]
        env = env_cls(cfg.task, cfg.device, headless=cfg.headless)
        env.set_seed(cfg.seed)
        env.eval()
        td: TensorDictBase = env.reset()
        robot = env.robot
        wrapper = env.robot_wrapper
        uw = wrapper.data
        command = env.command_manager
        action = env.action_manager

        _apply_simulation_inertia(robot, cfg.controller.inertia_diag)
        if cfg.stability.center_of_buoyancy_offset <= 0.0:
            raise ValueError("stability.center_of_buoyancy_offset must be positive")
        if cfg.stability.roll_linear_damping < 0.0 or cfg.stability.pitch_linear_damping < 0.0:
            raise ValueError("stability angular damping values must be non-negative")
        uw.coBM.fill_(cfg.stability.center_of_buoyancy_offset)
        uw.linear_damping_matrix[:, 3, 3] = cfg.stability.roll_linear_damping
        uw.linear_damping_matrix[:, 4, 4] = cfg.stability.pitch_linear_damping

        env_ids = torch.arange(env.num_envs, device=env.device)
        if cfg.flow.enabled:
            wrapper.set_flow_velocities(
                env_ids, cfg.flow.max_velocity, cfg.flow.noise_std
            )
        else:
            wrapper.set_flow_velocities(env_ids, (0.0,) * 6, (0.0,) * 6)
        wrapper.reset(env_ids)
        wrapper.write_data_to_sim()

        controller = action.controller
        mass = action.controller_mass
        inertia = controller.wrench_controller.inertia_b
        added_mass = (
            controller.wrench_controller.added_mass
            if cfg.controller.use_added_mass
            else None
        )
        thruster_positions_b = controller.allocator.positions_b
        thruster_directions_b = controller.allocator.directions_b

        initial_position_w = robot.data.root_link_pos_w.clone()
        initial_orientation_wb = robot.data.root_link_quat_w.clone()
        raw_target_position_w = command.goal_position_w.clone()
        raw_target_orientation_wb = command.goal_orientation_wb.clone()
        target_prim = None
        if cfg.target.mode == "interactive":
            target_prim = InteractiveTargetPrim(
                env.sim.get_initial_stage(),
                cfg.target.prim_path,
                raw_target_position_w,
                raw_target_orientation_wb,
                env.device,
            )

        reference_marker = None
        if env.sim.has_gui():
            reference_marker = env.scene.create_sphere_marker(
                "/Visuals/BlueROVTrajectoryReference",
                (1.0, 0.4, 0.05),
                radius=0.055,
            )

            def draw_controller_debug() -> None:
                position = robot.data.root_link_pos_w
                reference_marker.visualize(command.target_position_w)
                env.debug_draw.vector(
                    position,
                    raw_target_position_w - position,
                    size=3.0,
                    color=(0.1, 1.0, 0.2, 1.0),
                )
                target_x_w = quat_rotate(
                    raw_target_orientation_wb,
                    torch.tensor([[0.35, 0.0, 0.0]], device=env.device),
                )
                env.debug_draw.vector(
                    raw_target_position_w,
                    target_x_w,
                    size=3.0,
                    color=(1.0, 0.8, 0.1, 1.0),
                )
                if action.latest_output is not None:
                    force_w = quat_rotate(
                        robot.data.root_link_quat_w,
                        action.latest_output.realized_wrench_b[:, :3] * 0.02,
                    )
                    env.debug_draw.vector(
                        position,
                        force_w,
                        size=2.0,
                        color=(0.1, 0.5, 1.0, 1.0),
                    )

            env._debug_draw_callbacks.append(draw_controller_debug)

        output_dir = (
            Path(cfg.output_dir)
            if cfg.output_dir
            else Path(HydraConfig.get().runtime.output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "bluerov_explicit_controller.mp4"
        rows: list[dict[str, float | str]] = []
        wall_start = time.perf_counter()
        import omni.kit.app

        app = omni.kit.app.get_app()
        if cfg.target.mode == "scripted":
            max_steps = int(
                3.0 * cfg.target.scripted_segment_duration / env.step_dt
            )
        elif cfg.target.max_duration > 0.0:
            max_steps = int(cfg.target.max_duration / env.step_dt)
        else:
            max_steps = None

        print("BlueROV PoseReferenceCommand -> BlueROVPoseTrackingAction validation")
        print(f"  target mode: {cfg.target.mode}")
        print(f"  minimum-jerk duration: {cfg.target.trajectory_duration:.2f} s")
        print(f"  pitch control: {cfg.controller.control_pitch}")
        print(f"  flow enabled: {cfg.flow.enabled}, sampled flow: {uw.flow_vels[0].tolist()}")
        print(f"  mass: {mass:.6f} kg, allocator rank: {controller.allocator.rank}")
        print(f"  inertia diagonal: {torch.diag(inertia).detach().cpu().tolist()}")
        print(
            "  stability: coBM="
            f"{cfg.stability.center_of_buoyancy_offset:.3f} m, "
            f"roll/pitch damping=({cfg.stability.roll_linear_damping:.2f}, "
            f"{cfg.stability.pitch_linear_damping:.2f})"
        )
        print("  controller path: command.set_goal() -> Action -> throttle_cmd")
        if target_prim is not None:
            print(f"  drag target prim in Isaac Sim: {cfg.target.prim_path}")

        try:
            with env.get_recorder(
                video_path, enabled=cfg.record_video
            ) as recorder, torch.inference_mode():
                for step in itertools.count():
                    if max_steps is not None and step >= max_steps:
                        break
                    if max_steps is None and app is not None and not app.is_running():
                        break

                    if target_prim is not None:
                        phase_name = "interactive"
                        new_position, new_orientation = target_prim.get_world_pose()
                    else:
                        phase = min(
                            int(step * env.step_dt / cfg.target.scripted_segment_duration),
                            2,
                        )
                        phase_name, new_position, new_orientation = _scripted_target(
                            phase,
                            initial_position_w,
                            initial_orientation_wb,
                            cfg.target,
                        )

                    position_changed = torch.linalg.vector_norm(
                        new_position - command.goal_position_w, dim=-1
                    ).max() > cfg.target.position_change_threshold
                    orientation_changed = _quaternion_distance(
                        new_orientation, command.goal_orientation_wb
                    ).max() > math.radians(cfg.target.orientation_change_threshold_deg)
                    if bool(position_changed or orientation_changed):
                        command.set_goal(new_position, new_orientation)
                    raw_target_position_w = new_position
                    raw_target_orientation_wb = new_orientation
                    td["action"] = torch.zeros(
                        env.num_envs,
                        action.action_dim,
                        device=env.device,
                    )
                    transition = env.step(td)
                    td = transition["next"]
                    if cfg.record_video:
                        recorder.add_frame()

                    latest_output = action.latest_output
                    if latest_output is None:
                        continue
                    attitude_error_q = quat_mul(
                        quat_conjugate(robot.data.root_link_quat_w),
                        raw_target_orientation_wb,
                    )
                    attitude_error_vector = axis_angle_from_quat(attitude_error_q)
                    allocation_matrix = controller.allocator.allocation_matrix.to(
                        device=env.device, dtype=uw.thrusts_b.dtype
                    )
                    applied_thruster_wrench_b = torch.matmul(
                        uw.thrusts_b[..., 0], allocation_matrix.T
                    )
                    row: dict[str, float | str] = {
                        "time_s": step * env.step_dt,
                        "phase": phase_name,
                        "target_position_error_m": float(
                            torch.linalg.vector_norm(
                                raw_target_position_w - robot.data.root_link_pos_w,
                                dim=-1,
                            )[0].item()
                        ),
                        "reference_position_error_m": float(
                            torch.linalg.vector_norm(
                                command.target_position_w
                                - robot.data.root_link_pos_w,
                                dim=-1,
                            )[0].item()
                        ),
                        "roll_error_deg": math.degrees(
                            abs(float(attitude_error_vector[0, 0].item()))
                        ),
                        "pitch_error_deg": math.degrees(
                            abs(float(attitude_error_vector[0, 1].item()))
                        ),
                        "yaw_error_deg": math.degrees(
                            abs(float(attitude_error_vector[0, 2].item()))
                        ),
                        "signed_roll_error_deg": math.degrees(
                            float(attitude_error_vector[0, 0].item())
                        ),
                        "signed_pitch_error_deg": math.degrees(
                            float(attitude_error_vector[0, 1].item())
                        ),
                        "signed_yaw_error_deg": math.degrees(
                            float(attitude_error_vector[0, 2].item())
                        ),
                        "actuation_residual_norm": float(
                            torch.linalg.vector_norm(
                                latest_output.actuation_residual_wrench_b[0]
                            ).item()
                        ),
                        "max_abs_throttle": float(
                            latest_output.throttle[0].abs().max().item()
                        ),
                        "angular_speed_rad_s": float(
                            torch.linalg.vector_norm(
                                robot.data.root_link_ang_vel_b[0]
                            ).item()
                        ),
                        "angular_velocity_z_rad_s": float(
                            robot.data.root_link_ang_vel_b[0, 2].item()
                        ),
                        "desired_mz": float(latest_output.desired_wrench_b[0, 5].item()),
                        "realized_mz": float(latest_output.realized_wrench_b[0, 5].item()),
                        "applied_thruster_mz": float(
                            applied_thruster_wrench_b[0, 5].item()
                        ),
                        "hydro_mz": float(uw.hydro_torques_b[0, 2].item()),
                        "flow_x": float(uw.flow_vels[0, 0].item()),
                        "flow_y": float(uw.flow_vels[0, 1].item()),
                        "flow_z": float(uw.flow_vels[0, 2].item()),
                    }
                    for rotor_id in range(wrapper.num_rotors):
                        row[f"throttle_{rotor_id}"] = float(
                            uw.throttle[0, rotor_id].item()
                        )
                    rows.append(row)
                    if step % cfg.print_interval == 0:
                        print(
                            f"  t={row['time_s']:7.2f}s target_err="
                            f"{row['target_position_error_m']:.3f}m ref_err="
                            f"{row['reference_position_error_m']:.3f}m pitch="
                            f"{row['pitch_error_deg']:.2f}deg throttle="
                            f"{row['max_abs_throttle']:.3f}"
                        )
                    if cfg.realtime and not cfg.headless:
                        deadline = wall_start + (step + 1) * env.step_dt
                        time.sleep(max(0.0, deadline - time.perf_counter()))
        except KeyboardInterrupt:
            print("Interrupted by user.")

        if rows:
            csv_path = output_dir / "metrics.csv"
            with csv_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = {
                "controller": OmegaConf.to_container(cfg.controller, resolve=True),
                "target": OmegaConf.to_container(cfg.target, resolve=True),
                "flow": OmegaConf.to_container(cfg.flow, resolve=True),
                "stability": OmegaConf.to_container(cfg.stability, resolve=True),
                "sampled_flow": uw.flow_vels[0].detach().cpu().tolist(),
                "sim_parameters": {
                    "mass": mass,
                    "inertia_b": inertia.detach().cpu().tolist(),
                    "added_mass": added_mass.detach().cpu().tolist()
                    if added_mass is not None
                    else None,
                    "allocator_rank": controller.allocator.rank,
                    "runtime_thruster_positions_b": thruster_positions_b.detach()
                    .cpu()
                    .tolist(),
                    "runtime_thruster_directions_b": thruster_directions_b.detach()
                    .cpu()
                    .tolist(),
                    "shared_action_wrapper_thruster_model": (
                        controller.thruster_model is wrapper.thruster_model
                    ),
                    "command_action_path": True,
                    "step_dt": env.step_dt,
                },
                "summary": _summarize(rows),
            }
            report_path = output_dir / "report.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"Saved metrics to {csv_path}")
            print(f"Saved report to {report_path}")
            if cfg.record_video:
                print(f"Saved video to {video_path}")
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close(raise_if_closed=False)


if __name__ == "__main__":
    main()
