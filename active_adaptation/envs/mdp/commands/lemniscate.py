"""Analytic Lemniscate 3D pose-reference command."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

from active_adaptation.control.trajectory import (
    Lemniscate3DReference,
    Lemniscate3DTrajectory,
)
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    euler_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)
from active_adaptation.utils.symmetry import SymmetryTransform

from .base import CommandV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class Lemniscate3DCommand(CommandV2):
    """Generate a smooth 3D lemniscate pose reference.

    Translation follows :class:`Lemniscate3DTrajectory`. By default, target
    yaw follows the horizontal path tangent while roll and pitch stay fixed.
    This matches the five-axis BlueROV allocation, which does not command pitch.
    """

    def __init__(
        self,
        semi_axis_x: float,
        semi_axis_y: float,
        vertical_amplitude: float,
        period: float,
        entry_duration: float = 4.0,
        initial_phase: float = 0.0,
        direction: float = 1.0,
        curve_yaw_offset: float = 0.0,
        rpy_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        heading_mode: str = "tangent_yaw",
    ) -> None:
        super().__init__()
        scalar_values = {
            "semi_axis_x": semi_axis_x,
            "semi_axis_y": semi_axis_y,
            "vertical_amplitude": vertical_amplitude,
            "period": period,
            "entry_duration": entry_duration,
            "initial_phase": initial_phase,
            "direction": direction,
            "curve_yaw_offset": curve_yaw_offset,
        }
        if not all(math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError("lemniscate command parameters must be finite")
        if semi_axis_x <= 0.0 or semi_axis_y <= 0.0:
            raise ValueError("semi axes must be positive")
        if vertical_amplitude < 0.0:
            raise ValueError("vertical_amplitude must be non-negative")
        if period <= 0.0:
            raise ValueError("period must be positive")
        if entry_duration < 0.0:
            raise ValueError("entry_duration must be non-negative")
        if direction not in (-1.0, 1.0):
            raise ValueError("direction must be either -1 or 1")
        if len(rpy_offset) != 3 or not all(
            math.isfinite(float(value)) for value in rpy_offset
        ):
            raise ValueError("rpy_offset must contain three finite values")
        if heading_mode not in ("fixed", "tangent_yaw"):
            raise ValueError("heading_mode must be 'fixed' or 'tangent_yaw'")

        self.semi_axis_x = float(semi_axis_x)
        self.semi_axis_y = float(semi_axis_y)
        self.vertical_amplitude = float(vertical_amplitude)
        self.period = float(period)
        self.entry_duration = float(entry_duration)
        self.initial_phase = float(initial_phase)
        self.direction = float(direction)
        self.curve_yaw_offset = float(curve_yaw_offset)
        self.rpy_offset = tuple(float(value) for value in rpy_offset)
        self.heading_mode = heading_mode

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        with torch.device(self.device):
            initial_position_w = self.asset.data.root_link_pos_w.clone()
            initial_orientation_wb = self.asset.data.root_link_quat_w.clone()
            curve_yaw_w = self._curve_yaw_for_initial_heading(
                initial_orientation_wb
            )
            self.trajectory = Lemniscate3DTrajectory(
                initial_position_w=initial_position_w,
                semi_axis_x=self.semi_axis_x,
                semi_axis_y=self.semi_axis_y,
                vertical_amplitude=self.vertical_amplitude,
                period=self.period,
                entry_duration=self.entry_duration,
                initial_phase=self.initial_phase,
                direction=self.direction,
                curve_yaw_w=curve_yaw_w,
            )

            self.target_position_w = initial_position_w.clone()
            self.target_orientation_wb = initial_orientation_wb.clone()
            self.target_linear_velocity_w = torch.zeros(self.num_envs, 3)
            self.target_angular_velocity_b = torch.zeros(self.num_envs, 3)
            self.feedforward_linear_acceleration_w = torch.zeros(self.num_envs, 3)
            self.feedforward_angular_acceleration_b = torch.zeros(self.num_envs, 3)
            self.phase = torch.zeros(self.num_envs)
            self.phase_rate = torch.zeros(self.num_envs)
            self.phase_acceleration = torch.zeros(self.num_envs)
            self._target_roll_pitch_w = euler_from_quat(
                initial_orientation_wb
            )[:, :2].clone()

            self.position_error_b = torch.zeros(self.num_envs, 3)
            self.attitude_error_b = torch.zeros(self.num_envs, 3)
            self.linear_velocity_error_b = torch.zeros(self.num_envs, 3)
            self.angular_velocity_error_b = torch.zeros(self.num_envs, 3)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.position_error_b,
                self.attitude_error_b,
                self.linear_velocity_error_b,
                self.angular_velocity_error_b,
            ],
            dim=-1,
        )

    @override
    def reset(
        self, env_ids: torch.Tensor, tensordict: TensorDictBase | None = None
    ) -> None:
        orientation_wb = self.asset.data.root_link_quat_w[env_ids]
        rpy_offset = torch.tensor(
            self.rpy_offset,
            dtype=orientation_wb.dtype,
            device=self.device,
        ).expand(len(env_ids), -1)
        curve_yaw_w = self._curve_yaw_for_initial_heading(orientation_wb)
        self._target_roll_pitch_w[env_ids] = (
            euler_from_quat(orientation_wb)[:, :2] + rpy_offset[:, :2]
        )
        self.trajectory.reset(
            self.asset.data.root_link_pos_w[env_ids],
            env_ids=env_ids,
            initial_phase=self.initial_phase,
            curve_yaw_w=curve_yaw_w,
        )
        self.target_orientation_wb[env_ids] = quat_mul(
            orientation_wb, quat_from_euler_xyz(rpy_offset)
        )
        self._write_reference(self.trajectory.sample(), env_ids)
        self._sync_tracking(env_ids)

    @override
    def sync_state(self) -> None:
        self._sync_tracking(slice(None))

    @override
    def update(self) -> None:
        self._write_reference(self.trajectory.advance(self.env.step_dt))
        self._sync_tracking(slice(None))

    def _write_reference(
        self,
        reference: Lemniscate3DReference,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        self.target_position_w[env_ids] = reference.position_w[env_ids]
        self.target_linear_velocity_w[env_ids] = reference.linear_velocity_w[env_ids]
        self.feedforward_linear_acceleration_w[env_ids] = (
            reference.linear_acceleration_w[env_ids]
        )
        if self.heading_mode == "tangent_yaw":
            heading_yaw_w, heading_yaw_rate = (
                self.trajectory.horizontal_tangent_heading(
                    reference.phase[env_ids],
                    reference.phase_rate[env_ids],
                    env_ids,
                )
            )
            target_rpy_w = torch.cat(
                [
                    self._target_roll_pitch_w[env_ids],
                    (heading_yaw_w + self.rpy_offset[2]).unsqueeze(-1),
                ],
                dim=-1,
            )
            self.target_orientation_wb[env_ids] = quat_from_euler_xyz(target_rpy_w)
            self.target_angular_velocity_b[env_ids] = 0.0
            self.target_angular_velocity_b[env_ids, 2] = heading_yaw_rate
        else:
            self.target_angular_velocity_b[env_ids] = 0.0
        self.feedforward_angular_acceleration_b[env_ids] = 0.0
        self.phase[env_ids] = reference.phase[env_ids]
        self.phase_rate[env_ids] = reference.phase_rate[env_ids]
        self.phase_acceleration[env_ids] = reference.phase_acceleration[env_ids]

    def _curve_yaw_for_initial_heading(
        self,
        orientation_wb: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate the curve so its initial travel tangent matches the nose."""
        phase = orientation_wb.new_full(
            orientation_wb.shape[:-1], self.initial_phase
        )
        tangent_x = self.direction * self.semi_axis_x * torch.cos(phase)
        tangent_y = (
            self.direction * 2.0 * self.semi_axis_y * torch.cos(2.0 * phase)
        )
        local_tangent_yaw = torch.atan2(tangent_y, tangent_x)
        initial_yaw_w = euler_from_quat(orientation_wb)[..., 2]
        return initial_yaw_w + self.curve_yaw_offset - local_tangent_yaw

    def _sync_tracking(self, env_ids: torch.Tensor | slice) -> None:
        orientation_wb = self.asset.data.root_link_quat_w[env_ids]
        position_error_w = (
            self.target_position_w[env_ids]
            - self.asset.data.root_link_pos_w[env_ids]
        )
        self.position_error_b[env_ids] = quat_rotate_inverse(
            orientation_wb, position_error_w
        )
        attitude_error_quat_b = quat_mul(
            quat_conjugate(orientation_wb),
            self.target_orientation_wb[env_ids],
        )
        self.attitude_error_b[env_ids] = axis_angle_from_quat(
            attitude_error_quat_b
        )
        linear_velocity_error_w = (
            self.target_linear_velocity_w[env_ids]
            - self.asset.data.root_link_lin_vel_w[env_ids]
        )
        self.linear_velocity_error_b[env_ids] = quat_rotate_inverse(
            orientation_wb, linear_velocity_error_w
        )
        self.angular_velocity_error_b[env_ids] = (
            self.target_angular_velocity_b[env_ids]
            - self.asset.data.root_link_ang_vel_b[env_ids]
        )

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        return SymmetryTransform(
            perm=torch.arange(12),
            signs=[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        )

    @override
    def get_state(self) -> TensorDict:
        return TensorDict(
            {
                "target_position_w": self.target_position_w,
                "target_orientation_wb": self.target_orientation_wb,
                "target_linear_velocity_w": self.target_linear_velocity_w,
                "target_angular_velocity_b": self.target_angular_velocity_b,
                "feedforward_linear_acceleration_w": (
                    self.feedforward_linear_acceleration_w
                ),
                "feedforward_angular_acceleration_b": (
                    self.feedforward_angular_acceleration_b
                ),
                "phase": self.phase,
                "phase_rate": self.phase_rate,
                "phase_acceleration": self.phase_acceleration,
            },
            [self.num_envs],
            device=self.device,
        )

    @override
    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        return tensordict


__all__ = ["Lemniscate3DCommand"]
