"""Pose-reference commands for underwater vehicle control."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

from active_adaptation.control.trajectory import (
    MinimumJerkPoseTrajectory,
    PoseReference,
)
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)
from active_adaptation.utils.symmetry import SymmetryTransform

from .base import CommandV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase


class PoseReferenceCommand(CommandV2):
    """Generate a smooth pose reference for an underwater tracking Action."""

    def __init__(
        self,
        position_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rpy_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        trajectory_duration: float = 4.0,
    ) -> None:
        super().__init__()
        if len(position_offset_b) != 3:
            raise ValueError("position_offset_b must contain three values")
        if len(rpy_offset) != 3:
            raise ValueError("rpy_offset must contain three values")
        self.position_offset_b = tuple(float(value) for value in position_offset_b)
        self.rpy_offset = tuple(float(value) for value in rpy_offset)
        self.trajectory_duration = float(trajectory_duration)
        if not all(math.isfinite(value) for value in self.position_offset_b):
            raise ValueError("position_offset_b must contain only finite values")
        if not all(math.isfinite(value) for value in self.rpy_offset):
            raise ValueError("rpy_offset must contain only finite values")
        if (
            not math.isfinite(self.trajectory_duration)
            or self.trajectory_duration <= 0.0
        ):
            raise ValueError("trajectory_duration must be positive and finite")

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        with torch.device(self.device):
            self.target_position_w = torch.zeros(self.num_envs, 3)
            self.target_orientation_wb = torch.zeros(self.num_envs, 4)
            self.target_orientation_wb[:, 0] = 1.0
            self.target_linear_velocity_w = torch.zeros(self.num_envs, 3)
            self.target_angular_velocity_b = torch.zeros(self.num_envs, 3)
            self.feedforward_linear_acceleration_w = torch.zeros(self.num_envs, 3)
            self.feedforward_angular_acceleration_b = torch.zeros(self.num_envs, 3)

            self.trajectory = MinimumJerkPoseTrajectory(
                self.asset.data.root_link_pos_w.clone(),
                self.asset.data.root_link_quat_w.clone(),
                duration=self.trajectory_duration,
            )
            self.goal_position_w = self.trajectory.goal_position_w
            self.goal_orientation_wb = self.trajectory.goal_orientation_wb

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
        position_offset_b = torch.tensor(
            self.position_offset_b,
            dtype=orientation_wb.dtype,
            device=self.device,
        ).expand(len(env_ids), -1)
        rpy_offset = torch.tensor(
            self.rpy_offset,
            dtype=orientation_wb.dtype,
            device=self.device,
        ).expand(len(env_ids), -1)
        goal_position_w = (
            self.asset.data.root_link_pos_w[env_ids]
            + quat_rotate(orientation_wb, position_offset_b)
        )
        goal_orientation_wb = quat_mul(
            orientation_wb,
            quat_from_euler_xyz(rpy_offset),
        )
        self.trajectory.reset(
            self.asset.data.root_link_pos_w[env_ids],
            orientation_wb,
            env_ids,
        )
        self.trajectory.retarget(
            goal_position_w,
            goal_orientation_wb,
            env_ids,
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

    def set_goal(
        self,
        position_w: torch.Tensor,
        orientation_wb: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        """Replan the smooth reference from its current state to a new goal."""
        self.trajectory.retarget(position_w, orientation_wb, env_ids)
        self._write_reference(self.trajectory.sample(), env_ids)
        self._sync_tracking(env_ids)

    def _write_reference(
        self,
        reference: PoseReference,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        self.target_position_w[env_ids] = reference.position_w[env_ids]
        self.target_orientation_wb[env_ids] = reference.orientation_wb[env_ids]
        self.target_linear_velocity_w[env_ids] = reference.linear_velocity_w[env_ids]
        self.target_angular_velocity_b[env_ids] = 0.0
        self.feedforward_linear_acceleration_w[env_ids] = (
            reference.linear_acceleration_w[env_ids]
        )
        self.feedforward_angular_acceleration_b[env_ids] = 0.0

    def _sync_tracking(self, env_ids: torch.Tensor | slice) -> None:
        orientation_wb = self.asset.data.root_link_quat_w[env_ids]
        position_error_w = (
            self.target_position_w[env_ids]
            - self.asset.data.root_link_pos_w[env_ids]
        )
        self.position_error_b[env_ids] = quat_rotate_inverse(
            orientation_wb,
            position_error_w,
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
            orientation_wb,
            linear_velocity_error_w,
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
            },
            [self.num_envs],
            device=self.device,
        )

    @override
    def relabel_command(self, tensordict: TensorDict) -> TensorDict:
        return tensordict


__all__ = ["PoseReferenceCommand"]
