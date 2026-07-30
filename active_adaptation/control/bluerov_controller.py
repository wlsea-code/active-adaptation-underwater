"""Integrated six-DoF pose controller for underwater robots.

All world-frame quaternions use ``(w, x, y, z)``. Body twists and wrenches use
``[x, y, z]`` and ``[Fx, Fy, Fz, Mx, My, Mz]`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)
from active_adaptation.control.thruster import (
    AllocationResult,
    BlueROVThrusterModel,
    ThrusterAllocator,
)


def _tensor(value, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _clamp_components(value: torch.Tensor, limits: Sequence[float]) -> torch.Tensor:
    limit = _tensor(limits, like=value)
    return torch.maximum(torch.minimum(value, limit), -limit)


@dataclass(frozen=True)
class PoseControllerCfg:
    """PD gains and acceleration limits, ordered along body/world XYZ axes."""

    position_kp: tuple[float, float, float] = (1.5, 1.5, 1.5)
    position_kd: tuple[float, float, float] = (2.0, 2.0, 2.0)
    attitude_kp: tuple[float, float, float] = (2.0, 2.0, 2.0)
    attitude_kd: tuple[float, float, float] = (1.5, 1.5, 1.5)
    max_linear_acceleration: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_angular_acceleration: tuple[float, float, float] = (1.0, 1.0, 1.0)



class PoseAccelerationController:
    """World-position/body-attitude PD controller producing desired accelerations."""

    def __init__(self, cfg: PoseControllerCfg = PoseControllerCfg()):
        self.cfg = cfg

    def compute(
        self,
        position_w: torch.Tensor,
        orientation_wb: torch.Tensor,
        linear_velocity_w: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        target_position_w: torch.Tensor,
        target_orientation_wb: torch.Tensor,
        target_linear_velocity_w: torch.Tensor | None = None,
        target_angular_velocity_b: torch.Tensor | None = None,
        feedforward_linear_acceleration_w: torch.Tensor | None = None,
        feedforward_angular_acceleration_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical linear acceleration and angular acceleration in body frame."""
        zeros3 = torch.zeros_like(position_w)
        target_linear_velocity_w = (
            zeros3 if target_linear_velocity_w is None else target_linear_velocity_w
        )
        target_angular_velocity_b = (
            zeros3 if target_angular_velocity_b is None else target_angular_velocity_b
        )
        feedforward_linear_acceleration_w = (
            zeros3
            if feedforward_linear_acceleration_w is None
            else feedforward_linear_acceleration_w
        )
        feedforward_angular_acceleration_b = (
            zeros3
            if feedforward_angular_acceleration_b is None
            else feedforward_angular_acceleration_b
        )

        position_error_w = target_position_w - position_w
        velocity_error_w = target_linear_velocity_w - linear_velocity_w
        linear_acceleration_w = (
            _tensor(self.cfg.position_kp, like=position_w) * position_error_w
            + _tensor(self.cfg.position_kd, like=position_w) * velocity_error_w
            + feedforward_linear_acceleration_w
        )
        linear_acceleration_w = _clamp_components(
            linear_acceleration_w, self.cfg.max_linear_acceleration
        )
        linear_acceleration_b = quat_rotate_inverse(
            orientation_wb, linear_acceleration_w
        )

        # q_current^-1 * q_target expresses the shortest desired rotation in body axes.
        attitude_error_quat_b = quat_mul(
            quat_conjugate(orientation_wb), target_orientation_wb
        )
        attitude_error_b = axis_angle_from_quat(attitude_error_quat_b)
        angular_velocity_error_b = target_angular_velocity_b - angular_velocity_b
        angular_acceleration_b = (
            _tensor(self.cfg.attitude_kp, like=position_w) * attitude_error_b
            + _tensor(self.cfg.attitude_kd, like=position_w)
            * angular_velocity_error_b
            + feedforward_angular_acceleration_b
        )
        angular_acceleration_b = _clamp_components(
            angular_acceleration_b, self.cfg.max_angular_acceleration
        )
        return linear_acceleration_b, angular_acceleration_b


class RigidBodyWrenchController:
    """Convert desired accelerations into a body wrench.

    ``external_wrench_b`` is the modeled non-thruster wrench to compensate. When
    ``added_mass`` is configured, do not also include the measured added-mass
    wrench in this argument; doing both would compensate added mass twice.
    """

    def __init__(
        self,
        mass: float,
        inertia_b: Sequence[Sequence[float]] | Sequence[float],
        added_mass: Sequence[Sequence[float]] | Sequence[float] | None = None,
    ):
        if mass <= 0:
            raise ValueError("mass must be positive")
        self.mass = float(mass)
        inertia = torch.as_tensor(inertia_b, dtype=torch.float64)
        self.inertia_b = torch.diag(inertia) if inertia.ndim == 1 else inertia
        if self.inertia_b.shape != (3, 3):
            raise ValueError("inertia_b must contain 3 diagonal values or a 3x3 matrix")
        if added_mass is None:
            self.added_mass = torch.zeros(6, 6, dtype=torch.float64)
        else:
            added = torch.as_tensor(added_mass, dtype=torch.float64)
            self.added_mass = torch.diag(added) if added.ndim == 1 else added
        if self.added_mass.shape != (6, 6):
            raise ValueError("added_mass must contain 6 diagonal values or a 6x6 matrix")

    def compute(
        self,
        linear_acceleration_b: torch.Tensor,
        angular_acceleration_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        external_wrench_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        acceleration = torch.cat(
            [linear_acceleration_b, angular_acceleration_b], dim=-1
        )
        inertia = self.inertia_b.to(
            dtype=acceleration.dtype, device=acceleration.device
        )
        added_mass = self.added_mass.to(
            dtype=acceleration.dtype, device=acceleration.device
        )
        rigid_body_mass = torch.zeros(
            6, 6, dtype=acceleration.dtype, device=acceleration.device
        )
        rigid_body_mass[:3, :3] = torch.eye(
            3, dtype=acceleration.dtype, device=acceleration.device
        ) * self.mass
        rigid_body_mass[3:, 3:] = inertia
        desired_wrench = torch.matmul(
            rigid_body_mass + added_mass, acceleration.unsqueeze(-1)
        ).squeeze(-1)

        # Euler rigid-body gyroscopic torque, omega x (I omega).
        angular_momentum = torch.matmul(
            inertia, angular_velocity_b.unsqueeze(-1)
        ).squeeze(-1)
        desired_wrench[..., 3:] += torch.cross(
            angular_velocity_b, angular_momentum, dim=-1
        )
        # Dynamics convention: M * acceleration = thruster + external.
        if external_wrench_b is not None:
            desired_wrench = desired_wrench - external_wrench_b
        return desired_wrench



@dataclass(frozen=True)
class ExplicitControlOutput:
    linear_acceleration_b: torch.Tensor
    angular_acceleration_b: torch.Tensor
    desired_wrench_b: torch.Tensor
    allocation: AllocationResult
    rpm: torch.Tensor
    throttle: torch.Tensor
    realized_thrust: torch.Tensor
    realized_wrench_b: torch.Tensor
    actuation_residual_wrench_b: torch.Tensor


class BlueROVExplicitController:
    """Compose pose control, rigid-body dynamics, allocation and RPM inversion."""

    def __init__(
        self,
        pose_controller: PoseAccelerationController,
        wrench_controller: RigidBodyWrenchController,
        allocator: ThrusterAllocator,
        thruster_model: BlueROVThrusterModel,
        wrench_command_mask: Sequence[float] | None = None,
    ):
        self.pose_controller = pose_controller
        self.wrench_controller = wrench_controller
        self.allocator = allocator
        self.thruster_model = thruster_model
        self.wrench_command_mask = wrench_command_mask

    def compute(self, **pose_inputs) -> ExplicitControlOutput:
        external_wrench_b = pose_inputs.pop("external_wrench_b", None)
        linear_acceleration_b, angular_acceleration_b = self.pose_controller.compute(
            **pose_inputs
        )
        desired_wrench_b = self.wrench_controller.compute(
            linear_acceleration_b,
            angular_acceleration_b,
            pose_inputs["angular_velocity_b"],
            external_wrench_b,
        )
        if self.wrench_command_mask is not None:
            desired_wrench_b = desired_wrench_b * _tensor(
                self.wrench_command_mask, like=desired_wrench_b
            )
        allocation = self.allocator.allocate(desired_wrench_b)
        rpm = self.thruster_model.thrust_to_rpm(allocation.thrust)
        throttle = self.thruster_model.rpm_to_throttle(rpm)
        realized_thrust = self.thruster_model.rpm_to_thrust(rpm)
        allocation_matrix = self.allocator.allocation_matrix.to(
            dtype=desired_wrench_b.dtype, device=desired_wrench_b.device
        )
        realized_wrench_b = torch.matmul(realized_thrust, allocation_matrix.T)
        return ExplicitControlOutput(
            linear_acceleration_b=linear_acceleration_b,
            angular_acceleration_b=angular_acceleration_b,
            desired_wrench_b=desired_wrench_b,
            allocation=allocation,
            rpm=rpm,
            throttle=throttle,
            realized_thrust=realized_thrust,
            realized_wrench_b=realized_wrench_b,
            actuation_residual_wrench_b=desired_wrench_b - realized_wrench_b,
        )


__all__ = [
    "BlueROVExplicitController",
    "ExplicitControlOutput",
    "PoseAccelerationController",
    "PoseControllerCfg",
    "RigidBodyWrenchController",
]
