"""Smooth pose-reference trajectories for robot commands."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from active_adaptation.utils.math import slerp

@dataclass(frozen=True)
class PoseReference:
    position_w: torch.Tensor
    orientation_wb: torch.Tensor
    linear_velocity_w: torch.Tensor
    linear_acceleration_w: torch.Tensor


@dataclass(frozen=True)
class Lemniscate3DReference:
    """A translational 3D lemniscate reference and phase derivatives."""

    position_w: torch.Tensor
    linear_velocity_w: torch.Tensor
    linear_acceleration_w: torch.Tensor
    phase: torch.Tensor
    phase_rate: torch.Tensor
    phase_acceleration: torch.Tensor


class Lemniscate3DTrajectory:
    """Batched analytic 3D lemniscate with a smooth startup phase ramp.

    The local curve is ``[a sin(theta), b sin(2 theta), h cos(theta)]``.
    Resetting anchors its first point at the supplied world position. Phase
    speed ramps smoothly from zero to its steady value.
    """

    def __init__(
        self,
        initial_position_w: torch.Tensor,
        semi_axis_x: float,
        semi_axis_y: float,
        vertical_amplitude: float,
        period: float,
        entry_duration: float = 4.0,
        initial_phase: float | torch.Tensor = 0.0,
        direction: float = 1.0,
        curve_yaw_w: float | torch.Tensor = 0.0,
    ) -> None:
        if initial_position_w.ndim < 2 or initial_position_w.shape[-1] != 3:
            raise ValueError(
                "initial_position_w must have at least one batch dimension "
                "and end with three coordinates"
            )
        if not initial_position_w.is_floating_point():
            raise ValueError("initial_position_w must use a floating-point dtype")
        if not torch.isfinite(initial_position_w).all():
            raise ValueError("initial_position_w must contain only finite values")
        scalar_values = {
            "semi_axis_x": semi_axis_x,
            "semi_axis_y": semi_axis_y,
            "vertical_amplitude": vertical_amplitude,
            "period": period,
            "entry_duration": entry_duration,
            "direction": direction,
        }
        if not all(math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError("lemniscate parameters must be finite")
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

        self.semi_axis_x = float(semi_axis_x)
        self.semi_axis_y = float(semi_axis_y)
        self.vertical_amplitude = float(vertical_amplitude)
        self.period = float(period)
        self.entry_duration = float(entry_duration)
        self.direction = float(direction)
        self.angular_speed = 2.0 * math.pi / self.period

        batch_shape = initial_position_w.shape[:-1]
        self.elapsed = initial_position_w.new_zeros(batch_shape)
        self.initial_phase = initial_position_w.new_zeros(batch_shape)
        self.curve_yaw_w = initial_position_w.new_zeros(batch_shape)
        self.center_w = initial_position_w.new_zeros(*batch_shape, 3)
        self.reset(
            initial_position_w,
            initial_phase=initial_phase,
            curve_yaw_w=curve_yaw_w,
        )

    def sample(self) -> Lemniscate3DReference:
        return self.sample_at_time(self.elapsed)

    def curve_points(self, num_points: int = 241) -> torch.Tensor:
        """Return one closed geometric loop for visualization."""
        if (
            isinstance(num_points, bool)
            or num_points < 2
            or int(num_points) != num_points
        ):
            raise ValueError("num_points must be an integer greater than one")
        phase_offset = torch.linspace(
            0.0,
            self.direction * math.tau,
            int(num_points),
            dtype=self.elapsed.dtype,
            device=self.elapsed.device,
        )
        phase = self.initial_phase.unsqueeze(-1) + phase_offset
        local_position, _, _ = self._local_geometry(phase)
        return self.center_w.unsqueeze(-2) + self._rotate_local(local_position)

    def horizontal_tangent_heading(
        self,
        phase: torch.Tensor,
        phase_rate: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return world yaw and yaw rate along the horizontal travel tangent."""
        selected_shape = self.elapsed[env_ids].shape
        if phase.shape != selected_shape or phase_rate.shape != selected_shape:
            raise ValueError("phase and phase_rate must match the selected batch")
        _, local_first, local_second = self._local_geometry(phase)
        first_w = self._rotate_local(local_first, env_ids)
        second_w = self._rotate_local(local_second, env_ids)

        motion_tangent_w = self.direction * first_w
        yaw_w = torch.atan2(motion_tangent_w[..., 1], motion_tangent_w[..., 0])
        horizontal_speed_sq = first_w[..., :2].square().sum(dim=-1).clamp_min(1.0e-12)
        yaw_per_phase = (
            first_w[..., 0] * second_w[..., 1]
            - first_w[..., 1] * second_w[..., 0]
        ) / horizontal_speed_sq
        return yaw_w, yaw_per_phase * phase_rate

    def sample_at_time(
        self,
        elapsed: float | torch.Tensor,
    ) -> Lemniscate3DReference:
        elapsed_tensor = torch.as_tensor(
            elapsed,
            dtype=self.elapsed.dtype,
            device=self.elapsed.device,
        )
        try:
            elapsed_tensor = torch.broadcast_to(elapsed_tensor, self.elapsed.shape)
        except RuntimeError as error:
            raise ValueError(
                "elapsed must broadcast to the trajectory batch"
            ) from error
        if torch.any(elapsed_tensor < 0.0):
            raise ValueError("elapsed must be non-negative")

        phase, phase_rate, phase_acceleration = self._phase_state(elapsed_tensor)
        local_position, local_first, local_second = self._local_geometry(phase)
        local_velocity = local_first * phase_rate.unsqueeze(-1)
        local_acceleration = (
            local_second * phase_rate.square().unsqueeze(-1)
            + local_first * phase_acceleration.unsqueeze(-1)
        )
        return Lemniscate3DReference(
            position_w=self.center_w + self._rotate_local(local_position),
            linear_velocity_w=self._rotate_local(local_velocity),
            linear_acceleration_w=self._rotate_local(local_acceleration),
            phase=phase,
            phase_rate=phase_rate,
            phase_acceleration=phase_acceleration,
        )

    def advance(self, dt: float) -> Lemniscate3DReference:
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        self.elapsed.add_(float(dt))
        return self.sample()

    def reset(
        self,
        initial_position_w: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
        initial_phase: float | torch.Tensor | None = None,
        curve_yaw_w: float | torch.Tensor | None = None,
    ) -> None:
        """Reset selected trajectories and anchor their first point in world space."""
        selected_elapsed = self.elapsed[env_ids]
        expected_position_shape = (*selected_elapsed.shape, 3)
        if initial_position_w.shape != expected_position_shape:
            raise ValueError(
                f"initial_position_w must have shape {expected_position_shape}"
            )
        if not initial_position_w.is_floating_point():
            raise ValueError("initial_position_w must use a floating-point dtype")
        if not torch.isfinite(initial_position_w).all():
            raise ValueError("initial_position_w must contain only finite values")
        if initial_phase is not None:
            self.initial_phase[env_ids] = self._selected_parameter(
                "initial_phase", initial_phase, selected_elapsed
            )
        if curve_yaw_w is not None:
            self.curve_yaw_w[env_ids] = self._selected_parameter(
                "curve_yaw_w", curve_yaw_w, selected_elapsed
            )
        self.elapsed[env_ids] = 0.0
        phase = self.initial_phase[env_ids]
        local_position, _, _ = self._local_geometry(phase)
        self.center_w[env_ids] = initial_position_w - self._rotate_local(
            local_position, env_ids
        )

    def _phase_state(
        self,
        elapsed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signed_speed = self.direction * self.angular_speed
        if self.entry_duration == 0.0:
            return (
                self.initial_phase + signed_speed * elapsed,
                torch.full_like(elapsed, signed_speed),
                torch.zeros_like(elapsed),
            )

        duration = self.entry_duration
        u = (elapsed / duration).clamp(0.0, 1.0)
        speed_blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        speed_blend_derivative = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        integrated_blend = 2.5 * u**4 - 3.0 * u**5 + u**6
        entry_phase = signed_speed * duration * integrated_blend
        steady_phase = signed_speed * (elapsed - 0.5 * duration)
        in_entry = elapsed < duration
        return (
            self.initial_phase + torch.where(in_entry, entry_phase, steady_phase),
            torch.where(
                in_entry,
                signed_speed * speed_blend,
                torch.full_like(elapsed, signed_speed),
            ),
            torch.where(
                in_entry,
                signed_speed / duration * speed_blend_derivative,
                torch.zeros_like(elapsed),
            ),
        )

    def _local_geometry(
        self,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        sin_double_phase = torch.sin(2.0 * phase)
        cos_double_phase = torch.cos(2.0 * phase)
        return (
            torch.stack(
                [
                    self.semi_axis_x * sin_phase,
                    self.semi_axis_y * sin_double_phase,
                    self.vertical_amplitude * cos_phase,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    self.semi_axis_x * cos_phase,
                    2.0 * self.semi_axis_y * cos_double_phase,
                    -self.vertical_amplitude * sin_phase,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    -self.semi_axis_x * sin_phase,
                    -4.0 * self.semi_axis_y * sin_double_phase,
                    -self.vertical_amplitude * cos_phase,
                ],
                dim=-1,
            ),
        )

    def _rotate_local(
        self,
        value: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> torch.Tensor:
        yaw = self.curve_yaw_w[env_ids]
        while yaw.ndim < value.ndim - 1:
            yaw = yaw.unsqueeze(-1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.stack(
            [
                cos_yaw * value[..., 0] - sin_yaw * value[..., 1],
                sin_yaw * value[..., 0] + cos_yaw * value[..., 1],
                value[..., 2],
            ],
            dim=-1,
        )

    def _selected_parameter(
        self,
        name: str,
        value: float | torch.Tensor,
        selected_elapsed: torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=self.elapsed.dtype,
            device=self.elapsed.device,
        )
        try:
            tensor = torch.broadcast_to(tensor, selected_elapsed.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} must broadcast to the selected trajectory batch"
            ) from error
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor


class MinimumJerkPoseTrajectory:
    """Batched quintic pose reference with C2-continuous translation."""

    def __init__(
        self,
        position_w: torch.Tensor,
        orientation_wb: torch.Tensor,
        duration: float = 3.0,
    ) -> None:
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if position_w.shape[-1] != 3:
            raise ValueError("position_w must end with three coordinates")
        if orientation_wb.shape != (*position_w.shape[:-1], 4):
            raise ValueError("orientation_wb batch shape must match position_w")
        self.duration = float(duration)
        self.elapsed = position_w.new_full(position_w.shape[:-1], self.duration)
        self._coefficients = position_w.new_zeros(*position_w.shape[:-1], 6, 3)
        self._orientation_start = orientation_wb.clone()
        self.goal_position_w = position_w.clone()
        self.goal_orientation_wb = orientation_wb.clone()
        self.reset(position_w, orientation_wb)

    def sample(self) -> PoseReference:
        t = self.elapsed.clamp(max=self.duration)
        powers = torch.stack(
            [torch.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1
        )
        velocity_powers = torch.stack(
            [
                torch.zeros_like(t),
                torch.ones_like(t),
                2.0 * t,
                3.0 * t**2,
                4.0 * t**3,
                5.0 * t**4,
            ],
            dim=-1,
        )
        acceleration_powers = torch.stack(
            [
                torch.zeros_like(t),
                torch.zeros_like(t),
                2.0 * torch.ones_like(t),
                6.0 * t,
                12.0 * t**2,
                20.0 * t**3,
            ],
            dim=-1,
        )
        position = torch.einsum("...i,...ij->...j", powers, self._coefficients)
        velocity = torch.einsum(
            "...i,...ij->...j", velocity_powers, self._coefficients
        )
        acceleration = torch.einsum(
            "...i,...ij->...j", acceleration_powers, self._coefficients
        )
        u = (t / self.duration).unsqueeze(-1)
        blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        orientation = slerp(
            self._orientation_start, self.goal_orientation_wb, blend
        )
        return PoseReference(position, orientation, velocity, acceleration)

    def advance(self, dt: float) -> PoseReference:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        self.elapsed.add_(float(dt)).clamp_(max=self.duration)
        return self.sample()

    def reset(
        self,
        position_w: torch.Tensor,
        orientation_wb: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        """Place selected trajectories at rest at the supplied pose."""
        self._coefficients[env_ids] = 0.0
        self._coefficients[env_ids, 0, :] = position_w
        self._orientation_start[env_ids] = orientation_wb
        self.goal_position_w[env_ids] = position_w
        self.goal_orientation_wb[env_ids] = orientation_wb
        self.elapsed[env_ids] = self.duration

    def retarget(
        self,
        target_position_w: torch.Tensor,
        target_orientation_wb: torch.Tensor,
        env_ids: torch.Tensor | slice = slice(None),
    ) -> None:
        """Replan selected trajectories while preserving translation through C2."""
        current = self.sample()
        duration = self.duration
        a0 = current.position_w[env_ids]
        a1 = current.linear_velocity_w[env_ids]
        a2 = 0.5 * current.linear_acceleration_w[env_ids]
        residual_position = target_position_w - (
            a0 + a1 * duration + a2 * duration**2
        )
        residual_velocity = -(a1 + 2.0 * a2 * duration)
        residual_acceleration = -2.0 * a2
        rhs = torch.stack(
            [residual_position, residual_velocity, residual_acceleration], dim=-2
        )
        matrix = self._coefficients.new_tensor(
            [
                [duration**3, duration**4, duration**5],
                [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
                [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
            ]
        )
        high_order = torch.linalg.solve(matrix, rhs)
        self._coefficients[env_ids] = torch.stack(
            [
                a0,
                a1,
                a2,
                high_order[..., 0, :],
                high_order[..., 1, :],
                high_order[..., 2, :],
            ],
            dim=-2,
        )
        self._orientation_start[env_ids] = current.orientation_wb[env_ids]
        self.goal_position_w[env_ids] = target_position_w
        self.goal_orientation_wb[env_ids] = target_orientation_wb
        self.elapsed[env_ids] = 0.0


__all__ = [
    "Lemniscate3DReference",
    "Lemniscate3DTrajectory",
    "MinimumJerkPoseTrajectory",
    "PoseReference",
]
