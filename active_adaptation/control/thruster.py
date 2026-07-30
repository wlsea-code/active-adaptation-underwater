"""Thruster allocation and bidirectional throttle/RPM/thrust conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

def _tensor(value, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@dataclass(frozen=True)
class AllocationResult:
    thrust: torch.Tensor
    achieved_wrench_b: torch.Tensor
    residual_wrench_b: torch.Tensor


class ThrusterAllocator:
    """Bounded, weighted damped-least-squares control allocator."""

    def __init__(
        self,
        positions_b: Sequence[Sequence[float]],
        directions_b: Sequence[Sequence[float]],
        min_thrust: Sequence[float] | float,
        max_thrust: Sequence[float] | float,
        reaction_torque_per_thrust: Sequence[float] | float = 0.0,
        wrench_weights: Sequence[float] = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0),
        damping: float = 1.0e-6,
    ):
        positions = torch.as_tensor(positions_b, dtype=torch.float64)
        directions = torch.as_tensor(directions_b, dtype=torch.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions_b must have shape (num_thrusters, 3)")
        if directions.shape != positions.shape:
            raise ValueError("directions_b must match positions_b")
        directions = directions / torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        self.num_thrusters = positions.shape[0]
        self.positions_b = positions.clone()
        self.directions_b = directions.clone()
        reaction = torch.as_tensor(reaction_torque_per_thrust, dtype=torch.float64)
        reaction = torch.broadcast_to(reaction, (self.num_thrusters,))
        moments = torch.cross(positions, directions, dim=-1) + reaction[:, None] * directions
        self.allocation_matrix = torch.cat([directions.T, moments.T], dim=0)
        self.min_thrust = torch.broadcast_to(
            torch.as_tensor(min_thrust, dtype=torch.float64), (self.num_thrusters,)
        ).clone()
        self.max_thrust = torch.broadcast_to(
            torch.as_tensor(max_thrust, dtype=torch.float64), (self.num_thrusters,)
        ).clone()
        if torch.any(self.min_thrust >= self.max_thrust):
            raise ValueError("each min_thrust must be less than max_thrust")
        self.wrench_weights = torch.as_tensor(wrench_weights, dtype=torch.float64)
        if self.wrench_weights.shape != (6,):
            raise ValueError("wrench_weights must contain six values")
        self.damping = float(damping)

    @property
    def rank(self) -> int:
        return int(torch.linalg.matrix_rank(self.allocation_matrix).item())

    @property
    def singular_values(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.allocation_matrix)

    def allocate(self, desired_wrench_b: torch.Tensor) -> AllocationResult:
        original_shape = desired_wrench_b.shape
        if original_shape[-1] != 6:
            raise ValueError("desired_wrench_b must end in six wrench components")
        flat_wrench = desired_wrench_b.reshape(-1, 6)
        outputs = [self._allocate_one(wrench) for wrench in flat_wrench]
        thrust = torch.stack(outputs).reshape(*original_shape[:-1], self.num_thrusters)
        matrix = self.allocation_matrix.to(
            dtype=desired_wrench_b.dtype, device=desired_wrench_b.device
        )
        achieved = torch.matmul(thrust, matrix.T)
        return AllocationResult(
            thrust=thrust,
            achieved_wrench_b=achieved,
            residual_wrench_b=desired_wrench_b - achieved,
        )

    def _allocate_one(self, desired_wrench_b: torch.Tensor) -> torch.Tensor:
        device, dtype = desired_wrench_b.device, desired_wrench_b.dtype
        matrix = self.allocation_matrix.to(device=device, dtype=dtype)
        lower = self.min_thrust.to(device=device, dtype=dtype)
        upper = self.max_thrust.to(device=device, dtype=dtype)
        weights = self.wrench_weights.to(device=device, dtype=dtype)
        thrust = torch.zeros(self.num_thrusters, device=device, dtype=dtype)
        free = torch.ones(self.num_thrusters, device=device, dtype=torch.bool)

        # Active-set saturation: solve over remaining free thrusters, clamp any
        # violations, then reallocate the residual wrench.
        for _ in range(self.num_thrusters + 1):
            if not torch.any(free):
                break
            fixed = ~free
            residual = desired_wrench_b
            if torch.any(fixed):
                residual = residual - matrix[:, fixed] @ thrust[fixed]
            weighted_matrix = weights[:, None] * matrix[:, free]
            weighted_residual = weights * residual
            normal = weighted_matrix.T @ weighted_matrix
            normal += self.damping * torch.eye(
                normal.shape[0], device=device, dtype=dtype
            )
            solution = torch.linalg.solve(
                normal, weighted_matrix.T @ weighted_residual
            )
            free_ids = torch.nonzero(free, as_tuple=False).squeeze(-1)
            thrust[free_ids] = solution
            violations = (thrust < lower) | (thrust > upper)
            violations &= free
            if not torch.any(violations):
                break
            thrust[violations] = torch.clamp(
                thrust[violations], lower[violations], upper[violations]
            )
            free[violations] = False
        return torch.clamp(thrust, lower, upper)


@dataclass(frozen=True)
class ThrusterModelCfg:
    """Bidirectional throttle, RPM and thrust calibration."""

    min_rpm: float
    max_rpm: float
    throttle_deadband: float
    positive_rpm_slope: float
    positive_rpm_intercept: float
    negative_rpm_slope: float
    negative_rpm_intercept: float
    positive_thrust_coefficients: tuple[float, float, float]
    negative_thrust_coefficients: tuple[float, float, float]
    thrust_scale: float
    nominal_force_constant: float
    inversion_iterations: int = 40

    def __post_init__(self) -> None:
        scalar_values = (
            self.min_rpm,
            self.max_rpm,
            self.throttle_deadband,
            self.positive_rpm_slope,
            self.positive_rpm_intercept,
            self.negative_rpm_slope,
            self.negative_rpm_intercept,
            self.thrust_scale,
            self.nominal_force_constant,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("thruster calibration values must be finite")
        if not self.min_rpm < 0.0 < self.max_rpm:
            raise ValueError("min_rpm and max_rpm must straddle zero")
        if not 0.0 <= self.throttle_deadband < 1.0:
            raise ValueError("throttle_deadband must be in [0, 1)")
        if self.positive_rpm_slope <= 0.0 or self.negative_rpm_slope <= 0.0:
            raise ValueError("RPM slopes must be positive")
        if self.thrust_scale <= 0.0 or self.nominal_force_constant <= 0.0:
            raise ValueError("thrust scales must be positive")
        if self.inversion_iterations <= 0:
            raise ValueError("inversion_iterations must be positive")
        if (
            len(self.positive_thrust_coefficients) != 3
            or len(self.negative_thrust_coefficients) != 3
        ):
            raise ValueError("thrust coefficient sets must contain three values")
        positive_coefficients = torch.tensor(self.positive_thrust_coefficients)
        negative_coefficients = torch.tensor(self.negative_thrust_coefficients)
        positive_active_rpm = (
            self.positive_rpm_slope * self.throttle_deadband
            + self.positive_rpm_intercept
        )
        negative_active_rpm = (
            -self.negative_rpm_slope * self.throttle_deadband
            + self.negative_rpm_intercept
        )
        if not 0.0 < positive_active_rpm < self.max_rpm:
            raise ValueError("positive deadband edge must lie inside the RPM range")
        if not self.min_rpm < negative_active_rpm < 0.0:
            raise ValueError("negative deadband edge must lie inside the RPM range")

        positive_rpm = torch.tensor([positive_active_rpm, self.max_rpm])
        negative_rpm = torch.tensor([self.min_rpm, negative_active_rpm])
        positive_derivative = (
            2.0 * positive_coefficients[0] * positive_rpm
            + positive_coefficients[1]
        )
        negative_derivative = (
            2.0 * negative_coefficients[0] * negative_rpm
            + negative_coefficients[1]
        )
        if torch.any(positive_derivative <= 0.0) or torch.any(
            negative_derivative <= 0.0
        ):
            raise ValueError("thrust curves must increase monotonically with RPM")
        positive_thrust = self.thrust_scale * (
            positive_coefficients[0] * positive_rpm.square()
            + positive_coefficients[1] * positive_rpm
            + positive_coefficients[2]
        )
        negative_thrust = self.thrust_scale * (
            negative_coefficients[0] * negative_rpm.square()
            + negative_coefficients[1] * negative_rpm
            + negative_coefficients[2]
        )
        if torch.any(positive_thrust <= 0.0) or torch.any(
            negative_thrust >= 0.0
        ):
            raise ValueError("RPM branches must preserve the thrust sign")


class BlueROVThrusterModel:
    """Invertible actuator model constructed from explicit calibration."""

    def __init__(self, cfg: ThrusterModelCfg) -> None:
        self.cfg = cfg

    @property
    def positive_deadband_rpm(self) -> float:
        return (
            self.cfg.positive_rpm_slope * self.cfg.throttle_deadband
            + self.cfg.positive_rpm_intercept
        )

    @property
    def negative_deadband_rpm(self) -> float:
        return (
            -self.cfg.negative_rpm_slope * self.cfg.throttle_deadband
            + self.cfg.negative_rpm_intercept
        )

    def throttle_to_rpm(self, throttle: torch.Tensor) -> torch.Tensor:
        throttle = torch.clamp(throttle, -1.0, 1.0)
        positive = (
            self.cfg.positive_rpm_slope * throttle
            + self.cfg.positive_rpm_intercept
        )
        negative = (
            self.cfg.negative_rpm_slope * throttle
            + self.cfg.negative_rpm_intercept
        )
        rpm = torch.where(
            throttle > self.cfg.throttle_deadband,
            positive,
            torch.where(
                throttle < -self.cfg.throttle_deadband,
                negative,
                torch.zeros_like(throttle),
            ),
        )
        return torch.clamp(rpm, self.cfg.min_rpm, self.cfg.max_rpm)

    def rpm_to_thrust(self, rpm: torch.Tensor) -> torch.Tensor:
        positive = self._evaluate_curve(
            rpm, self.cfg.positive_thrust_coefficients
        )
        negative = self._evaluate_curve(
            rpm, self.cfg.negative_thrust_coefficients
        )
        return torch.where(
            rpm > 0.0,
            positive,
            torch.where(rpm < 0.0, negative, torch.zeros_like(rpm)),
        )

    def thrust_limits(self, *, dtype=torch.float64) -> tuple[float, float]:
        rpm = torch.tensor(
            [self.cfg.min_rpm, self.cfg.max_rpm],
            dtype=dtype,
        )
        limits = self.rpm_to_thrust(rpm)
        return float(limits[0]), float(limits[1])

    def thrust_to_rpm(self, thrust: torch.Tensor) -> torch.Tensor:
        """Invert by bisection, including saturation and the actuator dead zone."""
        min_thrust, max_thrust = self.thrust_limits(dtype=thrust.dtype)
        target = torch.clamp(thrust, min_thrust, max_thrust)
        is_positive = target > 0.0
        low = torch.where(
            is_positive,
            torch.full_like(target, self.positive_deadband_rpm),
            torch.full_like(target, self.cfg.min_rpm),
        )
        high = torch.where(
            is_positive,
            torch.full_like(target, self.cfg.max_rpm),
            torch.full_like(target, self.negative_deadband_rpm),
        )
        for _ in range(self.cfg.inversion_iterations):
            mid = (low + high) * 0.5
            move_low = self.rpm_to_thrust(mid) < target
            low = torch.where(move_low, mid, low)
            high = torch.where(move_low, high, mid)
        active_rpm = (low + high) * 0.5
        active_error = torch.abs(self.rpm_to_thrust(active_rpm) - target)
        use_zero = torch.abs(target) <= active_error
        return torch.where(use_zero, torch.zeros_like(active_rpm), active_rpm)

    def rpm_to_throttle(self, rpm: torch.Tensor) -> torch.Tensor:
        positive = (
            rpm - self.cfg.positive_rpm_intercept
        ) / self.cfg.positive_rpm_slope
        negative = (
            rpm - self.cfg.negative_rpm_intercept
        ) / self.cfg.negative_rpm_slope
        throttle = torch.where(
            rpm > 0.0,
            positive,
            torch.where(rpm < 0.0, negative, torch.zeros_like(rpm)),
        )
        deadband = torch.full_like(throttle, self.cfg.throttle_deadband)
        positive_edge = torch.nextafter(deadband, torch.ones_like(deadband))
        negative_edge = torch.nextafter(-deadband, -torch.ones_like(deadband))
        throttle = torch.where(
            rpm > 0.0,
            torch.maximum(throttle, positive_edge),
            torch.where(
                rpm < 0.0,
                torch.minimum(throttle, negative_edge),
                throttle,
            ),
        )
        return torch.clamp(throttle, -1.0, 1.0)

    def _evaluate_curve(
        self,
        rpm: torch.Tensor,
        coefficients: Sequence[float],
    ) -> torch.Tensor:
        quadratic, linear, constant = _tensor(coefficients, like=rpm)
        return self.cfg.thrust_scale * (
            quadratic * rpm.square() + linear * rpm + constant
        )


__all__ = [
    "AllocationResult",
    "BlueROVThrusterModel",
    "ThrusterAllocator",
    "ThrusterModelCfg",
]
