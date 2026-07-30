from __future__ import annotations

import math
import torch

from typing import TYPE_CHECKING, Mapping, Sequence, Tuple, cast
from typing_extensions import override
from tensordict import TensorDictBase

from active_adaptation.control.contracts import missing_pose_reference_fields
from active_adaptation.control.bluerov_controller import (
    BlueROVExplicitController,
    PoseAccelerationController,
    PoseControllerCfg,
    RigidBodyWrenchController,
)
from active_adaptation.control.thruster import (
    BlueROVThrusterModel,
    ThrusterAllocator,
    ThrusterModelCfg,
)
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.utils.symmetry import SymmetryTransform

from .base import ActionV2

if TYPE_CHECKING:
    from active_adaptation.envs.env_base import _EnvBase
    from active_adaptation.envs.robots.underwater import UnderwaterRobotData


def _pose_controller_cfg(
    values: Mapping[str, Sequence[float]],
) -> PoseControllerCfg:
    values = dict(values)
    return PoseControllerCfg(
        position_kp=tuple(values["position_kp"]),
        position_kd=tuple(values["position_kd"]),
        attitude_kp=tuple(values["attitude_kp"]),
        attitude_kd=tuple(values["attitude_kd"]),
        max_linear_acceleration=tuple(values["max_linear_acceleration"]),
        max_angular_acceleration=tuple(values["max_angular_acceleration"]),
    )


def _thruster_model_cfg(values: Mapping[str, object]) -> ThrusterModelCfg:
    values = dict(values)
    values["positive_thrust_coefficients"] = tuple(
        values["positive_thrust_coefficients"]
    )
    values["negative_thrust_coefficients"] = tuple(
        values["negative_thrust_coefficients"]
    )
    return ThrusterModelCfg(**values)


def _added_mass_applied_wrench(
    added_mass_wrench: torch.Tensor,
) -> torch.Tensor:
    wrench = -added_mass_wrench.clone()
    wrench[..., [1, 2, 4, 5]] *= -1.0
    return wrench


class UnderwaterThrottle(ActionV2):
    """Throttle action for underwater robots.

    The action directly controls per-rotor normalized throttle in ``[-1, 1]``.
    Throttle-to-thrust conversion is handled by ``UnderwaterRobot.write_data_to_sim``.
    """
    uw: "UnderwaterRobotData"

    def __init__(
        self,
        action_scaling: float = 1.0,
        alpha_range: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.action_scaling = float(action_scaling)
        self.alpha_range = tuple(alpha_range)

    @override
    def _initialize(self, env: "_EnvBase"):
        super()._initialize(env)
        if not hasattr(self.asset, "data_underwater"):
            raise RuntimeError(
                "UnderwaterThrottle requires robot.data_underwater to be initialized."
            )
        self.uw = cast("UnderwaterRobotData", self.asset.data_underwater)
        self.action_dim = int(self.uw.throttle_cmd.shape[-1])
        self.action_buf = torch.zeros(
            self.num_envs, 4, self.action_dim, device=self.device
        )
        self.applied_action = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.alpha = torch.ones(self.num_envs, 1, device=self.device)

    @override
    def reset(
        self, env_ids: torch.Tensor, tensordict: TensorDictBase | None = None
    ) -> None:
        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha
        self.action_buf[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.uw.throttle_cmd[env_ids] = 0.0
        self.uw.throttle[env_ids] = 0.0

    @override
    def process_action(self, action: torch.Tensor | None):
        if action is None:
            return
        self.action_buf = self.action_buf.roll(1, dims=1)
        self.action_buf[:, 0] = action

    @override
    def apply_action(self, substep: int):
        self.applied_action.lerp_(self.action_buf[:, 0], self.alpha)
        self.uw.throttle_cmd.copy_(
            torch.clamp(self.applied_action * self.action_scaling, -1.0, 1.0)
        )

    @override
    def symmetry_transform(self):
        return SymmetryTransform(
            perm=torch.arange(self.action_dim),
            signs=[1] * self.action_dim,
        )


class PoseReferenceTrackingActionBase(UnderwaterThrottle):
    """Shared pose-reference controller that writes rotor throttle.

    The policy action is intentionally ignored. Pose and velocity references
    are owned by the Command and exposed through the pose-reference fields.
    """

    supported_backends = ("isaac",)

    def __init__(
        self,
        pose_controller: Mapping[str, Sequence[float]],
        thruster_model: Mapping[str, object],
        wrench_weights: Sequence[float],
        wrench_command_mask: Sequence[float],
        reaction_torque_per_thrust: Sequence[float] | float = 0.0,
        allocation_damping: float = 1.0e-6,
        gravity_w: Sequence[float] = (0.0, 0.0, -9.81),
        use_added_mass: bool = True,
        compensate_hydrodynamics: bool = False,
        mass: float | None = None,
        inertia_b: Sequence[Sequence[float]] | Sequence[float] | None = None,
        base_body_name: str = "base_link",
        action_scaling: float = 1.0,
        alpha_range: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            alpha_range=alpha_range,
        )
        self.pose_cfg = _pose_controller_cfg(pose_controller)
        self.thruster_cfg = _thruster_model_cfg(thruster_model)
        self.wrench_weights = tuple(float(value) for value in wrench_weights)
        self.wrench_command_mask = tuple(
            float(value) for value in wrench_command_mask
        )
        self.reaction_torque_per_thrust = reaction_torque_per_thrust
        self.allocation_damping = float(allocation_damping)
        self.gravity_w = tuple(float(value) for value in gravity_w)
        if len(self.gravity_w) != 3 or not all(
            math.isfinite(value) for value in self.gravity_w
        ):
            raise ValueError("gravity_w must contain three finite values")
        self.use_added_mass = bool(use_added_mass)
        self.compensate_hydrodynamics = bool(compensate_hydrodynamics)
        self.mass_override = None if mass is None else float(mass)
        self.inertia_override = inertia_b
        self.base_body_name = base_body_name

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        command = self.env.command_manager
        missing_fields = missing_pose_reference_fields(command)
        if missing_fields:
            raise TypeError(
                f"{type(self).__name__} requires a pose-reference command; "
                f"missing fields: {missing_fields}"
            )
        self._validate_command_type(command)

        wrapper = getattr(self.env, "robot_wrapper", None)
        if wrapper is None or getattr(wrapper, "rotor_indices", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an initialized underwater "
                "robot wrapper"
            )
        if wrapper.num_rotors != self.action_dim:
            raise RuntimeError(
                "Underwater wrapper rotor count does not match the "
                "throttle action dimension"
            )

        mass, inertia_b = self._rigid_body_parameters()
        self.controller_mass = mass
        added_mass = None
        if self.use_added_mass:
            added_mass_matrices = self.uw.added_mass_matrix
            reference_added_mass = added_mass_matrices[0]
            if not torch.allclose(
                added_mass_matrices,
                reference_added_mass.expand_as(added_mass_matrices),
            ):
                raise ValueError(
                    "Per-environment added-mass variation is not supported"
                )
            added_mass = reference_added_mass.detach().cpu()

        positions_b, directions_b = self._thruster_geometry(wrapper)
        thruster_model = BlueROVThrusterModel(self.thruster_cfg)
        min_thrust, max_thrust = thruster_model.thrust_limits()
        allocator = ThrusterAllocator(
            positions_b=positions_b,
            directions_b=directions_b,
            min_thrust=min_thrust,
            max_thrust=max_thrust,
            reaction_torque_per_thrust=self.reaction_torque_per_thrust,
            wrench_weights=self.wrench_weights,
            damping=self.allocation_damping,
        )
        self.controller = BlueROVExplicitController(
            pose_controller=PoseAccelerationController(self.pose_cfg),
            wrench_controller=RigidBodyWrenchController(
                mass=mass,
                inertia_b=inertia_b,
                added_mass=added_mass,
            ),
            allocator=allocator,
            thruster_model=thruster_model,
            wrench_command_mask=self.wrench_command_mask,
        )
        self.gravity_force_w = torch.tensor(
            self.gravity_w,
            dtype=self.asset.data.root_link_pos_w.dtype,
            device=self.device,
        ) * mass
        self.latest_output = None

    def _validate_command_type(self, command: object) -> None:
        """Hook for Actions that require a concrete Command type."""

    def _rigid_body_parameters(self) -> tuple[float, torch.Tensor]:
        masses = self.asset.root_physx_view.get_masses()[0]
        mass = (
            float(masses.sum().item())
            if self.mass_override is None
            else self.mass_override
        )
        if self.inertia_override is not None:
            inertia_b = torch.as_tensor(self.inertia_override, dtype=torch.float64)
        else:
            base_id = self.asset.body_names.index(self.base_body_name)
            inertia_b = self.asset.root_physx_view.get_inertias()[
                0, base_id
            ].reshape(3, 3)
        return mass, inertia_b.detach().cpu()

    def _thruster_geometry(self, wrapper) -> tuple[torch.Tensor, torch.Tensor]:
        base_id = self.asset.body_names.index(self.base_body_name)
        base_position_w = self.asset.data.body_link_pos_w[:, base_id, None, :]
        base_orientation_wb = self.asset.data.body_link_quat_w[
            :, base_id, None, :
        ].expand(-1, wrapper.num_rotors, -1)
        rotor_position_w = self.asset.data.body_link_pos_w[
            :, wrapper.rotor_indices
        ]
        rotor_orientation_w = self.asset.data.body_link_quat_w[
            :, wrapper.rotor_indices
        ]
        positions_b = quat_rotate_inverse(
            base_orientation_wb,
            rotor_position_w - base_position_w,
        )[0]
        local_x = torch.zeros_like(positions_b)
        local_x[:, 0] = 1.0
        directions_w = quat_rotate(
            rotor_orientation_w,
            local_x.unsqueeze(0).expand(self.num_envs, -1, -1),
        )
        directions_b = quat_rotate_inverse(
            base_orientation_wb,
            directions_w,
        )[0]
        return positions_b.detach().cpu(), directions_b.detach().cpu()

    @override
    def process_action(self, _: torch.Tensor | None) -> None:
        command = self.env.command_manager
        orientation_wb = self.asset.data.root_link_quat_w
        gravity_b = quat_rotate_inverse(
            orientation_wb,
            self.gravity_force_w.expand(self.num_envs, -1),
        )
        gravity_wrench_b = torch.cat(
            [gravity_b, torch.zeros_like(gravity_b)], dim=-1
        )
        external_wrench_b = self.uw.buoyancy + gravity_wrench_b
        if self.compensate_hydrodynamics:
            external_wrench_b = external_wrench_b + self.uw.hydro
            if self.use_added_mass:
                external_wrench_b = (
                    external_wrench_b
                    - _added_mass_applied_wrench(self.uw.added_mass)
                )

        self.latest_output = self.controller.compute(
            position_w=self.asset.data.root_link_pos_w,
            orientation_wb=orientation_wb,
            linear_velocity_w=self.asset.data.root_link_lin_vel_w,
            angular_velocity_b=self.asset.data.root_link_ang_vel_b,
            target_position_w=command.target_position_w,
            target_orientation_wb=command.target_orientation_wb,
            target_linear_velocity_w=command.target_linear_velocity_w,
            target_angular_velocity_b=command.target_angular_velocity_b,
            feedforward_linear_acceleration_w=(
                command.feedforward_linear_acceleration_w
            ),
            feedforward_angular_acceleration_b=(
                command.feedforward_angular_acceleration_b
            ),
            external_wrench_b=external_wrench_b,
        )
        super().process_action(self.latest_output.throttle)

    @override
    def diagnostics(self) -> dict[str, torch.Tensor | float]:
        if self.latest_output is None:
            return {}
        return {
            "controller/allocator_rank": float(self.controller.allocator.rank),
            "controller/residual_wrench": torch.linalg.vector_norm(
                self.latest_output.actuation_residual_wrench_b, dim=-1
            ).mean(),
            "controller/max_abs_throttle": self.latest_output.throttle.abs().max(),
        }


class BlueROVPoseTrackingAction(PoseReferenceTrackingActionBase):
    """Track any Command that satisfies the pose-reference contract."""


class Lemniscate3DTrackingAction(PoseReferenceTrackingActionBase):
    """Track references produced specifically by ``Lemniscate3DCommand``."""

    @override
    def _validate_command_type(self, command: object) -> None:
        from active_adaptation.envs.mdp.commands.lemniscate import (
            Lemniscate3DCommand,
        )

        if not isinstance(command, Lemniscate3DCommand):
            raise TypeError(
                f"{type(self).__name__} requires Lemniscate3DCommand; "
                f"got {type(command).__name__}"
            )


def _validate_bluerov_heavy(action: PoseReferenceTrackingActionBase) -> None:
    expected_thrusters = 8
    actual_thrusters = action.controller.allocator.num_thrusters
    if actual_thrusters != expected_thrusters:
        raise RuntimeError(
            f"{type(action).__name__} requires {expected_thrusters} thrusters; "
            f"the initialized asset exposes {actual_thrusters}"
        )


class BlueROVHeavyPoseTrackingAction(PoseReferenceTrackingActionBase):
    """Track pose references with the eight-thruster BlueROVHeavy."""

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        _validate_bluerov_heavy(self)


class BlueROVHeavyLemniscateTrackingAction(Lemniscate3DTrackingAction):
    """Track Lemniscate references with the eight-thruster BlueROVHeavy."""

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        _validate_bluerov_heavy(self)


def _validate_bluerov_heavy_arm(
    action: PoseReferenceTrackingActionBase,
) -> None:
    _validate_bluerov_heavy(action)
    expected_arm_joints = {f"arm_joint{index}" for index in range(1, 9)}
    missing = sorted(expected_arm_joints.difference(action.asset.joint_names))
    if missing:
        raise RuntimeError(
            f"{type(action).__name__} requires the X5A arm joints; "
            f"missing {missing}"
        )


class BlueROVHeavyArmPoseTrackingAction(PoseReferenceTrackingActionBase):
    """Track pose references while the X5A arm holds its configured pose."""

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        _validate_bluerov_heavy_arm(self)


class BlueROVHeavyArmLemniscateTrackingAction(Lemniscate3DTrackingAction):
    """Track Lemniscate references with the BlueROVHeavy-X5A assembly."""

    @override
    def _initialize(self, env: "_EnvBase") -> None:
        super()._initialize(env)
        _validate_bluerov_heavy_arm(self)


__all__ = [
    "BlueROVPoseTrackingAction",
    "BlueROVHeavyArmLemniscateTrackingAction",
    "BlueROVHeavyArmPoseTrackingAction",
    "BlueROVHeavyLemniscateTrackingAction",
    "BlueROVHeavyPoseTrackingAction",
    "Lemniscate3DTrackingAction",
    "PoseReferenceTrackingActionBase",
    "UnderwaterThrottle",
]
