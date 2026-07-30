"""Shared contracts between pose-reference Commands and Actions."""

POSE_REFERENCE_FIELDS = (
    "target_position_w",
    "target_orientation_wb",
    "target_linear_velocity_w",
    "target_angular_velocity_b",
    "feedforward_linear_acceleration_w",
    "feedforward_angular_acceleration_b",
)


def missing_pose_reference_fields(command: object) -> tuple[str, ...]:
    """Return pose-reference fields not exposed by ``command``."""
    return tuple(
        field for field in POSE_REFERENCE_FIELDS if not hasattr(command, field)
    )


__all__ = ["POSE_REFERENCE_FIELDS", "missing_pose_reference_fields"]
