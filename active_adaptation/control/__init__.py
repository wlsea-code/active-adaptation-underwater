"""Classical controllers that do not depend on a learned policy."""

from .thruster import (
    AllocationResult,
    BlueROVThrusterModel,
    ThrusterAllocator,
    ThrusterModelCfg,
)
from .trajectory import (
    Lemniscate3DReference,
    Lemniscate3DTrajectory,
    MinimumJerkPoseTrajectory,
    PoseReference,
)
from .bluerov_controller import (
    BlueROVExplicitController,
    ExplicitControlOutput,
    PoseAccelerationController,
    PoseControllerCfg,
    RigidBodyWrenchController,
)

__all__ = [
    "AllocationResult",
    "BlueROVExplicitController",
    "BlueROVThrusterModel",
    "ExplicitControlOutput",
    "Lemniscate3DReference",
    "Lemniscate3DTrajectory",
    "MinimumJerkPoseTrajectory",
    "PoseAccelerationController",
    "PoseControllerCfg",
    "PoseReference",
    "RigidBodyWrenchController",
    "ThrusterAllocator",
    "ThrusterModelCfg",
]
