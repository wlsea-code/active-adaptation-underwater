"""BlueROVHeavy and X5A arm assembly for explicit underwater control."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.control.thruster import (
    BlueROVThrusterModel,
    ThrusterModelCfg,
)
from active_adaptation.envs.robots.underwater import (
    HydrodynamicsCfg,
    UnderwaterRobot,
)
from active_adaptation.registry import Registry


NUM_ROTORS = 8
ROTOR_FORCE_CONSTANT = 0.8e-7
NOMINAL_T200_FORCE_CONSTANT = 4.4e-7

MODEL_DIR = ROBOT_MODEL_DIR / "underwater" / "rov_arx"
WORKSPACE_MODEL_DIR = ROBOT_MODEL_DIR.parents[2] / "assetx" / "artifacts" / "rov_arx"
MJCF_PATH = MODEL_DIR / "model.xml"
if not MJCF_PATH.is_file() and WORKSPACE_MODEL_DIR.is_dir():
    MJCF_PATH = WORKSPACE_MODEL_DIR / "model.xml"

# Composite rigid-body values calculated from model.xml at the zero arm pose.
COMPOSITE_MASS = 15.034278
COMPOSITE_COM_B = (-0.00113973012, -0.0000163373282, -0.0524325795)
COMPOSITE_INERTIA_COM_B = (
    (0.364470680, -0.000378837927, 0.00466458665),
    (-0.000378837927, 0.449679313, 0.0000975358076),
    (0.00466458665, 0.0000975358076, 0.305379964),
)

# Arm hydrodynamics have not been measured. Use the Heavy drag model and a
# neutral-buoyancy volume for the assembled vehicle.
WATER_DENSITY = 997.0
VOLUME = COMPOSITE_MASS / WATER_DENSITY
COBM = 0.01
ADDED_MASS = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
LINEAR_DAMPING = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
QUADRATIC_DAMPING = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

ROTOR_NAMES = [f"rotor_{index}" for index in range(NUM_ROTORS)]
ROTOR_JOINT_NAMES = [f"rotor_{index}_joint" for index in range(NUM_ROTORS)]
ARM_JOINT_NAMES = [f"arm_joint{index}" for index in range(1, 9)]
ARM_BODY_NAMES = [
    "arm_base_link",
    "arm_link1",
    "arm_link2",
    "arm_link3",
    "arm_link4",
    "arm_link5",
    "gripper_base",
    "gripper_right",
    "gripper_left",
]
JOINT_NAMES = [*ROTOR_JOINT_NAMES, *ARM_JOINT_NAMES]
BODY_NAMES = ["base_link", *ROTOR_NAMES, *ARM_BODY_NAMES]

BODY_COLORS = {
    "base_link": (0.04, 0.30, 0.72),
    **{name: (0.025, 0.03, 0.04) for name in ROTOR_NAMES},
    "arm_base_link": (0.75294, 0.75294, 0.75294),
    "arm_link1": (0.79216, 0.81961, 0.93333),
    "arm_link2": (0.75294, 0.75294, 0.75294),
    "arm_link3": (1.0, 0.98431, 0.96471),
    "arm_link4": (0.79216, 0.81961, 0.92941),
    "arm_link5": (0.69804, 0.69804, 0.69804),
    "gripper_base": (0.89804, 0.91765, 0.92941),
    "gripper_right": (1.0, 1.0, 1.0),
    "gripper_left": (1.0, 1.0, 1.0),
}

ROTOR_TIME_CONSTANTS = {name: 0.01 for name in ROTOR_NAMES}
ROTOR_FORCE_CONSTANTS = {name: ROTOR_FORCE_CONSTANT for name in ROTOR_NAMES}
ROTOR_MAX_RPM = 3900.0
ROTOR_MAX_RAD_S = ROTOR_MAX_RPM * 2.0 * math.pi / 60.0

THRUSTER_MODEL_CFG = ThrusterModelCfg(
    min_rpm=-ROTOR_MAX_RPM,
    max_rpm=ROTOR_MAX_RPM,
    throttle_deadband=0.075,
    positive_rpm_slope=3659.9,
    positive_rpm_intercept=345.21,
    negative_rpm_slope=3494.4,
    negative_rpm_intercept=-433.50,
    positive_thrust_coefficients=(4.7368e-7, -1.9275e-4, 8.4452e-2),
    negative_thrust_coefficients=(-3.8442e-7, -1.6186e-4, -3.9139e-2),
    thrust_scale=9.81,
    nominal_force_constant=NOMINAL_T200_FORCE_CONSTANT,
    inversion_iterations=40,
)


def _spawn_rov_arx_mjcf(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn the MJCF and bind stable materials before tensor view creation."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_mjcf
    from pxr import UsdShade

    prim = spawn_from_mjcf(prim_path, cfg, translation, orientation)
    stage = sim_utils.get_current_stage()
    material_root = "/World/Looks/ROVArx"
    material_paths: dict[str, str] = {}
    for body_name, color in BODY_COLORS.items():
        material_path = f"{material_root}/{body_name}"
        material_paths[body_name] = material_path
        if not stage.GetPrimAtPath(material_path).IsValid():
            material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.42,
                metallic=0.15 if body_name == "base_link" else 0.0,
            )
            material_cfg.func(material_path, material_cfg)

    for robot_path in sim_utils.find_matching_prim_paths(str(prim_path)):
        geometry_root = f"{robot_path}/base_link"
        for body_name, material_path in material_paths.items():
            body_path = f"{geometry_root}/{body_name}"
            body_prim = stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                raise RuntimeError(f"ROV-arm body prim was not found: {body_path}")
            material = UsdShade.Material(stage.GetPrimAtPath(material_path))
            UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
    return prim


def make_isaaclab_cfg(self_collisions: bool = False):
    from isaacsim.core.utils.extensions import enable_extension

    from active_adaptation.assets.asset_cfg import (
        AssetSpec,
        ArticulationCfg,
        ImplicitActuatorCfg,
        sim_utils,
    )

    if not MJCF_PATH.is_file():
        raise FileNotFoundError(
            "BlueROVHeavy-X5A model was not found. Expected "
            f"{MODEL_DIR / 'model.xml'}"
        )

    enable_extension("isaacsim.asset.importer.mjcf")
    asset_cfg = ArticulationCfg(
        articulation_root_prim_path="/base_link/base_link",
        spawn=sim_utils.MjcfFileCfg(
            func=_spawn_rov_arx_mjcf,
            asset_path=str(MJCF_PATH),
            fix_base=False,
            self_collision=self_collisions,
            import_inertia_tensor=True,
            import_sites=True,
            make_instanceable=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                fix_root_link=False,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["rotor_.*_joint"],
                effort_limit_sim=0.0,
                velocity_limit_sim=ROTOR_MAX_RAD_S,
                stiffness=0.0,
                damping=0.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=20.0,
                stiffness=200.0,
                damping=5.0,
                friction=0.01,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["arm_joint[7-8]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=200.0,
                damping=5.0,
                friction=0.01,
                armature=0.01,
            ),
        },
        joint_names_simulation=JOINT_NAMES,
        body_names_simulation=BODY_NAMES,
    )
    return AssetSpec(
        config=asset_cfg,
        sensors={},
        wrapper=UnderwaterRobot(
            cfg=HydrodynamicsCfg(
                volume=VOLUME,
                coBM=COBM,
                added_mass=ADDED_MASS,
                linear_damping=LINEAR_DAMPING,
                quadratic_damping=QUADRATIC_DAMPING,
                water_density=WATER_DENSITY,
            ),
            rotor_time_constants=ROTOR_TIME_CONSTANTS,
            rotor_force_constants=ROTOR_FORCE_CONSTANTS,
            thruster_model=BlueROVThrusterModel(THRUSTER_MODEL_CFG),
        ),
    )


def make_cfg(backend: Literal["isaaclab", "mjlab", "motrix"]):
    if backend != "isaaclab":
        raise NotImplementedError("BlueROVHeavy-X5A currently requires IsaacLab")
    return make_isaaclab_cfg()


Registry.instance().register("asset", "bluerov_heavy_arm", make_cfg)


__all__ = [
    "ARM_JOINT_NAMES",
    "BODY_COLORS",
    "COMPOSITE_COM_B",
    "COMPOSITE_INERTIA_COM_B",
    "COMPOSITE_MASS",
    "MJCF_PATH",
    "NUM_ROTORS",
    "VOLUME",
    "make_cfg",
    "make_isaaclab_cfg",
]
