import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from active_adaptation.control.contracts import missing_pose_reference_fields
from active_adaptation.utils.math import quat_rotate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MDP_ROOT = PROJECT_ROOT / "active_adaptation" / "envs" / "mdp"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_lemniscate_command_class():
    package_names = (
        "active_adaptation.envs",
        "active_adaptation.envs.mdp",
        "active_adaptation.envs.mdp.commands",
    )
    previous = {name: sys.modules.get(name) for name in package_names}
    loaded_names = (
        *package_names,
        "active_adaptation.envs.mdp.base",
        "active_adaptation.envs.mdp.commands.base",
        "active_adaptation.envs.mdp.commands.lemniscate",
    )
    try:
        for name, path in zip(
            package_names,
            (MDP_ROOT.parent, MDP_ROOT, MDP_ROOT / "commands"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        load_module("active_adaptation.envs.mdp.base", MDP_ROOT / "base.py")
        load_module(
            "active_adaptation.envs.mdp.commands.base",
            MDP_ROOT / "commands" / "base.py",
        )
        module = load_module(
            "active_adaptation.envs.mdp.commands.lemniscate",
            MDP_ROOT / "commands" / "lemniscate.py",
        )
        return module.Lemniscate3DCommand
    finally:
        for name in loaded_names:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module


Lemniscate3DCommand = load_lemniscate_command_class()


def make_command(num_envs: int = 2) -> tuple[Lemniscate3DCommand, SimpleNamespace]:
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(num_envs, 3),
        root_link_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        ).expand(num_envs, -1).clone(),
        root_link_lin_vel_w=torch.zeros(num_envs, 3),
        root_link_ang_vel_b=torch.zeros(num_envs, 3),
        default_root_state=torch.zeros(num_envs, 13),
        default_joint_pos=torch.empty(num_envs, 0),
        default_joint_vel=torch.empty(num_envs, 0),
    )
    asset = SimpleNamespace(data=data)
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.02,
        scene=SimpleNamespace(articulations={"robot": asset}),
    )
    command = Lemniscate3DCommand(
        semi_axis_x=1.5,
        semi_axis_y=0.9,
        vertical_amplitude=0.45,
        period=45.0,
        entry_duration=6.0,
    )
    command._initialize(env)
    command.reset(torch.arange(num_envs))
    return command, env


def test_command_exposes_frozen_pose_reference_contract():
    command, _ = make_command()
    assert missing_pose_reference_fields(command) == ()
    assert command.command.shape == (2, 12)
    assert command.get_state().batch_size == torch.Size([2])


def test_sync_state_does_not_advance_and_update_advances_once():
    command, env = make_command()
    elapsed = command.trajectory.elapsed.clone()
    phase = command.phase.clone()
    command.sync_state()
    torch.testing.assert_close(command.trajectory.elapsed, elapsed)
    torch.testing.assert_close(command.phase, phase)

    command.update()
    torch.testing.assert_close(
        command.trajectory.elapsed,
        elapsed + env.step_dt,
    )


def test_initial_heading_and_motion_follow_vehicle_nose():
    command, _ = make_command(num_envs=1)
    forward_b = torch.tensor([[1.0, 0.0, 0.0]])
    target_forward_w = quat_rotate(command.target_orientation_wb, forward_b)
    torch.testing.assert_close(
        target_forward_w[..., :2],
        torch.tensor([[1.0, 0.0]]),
        atol=1e-6,
        rtol=0.0,
    )

    for _ in range(50):
        command.update()
    target_forward_w = quat_rotate(command.target_orientation_wb, forward_b)
    velocity_xy = torch.nn.functional.normalize(
        command.target_linear_velocity_w[..., :2], dim=-1
    )
    torch.testing.assert_close(
        target_forward_w[..., :2], velocity_xy, atol=1e-5, rtol=1e-5
    )


def test_invalid_command_parameters_are_rejected():
    try:
        Lemniscate3DCommand(
            semi_axis_x=1.0,
            semi_axis_y=0.6,
            vertical_amplitude=0.3,
            period=20.0,
            heading_mode="invalid",
        )
    except ValueError:
        return
    raise AssertionError("invalid heading_mode was accepted")
