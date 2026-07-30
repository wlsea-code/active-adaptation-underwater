import math

import torch

from active_adaptation.control.trajectory import Lemniscate3DTrajectory


def make_trajectory(direction: float = 1.0) -> Lemniscate3DTrajectory:
    initial_position = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]], dtype=torch.float64
    )
    return Lemniscate3DTrajectory(
        initial_position,
        semi_axis_x=1.0,
        semi_axis_y=0.6,
        vertical_amplitude=0.3,
        period=20.0,
        entry_duration=4.0,
        initial_phase=torch.tensor([0.0, 0.7], dtype=torch.float64),
        direction=direction,
        curve_yaw_w=torch.tensor([0.0, 0.4], dtype=torch.float64),
    )


def test_reset_anchor_starts_at_rest():
    initial_position = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]], dtype=torch.float64
    )
    reference = make_trajectory().sample()
    torch.testing.assert_close(reference.position_w, initial_position)
    torch.testing.assert_close(
        reference.linear_velocity_w,
        torch.zeros_like(initial_position),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        reference.linear_acceleration_w,
        torch.zeros_like(initial_position),
        atol=1e-12,
        rtol=0.0,
    )


def test_entry_is_continuous_and_steady_reference_is_periodic():
    trajectory = make_trajectory()
    at_entry_end = trajectory.sample_at_time(4.0)
    just_before = trajectory.sample_at_time(4.0 - 1e-6)
    torch.testing.assert_close(
        at_entry_end.phase_rate,
        torch.full_like(at_entry_end.phase_rate, math.tau / 20.0),
    )
    torch.testing.assert_close(
        at_entry_end.phase_acceleration,
        torch.zeros_like(at_entry_end.phase_acceleration),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        just_before.position_w, at_entry_end.position_w, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        just_before.linear_velocity_w,
        at_entry_end.linear_velocity_w,
        atol=1e-6,
        rtol=0.0,
    )

    first = trajectory.sample_at_time(7.25)
    second = trajectory.sample_at_time(27.25)
    torch.testing.assert_close(first.position_w, second.position_w)
    torch.testing.assert_close(first.linear_velocity_w, second.linear_velocity_w)
    torch.testing.assert_close(
        first.linear_acceleration_w, second.linear_acceleration_w
    )


def test_analytic_derivatives_match_finite_difference():
    trajectory = make_trajectory()
    time = 8.3
    step = 1e-4
    before = trajectory.sample_at_time(time - step)
    current = trajectory.sample_at_time(time)
    after = trajectory.sample_at_time(time + step)
    numerical_velocity = (after.position_w - before.position_w) / (2.0 * step)
    numerical_acceleration = (
        after.linear_velocity_w - before.linear_velocity_w
    ) / (2.0 * step)
    torch.testing.assert_close(
        current.linear_velocity_w, numerical_velocity, atol=1e-8, rtol=1e-7
    )
    torch.testing.assert_close(
        current.linear_acceleration_w,
        numerical_acceleration,
        atol=1e-8,
        rtol=1e-7,
    )


def test_tangent_heading_matches_forward_and_reverse_motion():
    initial_position = torch.zeros(1, 3, dtype=torch.float64)
    for direction in (-1.0, 1.0):
        trajectory = Lemniscate3DTrajectory(
            initial_position,
            semi_axis_x=1.0,
            semi_axis_y=0.6,
            vertical_amplitude=0.3,
            period=20.0,
            entry_duration=0.0,
            direction=direction,
            curve_yaw_w=0.35,
        )
        reference = trajectory.sample_at_time(3.7)
        yaw_w, yaw_rate = trajectory.horizontal_tangent_heading(
            reference.phase, reference.phase_rate
        )
        heading_xy = torch.stack((torch.cos(yaw_w), torch.sin(yaw_w)), dim=-1)
        velocity_xy = torch.nn.functional.normalize(
            reference.linear_velocity_w[..., :2], dim=-1
        )
        torch.testing.assert_close(heading_xy, velocity_xy)
        assert torch.isfinite(yaw_rate).all()


def test_subset_reset_and_closed_curve_points():
    trajectory = make_trajectory()
    points = trajectory.curve_points(121)
    assert points.shape == (2, 121, 3)
    torch.testing.assert_close(points[:, 0], points[:, -1])

    trajectory.advance(6.0)
    before = trajectory.sample()
    new_anchor = torch.tensor([[4.0, -1.0, 2.5]], dtype=torch.float64)
    trajectory.reset(
        new_anchor,
        env_ids=torch.tensor([0]),
        initial_phase=1.2,
        curve_yaw_w=-0.3,
    )
    after = trajectory.sample()
    torch.testing.assert_close(after.position_w[0], new_anchor[0])
    torch.testing.assert_close(after.position_w[1], before.position_w[1])


def test_invalid_geometry_is_rejected():
    initial_position = torch.zeros(1, 3)
    for kwargs in (
        {"semi_axis_x": 0.0},
        {"vertical_amplitude": -0.1},
        {"period": 0.0},
        {"entry_duration": -1.0},
        {"direction": 0.0},
    ):
        values = {
            "semi_axis_x": 1.0,
            "semi_axis_y": 0.6,
            "vertical_amplitude": 0.3,
            "period": 20.0,
            **kwargs,
        }
        try:
            Lemniscate3DTrajectory(initial_position, **values)
        except ValueError:
            continue
        raise AssertionError(f"invalid parameters were accepted: {kwargs}")
