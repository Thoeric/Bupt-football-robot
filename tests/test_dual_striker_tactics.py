import math

import pytest

from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    MoveIntent,
    PlayContext,
    Pose2D,
    ReadySlot,
    RobotCommand,
    RobotState,
    SetPlay,
    SoccerConfig,
)
from src.tactics.geometry import TeamFieldFrame
from src.tactics.kick_hysteresis import KickHysteresis
from src.tactics.motion import MotionController
from src.tactics.navigation import ObstacleCollector
from src.tactics.targeting.restart import (
    behind_ball_target,
    opponent_restart_target,
    own_restart_kick_target,
    own_restart_position,
    restart_kicker_role,
)


@pytest.fixture
def setup():
    config = SoccerConfig(team_id=1)
    field = TeamFieldFrame(config)
    obstacles = ObstacleCollector(config, field)
    return config, field, obstacles


@pytest.mark.parametrize(
    ("ball_y", "role"),
    [(2.2, "left_chaser"), (-2.2, "right_chaser")],
)
def test_side_restarts_use_side_forward(ball_y, role):
    ball = BallState(-6.0, ball_y)
    for set_play in (
        SetPlay.GOAL_KICK,
        SetPlay.CORNER_KICK,
        SetPlay.THROW_IN,
        SetPlay.DIRECT_FREE_KICK,
        SetPlay.INDIRECT_FREE_KICK,
    ):
        assert restart_kicker_role(set_play, ball) == role


def test_indirect_first_touch_is_infield(setup):
    config, field, _ = setup
    ball = BallState(1.0, 4.35)
    target = own_restart_kick_target(
        config, field, SetPlay.INDIRECT_FREE_KICK, ball,
    )
    assert target.x > ball.x
    assert abs(target.y) < abs(ball.y)
    assert target.x <= config.field_length / 2 - 0.35
    assert abs(target.y) <= config.field_width / 2 - 0.35


def test_opponent_half_corner_targets_penalty_spot(setup):
    config, field, _ = setup
    ball = BallState(5.8, 4.35)
    kick_target = own_restart_kick_target(
        config, field, SetPlay.CORNER_KICK, ball,
    )
    receiver = own_restart_position(
        config, field, "right_chaser", SetPlay.CORNER_KICK, ball,
    )
    assert kick_target.x == pytest.approx(4.9)
    assert kick_target.y == pytest.approx(0.0)
    assert receiver.x == pytest.approx(4.55)
    assert receiver.y == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("ball_x", "expected_x"),
    [(-4.0, 4.0), (5.8, 4.9)],
)
def test_throw_in_uses_a_two_touch_pass_target(setup, ball_x, expected_x):
    config, field, _ = setup
    ball = BallState(ball_x, 4.35)
    kick_target = own_restart_kick_target(
        config, field, SetPlay.THROW_IN, ball,
    )
    assert kick_target.x == pytest.approx(expected_x)
    assert kick_target.y == pytest.approx(0.0)


def test_restart_receiver_is_clear_of_ball_and_inside_field(setup):
    config, field, _ = setup
    ball = BallState(6.7, 4.35)
    receiver = own_restart_position(
        config, field, "right_chaser", SetPlay.CORNER_KICK, ball,
    )
    assert math.hypot(receiver.x - ball.x, receiver.y - ball.y) >= 1.20 - 1e-9
    assert receiver.x <= config.field_length / 2 - 0.35
    assert abs(receiver.y) <= config.field_width / 2 - 0.35


def test_behind_ball_uses_real_kick_line(setup):
    _, field, _ = setup
    ball = BallState(0.0, 0.0)
    target = behind_ball_target(field, ball, Pose2D(5.0, 2.0), offset=0.4)
    # The target is opposite the diagonal aim, not simply ball.x - offset.
    assert target.x < 0.0
    assert target.y < 0.0
    assert math.hypot(target.x, target.y) == pytest.approx(0.4)


def test_opponent_restart_keeps_rule_distance(setup):
    config, field, _ = setup
    ball = BallState(0.0, 0.0)
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.DIRECT_FREE_KICK,
        kicking_team=2,
    )
    context = PlayContext(
        game_state=game,
        ball=ball,
        teammates={1: RobotState(1, Pose2D(0.1, 0.0, 0.0))},
    )
    target = opponent_restart_target(
        config,
        field,
        1,
        ReadySlot.CENTER,
        context,
        lambda slot, own: Pose2D(0.0, 0.0, 0.0),
    )
    assert math.hypot(target.x - ball.x, target.y - ball.y) >= 2.15 - 1e-9
    assert config.strategy.opponent_restart_avoid_distance_m == pytest.approx(2.2)


@pytest.mark.parametrize("x", [-7.20, 7.20])
def test_goal_escape_points_toward_field(setup, x):
    _, _, obstacles = setup
    target = obstacles.goal_escape_target(Pose2D(x, 0.3, 0.0))
    assert target is not None
    assert abs(target.x) < 7.0
    assert target.y == pytest.approx(0.3)
    assert target.theta == pytest.approx(0.0 if x < 0.0 else math.pi)


def test_motion_overrides_target_when_robot_is_inside_goal(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.PLAYING),
        ball=BallState(0.0, 0.0),
        teammates={1: RobotState(1, Pose2D(7.20, 0.0, math.pi))},
    )
    command = motion.move_to_target(1, context, Pose2D(6.8, 3.0), "test")
    assert "escape goal" in command.reason
    assert isinstance(command.intent, MoveIntent)
    assert command.intent.vx > 0.0


def test_farther_robot_yields_in_close_teammate_conflict(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.PLAYING),
        ball=BallState(1.0, 0.0),
        teammates={
            1: RobotState(1, Pose2D(0.0, 0.0, 0.0)),
            2: RobotState(2, Pose2D(0.60, 0.0, 0.0)),
        },
    )
    command = motion._apply_close_teammate_yield(
        1, context, RobotCommand(MoveIntent(vx=1.0), "test"),
    )
    assert isinstance(command.intent, MoveIntent)
    assert command.intent.vx == pytest.approx(0.22)
    assert "yield teammate" in command.reason


def test_close_teammate_logic_is_safe_when_ready_has_no_ball(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.READY),
        teammates={
            1: RobotState(1, Pose2D(0.0, 0.0, 0.0)),
            2: RobotState(2, Pose2D(0.60, 0.0, 0.0)),
        },
        ball=None,
    )
    command = motion.move_to_target(1, context, Pose2D(2.0, 0.0), "ready")
    assert isinstance(command.intent, MoveIntent)


def test_agile_motion_reaches_configured_speed_and_turn_limits(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.PLAYING),
        ball=BallState(3.0, 0.0),
        teammates={1: RobotState(1, Pose2D(0.0, 0.0, 0.0))},
    )
    sprint = motion.move_to_target(1, context, Pose2D(5.0, 0.0), "sprint")
    assert isinstance(sprint.intent, MoveIntent)
    assert sprint.intent.vx == pytest.approx(config.strategy.max_linear_speed)

    turn = motion.move_to_target(1, context, Pose2D(0.0, 3.0), "turn")
    assert isinstance(turn.intent, MoveIntent)
    assert turn.intent.vyaw == pytest.approx(config.strategy.max_angular_speed)


def test_final_shot_approach_walks_and_turns_but_caps_speed(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.PLAYING),
        ball=BallState(0.50, 0.0),
        teammates={1: RobotState(1, Pose2D(0.0, 0.0, 0.0))},
    )
    target = Pose2D(math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)

    command = motion.move_to_target(
        1, context, target, "final shot", final_ball_approach=True,
    )

    assert isinstance(command.intent, MoveIntent)
    assert 0.0 < command.intent.vx <= 0.80
    assert command.intent.vyaw > 0.0


def test_final_shot_approach_skips_opponent_detour_but_keeps_teammate_avoidance(setup):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(state=GameState.PLAYING),
        ball=BallState(1.0, 0.0),
        teammates={1: RobotState(1, Pose2D(-0.1, 0.0, 0.0))},
        opponents={4: RobotState(4, Pose2D(0.25, 0.25, 0.0))},
    )
    target = Pose2D(0.55, 0.50, 0.0)

    opponent_only = motion.move_to_target(
        1, context, target, "final shot", final_ball_approach=True,
    )
    assert "via obstacle" not in opponent_only.reason

    context.teammates[2] = RobotState(2, Pose2D(0.25, 0.25, 0.0))
    teammate_blocked = motion.move_to_target(
        1, context, target, "final shot", final_ball_approach=True,
    )
    assert "via obstacle" in teammate_blocked.reason


@pytest.mark.parametrize(
    ("heading", "expected_vx"),
    [(0.0, 1.20), (math.pi, -1.00)],
)
def test_emergency_restart_escape_uses_shorter_forward_or_backward_motion(
    setup, heading, expected_vx,
):
    config, field, obstacles = setup
    motion = MotionController(
        config,
        field,
        KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5),
        obstacles,
    )
    context = PlayContext(
        game_state=GameControlState(
            state=GameState.PLAYING,
            set_play=SetPlay.DIRECT_FREE_KICK,
            kicking_team=2,
        ),
        ball=BallState(0.0, 0.0),
        teammates={1: RobotState(1, Pose2D(0.20, 0.0, heading))},
    )

    command = motion.escape_restart_ball(1, context, "emergency escape")

    assert isinstance(command.intent, MoveIntent)
    assert command.intent.vx == pytest.approx(expected_vx)
    assert command.intent.vyaw == pytest.approx(0.0)
