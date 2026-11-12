# ruff: noqa: E402

import sys
import types
import math
from types import SimpleNamespace

import py_trees
import pytest

# The Booster runtime imports ROS at module import time.  Strategy tests only
# need the SoccerKit annotation used by playbook.py, so supply that narrow type
# before importing the play package on a non-Booster development machine.
runtime_stub = types.ModuleType("src.runtime")
runtime_stub.SoccerKit = object
sys.modules.setdefault("src.runtime", runtime_stub)

from src.play.dual_striker_playbook import DualStrikerPlaybook
from src.play.dual_striker_roles import (
    _kick_power_for_context,
    _normal_shot_power,
)
from src.play.nodes import GoSetPlayTarget
from src.play.play_subtree import create_play_subtree
from src.play.shot_execution import ShotExecution
from src.behavior_tree.safety_subtree import create_safety_subtree
from src.behavior_tree.blackboard import BlackboardClient, BlackboardKeys, cmd_key
from src.behavior_tree.nodes.conditions import IsRobotFallen
from src.soccer_framework import (
    BallState,
    GameControlState,
    GameState,
    KickIntent,
    MoveIntent,
    PlayContext,
    Pose2D,
    RobotState,
    RobotRuntimeStatus,
    SetPlay,
    SoccerConfig,
    StopIntent,
)
from src.behavior_tree.blackboard import robot_status_key
from src.tactics.geometry import TeamFieldFrame
from src.tactics.kick_hysteresis import KickHysteresis
from src.tactics.motion import MotionController
from src.tactics.navigation import ObstacleCollector
from src.tactics.ready_stance import ReadyStance
from src.tactics.targeting import Targeting


def make_playbook():
    config = SoccerConfig(team_id=1)
    field = TeamFieldFrame(config)
    obstacles = ObstacleCollector(config, field)
    kicker = KickHysteresis(enter=2.5, exit=3.0, exit_delay=1.5)
    kit = SimpleNamespace(
        config=config,
        field=field,
        obstacles=obstacles,
        targeting=Targeting(config, field, obstacles),
        ready_stance=ReadyStance(config, field),
        kicker=kicker,
        motion=MotionController(config, field, kicker, obstacles),
        is_player_allowed=lambda game, player_id: game.is_active_player(1, player_id),
    )
    return DualStrikerPlaybook(kit), kit


def context(game, x=0.0, y=0.0):
    return PlayContext(game_state=game, ball=BallState(x, y))


def test_base_role_mapping_is_unchanged():
    playbook, _ = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    assignment = playbook.assign_roles(context(game))
    assert dict(assignment.by_player) == {
        1: "right_chaser",
        2: "left_chaser",
        3: "goalkeeper",
    }


def test_complete_play_subtree_builds_with_all_three_roles():
    playbook, kit = make_playbook()
    tree = create_play_subtree(kit, playbook)
    assert tree.name == "PlayingPhase"
    assert [role.name for role in playbook.role_registry] == [
        "left_chaser",
        "right_chaser",
        "goalkeeper",
    ]


def test_active_forward_has_dead_band():
    playbook, _ = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    playbook.assign_roles(context(game, y=0.00))
    assert playbook.active_striker() == "right_chaser"
    playbook.assign_roles(context(game, y=1.30))
    assert playbook.active_striker() == "left_chaser"
    playbook.assign_roles(context(game, y=0.90))
    assert playbook.active_striker() == "left_chaser"
    playbook.assign_roles(context(game, y=0.50))
    assert playbook.active_striker() == "right_chaser"

    fresh_playbook, _ = make_playbook()
    fresh_playbook.assign_roles(context(game, y=1.00))
    assert fresh_playbook.active_striker() == "left_chaser"


def test_kickoff_waits_then_releases_right_forward():
    playbook, kit = make_playbook()
    waiting = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.NONE,
        kicking_team=1,
        secondary_time=10,
    )
    waiting_context = context(waiting)
    playbook.assign_roles(waiting_context)
    assert playbook.is_own_kickoff(waiting_context)
    assert not playbook._right_role.wants_to_kick(kit, 1, waiting_context)

    released = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.NONE,
        kicking_team=255,
        secondary_time=0,
    )
    released_context = context(released)
    playbook.assign_roles(released_context)
    assert playbook.is_kickoff_cannon_released(released_context)
    assert playbook._right_role.wants_to_kick(kit, 1, released_context)
    assert not playbook._left_role.wants_to_kick(kit, 2, released_context)


def test_goal_kick_uses_side_forward_not_goalkeeper():
    playbook, _ = make_playbook()
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.GOAL_KICK,
        kicking_team=1,
    )
    left = context(game, x=-6.0, y=1.8)
    assert playbook.is_set_play_kicker("left_chaser", left)
    assert not playbook.is_set_play_kicker("goalkeeper", left)
    right = context(game, x=-6.0, y=-1.8)
    assert playbook.is_set_play_kicker("right_chaser", right)


def test_opponent_set_play_table_always_passes_safety_chain():
    playbook, kit = make_playbook()
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.DIRECT_FREE_KICK,
        kicking_team=2,
    )
    restart_context = context(game, x=6.6, y=4.2)
    for role in ("left_chaser", "right_chaser", "goalkeeper"):
        target = playbook.get_set_play_target(role, restart_context)
        assert math.hypot(target.x - 6.6, target.y - 4.2) >= 2.00 - 1e-9
        assert abs(target.x) <= kit.config.field_length / 2 - 0.35 + 1e-9
        assert abs(target.y) <= kit.config.field_width / 2 - 0.35 + 1e-9


def test_goalkeeper_stays_home_until_ball_enters_danger_zone():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    safe = context(game, x=-2.0, y=2.0)
    safe_target = playbook._gk_role.target(kit, 3, safe)
    assert safe_target.x == -6.10
    assert abs(safe_target.y) <= 0.90

    danger = context(game, x=-5.2, y=1.0)
    danger_target = playbook._gk_role.target(kit, 3, danger)
    assert danger_target.x > -6.82
    assert danger_target.x <= -4.15
    assert abs(danger_target.y) <= 2.45


def test_non_owner_waits_outside_the_near_ball_inactivity_radius():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    for ball_x in (0.0, 4.0, 6.5):
        play_context = context(game, x=ball_x, y=1.4)
        playbook.assign_roles(play_context)
        target = playbook._right_role.target(kit, 1, play_context)
        assert math.hypot(target.x - ball_x, target.y - 1.4) >= 3.20


def test_forwards_receive_ahead_of_goalkeeper_zone():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=-4.0, y=0.0)
    playbook.assign_roles(play_context)
    left = playbook._left_role.target(kit, 2, play_context)
    right = playbook._right_role.target(kit, 1, play_context)
    assert left.x == pytest.approx(-0.7)
    assert right.x == pytest.approx(-0.7)
    assert left.y == pytest.approx(1.8)
    assert right.y == pytest.approx(-1.8)
    assert math.hypot(left.x + 4.0, left.y) >= 3.20
    assert math.hypot(right.x + 4.0, right.y) >= 3.20
    assert not playbook._left_role.wants_to_kick(kit, 2, play_context)
    assert not playbook._right_role.wants_to_kick(kit, 1, play_context)


def test_shot_aim_ignores_small_keeper_offset():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=0.0, y=0.6)
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, 0.45, 0.0))
    target = playbook._left_role.kick_target(kit, 2, play_context)
    assert target.y == 0.0
    assert target.x == 7.0


def test_unblocked_forwards_prefer_goal_centre():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    left_context = context(game, x=2.0, y=2.0)
    right_context = context(game, x=2.0, y=-2.0)
    assert playbook._left_role.kick_target(kit, 2, left_context).y == 0.0
    assert playbook._right_role.kick_target(kit, 1, right_context).y == 0.0


@pytest.mark.parametrize(
    ("keeper_y", "expected_y"),
    [(0.45, 0.0), (-0.45, 0.0), (0.70, -0.20), (-0.70, 0.20)],
)
def test_close_shot_aims_away_from_goalkeeper(keeper_y, expected_y):
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=5.0, y=0.0)
    play_context.opponents[4] = RobotState(
        4, Pose2D(6.25, keeper_y, 0.0),
    )

    target = playbook._right_role.kick_target(kit, 1, play_context)

    assert target.y == pytest.approx(expected_y)


@pytest.mark.parametrize(
    "ball_x",
    [5.0, 3.0, 0.0, -2.0, -6.0],
)
def test_shot_defaults_to_goal_centre_at_every_distance(ball_x):
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=ball_x, y=0.0)
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, 0.4, 0.0))

    target = playbook._right_role.kick_target(kit, 1, play_context)

    assert target.y == 0.0
    assert target.x == pytest.approx(7.0)


def test_shot_target_locks_inside_final_2_metres():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=0.0, y=0.0)
    play_context.teammates[1] = RobotState(1, Pose2D(-0.8, 0.0, 0.0))
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, 0.45, 0.0))

    locked = playbook._right_role.kick_target(kit, 1, play_context)
    assert locked.y == 0.0

    play_context.opponents.clear()
    still_locked = playbook._right_role.kick_target(kit, 1, play_context)
    assert still_locked == locked

    play_context.ball = BallState(0.5, 0.0)
    unlocked = playbook._right_role.kick_target(kit, 1, play_context)
    assert unlocked.y == 0.0


def test_diagonal_shot_uses_lane_geometry_instead_of_keeper_side():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=3.0, y=2.0)
    # The keeper is above centre but below the diagonal shot line.  A positive
    # target passes farther from the keeper than the old negative-side rule.
    play_context.opponents[4] = RobotState(4, Pose2D(5.0, 0.8, 0.0))

    target = playbook._left_role.kick_target(kit, 2, play_context)

    assert target.y == pytest.approx(-0.20)


def test_goalkeeper_position_selects_the_opposite_side_lane():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=0.0)
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, 1.2, 0.0))

    assert playbook._right_role.kick_target(kit, 1, play_context).y == -0.20


def test_midfield_blocker_counts_without_a_goalkeeper():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=0.0)
    play_context.opponents[4] = RobotState(4, Pose2D(3.0, 0.0, 0.0))

    # Aim remains stable at the centre; the robot does not invent a side lane.
    assert playbook._right_role.kick_target(kit, 1, play_context).y == 0.0


def test_teammate_blocks_lane_but_shooter_does_not():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=0.0)
    play_context.teammates[1] = RobotState(1, Pose2D(-0.8, 0.0, 0.0))
    play_context.teammates[2] = RobotState(2, Pose2D(4.0, 0.0, 0.0))

    assert playbook._right_role.kick_target(kit, 1, play_context).y == 0.0


def test_small_clearance_difference_prefers_centre():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=0.0)
    play_context.opponents[4] = RobotState(4, Pose2D(0.5, 0.0, 0.0))

    # Near the ball, the narrow candidate rays differ by under 3 cm.
    assert playbook._right_role.kick_target(kit, 1, play_context).y == 0.0


@pytest.mark.parametrize(
    ("ball_x", "expected_power"),
    [(5.0, 5.0), (3.0, 7.0), (0.0, 9.0), (-2.0, 10.0), (-6.0, 10.0)],
)
def test_normal_shot_power_uses_distance_bands(ball_x, expected_power):
    _, kit = make_playbook()
    ball = BallState(ball_x, 0.0)
    target = Pose2D(7.0, 0.0, 0.0)

    assert _normal_shot_power(kit, ball, target) == expected_power


def test_shot_power_locks_with_target_during_final_approach():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=1.9, y=0.0)
    play_context.teammates[1] = RobotState(1, Pose2D(1.1, 0.0, 0.0))

    assert playbook._right_role._shot_lock.power(kit, 1, play_context) == 9.0

    play_context.ball = BallState(2.2, 0.0)
    assert playbook._right_role._shot_lock.power(kit, 1, play_context) == 9.0

    play_context.ball = BallState(2.4, 0.0)
    assert playbook._right_role._shot_lock.power(kit, 1, play_context) == 7.0


def test_near_opponent_caps_only_a_normal_long_shot():
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=-2.0)
    play_context.opponents[4] = RobotState(4, Pose2D(-1.2, 0.0, 0.0))

    assert playbook._right_role._shot_lock.power(kit, 1, play_context) == 7.0


@pytest.mark.parametrize(
    ("opponent_distance", "expected_power"),
    [(0.80, 7.0), (1.20, 8.0), (1.70, 10.0)],
)
def test_long_shot_power_uses_two_simple_pressure_caps(
    opponent_distance, expected_power,
):
    _, kit = make_playbook()
    ball = BallState(-2.0, 0.0)
    play_context = context(GameControlState(state=GameState.PLAYING), x=ball.x)
    play_context.opponents[4] = RobotState(
        4, Pose2D(ball.x + opponent_distance, ball.y, 0.0),
    )

    power = _normal_shot_power(
        kit, ball, Pose2D(7.0, 0.0, 0.0), play_context,
    )

    assert power == expected_power


def test_extreme_wide_ball_uses_penalty_spot_cutback_and_receiver():
    playbook, kit = make_playbook()
    play_context = context(
        GameControlState(state=GameState.PLAYING), x=3.0, y=3.3,
    )
    playbook.assign_roles(play_context)

    target = playbook._left_role.kick_target(kit, 2, play_context)
    power = playbook._left_role._shot_lock.power(kit, 2, play_context)
    receiver = playbook._right_role.target(kit, 1, play_context)

    assert target == Pose2D(4.9, 0.0, 0.0)
    assert power == 2.5
    assert receiver.x == pytest.approx(4.55)
    assert receiver.y == 0.0


def test_almost_out_ball_keeps_chaser_away_from_restart_point():
    playbook, kit = make_playbook()
    play_context = context(
        GameControlState(state=GameState.PLAYING), x=1.0, y=4.31,
    )
    playbook.assign_roles(play_context)

    target = playbook._left_role.target(kit, 2, play_context)

    assert target.y == pytest.approx(2.20)


def test_goalkeeper_clearance_is_straight_forward():
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING)
    play_context = context(game, x=-4.0, y=2.3)
    target = playbook._gk_role.kick_target(kit, 3, play_context)
    assert target.x == 7.0
    assert target.y == 2.3


@pytest.mark.parametrize(
    ("set_play", "expected"),
    [
        (SetPlay.GOAL_KICK, 10.0),
        (SetPlay.THROW_IN, 2.5),
        (SetPlay.CORNER_KICK, 2.0),
        (SetPlay.INDIRECT_FREE_KICK, 1.5),
        (SetPlay.DIRECT_FREE_KICK, 6.5),
        (SetPlay.PENALTY_KICK, 6.5),
    ],
)
def test_restart_kick_power_table(set_play, expected):
    playbook, _ = make_playbook()
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=set_play,
        kicking_team=1,
    )
    assert _kick_power_for_context(playbook, context(game), 6.5) == expected


def test_throw_in_power_matches_first_pass_target_and_field_half():
    playbook, _ = make_playbook()
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.THROW_IN,
        kicking_team=1,
    )
    assert _kick_power_for_context(
        playbook, context(game, x=4.9, y=2.5), 6.5,
    ) == 2.0
    assert _kick_power_for_context(
        playbook, context(game, x=-4.0, y=4.4), 6.5,
    ) == 7.0


@pytest.fixture
def shot_clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("src.play.shot_execution.time.monotonic", lambda: clock.now)
    monkeypatch.setattr(
        "src.play.dual_striker_playbook.time.monotonic", lambda: clock.now,
    )
    return clock


def test_front_restart_receiver_waits_then_owns_ball_for_two_seconds(shot_clock):
    playbook, kit = make_playbook()
    restart_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.CORNER_KICK,
        kicking_team=1,
    )
    restart_context = context(restart_game, x=6.9, y=4.4)
    playbook.assign_roles(restart_context)
    assert playbook._right_role.target(kit, 1, restart_context).x == pytest.approx(4.55)

    released_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.NONE,
        kicking_team=255,
    )
    before_arrival = context(released_game, x=6.5, y=3.8)
    playbook.assign_roles(before_arrival)
    assert playbook.restart_receiver_role(before_arrival) == "right_chaser"
    assert not playbook._right_role.wants_to_kick(kit, 1, before_arrival)
    assert not playbook._left_role.wants_to_kick(kit, 2, before_arrival)
    assert playbook._right_role.target(kit, 1, before_arrival).x == pytest.approx(4.55)

    # A travelling pass does not consume the receiver's shooting window.
    shot_clock.now += 3.0
    in_transit = context(released_game, x=5.8, y=2.5)
    playbook.assign_roles(in_transit)
    assert playbook.restart_receiver_role(in_transit) == "right_chaser"
    assert not playbook._right_role.wants_to_kick(kit, 1, in_transit)

    arrived = context(released_game, x=4.9, y=1.0)
    playbook.assign_roles(arrived)
    assert playbook._right_role.wants_to_kick(kit, 1, arrived)
    assert not playbook._left_role.wants_to_kick(kit, 2, arrived)

    shot_clock.now += 2.01
    playbook.assign_roles(arrived)
    assert playbook.restart_receiver_role(arrived) is None


def test_own_half_throw_receiver_takes_second_touch_at_central_pass(shot_clock):
    playbook, kit = make_playbook()
    restart_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.THROW_IN,
        kicking_team=1,
    )
    restart_context = context(restart_game, x=-4.0, y=4.35)
    playbook.assign_roles(restart_context)
    assert playbook._right_role.target(kit, 1, restart_context).x == pytest.approx(3.65)

    released_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.NONE,
        kicking_team=255,
    )
    before_arrival = context(released_game, x=1.0, y=1.0)
    playbook.assign_roles(before_arrival)
    assert playbook.restart_receiver_role(before_arrival) == "right_chaser"
    assert not playbook._right_role.wants_to_kick(kit, 1, before_arrival)

    arrived = context(released_game, x=4.0, y=0.0)
    playbook.assign_roles(arrived)
    assert playbook.restart_receiver_role(arrived) == "right_chaser"
    assert playbook._right_role.wants_to_kick(kit, 1, arrived)


def test_front_restart_receiver_finishes_committed_shot_after_window(shot_clock):
    playbook, _ = make_playbook()
    restart_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.CORNER_KICK,
        kicking_team=1,
    )
    playbook.assign_roles(context(restart_game, x=6.9, y=4.4))

    released_game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.NONE,
        kicking_team=255,
    )
    arrived = context(released_game, x=4.9, y=1.0)
    shot_theta = math.atan2(-1.0, 2.1)
    arrived.teammates[1] = RobotState(
        1,
        Pose2D(
            4.9 - 0.5 * math.cos(shot_theta),
            1.0 - 0.5 * math.sin(shot_theta),
            shot_theta,
        ),
    )
    playbook.assign_roles(arrived)
    assert playbook._right_role._shot_execution.ready(
        1, arrived, Pose2D(7.0, 0.0, 0.0),
    )

    shot_clock.now += 2.01
    playbook.assign_roles(arrived)
    assert playbook.restart_receiver_role(arrived) == "right_chaser"

    ball_gone = context(released_game, x=6.9, y=4.4)
    ball_gone.teammates[1] = arrived.teammates[1]
    assert not playbook._right_role._shot_execution.ready(
        1, ball_gone, Pose2D(7.0, 0.0, 0.0),
    )
    playbook.assign_roles(ball_gone)
    assert playbook.restart_receiver_role(ball_gone) is None


@pytest.mark.parametrize(
    ("pose", "ready"),
    [
        (Pose2D(-2.0, 0.0, 0.0), False),
        (Pose2D(-0.6, 0.0, 0.0), True),
        (Pose2D(0.6, 0.0, 0.0), False),
        (Pose2D(-0.6, 0.5, 0.0), False),
        (Pose2D(-0.6, 0.0, math.pi / 2), False),
        (Pose2D(-0.3, 0.05, math.radians(15)), True),
        (Pose2D(-0.6, 0.0, math.radians(20)), True),
        (Pose2D(-0.7, 0.0, 0.0), True),
        (Pose2D(0.10, 0.0, 0.0), False),
    ],
)
def test_shot_handover_needs_rough_alignment_not_perfect_aim(pose, ready, shot_clock):
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, pose)

    assert ShotExecution().ready(1, play_context, Pose2D(7.0, 0.0)) is ready


def test_pressure_allows_a_faster_but_still_bounded_handover(shot_clock):
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(
        1, Pose2D(-0.70, 0.20, math.radians(24.0)),
    )
    play_context.opponents[4] = RobotState(4, Pose2D(0.80, 0.0, 0.0))

    assert ShotExecution().ready(1, play_context, Pose2D(7.0, 0.0))


def test_opponent_restart_stops_immediately_once_2_2m_clear():
    playbook, kit = make_playbook()
    game = GameControlState(
        state=GameState.PLAYING,
        set_play=SetPlay.DIRECT_FREE_KICK,
        kicking_team=2,
    )
    play_context = context(game)
    play_context.teammates[1] = RobotState(1, Pose2D(2.20, 0.0, 0.0))
    blackboard = BlackboardClient(name="legal restart stop")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    blackboard.write(BlackboardKeys.ROLES, playbook.assign_roles(play_context))
    node = GoSetPlayTarget(kit, playbook, 1)

    node.tick_once()
    command = blackboard.read(cmd_key(1))

    assert isinstance(command.intent, StopIntent)
    assert "legal and still" in command.reason


def test_stopped_flag_still_stops_all_players():
    _, kit = make_playbook()
    play_context = context(
        GameControlState(state=GameState.PLAYING, stopped=True),
    )
    blackboard = BlackboardClient(name="stopped regression")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    tree = create_safety_subtree(kit)

    tree.tick_once()

    for player_id in kit.config.player_ids:
        command = blackboard.read(cmd_key(player_id))
        assert isinstance(command.intent, StopIntent)
        assert command.reason == "game stopped"


def test_fallen_robot_waits_to_get_up_until_stopped_phase_ends():
    play_context = context(
        GameControlState(state=GameState.PLAYING, stopped=True),
    )
    blackboard = BlackboardClient(name="fallen during stopped phase")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    blackboard.write(
        robot_status_key(1),
        RobotRuntimeStatus(mode="damping", fall_down_state="fallen"),
    )

    condition = IsRobotFallen(1)
    condition.tick_once()

    assert condition.status == py_trees.common.Status.FAILURE


def test_diagonal_handover_uses_shot_line_not_field_x_axis(shot_clock):
    play_context = context(GameControlState(state=GameState.PLAYING), x=2.0, y=2.0)
    theta = math.atan2(-2.0, 5.0)
    play_context.teammates[1] = RobotState(
        1, Pose2D(2.0 - 0.6 * math.cos(theta), 2.0 - 0.6 * math.sin(theta), theta),
    )

    assert ShotExecution().ready(1, play_context, Pose2D(7.0, 0.0))


def test_running_shot_does_not_recheck_heading_each_frame(shot_clock):
    attempt = ShotExecution()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(-0.6, 0.0, 0.0))
    assert attempt.ready(1, play_context, Pose2D(7.0, 0.0))

    shot_clock.now += 0.1
    play_context.teammates[1] = RobotState(1, Pose2D(-0.7, 0.0, math.radians(40)))
    assert attempt.ready(1, play_context, Pose2D(7.0, 0.0))


def test_stalled_shot_releases_then_retries(shot_clock):
    _, kit = make_playbook()
    attempt = ShotExecution()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(-0.6, 0.0, 0.0))
    aim = Pose2D(7.0, 0.0)
    assert attempt.ready(1, play_context, aim)
    for _ in range(3):
        shot_clock.now += 0.5
        assert attempt.ready(1, play_context, aim)

    shot_clock.now += 0.5
    assert not attempt.ready(1, play_context, aim)
    assert attempt.phase == "shot retry"
    assert attempt.approach_target(kit, 1, play_context, aim).x == pytest.approx(-0.60)
    shot_clock.now += 0.2
    assert not attempt.ready(1, play_context, aim)
    shot_clock.now += 0.2
    assert attempt.ready(1, play_context, aim)


def test_progress_extends_attempt_but_total_duration_is_bounded(shot_clock):
    attempt = ShotExecution()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(-0.6, 0.0, 0.0))
    aim = Pose2D(7.0, 0.0)
    assert attempt.ready(1, play_context, aim)
    for tick in range(1, 7):
        shot_clock.now += 0.5
        # Continuous nearby movement prevents the 1.8 s stall timeout.
        play_context.ball = BallState(0.10 * tick, 0.0)
        play_context.teammates[1] = RobotState(1, Pose2D(0.10 * tick - 0.6, 0.0, 0.0))
        assert attempt.ready(1, play_context, aim)
    # The independent 4 s ceiling still terminates a continually moving try.
    shot_clock.now += 0.5
    assert attempt.ready(1, play_context, aim)
    shot_clock.now += 0.5
    assert not attempt.ready(1, play_context, aim)
    assert attempt.phase == "shot retry"


def test_shot_releases_when_ball_leaves_and_after_a_phase_gap(shot_clock):
    attempt = ShotExecution()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(-0.6, 0.0, 0.0))
    aim = Pose2D(7.0, 0.0)
    assert attempt.ready(1, play_context, aim)
    shot_clock.now += 0.1
    play_context.ball = BallState(2.0, 0.0)
    assert not attempt.ready(1, play_context, aim)
    play_context.ball = BallState(0.0, 0.0)
    assert attempt.ready(1, play_context, aim)
    shot_clock.now += 10.0
    play_context.teammates[1] = RobotState(1, Pose2D(-0.6, 0.0, math.pi))
    assert not attempt.ready(1, play_context, aim)
    assert attempt.phase == "shot approach"


def test_front_approach_goes_around_ball(shot_clock):
    _, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(0.7, 0.0, math.pi))
    target = ShotExecution().approach_target(kit, 1, play_context, Pose2D(7.0, 0.0))
    assert target.x < 0.0
    assert abs(target.y) >= 0.6


def test_central_goalmouth_approaches_directly_from_field_side(shot_clock):
    _, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=5.5, y=0.2)
    target = ShotExecution().approach_target(
        kit, 1, play_context, Pose2D(7.0, 0.0),
    )
    assert target.x == pytest.approx(5.05)
    assert target.y == pytest.approx(0.2)


def test_goalmouth_shortcut_falls_back_when_lateral_error_is_too_large(shot_clock):
    _, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), x=6.9, y=0.9)
    target = ShotExecution().approach_target(
        kit, 1, play_context, Pose2D(7.0, 0.0),
    )
    assert target.y > 0.9


def test_shot_plan_does_not_expire_mid_preparation(shot_clock):
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING))
    play_context.teammates[1] = RobotState(1, Pose2D(-0.8, 0.0, 0.0))
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, 0.80, 0.0))
    target = playbook._right_role.kick_target(kit, 1, play_context)
    assert target.y < 0.0
    play_context.opponents[4] = RobotState(4, Pose2D(6.25, -0.80, 0.0))
    shot_clock.now += 2.0
    assert playbook._right_role.kick_target(kit, 1, play_context) == target
    play_context.teammates[1] = RobotState(1, Pose2D(-2.5, 0.0, 0.0))
    assert playbook._right_role.kick_target(kit, 1, play_context).y > 0.0


@pytest.mark.parametrize(("player_id", "ball_y"), [(1, 0.0), (2, 1.5)])
def test_forward_tree_approaches_before_handover_then_issues_kick(player_id, ball_y, shot_clock):
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), y=ball_y)
    play_context.teammates[player_id] = RobotState(player_id, Pose2D(-2.0, ball_y, 0.0))
    blackboard = BlackboardClient(name="shot execution test")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    blackboard.write(BlackboardKeys.ROLES, playbook.assign_roles(play_context))
    blackboard.write(BlackboardKeys.NOW, shot_clock.now)
    role = playbook._right_role if player_id == 1 else playbook._left_role
    tree = role.build_subtree(kit, player_id)
    tree.tick_once()
    assert isinstance(blackboard.read(cmd_key(player_id)).intent, MoveIntent)

    theta = math.atan2(-ball_y, 7.0)
    play_context.teammates[player_id] = RobotState(
        player_id, Pose2D(-0.55 * math.cos(theta), ball_y - 0.55 * math.sin(theta), theta),
    )
    shot_clock.now += 0.1
    blackboard.write(BlackboardKeys.NOW, shot_clock.now)
    tree.tick_once()
    command = blackboard.read(cmd_key(player_id))
    assert isinstance(command.intent, KickIntent)
    assert command.intent.power == 9.0


def test_restart_tree_retains_old_handover_and_power(shot_clock):
    playbook, kit = make_playbook()
    game = GameControlState(state=GameState.PLAYING, set_play=SetPlay.THROW_IN, kicking_team=1)
    play_context = context(game, x=2.0, y=4.45)
    play_context.teammates[2] = RobotState(2, Pose2D(2.0, 3.0, 0.0))
    blackboard = BlackboardClient(name="restart regression test")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    blackboard.write(BlackboardKeys.ROLES, playbook.assign_roles(play_context))
    blackboard.write(BlackboardKeys.NOW, shot_clock.now)
    tree = playbook._left_role.build_subtree(kit, 2)
    tree.tick_once()
    command = blackboard.read(cmd_key(2))
    assert isinstance(command.intent, KickIntent)
    assert command.intent.power == 2.5


@pytest.mark.parametrize(
    ("ball_xy", "pose"),
    [
        ((0.0, 0.0), Pose2D(-3.0, 0.0, 0.0)),
        ((0.0, 0.0), Pose2D(0.8, 0.0, math.pi)),
        ((0.0, 0.0), Pose2D(-0.3, 0.7, -math.pi / 2)),
        ((2.0, 1.5), Pose2D(0.0, 3.0, 0.0)),
        ((-2.05, 0.0), Pose2D(-1.5, -0.3, math.pi)),
        ((6.7, -1.0), Pose2D(5.0, -2.0, 0.0)),
    ],
)
def test_closed_loop_approach_reaches_handover_without_waiting_forever(ball_xy, pose, shot_clock):
    # A point-chassis check for the strategy's approach, not a Booster kick or
    # accuracy simulation.  It catches unreachable guards and approach loops.
    playbook, kit = make_playbook()
    play_context = context(GameControlState(state=GameState.PLAYING), *ball_xy)
    player_id = 2 if ball_xy[1] >= 0.90 else 1
    blackboard = BlackboardClient(name="closed loop shot approach")
    blackboard.write(BlackboardKeys.PLAY_CONTEXT, play_context)
    blackboard.write(BlackboardKeys.ROLES, playbook.assign_roles(play_context))
    role = playbook._left_role if player_id == 2 else playbook._right_role
    tree = role.build_subtree(kit, player_id)
    dt = 0.05
    for _ in range(250):
        play_context.teammates[player_id] = RobotState(player_id, pose)
        blackboard.write(BlackboardKeys.NOW, shot_clock.now)
        tree.tick_once()
        intent = blackboard.read(cmd_key(player_id)).intent
        if isinstance(intent, KickIntent):
            break
        if isinstance(intent, MoveIntent):
            pose = Pose2D(
                pose.x + intent.vx * math.cos(pose.theta) * dt,
                pose.y + intent.vx * math.sin(pose.theta) * dt,
                pose.theta + intent.vyaw * dt,
            )
        shot_clock.now += dt
    else:
        pytest.fail(f"approach never handed over at ball={ball_xy}, pose={pose}")
