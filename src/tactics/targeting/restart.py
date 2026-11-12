"""Avoidance targets during opponent restarts.

When GameController marks an opponent ``set_play``, all our players must stay
outside the configured ball-avoidance radius. This module passes position targets
through constraints for radius safety, penalty-area legality, and field bounds.

Control-flow shape:
Entry :func:`opponent_restart_target` chooses a base target from normal READY
position or escape mode, then runs :func:`_apply_safety_chain`.
:func:`_apply_safety_chain` composes radius safety then penalty-area legality.
Penalty-area pushout can move a target closer to the ball, so
:func:`_goal_kick_safe` calls :func:`_radius_safe` again as a fallback.
"""

from __future__ import annotations

import math

from ...soccer_framework import (
    BallState,
    GameControlState,
    Pose2D,
    ReadySlot,
    SetPlay,
    SoccerConfig,
    PlayContext,
)
from ..geometry import TeamFieldFrame
from .predicates import sideline_sign
from .recovery import BaseReadyTarget


__all__ = [
    "behind_ball_target",
    "own_restart_kick_target",
    "own_restart_position",
    "opponent_restart_hold_vyaw",
    "opponent_restart_target",
    "restart_kicker_role",
    "safe_restart_position",
]


def restart_kicker_role(set_play: SetPlay, ball: BallState) -> str | None:
    """Return the designated dual-striker kicker for an own restart.

    Restarts on one side are taken by the forward already responsible for that
    side.  This keeps crossings short and, importantly, makes goal kicks a
    forward's job so the goalkeeper can stay in front of the goal.
    """

    if set_play == SetPlay.PENALTY_KICK:
        return "left_chaser"
    if set_play in (
        SetPlay.DIRECT_FREE_KICK,
        SetPlay.INDIRECT_FREE_KICK,
        SetPlay.THROW_IN,
        SetPlay.GOAL_KICK,
        SetPlay.CORNER_KICK,
    ):
        return "left_chaser" if ball.y >= 0.0 else "right_chaser"
    return None


def behind_ball_target(
    field: TeamFieldFrame,
    ball: BallState,
    aim: Pose2D,
    offset: float = 0.30,
) -> Pose2D:
    """Place the robot behind the ball on the actual kick line."""

    kick_theta = math.atan2(aim.y - ball.y, aim.x - ball.x)
    return field.clamp_inside_field(
        Pose2D(
            ball.x - math.cos(kick_theta) * offset,
            ball.y - math.sin(kick_theta) * offset,
            kick_theta,
        ),
        margin=0.18,
    )


def own_restart_kick_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    set_play: SetPlay,
    ball: BallState,
) -> Pose2D:
    """Return the legal and deliberately simple first-touch target.

    Throw-ins use a two-touch pass: opponent-half throws target the penalty spot,
    while own-half throws target a central receiver. Goal kicks target the
    opponent goal directly. Indirect free kicks retain a short first touch.
    """

    penalty_spot = Pose2D(
        field.opponent_goal_x() - config.penalty_dist,
        0.0,
        0.0,
    )
    side = 1.0 if ball.y >= 0.0 else -1.0
    if set_play == SetPlay.THROW_IN:
        if ball.x >= 0.0:
            return penalty_spot
        return Pose2D(4.0, 0.0, 0.0)
    if set_play == SetPlay.INDIRECT_FREE_KICK:
        return field.clamp_inside_field(
            Pose2D(ball.x + 1.10, ball.y - side * 1.15, 0.0), margin=0.35,
        )
    if set_play == SetPlay.CORNER_KICK:
        return penalty_spot
    # Goal kicks may be directed straight at the opponent's goal.
    return Pose2D(field.opponent_goal_x(), 0.0, 0.0)


def own_restart_position(
    config: SoccerConfig,
    field: TeamFieldFrame,
    role_name: str,
    set_play: SetPlay,
    ball: BallState,
) -> Pose2D:
    """Position a non-kicker for each own restart using the ball location.

    The caller also uses this for the goalkeeper.  Targets are kept inside the
    field and separated from the ball by 1.20 m to avoid teammate congestion.
    """

    if role_name == "goalkeeper":
        target = Pose2D(-6.10, max(-0.85, min(0.85, ball.y * 0.25)), 0.0)
        return field.clamp_inside_field(target, margin=0.30)

    kicker = restart_kicker_role(set_play, ball)
    side = 1.0 if ball.y >= 0.0 else -1.0
    if role_name == kicker:
        target = behind_ball_target(
            field, ball, own_restart_kick_target(config, field, set_play, ball),
        )
    elif set_play == SetPlay.GOAL_KICK:
        target = Pose2D(max(-3.8, ball.x + 2.1), -side * 1.65, 0.0)
    elif set_play == SetPlay.CORNER_KICK or (
        set_play == SetPlay.THROW_IN and ball.x >= 0.0
    ):
        # Wait 0.35 m behind the penalty spot so the pass has a clear landing.
        target = Pose2D(
            field.opponent_goal_x() - config.penalty_dist - 0.35,
            0.0,
            0.0,
        )
    elif set_play == SetPlay.PENALTY_KICK:
        target = Pose2D(2.70, -2.00, 0.0)
    elif set_play == SetPlay.THROW_IN:
        # The own-half taker plays a legal first touch to the central receiver;
        # that receiver can take the second touch and shoot.
        target = Pose2D(3.65, 0.0, 0.0)
    elif set_play == SetPlay.INDIRECT_FREE_KICK:
        pass_target = own_restart_kick_target(config, field, set_play, ball)
        target = Pose2D(pass_target.x - 0.35, pass_target.y, 0.0)
    else:  # direct free kick
        target = Pose2D(
            max(-2.5, min(5.1, ball.x + 1.45)),
            max(-3.4, min(3.4, ball.y - side * 1.45)),
            0.0,
        )

    target = field.clamp_inside_field(target, margin=0.35)
    # The inactive forward must not occupy the kicker's immediate working area.
    if role_name != kicker:
        target = _radius_safe(field, target, ball, 1.20)
    return Pose2D(
        target.x,
        target.y,
        field.face_ball_theta(target.x, target.y, ball),
    )


def safe_restart_position(
    config: SoccerConfig,
    field: TeamFieldFrame,
    target: Pose2D,
    ball: BallState,
    game: GameControlState,
    min_distance: float = 1.60,
) -> Pose2D:
    """Apply the complete field/radius/goal-kick safety chain to a target."""

    return _apply_safety_chain(
        config, field, target, ball, game, min_distance,
    )


def opponent_restart_target(
    config: SoccerConfig,
    field: TeamFieldFrame,
    player_id: int,
    slot: ReadySlot,
    context: PlayContext,
    base_ready_target: BaseReadyTarget,
) -> Pose2D:
    """Position target for this tick during an opponent restart.

    Compute ``base_target`` through two possible paths, then run :func:`_apply_safety_chain`:

    If the player is already too close to the ball, use :func:`_escape_initial_target`
    and add ``min_distance + 0.35`` buffer in escape mode.
    Otherwise use ``base_ready_target`` directly.

    See :func:`_apply_safety_chain` for safety-chain details.
    """

    base_target = base_ready_target(slot, False)
    ball = context.known_ball
    game = context.known_game

    min_distance = config.strategy.opponent_restart_avoid_distance_m
    robot = context.teammates.get(player_id)
    if robot is not None and robot.pose is not None:
        distance_to_ball = math.hypot(
            robot.pose.x - ball.x,
            robot.pose.y - ball.y,
        )
        if distance_to_ball < min_distance + 0.25:
            escape_target = _escape_initial_target(
                field, robot.pose, ball, game,
                min_distance=min_distance + 0.35,
            )
            return _apply_safety_chain(
                config, field, escape_target, ball, game,
                min_distance=min_distance + 0.35,
            )

    return _apply_safety_chain(
        config, field, base_target, ball, game, min_distance,
    )


def opponent_restart_hold_vyaw(
    config: SoccerConfig,
    player_id: int,
    game: GameControlState,
) -> float:
    """Stay fully still after reaching a legal opponent-restart position."""

    return 0.0


# Safety chain


def _apply_safety_chain(
    config: SoccerConfig,
    field: TeamFieldFrame,
    target: Pose2D,
    ball: BallState,
    game: GameControlState,
    min_distance: float,
) -> Pose2D:
    """Safety constraint chain for restart positioning.

    Order:
    1. :func:`_radius_safe` ensures target is at least ``min_distance`` from the ball.
    2. :func:`_goal_kick_safe` pushes target out of the opponent penalty area on GOAL_KICK only.

    Note: ``_goal_kick_safe`` calls ``_radius_safe`` again because lateral pushout
    can move the target closer than min_distance. That nonlinear call is a property
    of the constraints, not a control-flow bug.
    """

    target = _radius_safe(field, target, ball, min_distance)
    target = _goal_kick_safe(config, field, target, ball, game, min_distance)
    return target


# Single-step constraints


def _escape_initial_target(
    field: TeamFieldFrame,
    robot_pose: Pose2D,
    ball: BallState,
    game: GameControlState,
    min_distance: float,
) -> Pose2D:
    """Compute an initial escape target away from the ball when a player is already too close.

    Move along "ball -> current pose" out to ``min_distance``; THROW_IN adds a
    backfield bias. The target still must pass :func:`_apply_safety_chain`.
    """

    dx = robot_pose.x - ball.x
    dy = robot_pose.y - ball.y
    if math.hypot(dx, dy) <= 1e-6:
        dx = -1.0
        dy = -sideline_sign(ball.y)

    if game.set_play == SetPlay.THROW_IN:
        dx -= 0.8
        dy -= sideline_sign(ball.y) * 0.9

    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        distance = 1.0
    target_x = ball.x + dx / distance * min_distance
    target_y = ball.y + dy / distance * min_distance
    target = Pose2D(
        x=target_x,
        y=target_y,
        theta=field.face_ball_theta(target_x, target_y, ball),
    )
    if game.set_play == SetPlay.THROW_IN:
        target = Pose2D(
            field.own_half_x(target.x, margin=0.45),
            target.y,
            target.theta,
        )
    return target


def _radius_safe(
    field: TeamFieldFrame,
    target: Pose2D,
    ball: BallState,
    min_distance: float,
) -> Pose2D:
    """Ensure target is at least ``min_distance`` from the ball.

    If too close, scale along the original direction. If scaling fails because
    target is exactly on the ball, fall back to "back from ball plus opposite sideline";
    if still invalid, fall back to our half.
    """

    candidate = field.clamp_inside_field(target, margin=0.35)
    dx = candidate.x - ball.x
    dy = candidate.y - ball.y
    distance = math.hypot(dx, dy)
    if distance >= min_distance:
        return candidate

    if distance <= 1e-6:
        dx = -1.0
        dy = -sideline_sign(ball.y)
        distance = math.hypot(dx, dy)

    candidate = field.clamp_inside_field(
        Pose2D(
            x=ball.x + dx / distance * min_distance,
            y=ball.y + dy / distance * min_distance,
            theta=target.theta,
        ),
        margin=0.35,
    )
    if math.hypot(candidate.x - ball.x, candidate.y - ball.y) >= min_distance:
        return candidate

    fallback_y = ball.y - sideline_sign(ball.y) * min_distance
    fallback_x = field.own_half_x(
        ball.x - min_distance,
        margin=0.45,
    )
    return field.clamp_inside_field(
        Pose2D(
            fallback_x,
            fallback_y,
            field.face_ball_theta(fallback_x, fallback_y, ball),
        ),
        margin=0.35,
    )


def _goal_kick_safe(
    config: SoccerConfig,
    field: TeamFieldFrame,
    target: Pose2D,
    ball: BallState,
    game: GameControlState,
    min_distance: float,
) -> Pose2D:
    """During opponent goal kicks, also avoid standing inside the opponent penalty area.

    If target is inside the penalty area, push it laterally outside the area, then
    run :func:`_radius_safe` again because lateral pushout may move it closer to the ball.
    """

    if game.set_play != SetPlay.GOAL_KICK:
        return target

    goal_sign = 1.0 if ball.x >= 0.0 else -1.0
    penalty_front_x = goal_sign * (
        config.field_length / 2.0
        - config.penalty_area_length
        - 0.55
    )
    inside_penalty_x = (
        target.x >= penalty_front_x
        if goal_sign > 0.0
        else target.x <= penalty_front_x
    )
    inside_penalty_y = (
        abs(target.y) <= config.penalty_area_width / 2.0 + 0.35
    )
    if not (inside_penalty_x and inside_penalty_y):
        return target

    lateral_sign = -1.0 if ball.y >= 0.0 else 1.0
    lateral_y = lateral_sign * (
        config.penalty_area_width / 2.0 + min_distance
    )
    safe_x = penalty_front_x - goal_sign * 0.45
    candidate = field.clamp_inside_field(
        Pose2D(
            safe_x,
            lateral_y,
            field.face_ball_theta(
                safe_x,
                lateral_y,
                ball,
            ),
        ),
        margin=0.35,
    )
    if math.hypot(candidate.x - ball.x, candidate.y - ball.y) >= min_distance:
        return candidate

    return _radius_safe(field, candidate, ball, min_distance)
