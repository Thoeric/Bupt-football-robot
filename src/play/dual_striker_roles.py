"""Dual Striker role implementations per the tactical spec.

Three roles with clear responsibilities:

Goalkeeper — robot3, covering the rear 35% of the field (x <= -2.1)
  - Standby: hold near (-6.1, 0) when the ball is outside its zone.
  - Rush: ball enters its zone → charge and clear exactly along +x.
  - SetPlay: use Playbook positioning table; kick only when designated.

LeftChaser — robot2, covers the upper/left 40% (y >= +0.9)
  - Chase: approach behind the ball and select the clearest narrow shot lane.
  - Support: wait one metre behind the ball on y=+0.9.
  - SetPlay: use Playbook positioning table; kick only when designated.

RightChaser — robot1, covers the lower/right 60% (y < +0.9)
  - Chase: approach behind the ball and select the clearest narrow shot lane.
  - Support: wait one metre behind the ball on y=+0.9.

Forwards stay ahead of x=-2.1; goalkeeper has priority behind that line.
§6.2 sent-off handling: when a forward is alone they cover the full field.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import py_trees

from ..behavior_tree.blackboard import BlackboardKeys, BlackboardClient
from ..soccer_framework import BallState, PlayContext, Pose2D, SetPlay
from .nodes import AttackSubtreeConfig, build_attack_subtree
from .playbook import RoleAssignment
from .role import RoleStrategy
from .shot_execution import ShotExecution

if TYPE_CHECKING:
    from ..runtime import SoccerKit
    from .dual_striker_playbook import DualStrikerPlaybook


# ---------------------------------------------------------------------------
# Constants (§3, §4)
# ---------------------------------------------------------------------------

_APPROACH_OFFSET = 0.2          # Distance behind ball when chasing.
_GK_ZONE_FRONT_X = -2.10        # 35% of a 14 m field from our x=-7 goal line.
_GK_DANGER_X = _GK_ZONE_FRONT_X
_GK_MAX_X = _GK_ZONE_FRONT_X
_GK_STANDBY_X = -6.10           # Goalkeeper stays close enough to cover the goal.
_GK_MAX_Y = 4.30                # The 35% goalkeeper zone spans the field width.
_FORWARD_MIN_X = -1.95          # Keep forwards just ahead of the goalkeeper zone.
_FORWARD_BOUNDARY_Y = 0.90      # Left 40% / right 60% boundary on a 9 m field.
_FORWARD_MAX_TARGET_X = 6.50    # Never walk a forward into the goal mouth/net.
_SHOT_SIDE_OFFSET = 0.20        # Small keeper-side offset; centre is the default.
_WIDE_CROSS_MIN_X = 2.50
_WIDE_CROSS_MIN_ABS_Y = 3.20
_WIDE_CROSS_PRESSURE_M = 0.90
_SIDELINE_DANGER_Y = 4.30
_SIDELINE_RETREAT_Y = 2.20

# Goal posts (opponent and own).
_GOAL_POSTS = (
    (7.0, 1.3),
    (7.0, -1.3),
    (-7.0, 1.3),
    (-7.0, -1.3),
)
_GOAL_POST_CLEARANCE = 0.5     # Min distance from any post.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avoid_goal_posts(x: float, y: float) -> tuple[float, float]:
    """Push *(x, y)* radially away from any goal post within 0.5 m."""
    for px, py in _GOAL_POSTS:
        dx = x - px
        dy = y - py
        d = math.hypot(dx, dy)
        if d < _GOAL_POST_CLEARANCE and d > 1e-6:
            scale = _GOAL_POST_CLEARANCE / d
            x = px + dx * scale
            y = py + dy * scale
    return x, y


def _clamp_out_of_own_half(x: float, y: float) -> tuple[float, float]:
    """Keep normal forward targets outside the goalkeeper's 35% zone."""
    if x < _FORWARD_MIN_X:
        x = _FORWARD_MIN_X
    return x, y


def _clamp_to_field(kit: "SoccerKit", x: float, y: float) -> tuple[float, float]:
    """Clamp to field boundaries with small margin."""
    hw = kit.config.field_width / 2.0 - 0.20
    hl = kit.config.field_length / 2.0 - 0.20
    return (
        max(-hl, min(_FORWARD_MAX_TARGET_X, x)),
        max(-hw, min(hw, y)),
    )


def _best_shot_target(
    kit: "SoccerKit",
    player_id: int,
    context: PlayContext,
) -> Pose2D:
    """Aim at goal centre, opening a small opposite lane for a clearly offset keeper."""

    goal_x = kit.field.opponent_goal_x()
    centre = Pose2D(goal_x, 0.0, 0.0)
    opponents = kit.obstacles.opponent_obstacles(context)
    keeper = min(
        opponents,
        key=lambda obstacle: math.hypot(goal_x - obstacle.x, obstacle.y),
        default=None,
    )
    if keeper is None or abs(keeper.y) < 0.55:
        return centre
    return Pose2D(goal_x, -math.copysign(_SHOT_SIDE_OFFSET, keeper.y), 0.0)


def _nearest_opponent_to_ball(context: PlayContext, ball: BallState) -> float:
    distances = [
        math.hypot(opponent.pose.x - ball.x, opponent.pose.y - ball.y)
        for opponent in context.opponents.values()
        if opponent.pose is not None
    ]
    return min(distances, default=math.inf)


def _use_wide_cross(kit: "SoccerKit", context: PlayContext) -> bool:
    """Replace a low-percentage extreme-angle shot with one simple cut-back."""

    ball = context.known_ball
    return (
        ball.x >= _WIDE_CROSS_MIN_X
        and abs(ball.y) >= _WIDE_CROSS_MIN_ABS_Y
        and _nearest_opponent_to_ball(context, ball) > _WIDE_CROSS_PRESSURE_M
    )


def _wide_cross_target(kit: "SoccerKit") -> Pose2D:
    return Pose2D(
        kit.field.opponent_goal_x() - kit.config.penalty_dist,
        0.0,
        0.0,
    )


def _normal_kick_target(
    kit: "SoccerKit",
    player_id: int,
    context: PlayContext,
) -> Pose2D:
    if _use_wide_cross(kit, context):
        return _wide_cross_target(kit)
    return _best_shot_target(kit, player_id, context)


def _normal_shot_power(
    kit: "SoccerKit",
    ball: BallState,
    target: Pose2D,
    context: PlayContext | None = None,
) -> float:
    """Return the five-band power, reduced only when pressure demands speed."""

    if context is not None and _use_wide_cross(kit, context):
        return 2.5

    in_penalty_area = (
        ball.x >= kit.field.opponent_goal_x() - kit.config.penalty_area_length
        and abs(ball.y) <= kit.config.penalty_area_width / 2.0
    )
    distance = math.hypot(kit.field.opponent_goal_x() - ball.x, ball.y)
    if in_penalty_area or distance <= 3.0:
        power = 5.0
    elif distance <= 5.0:
        power = 7.0
    elif distance <= 7.5:
        power = 9.0
    elif distance <= 10.0:
        power = 10.0
    else:
        power = 10.0

    if context is not None:
        opponent_distance = _nearest_opponent_to_ball(context, ball)
        if opponent_distance <= 0.90:
            power = min(power, 7.0)
        elif opponent_distance <= 1.60:
            power = min(power, 8.0)
    return power


class _ShotTargetLock:
    """Keep one player's target and power stable through the final approach."""

    def __init__(self) -> None:
        self._target: Pose2D | None = None
        self._power: float | None = None
        self._ball_origin: tuple[float, float] | None = None
        self._player_id: int | None = None

    def clear(self) -> None:
        self._target = None
        self._power = None
        self._ball_origin = None
        self._player_id = None

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        return self._plan(kit, player_id, context)[0]

    def power(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> float:
        return self._plan(kit, player_id, context)[1]

    def _plan(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> tuple[Pose2D, float]:
        ball = context.known_ball
        robot = context.teammates.get(player_id)
        distance = (
            math.hypot(ball.x - robot.pose.x, ball.y - robot.pose.y)
            if robot is not None and robot.pose is not None else math.inf
        )
        if (
            self._target is not None
            and self._power is not None
            and self._ball_origin is not None
        ):
            ball_moved = math.hypot(
                ball.x - self._ball_origin[0],
                ball.y - self._ball_origin[1],
            ) > 0.35
            # Keep one plan while preparing; a timer alone must not switch the
            # target mid-action.  Loss of proximity/player or a new ball clears it.
            if ball_moved or distance > 2.40 or self._player_id != player_id:
                self.clear()
            else:
                return self._target, self._power

        selected = _normal_kick_target(kit, player_id, context)
        power = _normal_shot_power(kit, ball, selected, context)
        if distance <= 2.00:
            self._target = selected
            self._power = power
            self._ball_origin = (ball.x, ball.y)
            self._player_id = player_id
        return selected, power


def _forward_clear_target(kit: "SoccerKit", ball) -> Pose2D:
    """Goalkeeper clearance: exactly forward (+x), without goal-point aiming."""

    return Pose2D(kit.field.opponent_goal_x(), ball.y, 0.0)


def _behind_ball(kit: "SoccerKit", ball, aim: Pose2D) -> Pose2D:
    """Approach on the intended kick line instead of always along the x-axis."""

    theta = math.atan2(aim.y - ball.y, aim.x - ball.x)
    return kit.field.clamp_inside_field(
        Pose2D(
            ball.x - math.cos(theta) * _APPROACH_OFFSET,
            ball.y - math.sin(theta) * _APPROACH_OFFSET,
            theta,
        ),
        margin=0.18,
    )


def _boundary_support_target(kit: "SoccerKit", ball) -> Pose2D:
    """Hold the 40/60 support lane outside the referee's near-ball inactivity zone."""

    x = max(_FORWARD_MIN_X, ball.x - 2.80)
    side = 1.0 if _FORWARD_BOUNDARY_Y >= ball.y else -1.0
    y = ball.y + side * 3.20
    x, y = _clamp_to_field(kit, x, y)
    x, y = _clamp_out_of_own_half(x, y)
    x, y = _avoid_goal_posts(x, y)
    return Pose2D(x, y, kit.field.face_ball_theta(x, y, ball))


def _goalkeeper_zone_receiver(
    kit: "SoccerKit",
    ball,
    role_side: float,
) -> Pose2D:
    """Hold ahead of the 35% line while the goalkeeper clears the ball."""

    x = max(_FORWARD_MIN_X, min(0.20, ball.x + 3.30))
    y = ball.y + role_side * 1.80
    x, y = _clamp_to_field(kit, x, y)
    return Pose2D(x, y, kit.field.face_ball_theta(x, y, ball))


def _restart_receiver_wait_target(kit: "SoccerKit", ball: BallState) -> Pose2D:
    """Wait 0.35 m behind the penalty spot so the incoming ball stays clear."""

    x = kit.field.opponent_goal_x() - kit.config.penalty_dist - 0.35
    y = 0.0
    return Pose2D(x, y, kit.field.face_ball_theta(x, y, ball))


def _wide_cross_receiver_target(kit: "SoccerKit", ball: BallState) -> Pose2D:
    """Reuse the proven penalty-spot receiver position during open play."""

    x = kit.field.opponent_goal_x() - kit.config.penalty_dist - 0.35
    return Pose2D(x, 0.0, kit.field.face_ball_theta(x, 0.0, ball))


def _sideline_safe_chase_target(ball: BallState, target: Pose2D) -> Pose2D:
    """Do not follow an almost-out ball onto its likely restart point."""

    if abs(ball.y) < _SIDELINE_DANGER_Y:
        return target
    safe_y = math.copysign(_SIDELINE_RETREAT_Y, ball.y)
    return Pose2D(target.x, safe_y, target.theta)


def _kick_power_for_context(
    playbook: "DualStrikerPlaybook",
    context: PlayContext,
    normal_power: float,
) -> float:
    """Apply the requested fixed powers without changing the BT structure."""

    if not playbook.is_own_set_play(context):
        return normal_power
    set_play = context.known_game.set_play
    ball = context.known_ball
    if set_play == SetPlay.GOAL_KICK:
        return 10.0
    if set_play == SetPlay.THROW_IN:
        if ball.x < 0.0:
            return 7.0
        target = playbook.get_set_play_kick_target(playbook.kit, context)
        distance = math.hypot(target.x - ball.x, target.y - ball.y)
        return 2.0 if distance <= 4.0 else 2.5
    if set_play == SetPlay.CORNER_KICK:
        target = playbook.get_set_play_kick_target(playbook.kit, context)
        distance = math.hypot(target.x - ball.x, target.y - ball.y)
        return 1.5 if distance <= 3.0 else 2.0
    if set_play == SetPlay.INDIRECT_FREE_KICK:
        return 1.5
    return normal_power


def _forward_kick_power(
    playbook: "DualStrikerPlaybook",
    kit: "SoccerKit",
    player_id: int,
    shot_lock: _ShotTargetLock,
    context: PlayContext,
) -> float:
    """Keep restart powers fixed and use distance power for normal shots."""

    if playbook.is_own_set_play(context):
        return _kick_power_for_context(playbook, context, 6.5)
    if playbook.is_kickoff_cannon_released(context):
        return 7.0
    return shot_lock.power(kit, player_id, context)


def _forward_shot_ready(
    playbook: "DualStrikerPlaybook",
    kit: "SoccerKit",
    player_id: int,
    context: PlayContext,
    shot_lock: _ShotTargetLock,
    execution: ShotExecution,
) -> bool:
    """Prepare normal forward shots without altering restart execution."""

    if playbook.is_set_play_active(context):
        execution.clear()
        return True
    aim = shot_lock.target(kit, player_id, context)
    ready = execution.ready(player_id, context, aim)
    if execution.phase == "shot retry":
        shot_lock.clear()
    return ready


# ---------------------------------------------------------------------------
# Goalkeeper  (§1.2, §2.2, §5)
# ---------------------------------------------------------------------------


class GoalkeeperRole(RoleStrategy):
    """Robot3 keeper: rear 35% ownership and straight-ahead clearances."""

    name = "goalkeeper"

    def __init__(self, playbook: "DualStrikerPlaybook"):
        self._playbook = playbook
        self._bb = BlackboardClient(name="goalkeeper")

    # -- targeting ----------------------------------------------------------

    @staticmethod
    def _in_own_half(ball) -> bool:
        return ball.x <= _GK_DANGER_X

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball

        # Set Play: kicker approaches ball; others use table positions.
        if self._playbook.is_set_play_active(context):
            if self._playbook.is_set_play_kicker("goalkeeper", context):
                aim = self._playbook.get_set_play_kick_target(kit, context)
                return self._capped(_behind_ball(kit, ball, aim))
            return self._capped(
                self._playbook.get_set_play_target("goalkeeper", context)
            )

        # In a one-robot emergency this role intentionally covers the full field.
        if self._is_solo():
            return _behind_ball(kit, ball, _forward_clear_target(kit, ball))

        # Challenge only in the defensive danger zone, and stay inside a bounded
        # keeper corridor instead of following the ball into midfield.
        if self._in_own_half(ball):
            aim = _forward_clear_target(kit, ball)
            return self._capped(_behind_ball(kit, ball, aim))

        standby_y = max(-0.90, min(0.90, ball.y * 0.25))
        theta = kit.field.face_ball_theta(_GK_STANDBY_X, standby_y, ball)
        return self._capped(Pose2D(_GK_STANDBY_X, standby_y, theta))

    def wants_to_kick(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> bool:
        if self._playbook.is_set_play_active(context):
            return self._playbook.is_set_play_kicker("goalkeeper", context)

        return self._is_solo() or self._in_own_half(context.known_ball)

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        if self._playbook.is_set_play_active(context):
            return self._playbook.get_set_play_kick_target(kit, context)
        return _forward_clear_target(kit, context.known_ball)

    # -- subtree ------------------------------------------------------------

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda ctx: self.target(kit, player_id, ctx),
                kick_target_fn=lambda ctx: self.kick_target(kit, player_id, ctx),
                wants_kick_fn=lambda ctx: self.wants_to_kick(kit, player_id, ctx),
                reason_fn=lambda: "goalkeeper sweep",
                kick_reason_fn=lambda t: "goalkeeper clear",
                kick_power=lambda ctx: _kick_power_for_context(
                    self._playbook, ctx, 7.0,
                ),
            ),
        )

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _capped(target: Pose2D) -> Pose2D:
        """Clamp normal keeper activity to the goal-side defensive corridor."""
        x = max(-6.82, min(_GK_MAX_X, target.x))
        y = max(-_GK_MAX_Y, min(_GK_MAX_Y, target.y))
        return Pose2D(x, y, target.theta)

    def _is_solo(self) -> bool:
        assignment = self._bb.read(BlackboardKeys.ROLES)
        return isinstance(assignment, RoleAssignment) and len(assignment.by_player) == 1


# ---------------------------------------------------------------------------
# LeftChaser  (§1.2, §2.1, §5)
# ---------------------------------------------------------------------------


class LeftChaserRole(RoleStrategy):
    """Robot2 left forward: owns the upper 40% and shoots across goal."""

    name = "left_chaser"

    def __init__(self, playbook: "DualStrikerPlaybook"):
        self._playbook = playbook
        self._bb = BlackboardClient(name="left_chaser")
        self._shot_lock = _ShotTargetLock()
        self._shot_execution = ShotExecution()

    def clear_transient_action(self) -> None:
        """Drop an unfinished normal shot when referee control takes priority."""

        self._shot_lock.clear()
        self._shot_execution.clear()

    def has_committed_shot(self) -> bool:
        return self._shot_execution.phase == "shot commit"

    # -- targeting ----------------------------------------------------------

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball

        # Set Play: kicker approaches ball (bypass own-half clamp);
        # others use table positions.
        if self._playbook.is_set_play_active(context):
            self._shot_lock.clear()
            self._shot_execution.clear()
            if self._playbook.is_set_play_kicker("left_chaser", context):
                aim = self._playbook.get_set_play_kick_target(kit, context)
                return _behind_ball(kit, ball, aim)
            return self._finalise_forward_target(
                kit, self._playbook.get_set_play_target("left_chaser", context)
            )

        # The right forward owns kickoff.  Left holds a clear receiving lane.
        if self._playbook.is_own_kickoff(context):
            self._shot_lock.clear()
            self._shot_execution.clear()
            return _boundary_support_target(kit, ball)
        if self._playbook.is_kickoff_cannon_released(context):
            self._shot_lock.clear()
            self._shot_execution.clear()
            return _boundary_support_target(kit, ball)

        if ball.x <= _GK_ZONE_FRONT_X:
            self._shot_lock.clear()
            self._shot_execution.clear()
            return _goalkeeper_zone_receiver(kit, ball, 1.0)

        receiver = self._playbook.restart_receiver_role(context)
        if receiver is not None:
            if receiver == self.name and self._playbook.restart_ball_ready(context):
                return self._chase_target(kit, player_id, context)
            self.clear_transient_action()
            if receiver == self.name:
                return _restart_receiver_wait_target(kit, ball)
            return _boundary_support_target(kit, ball)

        # Normal play.
        if self._playbook.active_striker() == self.name or self._is_solo():
            return self._chase_target(kit, player_id, context)
        self._shot_lock.clear()
        self._shot_execution.clear()
        if _use_wide_cross(kit, context):
            return _wide_cross_receiver_target(kit, ball)
        return _boundary_support_target(kit, ball)

    def wants_to_kick(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> bool:
        if self._playbook.is_set_play_active(context):
            return self._playbook.is_set_play_kicker("left_chaser", context)
        # Own kickoff: right_chaser kicks, not left_chaser.
        if self._playbook.is_own_kickoff(context):
            return False
        if self._playbook.is_kickoff_cannon_released(context):
            return False
        if context.known_ball.x <= _GK_ZONE_FRONT_X:
            return False
        receiver = self._playbook.restart_receiver_role(context)
        if receiver is not None:
            return receiver == self.name and self._playbook.restart_ball_ready(context)
        return self._playbook.active_striker() == self.name or self._is_solo()

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        if self._playbook.is_set_play_active(context):
            self._shot_lock.clear()
            return self._playbook.get_set_play_kick_target(kit, context)
        return self._shot_lock.target(kit, player_id, context)

    # -- subtree ------------------------------------------------------------

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda ctx: self.target(kit, player_id, ctx),
                kick_target_fn=lambda ctx: self.kick_target(kit, player_id, ctx),
                wants_kick_fn=lambda ctx: (
                    self.wants_to_kick(kit, player_id, ctx)
                    and _forward_shot_ready(
                        self._playbook, kit, player_id, ctx,
                        self._shot_lock, self._shot_execution,
                    )
                ),
                reason_fn=lambda: f"left_chaser {self._shot_execution.phase}",
                kick_reason_fn=lambda t: "left_chaser kick",
                kick_power=lambda ctx: _forward_kick_power(
                    self._playbook, kit, player_id, self._shot_lock, ctx,
                ),
                final_ball_approach=True,
            ),
        )

    # -- internal -----------------------------------------------------------

    def _is_solo(self) -> bool:
        """Check whether this is the only forward (sent-off scenario)."""
        val = self._bb.read(BlackboardKeys.ROLES)
        if not isinstance(val, RoleAssignment):
            return False
        return len(val.players_of("right_chaser")) == 0

    def _chase_target(
        self, kit: "SoccerKit", player_id: int, context: PlayContext,
    ) -> Pose2D:
        aim = self._shot_lock.target(kit, player_id, context)
        target = self._shot_execution.approach_target(
                kit, player_id, context, aim, min_x=_FORWARD_MIN_X,
            )
        return self._finalise_forward_target(
            kit, _sideline_safe_chase_target(context.known_ball, target),
        )

    @staticmethod
    def _finalise_forward_target(kit: "SoccerKit", target: Pose2D) -> Pose2D:
        """Apply penalty-area clamp, goal-post avoidance, and field clamp."""
        x, y = _clamp_out_of_own_half(target.x, target.y)
        x, y = _avoid_goal_posts(x, y)
        x, y = _clamp_to_field(kit, x, y)
        return Pose2D(x, y, target.theta)


# ---------------------------------------------------------------------------
# RightChaser  (§1.2, §2.1, §5)
# ---------------------------------------------------------------------------


class RightChaserRole(RoleStrategy):
    """Robot1 right forward: owns the lower 60% and shoots across goal."""

    name = "right_chaser"

    def __init__(self, playbook: "DualStrikerPlaybook"):
        self._playbook = playbook
        self._bb = BlackboardClient(name="right_chaser")
        self._shot_lock = _ShotTargetLock()
        self._shot_execution = ShotExecution()

    def clear_transient_action(self) -> None:
        """Drop an unfinished normal shot when referee control takes priority."""

        self._shot_lock.clear()
        self._shot_execution.clear()

    def has_committed_shot(self) -> bool:
        return self._shot_execution.phase == "shot commit"

    # -- targeting ----------------------------------------------------------

    def target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        ball = context.known_ball

        # Set Play: kicker approaches ball (bypass own-half clamp);
        # others use table positions.
        if self._playbook.is_set_play_active(context):
            self._shot_lock.clear()
            self._shot_execution.clear()
            if self._playbook.is_set_play_kicker("right_chaser", context):
                aim = self._playbook.get_set_play_kick_target(kit, context)
                return _behind_ball(kit, ball, aim)
            return self._finalise_forward_target(
                kit, self._playbook.get_set_play_target("right_chaser", context)
            )

        # During the protected ten seconds, line up but never request a kick.
        if self._playbook.is_own_kickoff(context):
            self._shot_lock.clear()
            self._shot_execution.clear()
            return _behind_ball(
                kit, ball, Pose2D(kit.field.opponent_goal_x() + 0.1, 0.0, 0.0),
            )
        if self._playbook.is_kickoff_cannon_released(context):
            return self._shot_execution.approach_target(
                kit, player_id, context,
                self._shot_lock.target(kit, player_id, context),
            )

        if ball.x <= _GK_ZONE_FRONT_X:
            self._shot_lock.clear()
            self._shot_execution.clear()
            return _goalkeeper_zone_receiver(kit, ball, -1.0)

        receiver = self._playbook.restart_receiver_role(context)
        if receiver is not None:
            if receiver == self.name and self._playbook.restart_ball_ready(context):
                return self._chase_target(kit, player_id, context)
            self.clear_transient_action()
            if receiver == self.name:
                return _restart_receiver_wait_target(kit, ball)
            return _boundary_support_target(kit, ball)

        if self._playbook.active_striker() == self.name or self._is_solo():
            return self._chase_target(kit, player_id, context)
        self._shot_lock.clear()
        self._shot_execution.clear()
        if _use_wide_cross(kit, context):
            return _wide_cross_receiver_target(kit, ball)
        return _boundary_support_target(kit, ball)

    def wants_to_kick(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> bool:
        if self._playbook.is_set_play_active(context):
            return self._playbook.is_set_play_kicker("right_chaser", context)
        # Waiting phase is protected: direct goals before release are invalid.
        if self._playbook.is_own_kickoff(context):
            return False
        if self._playbook.is_kickoff_cannon_released(context):
            return True
        if context.known_ball.x <= _GK_ZONE_FRONT_X:
            return False
        receiver = self._playbook.restart_receiver_role(context)
        if receiver is not None:
            return receiver == self.name and self._playbook.restart_ball_ready(context)
        return self._playbook.active_striker() == self.name or self._is_solo()

    def kick_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
    ) -> Pose2D:
        if self._playbook.is_set_play_active(context):
            self._shot_lock.clear()
            return self._playbook.get_set_play_kick_target(kit, context)
        return self._shot_lock.target(kit, player_id, context)

    # -- subtree ------------------------------------------------------------

    def build_subtree(
        self,
        kit: "SoccerKit",
        player_id: int,
    ) -> py_trees.behaviour.Behaviour:
        return build_attack_subtree(
            kit,
            player_id,
            AttackSubtreeConfig(
                target_fn=lambda ctx: self.target(kit, player_id, ctx),
                kick_target_fn=lambda ctx: self.kick_target(kit, player_id, ctx),
                wants_kick_fn=lambda ctx: (
                    self.wants_to_kick(kit, player_id, ctx)
                    and _forward_shot_ready(
                        self._playbook, kit, player_id, ctx,
                        self._shot_lock, self._shot_execution,
                    )
                ),
                reason_fn=lambda: f"right_chaser {self._shot_execution.phase}",
                kick_reason_fn=lambda t: "right_chaser kick",
                kick_power=lambda ctx: _forward_kick_power(
                    self._playbook, kit, player_id, self._shot_lock, ctx,
                ),
                final_ball_approach=True,
            ),
        )

    # -- internal -----------------------------------------------------------

    def _is_solo(self) -> bool:
        val = self._bb.read(BlackboardKeys.ROLES)
        if not isinstance(val, RoleAssignment):
            return False
        return len(val.players_of("left_chaser")) == 0

    def _chase_target(
        self, kit: "SoccerKit", player_id: int, context: PlayContext,
    ) -> Pose2D:
        aim = self._shot_lock.target(kit, player_id, context)
        target = self._shot_execution.approach_target(
                kit, player_id, context, aim, min_x=_FORWARD_MIN_X,
            )
        return self._finalise_forward_target(
            kit, _sideline_safe_chase_target(context.known_ball, target),
        )

    @staticmethod
    def _finalise_forward_target(kit: "SoccerKit", target: Pose2D) -> Pose2D:
        x, y = _clamp_out_of_own_half(target.x, target.y)
        x, y = _avoid_goal_posts(x, y)
        x, y = _clamp_to_field(kit, x, y)
        return Pose2D(x, y, target.theta)
