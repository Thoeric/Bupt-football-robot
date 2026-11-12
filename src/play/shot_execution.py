"""Small forward-only handover guard: approach, commit, and retry if stuck.

This does not control the kick animation.  Booster still handles foot selection
and final alignment after the strategy hands over a roughly aligned ball.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from ..soccer_framework import PlayContext, Pose2D
from ..tactics.geometry import normalize_angle

if TYPE_CHECKING:
    from ..runtime import SoccerKit


class ShotExecution:
    """Keep readiness checks out of an already-running kick attempt."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._player_id: int | None = None
        self._last_seen: float | None = None
        self._started_at: float | None = None
        self._progress_at = 0.0
        self._best_distance = math.inf
        self._progress_ball = (0.0, 0.0)
        self._retry_until = 0.0
        self._failed_attempts = 0
        self._escape_target: Pose2D | None = None
        self._near_ball_since: float | None = None
        self._near_ball_anchor: tuple[float, float] | None = None

    @property
    def phase(self) -> str:
        if self._escape_target is not None:
            return "shot escape"
        if time.monotonic() < self._retry_until:
            return "shot retry"
        return "shot commit" if self._started_at is not None else "shot approach"

    def ready(
        self,
        player_id: int,
        context: PlayContext,
        aim: Pose2D,
    ) -> bool:
        robot = context.teammates.get(player_id)
        if robot is None or robot.pose is None:
            self.clear()
            return False

        now = time.monotonic()
        # READY, penalties and other branches may skip this role for a while.
        # Never carry their old running attempt into a fresh phase or player.
        if self._player_id != player_id or (
            self._last_seen is not None and now - self._last_seen > 1.0
        ):
            self.clear()
        self._player_id = player_id
        self._last_seen = now

        ball = context.known_ball
        dx, dy = ball.x - robot.pose.x, ball.y - robot.pose.y
        distance = math.hypot(dx, dy)
        if self._escape_target is not None:
            return False
        if now < self._retry_until:
            return False

        if self._started_at is not None:
            if distance > 1.20:
                # The ball has left the working area: chase the next ball.
                self.clear()
                return False
            ball_progress = math.hypot(
                ball.x - self._progress_ball[0], ball.y - self._progress_ball[1],
            )
            if distance < self._best_distance - 0.10 or ball_progress > 0.20:
                self._progress_at = now
                self._best_distance = min(distance, self._best_distance)
                self._progress_ball = (ball.x, ball.y)
            if now - self._progress_at >= 2.00 or now - self._started_at >= 4.00:
                self._started_at = None
                self._failed_attempts += 1
                self._retry_until = now + 0.35
                if self._failed_attempts >= 2:
                    away_x, away_y = robot.pose.x - ball.x, robot.pose.y - ball.y
                    away_length = math.hypot(away_x, away_y)
                    if away_length < 1e-6:
                        away_x, away_y, away_length = -1.0, 0.0, 1.0
                    self._escape_target = Pose2D(
                        robot.pose.x + 0.60 * away_x / away_length,
                        robot.pose.y + 0.60 * away_y / away_length,
                        math.atan2(away_y, away_x),
                    )
                return False
            return True

        theta = math.atan2(aim.y - ball.y, aim.x - ball.x)
        ux, uy = math.cos(theta), math.sin(theta)
        behind = dx * ux + dy * uy
        lateral = abs(dx * uy - dy * ux)
        heading_error = abs(normalize_angle(theta - robot.pose.theta))
        opponent_distance = min(
            (
                math.hypot(
                    opponent.pose.x - ball.x,
                    opponent.pose.y - ball.y,
                )
                for opponent in context.opponents.values()
                if opponent.pose is not None
            ),
            default=math.inf,
        )
        pressured = opponent_distance <= 1.00
        if pressured:
            distance_limit = 0.85
            lateral_limit = 0.28
            angle_limit = math.radians(30.0)
            behind_limit = -0.05
        else:
            distance_limit = 0.75
            lateral_limit = 0.22
            angle_limit = math.radians(20.0)
            behind_limit = 0.10
        # Just in front of the keeper/forward ownership line, the forward's
        # x >= -1.95 activity clamp can leave it a few centimetres goal-side
        # of the ball.  Accept that narrow geometric exception; otherwise the
        # approach target is unreachable and the forward circles indefinitely.
        if (
            -2.10 < ball.x <= -1.80
            and robot.pose.x <= ball.x + 0.20
        ):
            behind_limit = -0.20
        if (
            distance > distance_limit
            or behind < behind_limit
            or lateral > lateral_limit
            or heading_error > angle_limit
        ):
            return False

        self._started_at = now
        self._progress_at = now
        self._best_distance = distance
        self._progress_ball = (ball.x, ball.y)
        return True

    def approach_target(
        self,
        kit: "SoccerKit",
        player_id: int,
        context: PlayContext,
        aim: Pose2D,
        *,
        min_x: float | None = None,
    ) -> Pose2D:
        ball = context.known_ball
        now = time.monotonic()
        robot = context.teammates.get(player_id)
        if robot is not None and robot.pose is not None:
            if self._escape_target is not None:
                if math.hypot(
                    robot.pose.x - self._escape_target.x,
                    robot.pose.y - self._escape_target.y,
                ) <= 0.20:
                    self._escape_target = None
                    self._failed_attempts = 0
                    self._near_ball_since = None
                    self._near_ball_anchor = None
                else:
                    return kit.field.clamp_inside_field(
                        self._escape_target, margin=0.18,
                    )

            robot_ball_distance = math.hypot(
                robot.pose.x - ball.x, robot.pose.y - ball.y,
            )
            if robot_ball_distance <= 3.0:
                if self._near_ball_since is None or self._near_ball_anchor is None:
                    self._near_ball_since = now
                    self._near_ball_anchor = (robot.pose.x, robot.pose.y)
                elif math.hypot(
                    robot.pose.x - self._near_ball_anchor[0],
                    robot.pose.y - self._near_ball_anchor[1],
                ) >= 0.15:
                    self._near_ball_since = now
                    self._near_ball_anchor = (robot.pose.x, robot.pose.y)
                elif now - self._near_ball_since >= 4.5:
                    away_x, away_y = robot.pose.x - ball.x, robot.pose.y - ball.y
                    away_length = math.hypot(away_x, away_y)
                    if away_length < 1e-6:
                        away_x, away_y, away_length = -1.0, 0.0, 1.0
                    self._escape_target = Pose2D(
                        robot.pose.x + 0.60 * away_x / away_length,
                        robot.pose.y + 0.60 * away_y / away_length,
                        math.atan2(away_y, away_x),
                    )
                    self._started_at = None
                    self._retry_until = now + 0.25
                    return kit.field.clamp_inside_field(
                        self._escape_target, margin=0.18,
                    )
            else:
                self._near_ball_since = None
                self._near_ball_anchor = None

        theta = math.atan2(aim.y - ball.y, aim.x - ball.x)
        ux, uy = math.cos(theta), math.sin(theta)
        offset = 0.60 if now < self._retry_until else 0.45
        x, y = ball.x - offset * ux, ball.y - offset * uy

        # In the central goalmouth, approach directly from field-side to avoid
        # tracing a diagonal around a post.  Keep the normal shot-line point if
        # this shortcut would exceed the same 0.30 m lateral handover limit.
        if offset == 0.45 and ball.x > 4.50 and abs(ball.y) < 1.00:
            direct_x, direct_y = ball.x - offset, ball.y
            direct_dx, direct_dy = ball.x - direct_x, ball.y - direct_y
            direct_lateral = abs(direct_dx * uy - direct_dy * ux)
            if direct_lateral <= 0.18:
                x, y = direct_x, direct_y
        can_reach_behind = min_x is None or x >= min_x

        if robot is not None and robot.pose is not None:
            rx, ry = robot.pose.x - ball.x, robot.pose.y - ball.y
            along = rx * ux + ry * uy
            lateral = -rx * uy + ry * ux
            # When arriving from in front, step around the ball rather than
            # walking straight through it on the way to the behind-ball point.
            if (
                can_reach_behind and along > 0.10
                and abs(lateral) < 0.45 and math.hypot(rx, ry) < 1.2
            ):
                side = 1.0 if lateral >= 0.0 else -1.0
                x = ball.x - 0.15 * ux - side * 0.65 * uy
                y = ball.y - 0.15 * uy + side * 0.65 * ux

        return kit.field.clamp_inside_field(Pose2D(x, y, theta), margin=0.18)
