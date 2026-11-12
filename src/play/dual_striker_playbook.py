"""Dual Striker attack system Playbook.

Fixed formation for 3v3 simulation soccer:
- Player 1 (robot1): right_chaser  — covers right half (y < 0)
- Player 2 (robot2): left_chaser   — covers left half  (y >= 0)
- Player 3 (robot3): goalkeeper   — penalty area only

Set Play positioning follows the full table from the tactical spec (§5.1).
Sent-off players trigger automatic role redistribution (§6.2).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from ..soccer_framework import (
    GameState,
    Penalty,
    PlayContext,
    Pose2D,
    RobotCommand,
    SetPlay,
)
from .dual_striker_roles import GoalkeeperRole, LeftChaserRole, RightChaserRole
from .playbook import Playbook, RoleAssignment
from ..tactics.targeting import restart

if TYPE_CHECKING:
    from ..runtime import SoccerKit


# ---------------------------------------------------------------------------
# Set Play target helpers
# ---------------------------------------------------------------------------

def _is_own_set_play(kit: "SoccerKit", game) -> bool:
    return game.has_kicking_team() and game.kicking_team == kit.config.team_id


def _is_opponent_set_play(kit: "SoccerKit", game) -> bool:
    return game.has_kicking_team() and game.kicking_team != kit.config.team_id


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------


class DualStrikerPlaybook(Playbook):
    """Dual Striker attack system — fixed 2-forward + 1-goalkeeper formation."""

    def __init__(self, kit: "SoccerKit"):
        super().__init__(kit)
        # Small cross-tick memories only: a dead band prevents the two forwards
        # swapping jobs on every noisy centre-line observation, and the kickoff
        # latch opens a short shot window after GameController releases the ball.
        # The y=+0.90 boundary gives robot2 40% of field width and robot1 60%.
        # The first observation uses the exact boundary; later ticks add hysteresis.
        self._active_striker: str | None = None
        self._kickoff_armed = False
        self._kickoff_release_ticks = 0
        self._kickoff_ball_origin: tuple[float, float] | None = None
        self._restart_receiver: str | None = None
        self._restart_kind: SetPlay | None = None
        self._restart_ball_origin: tuple[float, float] | None = None
        self._restart_released_at: float | None = None
        self._restart_receive_started_at: float | None = None
        self._left_role = LeftChaserRole(self)
        self._right_role = RightChaserRole(self)
        self._gk_role = GoalkeeperRole(self)
        self.register_role(self._left_role)
        self.register_role(self._right_role)
        self.register_role(self._gk_role)

    # ------------------------------------------------------------------
    # Role assignment (§1.1, §6.2)
    # ------------------------------------------------------------------

    def assign_roles(self, context: PlayContext) -> RoleAssignment:
        self._update_tactical_state(context)
        game = context.known_game
        team_id = self.kit.config.team_id
        all_ids = list(self.kit.config.player_ids)  # [1, 2, 3]

        # Collect available (non-penalised) players.
        available: list[int] = []
        for pid in all_ids:
            ps = game.get_player_state(team_id, pid)
            if ps is not None and ps.penalty in (Penalty.SENT_OFF, Penalty.SUBSTITUTE):
                continue
            if self.kit.is_player_allowed(game, pid):
                available.append(pid)

        n = len(available)
        mapping: dict[int, str] = {}

        if n == 0:
            return RoleAssignment({})

        if n == 1:
            # Only player becomes goalkeeper who abandons the box and
            # chases the ball everywhere.
            mapping[available[0]] = "goalkeeper"

        elif n == 2:
            # Prefer player 3 as goalkeeper.
            gk = 3 if 3 in available else available[0]
            mapping[gk] = "goalkeeper"
            fwd = [p for p in available if p != gk][0]
            # Single forward covers the full field → registered as left_chaser
            # (LeftChaserRole detects solo mode and ignores side split).
            mapping[fwd] = "left_chaser"

        else:
            # Full 3-player formation.
            # Player 3 → goalkeeper (fixed).
            mapping[3] = "goalkeeper"
            # Remaining two → forwards, prefer player 2 as left, player 1 as right.
            fwds = [p for p in available if p != 3]
            if 2 in fwds:
                mapping[2] = "left_chaser"
                fwds.remove(2)
            if fwds:
                mapping[fwds[0]] = "right_chaser"

        if (
            self._restart_receiver is not None
            and self._restart_receiver not in mapping.values()
        ):
            self._clear_restart_receive()
        return RoleAssignment(mapping)

    def _update_tactical_state(self, context: PlayContext) -> None:
        """Refresh the two intentionally tiny pieces of tactical memory."""

        game = context.known_game
        ball = context.known_ball

        # 0.30 m hysteresis around y=+0.90 avoids noisy ownership changes.
        if self._active_striker is None:
            self._active_striker = (
                "left_chaser" if ball.y >= 0.90 else "right_chaser"
            )
        elif ball.y > 1.20:
            self._active_striker = "left_chaser"
        elif ball.y < 0.60:
            self._active_striker = "right_chaser"

        # Every referee-controlled restart cancels a normal shot immediately.
        if self.is_set_play_active(context):
            self._left_role.clear_transient_action()
            self._right_role.clear_transient_action()
            self._kickoff_armed = False
            self._kickoff_release_ticks = 0
            self._kickoff_ball_origin = None
            qualifying = (
                self.is_own_set_play(context)
                and (
                    game.set_play == SetPlay.CORNER_KICK
                    or game.set_play == SetPlay.THROW_IN
                )
            )
            if qualifying:
                kicker = restart.restart_kicker_role(game.set_play, ball)
                receiver = (
                    "right_chaser" if kicker == "left_chaser" else "left_chaser"
                )
                if (
                    self._restart_receiver is None
                    or self._restart_kind != game.set_play
                    or self._restart_released_at is not None
                ):
                    self._restart_receiver = receiver
                    self._restart_kind = game.set_play
                    self._restart_ball_origin = (ball.x, ball.y)
                    self._restart_released_at = None
                    self._restart_receive_started_at = None
            else:
                self._clear_restart_receive()
            return

        if self._restart_receiver is not None:
            now = time.monotonic()
            if (
                game.state != GameState.PLAYING
                or game.stopped
                or game.has_kicking_team()
                or ball.x <= -2.10
            ):
                self._clear_restart_receive()
            else:
                if self._restart_released_at is None:
                    self._restart_released_at = now
                if (
                    self._restart_receive_started_at is None
                    and self.restart_ball_ready(context)
                ):
                    # Start the short shooting window when the pass arrives,
                    # not while a long own-half throw is still travelling.
                    self._restart_receive_started_at = now
                elif (
                    self._restart_receive_started_at is None
                    and now - self._restart_released_at > 6.0
                ):
                    self._clear_restart_receive()
                elif (
                    self._restart_receive_started_at is not None
                    and now - self._restart_receive_started_at > 2.0
                ):
                    receiver_role = (
                        self._right_role
                        if self._restart_receiver == "right_chaser"
                        else self._left_role
                    )
                    if not receiver_role.has_committed_shot():
                        self._clear_restart_receive()

        if self.is_own_kickoff(context):
            self._kickoff_armed = True
            self._kickoff_release_ticks = 0
            self._kickoff_ball_origin = (ball.x, ball.y)
            return

        # Some GameController packets briefly report secondary_time == 0 while
        # still retaining our team as the kicking team.  Preserve the latch but
        # do not shoot until kicking_team is explicitly cleared.
        if (
            self._kickoff_armed
            and game.state == GameState.PLAYING
            and not game.stopped
            and game.set_play == SetPlay.NONE
            and game.kicking_team == self.kit.config.team_id
        ):
            return

        released = (
            self._kickoff_armed
            and game.state == GameState.PLAYING
            and not game.stopped
            and game.set_play == SetPlay.NONE
            and not game.has_kicking_team()
            and game.secondary_time <= 0
        )
        origin = self._kickoff_ball_origin
        ball_still_central = (
            origin is not None
            and math.hypot(ball.x - origin[0], ball.y - origin[1]) <= 0.65
            and math.hypot(ball.x, ball.y) <= 0.90
        )
        if released and ball_still_central and self._kickoff_release_ticks < 90:
            self._kickoff_release_ticks += 1
            return

        self._kickoff_armed = False
        self._kickoff_release_ticks = 0
        self._kickoff_ball_origin = None

    def active_striker(self) -> str:
        """Return the forward owning the 40/60 zone with boundary hysteresis."""

        return self._active_striker or "right_chaser"

    def restart_receiver_role(self, context: PlayContext) -> str | None:
        """Return the two-second receiver owner after a front-half restart."""

        if self.is_set_play_active(context):
            return None
        if self._restart_released_at is None:
            return None
        return self._restart_receiver

    def restart_ball_ready(self, context: PlayContext) -> bool:
        """The receiver attacks only after the pass reaches its 1.5 m zone."""

        ball = context.known_ball
        if self._restart_kind == SetPlay.THROW_IN and (
            self._restart_ball_origin is not None
            and self._restart_ball_origin[0] < 0.0
        ):
            receive_x = 4.0
        else:
            receive_x = self.kit.field.opponent_goal_x() - self.kit.config.penalty_dist
        return math.hypot(ball.x - receive_x, ball.y) <= 1.50

    def _clear_restart_receive(self) -> None:
        self._restart_receiver = None
        self._restart_kind = None
        self._restart_ball_origin = None
        self._restart_released_at = None
        self._restart_receive_started_at = None

    # ------------------------------------------------------------------
    # Set Play helpers  (§5)
    # ------------------------------------------------------------------

    def is_set_play_active(self, context: PlayContext) -> bool:
        game = context.known_game
        return (
            game.state == GameState.PLAYING
            and not game.stopped
            and game.set_play != SetPlay.NONE
        )

    def is_own_set_play(self, context: PlayContext) -> bool:
        game = context.known_game
        return self.is_set_play_active(context) and _is_own_set_play(self.kit, game)

    def is_own_kickoff(self, context: PlayContext) -> bool:
        """Detect own centre-circle kick-off (start of half / after goal)."""
        game = context.known_game
        return (
            game.state == GameState.PLAYING
            and not game.stopped
            and game.set_play == SetPlay.NONE
            and game.secondary_time > 0
            and game.has_kicking_team()
            and game.kicking_team == self.kit.config.team_id
        )

    def is_kickoff_cannon_released(self, context: PlayContext) -> bool:
        """True only in the short, legal shot window after the 10 second hold."""

        game = context.known_game
        return (
            self._kickoff_release_ticks > 0
            and game.state == GameState.PLAYING
            and not game.stopped
            and game.set_play == SetPlay.NONE
            and not game.has_kicking_team()
            and game.secondary_time <= 0
        )

    def is_set_play_kicker(self, role_name: str, context: PlayContext) -> bool:
        """Return True when *role_name* is the designated kicker for this own set play."""
        if not self.is_own_set_play(context):
            return False
        game = context.known_game
        ball = context.known_ball
        sp = game.set_play

        return role_name == restart.restart_kicker_role(sp, ball)

    def get_set_play_kick_target(
        self,
        kit: "SoccerKit",
        context: PlayContext,
    ) -> Pose2D:
        """Return the kick-aim target for own set plays.

        Corner kick  → goal-area line midpoint (6.0, 0).
        Throw-in     → opponent penalty spot (4.9, 0).
        Other        → opponent goal centre.
        """
        game = context.known_game
        sp = game.set_play
        own = _is_own_set_play(self.kit, game)

        if own:
            return restart.own_restart_kick_target(
                kit.config, kit.field, sp, context.known_ball,
            )
        return Pose2D(kit.field.opponent_goal_x(), 0.0, 0.0)

    def get_set_play_target(
        self,
        role_name: str,
        context: PlayContext,
    ) -> Pose2D:
        """Return the set-play positioning target for *role_name* (§5.1 table)."""
        game = context.known_game
        ball = context.known_ball
        sp = game.set_play
        own = _is_own_set_play(self.kit, game)

        # ---- OWN set plays -------------------------------------------------
        if own:
            return restart.own_restart_position(
                self.kit.config, self.kit.field, role_name, sp, ball,
            )

        # ---- OPPONENT set plays --------------------------------------------
        target: Pose2D | None = None
        if sp == SetPlay.GOAL_KICK:
            target = _set_play_positions(
                gk=(-6.5, 0.0),
                lc=(0.0, 2.5),
                rc=(0.0, -2.5),
                role_name=role_name,
            )
        elif sp == SetPlay.CORNER_KICK:
            target = _set_play_positions(
                gk=(-7.0, 0.0),
                lc=(-5.0, 2.0),
                rc=(-5.0, -2.0),
                role_name=role_name,
            )
        elif sp == SetPlay.PENALTY_KICK:
            target = _set_play_positions(
                gk=(-7.0, 0.0),
                lc=(-4.0, 2.5),
                rc=(-4.0, -2.5),
                role_name=role_name,
            )
        # Opponent throw-in: 2 m behind ball, aligned with own goal posts.
        elif sp == SetPlay.THROW_IN:
            target = _set_play_positions(
                gk=(-7.0, 0.0),
                lc=(ball.x - 2.0, 1.3),
                rc=(ball.x - 2.0, -1.3),
                role_name=role_name,
            )
        # Opponent free kick: stand 3 m ahead of ball.
        elif sp in (SetPlay.DIRECT_FREE_KICK, SetPlay.INDIRECT_FREE_KICK):
            target = _set_play_positions(
                gk=(-7.0, 0.0),
                lc=(ball.x + 3.0, ball.y + 3.0),
                rc=(ball.x + 3.0, ball.y - 3.0),
                role_name=role_name,
            )

        if target is not None:
            return restart.safe_restart_position(
                self.kit.config,
                self.kit.field,
                target,
                ball,
                game,
                min_distance=max(
                    2.2,
                    self.kit.config.strategy.opponent_restart_avoid_distance_m,
                ),
            )

        # Fallback: stop where you are (return current position isn't
        # available here, so return a safe neutral position).
        return Pose2D(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def waiting_command(self, player_id: int, context: PlayContext) -> RobotCommand:
        slot = self.kit.config.ready_slot_for_player(player_id)
        return RobotCommand.stop(f"{slot.value} waiting for ball")


def _set_play_positions(
    *,
    gk: tuple[float, float],
    lc: tuple[float, float],
    rc: tuple[float, float],
    role_name: str,
) -> Pose2D:
    """Select one role target from a compact three-player restart table."""

    if role_name == "goalkeeper":
        x, y = gk
    elif role_name == "left_chaser":
        x, y = lc
    else:
        x, y = rc
    return Pose2D(x, y, 0.0)


