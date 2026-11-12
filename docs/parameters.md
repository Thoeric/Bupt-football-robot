# 3v3 Simulation Soccer Agent - Parameters Reference

This document collects all key parameters for the soccer agent, organized by category.

## 1. Motion Control Parameters (`src/tactics/motion.py`)

### Walking Control
| Parameter | Value | Description |
|-----------|-------|-------------|
| `_ARRIVE_DISTANCE` | 0.12 m | Distance threshold to consider "arrived" at target |
| `_ARRIVE_ANGLE` | 0.14 rad (~8°) | Angle error threshold for arrival |
| `_TURN_THRESHOLD` | 0.35 rad (~20°) | Angle error threshold to trigger pure turn instead of walking |
| `_ANGULAR_SPEED_FLOOR` | 0.35 rad/s | Minimum angular speed when turning |
| `_ANGULAR_DEAD_ZONE` | 0.08 rad | Below this angle error, skip floor and use pure proportional control |
| `_ANGULAR_GAIN` | 3.2 | Proportional angular gain |
| `_LINEAR_SPEED_FLOOR` | 0.40 m/s | Minimum linear speed when moving |
| `_LINEAR_GAIN` | 1.40 | Gain applied to distance * cos(angle_error) for vx calculation |
| `_FINAL_APPROACH_DISTANCE` | 1.20 m | Radius where the forward shot-approach profile applies |
| `_FINAL_APPROACH_TURN_THRESHOLD` | 35° | Walk-and-turn allowance during final shot approach |
| `_FINAL_APPROACH_LINEAR_CAP` | 0.80 m/s | Linear cap during final shot approach only |

### Approach
| Parameter | Value | Description |
|-----------|-------|-------------|
| Normal shot approach offset | 0.45 m | Distance behind the ball during a normal approach |
| Shot retry approach offset | 0.60 m | Distance behind the ball during the short retry phase |

## 2. Kick Hysteresis Parameters (`src/soccer_framework/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `soccer_kick_enter_distance` | 2.5 m | Enter kick mode when distance to ball is below this |
| `soccer_kick_exit_distance` | 3.0 m | Exit kick mode when distance to ball is above this |
| `soccer_kick_power` | 1.5 | Kick power output |
| `soccer_kick_min_active_sec` | 0.5 s | Minimum active kick duration to avoid instant switching |
| `soccer_kick_exit_delay_sec` | 1.5 s | Delay after exit condition before leaving kick mode |

## 3. Speed Limits (`src/soccer_framework/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_linear_speed` | 1.5 m/s | Maximum linear speed outside the final shot approach |
| `max_angular_speed` | 2.0 rad/s | Maximum angular speed |

## 4. Obstacle Avoidance Parameters (`src/soccer_framework/config.py`)

### Path Detour (First Avoidance Layer)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `opponent_obstacle_radius` | 0.55 m | Circular radius for opponent obstacle detection |
| `teammate_obstacle_radius` | 0.48 m | Circular radius for teammate obstacle detection |
| `obstacle_safety_margin` | 0.22 m | Extra safety margin outside obstacle radius |
| `obstacle_start_ignore_distance` | 0.35 m | Ignore obstacles this close to start position |
| `obstacle_target_ignore_distance` | 0.35 m | Ignore obstacles this close to target position |

### Yaw Avoidance (Second Avoidance Layer)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `yaw_avoid_horizon_sec` | 1.0 s | Prediction horizon for nearby-neighbor trajectories |
| `yaw_avoid_min_distance_m` | 0.78 m | Apply vyaw bias when current/predicted distance is below this |
| `yaw_avoid_bias_max` | 0.6 rad/s | Maximum vyaw bias per neighbor |

### Goal Structure
| Parameter | Value | Description |
|-----------|-------|-------------|
| `goal_depth` | 0.6 m | Goal depth (rule-fixed) |
| `post_radius` | 0.18 m | Goal post obstacle radius |
| `net_radius` | 0.20 m | Goal net obstacle radius |
| `net_step` | 0.35 m | Sampling interval for net obstacles |

## 5. Teammate Repulsion Parameters (`src/play/nodes.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `REPULSION_RADIUS` | 2.5 m | Trigger repulsion when estimated targets within this distance |
| `REPULSION_STRENGTH` | 1.2 m | Maximum repulsion offset applied to target |

### Role Priority (Lower number = Higher priority)
| Role | Priority | Description |
|------|----------|-------------|
| chaser | 0 | Highest priority, never repelled |
| corner_attacker | 1 | Second priority |
| supporter | 2 | Medium priority |
| defender | 3 | Lower priority |
| goalkeeper | 4 | Lowest priority |

## 6. Formation / Positioning Parameters (`src/soccer_framework/config.py`)

### Support Positioning
| Parameter | Value | Description |
|-----------|-------|-------------|
| `support_depth_m` | 1.05 m | Supporter depth behind the ball carrier |
| `support_lateral_m` | 1.25 m | Lateral spacing for supporters |
| `support_min_spacing_m` | 1.15 m | Minimum teammate spacing to avoid clustering |

### Dribbling
| Parameter | Value | Description |
|-----------|-------|-------------|
| `dribble_advance_m` | 1.15 m | Forward distance advanced by one dribble target |
| `dribble_center_pull` | 0.65 | Pull toward centerline while dribbling |

### Passing
| Parameter | Value | Description |
|-----------|-------|-------------|
| `pass_enabled` | True | Master pass switch |
| `pass_min_score` | 0.52 | Minimum pass-candidate score; below this, dribble instead |
| `pass_min_forward_m` | 0.35 m | Minimum forward progress for passes |
| `pass_lane_clearance` | 0.75 m | Required clearance around pass lane |

### Restart / Recovery
| Parameter | Value | Description |
|-----------|-------|-------------|
| `restart_touch_distance` | 0.45 m | Distance threshold for "touched the ball" |
| `opponent_restart_avoid_distance_m` | 2.2 m | Opponent-restart safety radius; inside 1.8 m uses emergency radial escape |
| `sideline_recovery_margin_m` | 0.90 m | Sideline distance threshold for recovery |
| `sideline_recovery_infield_m` | 1.60 m | Infield pull depth during recovery |
| `sideline_recovery_advance_m` | 0.75 m | Forward advance during recovery |
| `goal_line_recovery_margin_m` | 0.08 m | Goal-line recovery margin |

### Goalkeeping
| Parameter | Value | Description |
|-----------|-------|-------------|
| `goalkeeper_challenge_margin_m` | 0.70 m | Margin that triggers goalkeeper challenge |

## 7. Ball Claim Arbitration (`src/soccer_framework/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `teammate_challenge_tie_margin_m` | 0.15 m | Tie band for teammate ball-claim distances to prevent oscillating handoff |

## 8. Field Dimensions (Default Adult Field)

| Parameter | Description |
|-----------|-------------|
| `field_length` | Field length (from ADULT_FIELD_DIMENSIONS) |
| `field_width` | Field width (from ADULT_FIELD_DIMENSIONS) |
| `penalty_dist` | Penalty distance (from ADULT_FIELD_DIMENSIONS) |
| `goal_width` | Goal width (from ADULT_FIELD_DIMENSIONS) |
| `center_circle_radius` | Center circle radius (from ADULT_FIELD_DIMENSIONS) |
| `penalty_area_length` | Penalty area length (from ADULT_FIELD_DIMENSIONS) |
| `penalty_area_width` | Penalty area width (from ADULT_FIELD_DIMENSIONS) |
| `goal_area_length` | Goal area length (from ADULT_FIELD_DIMENSIONS) |
| `goal_area_width` | Goal area width (from ADULT_FIELD_DIMENSIONS) |

## 9. System Configuration (`src/soccer_framework/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `control_hz` | 30.0 Hz | Control loop frequency |
| `team_id` | 1 | Default team ID |
| `robot_names` | ("robot1", "robot2", "robot3") | Default robot names |
| `ready_slots` | {1: CENTER, 2: SIDE, 3: KEEPER} | Default ready slot assignments |

## Parameter Tuning Guide

### If players are too clustered:
- Increase `REPULSION_RADIUS` (currently 2.5m)
- Increase `REPULSION_STRENGTH` (currently 1.2m)
- Increase `support_min_spacing_m` (currently 1.15m)

### If players are too spread out:
- Decrease `support_depth_m` (currently 1.05m)
- Decrease `support_lateral_m` (currently 1.25m)

### If movement is too jerky:
- Increase `_ANGULAR_DEAD_ZONE` (currently 0.08 rad)
- Decrease `_ANGULAR_SPEED_FLOOR` (currently 0.35 rad/s)
- Increase `_ARRIVE_ANGLE` (currently 0.14 rad)

### If kicking is too aggressive/conservative:
- Adjust `soccer_kick_power` (currently 1.5)
- Adjust `soccer_kick_enter_distance` (currently 2.5m)

### If obstacle avoidance is too conservative:
- Decrease `obstacle_safety_margin` (currently 0.22m)
- Decrease `teammate_obstacle_radius` (currently 0.48m)
