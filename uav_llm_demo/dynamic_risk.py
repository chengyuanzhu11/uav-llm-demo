"""Shared metric contract for spherical dynamic-object envelopes.

Positions and obstacle velocities retain the scene's grid / grid-per-second API.
All radii, clearances, and vehicle velocities in this module are SI quantities.
These are simulation envelopes, not measured mesh contacts or physical bird sizes.
"""
from math import isfinite

from .motion import SCALE, norm

UAV_RADIUS_M = 0.4
SEPARATION_MARGIN_M = 0.6
AVOIDANCE_BUFFER_M = 0.3
NEAR_MISS_MARGIN_M = 1.0


def physical_radius_m(obstacle):
    value = getattr(obstacle, "radius_m", None)
    radius = float(obstacle.radius) * SCALE[0] if value is None else float(value)
    if not isfinite(radius) or radius < 0:
        raise ValueError("Dynamic-object radius must be finite and nonnegative.")
    return radius


def contact_distance_m(radius_m):
    """Center distance at which the object and UAV envelopes touch."""
    return radius_m + UAV_RADIUS_M


def separation_distance_m(radius_m):
    """Required center distance, excluding forecast uncertainty / planning buffer."""
    return contact_distance_m(radius_m) + SEPARATION_MARGIN_M


def envelope_fields(obstacle):
    radius = physical_radius_m(obstacle)
    return dict(radius_m=radius, radius_unit="horizontal_grid_cell", uav_radius_m=UAV_RADIUS_M,
                separation_margin_m=SEPARATION_MARGIN_M,
                contact_distance_m=contact_distance_m(radius),
                separation_distance_m=separation_distance_m(radius),
                near_miss_distance_m=separation_distance_m(radius) + NEAR_MISS_MARGIN_M)


def risk_levels(position, vehicle_position, velocity, vehicle_velocity=(0.0, 0.0, 0.0), *,
                radius_m, horizon_seconds=3.0):
    """Current and relative-motion closest-approach risk, in meters / seconds.

    This is a short-term constant-velocity proximity indicator, not the learned or
    history-based trajectory predictor used by avoidance. Vehicle velocity is m/s;
    the observed obstacle velocity is grid units/s.
    """
    relative = tuple((position[i] - vehicle_position[i]) * SCALE[i] for i in range(3))
    relative_velocity = tuple(velocity[i] * SCALE[i] - vehicle_velocity[i] for i in range(3))
    current_distance = norm(relative)
    relative_speed = norm(relative_velocity)
    dot = sum(relative[i] * relative_velocity[i] for i in range(3))
    closest_time = (max(0.0, min(float(horizon_seconds), -dot / relative_speed ** 2))
                    if relative_speed > 1e-9 else 0.0)
    closest_distance = norm(tuple(relative[i] + relative_velocity[i] * closest_time for i in range(3)))
    closing_speed = -dot / current_distance if current_distance > 1e-9 else 0.0
    contact = contact_distance_m(radius_m)
    separation = separation_distance_m(radius_m)
    current_contact = current_distance - contact
    current_clearance = current_distance - separation
    predicted_clearance = closest_distance - separation
    if current_contact < 0:
        threat = "critical"
    elif current_clearance < 0 or predicted_clearance < 0:
        threat = "high"
    elif current_clearance < NEAR_MISS_MARGIN_M or (closing_speed > 0 and predicted_clearance < NEAR_MISS_MARGIN_M):
        threat = "medium"
    else:
        threat = "low"
    return dict(distance=current_distance, predicted_distance=closest_distance,
                distance_m=current_distance, predicted_distance_m=closest_distance, distance_unit="m",
                contact_clearance_m=current_contact, separation_clearance_m=current_clearance,
                predicted_separation_clearance_m=predicted_clearance,
                relative_speed_mps=relative_speed, closing_speed_mps=closing_speed,
                closest_approach_seconds=closest_time, risk_horizon_seconds=float(horizon_seconds),
                risk_prediction="relative_constant_velocity", threat_level=threat)
