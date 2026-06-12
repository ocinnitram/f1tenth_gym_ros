"""
vehicle_model_patch.py - with identified parameters

Higher-fidelity single-track model: Pacejka pure-slip tires, AWD longitudinal
split, combined-slip (Kamm) saturation, and first-order tire relaxation lag.
Monkey-patches base_classes.RaceCar.update_pose at deployment-sim time.

State vector is fixed and indexed downstream by position:
    x = [X, Y, delta, v, psi, r, beta]
        0  1    2     3   4   5    6
    (world X, world Y, steering angle, speed, yaw, yaw rate, body slip angle)
"""

from math import sin, cos, sqrt, atan2, atan

import numpy as np
from numba import njit
from f110_gym.envs import base_classes
from f110_gym.envs.dynamic_models import (
    vehicle_dynamics_ks,
    vehicle_dynamics_st,
    accl_constraints,
    steering_constraint,
)

# --- Tire-curve constants, solved once at import ----------------------------
# B, C, D are pinned from three measured anchors rather than a rig fit.
#
# Friction-circle radius is the PEAK grip, not the sliding asymptote.
# peak = LAT_PEAK_SCALE * mu * Fz ; sliding = mu * Fz. Ratio measured ~1.20.
LAT_PEAK_SCALE = 1.20
# C: shape factor pinning the falling-branch asymptote to the sliding level.
#    sin(C*pi/2) = 1/LAT_PEAK_SCALE so F(alpha->inf) = mu*Fz when D = peak.
_C_CURVE = 2.0 * (np.pi - np.arcsin(1.0 / LAT_PEAK_SCALE)) / np.pi
# ALPHA_PK: rear slip angle of the Pacejka peak [rad], measured 2026-06-11.
ALPHA_PK = 0.16
# B (rear): atan(B*ALPHA_PK) = pi/(2C) puts the sin argument at its max (peak
#    at ALPHA_PK). Front B scales this by the stock C_Sf/C_Sr at the call site.
_B_R = np.tan(np.pi / (2.0 * _C_CURVE)) / ALPHA_PK

# --- Lag / low-speed constants ----------------------------------------------
# Tire relaxation length [m]; time-domain tau = SIGMA_FY / vx (lag inert above
# vx ~ SIGMA_FY/dt = 3 m/s with dt=0.01s).
SIGMA_FY = 0.03
# Floor on vx when forming tau = SIGMA_FY/vx, so standstill doesn't divide by 0.
_SIGMA_V_FLOOR = 0.5
# Beta self-heal time constant in the kinematic branch [s] (bleed beta -> 0).
TAU_BETA_KIN = 0.15


@njit(cache=True)
def _axle_forces(Fx_total, F_yf0, F_yr0, mu, Fz_f, Fz_r):
    """
    Apply the AWD longitudinal split and the combined-slip (Kamm) circle
    per axle, returning the saturated forces actually delivered to the road.

    Inputs:
        Fx_total : total commanded longitudinal force (drive +, brake -)
        F_yf0    : front PURE-SLIP lateral force (from Pacejka, pre-saturation)
        F_yr0    : rear  PURE-SLIP lateral force
        mu       : friction coefficient
        Fz_f     : front-axle normal load
        Fz_r     : rear-axle normal load

    Returns:
        Fx_f, F_yf, Fx_r, F_yr, Fx_transmitted
        (saturated per-axle long/lat forces + total long force the tires
         actually put down, which is the wheel-slip signal for _v_wheel)
    """
    # AWD split: share Fx_total across axles in proportion to normal load.
    Fx_f = Fx_total * Fz_f / (Fz_f + Fz_r)
    Fx_r = Fx_total - Fx_f

    # Per-axle friction-circle radii (peak grip).
    cap_f = LAT_PEAK_SCALE * mu * Fz_f
    cap_r = LAT_PEAK_SCALE * mu * Fz_r

    # Combined-force demand magnitudes.
    n_f = sqrt(pow(Fx_f, 2) + pow(F_yf0, 2))
    n_r = sqrt(pow(Fx_r, 2) + pow(F_yr0, 2))

    # Scale Fx and Fy by the same scalar so the force shrinks onto the circle
    # without rotating; pass through unchanged when inside it.
    s_f = cap_f / n_f if n_f > cap_f else 1.0
    s_r = cap_r / n_r if n_r > cap_r else 1.0

    return Fx_f*s_f, F_yf0*s_f, Fx_r*s_r, F_yr0*s_r, Fx_f*s_f + Fx_r*s_r


@njit(cache=True)
def _steady_state_lateral_forces(x, accl, mu, C_Sf, C_Sr, lf, lr, h, m):
    """
    Pure-slip Pacejka lateral forces at the current state (the LAG TARGETS).

    Nonlinear slip-angle geometry (arctan2-based, valid at any beta). Below the
    low-speed gate the tires bear no lateral load in this model, so return 0,0.

    x = [X, Y, delta, v, psi, r, beta]
    Returns: F_yf, F_yr   (front, rear pure-slip lateral forces)
    """
    g = 9.81
    v = x[3]

    # Low-speed gate (same threshold as the kinematic branch).
    if v*cos(x[6]) < 0.5:
        return 0.0, 0.0

    L = lf + lr

    # Slip-angle geometry. Rear has no steer term; only the front subtracts delta.
    vx = v * cos(x[6])
    vy_f = v*sin(x[6]) + lf*x[5]
    vy_r = v*sin(x[6]) - lr*x[5]
    alpha_f = x[2] - atan2(vy_f, vx)
    alpha_r = -atan2(vy_r, vx)

    # Normal loads with longitudinal load transfer.
    Fz_f = m*(g*lr - accl*h)/L
    Fz_r = m*(g*lf + accl*h)/L

    # Pacejka pure-slip lateral forces.
    D_f = LAT_PEAK_SCALE * mu * Fz_f
    D_r = LAT_PEAK_SCALE * mu * Fz_r
    B_f = _B_R * C_Sf/C_Sr
    Fy_f = D_f * sin(_C_CURVE * atan(B_f * alpha_f))
    Fy_r = D_r * sin(_C_CURVE * atan(_B_R * alpha_r))

    return Fy_f, Fy_r


@njit(cache=True)
def _dynamics_with_lagged_forces(
    x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I,
    s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,
    F_yf0, F_yr0,
):
    """
    7-element state derivative xdot, using EXTERNALLY-provided lagged pure-slip
    lateral forces F_yf0, F_yr0 (held constant across the RK4 sub-stages — their
    relaxation lag is integrated outside, in the orchestration step).

    Combined-slip (Kamm) scaling IS applied here, instantaneously, from the
    current longitudinal demand m*accl via _axle_forces.

    x = [X, Y, delta, v, psi, r, beta]
    Returns xdot, same 7 slots.
    """
    # Clamp the raw commands to the actuator limits before using them.
    u = np.array([
        steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
        accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max),
    ])

    # Low-speed kinematic branch. Below v_x = 0.5 m/s the slip-angle geometry
    # (atan2(vy, vx)) and the (v, beta) divisions become ill-conditioned. Fall
    # back to the no-tire-force kinematic single-track (KS) model, which
    # propagates pose purely geometrically. KS returns a 5-vector
    # [Xdot, Ydot, deltadot, vdot, psidot]; append the two missing slots
    # (r_dot, beta_dot) to refill the 7-state.
    if x[3] * np.cos(x[6]) < 0.5:
        lwb = lf + lr
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(
            x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I,
            s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,
        )
        # r_dot: differentiate the kinematic yaw rate psi_dot = v/lwb*tan(delta)
        #        wrt time -> chain rule on v (vdot=u[1]) and delta (ddot=u[0]).
        # beta_dot: bleed beta toward 0 over TAU_BETA_KIN (self-heal — stops a
        #        spun-out car from freezing at beta~pi forever, see note below).
        return np.hstack((
            f_ks,
            np.array([
                u[1] / lwb * np.tan(x[2]) + x[3] / (lwb * np.cos(x[2]) ** 2) * u[0],
                -x[6] / TAU_BETA_KIN,
            ]),
        ))

    # Dynamic branch: body-frame (v_x, v_y, r) force balance.
    vx = x[3]*cos(x[6])
    vy = x[3]*sin(x[6])

    # Normal loads with longitudinal load transfer (a_x = u[1]).
    g = 9.81
    L = lf + lr
    Fz_f = m*(g*lr - u[1]*h)/L
    Fz_r = m*(g*lf + u[1]*h)/L

    # Combined-slip saturation of the longitudinal demand.
    Fx_f, F_yf, Fx_r, F_yr, _ = _axle_forces(m*u[1], F_yf0, F_yr0, mu, Fz_f, Fz_r)

    # Force/moment balance. The +vy*r / -vx*r terms are the rotating-frame
    # cross terms moved to the derivative side.
    cd = cos(x[2])
    sd = sin(x[2])
    vx_dot = (Fx_f*cd - F_yf*sd + Fx_r)/m + vy*x[5]
    vy_dot = (Fx_f*sd + F_yf*cd + F_yr)/m - vx*x[5]
    r_dot = (lf*(F_yf*cd + Fx_f*sd) - lr*F_yr) / I

    # Convert body-frame (vx_dot, vy_dot) back to state (v_dot, beta_dot).
    v_dot = (vx*vx_dot + vy*vy_dot) / x[3]
    beta_dot = (vx*vy_dot - vy*vx_dot) / pow(x[3], 2)

    X_dot = x[3]*cos(x[6]+x[4])
    Y_dot = x[3]*sin(x[6]+x[4])

    return np.array([X_dot, Y_dot, u[0], v_dot, x[5], r_dot, beta_dot])


def _ensure_tire_state(car):
    """Lazily attach the persistent per-car tire/odometry state."""
    if not hasattr(car, '_F_yf'):
        car._F_yf = 0.0
    if not hasattr(car, '_F_yr'):
        car._F_yr = 0.0
    # Wheel/motor rotational velocity (VESC ERPM equivalent).
    if not hasattr(car, '_v_wheel'):
        car._v_wheel = 0.0


def _racecar_update_pose_lagged_sat(self, raw_steer, vel):
    """
    update_pose override with the Pacejka/AWD/combined-slip model.

    Inlines the actuator-delay + PID (actuator_patch logic) so the integrator
    can call our dynamics directly. This override replaces actuator_patch's
    update_pose, so apply it after that patch.
    """
    # Lazy import so this module doesn't depend on import order with actuator_patch.
    from f1tenth_gym_ros.actuator_patch import (
        _delayed_speed_cmd,
        _delayed_steer_cmd,
        better_pid,
        _reset_actuator_models,
    )
    if not hasattr(self, 'v_cmd_buffer'):
        _reset_actuator_models(self)
    _ensure_tire_state(self)

    # Actuator delay + PID -> (accl, sv).
    delayed_vel = _delayed_speed_cmd(self, vel)
    delayed_steer = _delayed_steer_cmd(self, raw_steer)
    accl, sv = better_pid(
        delayed_vel, delayed_steer, self.state[3], self.state[2],
        self.accel, self.delta_dot, self.time_step,
        self.params['sv_min'], self.params['sv_max'],
        max(float(self.params.get('tau_v', self.time_step)), 1e-9),
        max(float(self.params.get('tau_a', self.time_step)), 1e-9),
        self.params['a_max'],
        max(float(self.params['tau_delta']), 1e-9),
        max(float(self.params['tau_delta_dot']), 1e-9),
        max(float(self.params['delta_th']), 1e-9),
        float(self.params['steer_gamma']),
    )
    self.accel = accl
    self.delta_dot = sv

    p = self.params
    mu = p['mu']
    dyn_args = (
        mu, p['C_Sf'], p['C_Sr'], p['lf'], p['lr'], p['h'], p['m'], p['I'],
        p['s_min'], p['s_max'], p['sv_min'], p['sv_max'],
        p['v_switch'], p['a_max'], p['v_min'], p['v_max'],
    )

    # First-order relaxation lag on the pure-slip tire forces. Compute the
    # steady-state forces at the current state, then Euler-step the stored
    # forces toward them (tau = SIGMA_FY / vx).
    F_yf_ss, F_yr_ss = _steady_state_lateral_forces(self.state, accl, mu, p['C_Sf'], p['C_Sr'], p['lf'], p['lr'], p['h'], p['m'])
    vx_now = max(self.state[3] * np.cos(self.state[6]), _SIGMA_V_FLOOR) # floor before dividing
    tau_fy = SIGMA_FY / vx_now
    lag_alpha = min(1.0, self.time_step / tau_fy)           # clamp so dt can't overshoot
    self._F_yf += lag_alpha * (F_yf_ss - self._F_yf) 
    self._F_yr += lag_alpha * (F_yr_ss - self._F_yr)
    F_yf, F_yr = self._F_yf, self._F_yr
    

    # Wheel/motor rotational velocity (VESC ERPM equivalent). While the Kamm
    # circles transmit the full longitudinal demand, the wheels are grip-coupled
    # to the body (read v_x). When an axle saturates and the transmitted Fx falls
    # short of demand, the wheels slip and track the delayed commanded speed
    # through the motor's first-order response (tau_v).
    g = 9.81
    L = p['lf'] + p['lr']
    Fz_f_now = p['m'] * (g * p['lr'] - accl * p['h']) / L
    Fz_r_now = p['m'] * (g * p['lf'] + accl * p['h']) / L
    Fx_demand = p['m'] * accl

    _,_,_,_, Fx_transmitted = _axle_forces(Fx_demand, F_yf, F_yr, mu, Fz_f_now, Fz_r_now)
    if abs(Fx_demand - Fx_transmitted) > 0.05*abs(Fx_demand) + 1e-3:   # slipping
        tau_v = max(p.get('tau_v', self.time_step), self.time_step)
        self._v_wheel += (self.time_step/tau_v) * (delayed_vel - self._v_wheel)
    else:                                                             # gripping
        self._v_wheel = self.state[3]*cos(self.state[6])
    self._v_wheel = max(0.0, self._v_wheel)

    # Integrate the dynamics. The lagged forces are held constant across the
    # RK4 sub-stages.
    u_arr = np.array([sv, accl])

    if self.integrator is base_classes.Integrator.RK4:
        k1 = _dynamics_with_lagged_forces(self.state, u_arr, *dyn_args, F_yf, F_yr)
        k2_state = self.state + self.time_step * (k1 / 2)
        k2 = _dynamics_with_lagged_forces(k2_state, u_arr, *dyn_args, F_yf, F_yr)
        k3_state = self.state + self.time_step * (k2 / 2)
        k3 = _dynamics_with_lagged_forces(k3_state, u_arr, *dyn_args, F_yf, F_yr)
        k4_state = self.state + self.time_step * k3
        k4 = _dynamics_with_lagged_forces(k4_state, u_arr, *dyn_args, F_yf, F_yr)
        self.state = self.state + self.time_step * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    elif self.integrator is base_classes.Integrator.Euler:
        f = _dynamics_with_lagged_forces(self.state, u_arr, *dyn_args, F_yf, F_yr)
        self.state = self.state + self.time_step * f
    else:
        raise SyntaxError(
            f"Invalid Integrator Specified. Provided {self.integrator.name}. "
            f"Please choose RK4 or Euler"
        )
    # Yaw wrap, then scan from the front-mounted lidar.
    if self.state[4] > 2 * np.pi:
        self.state[4] = self.state[4] - 2 * np.pi
    elif self.state[4] < 0:
        self.state[4] = self.state[4] + 2 * np.pi

    scan_x = self.state[0] + self.lidar_dist * np.cos(self.state[4])
    scan_y = self.state[1] + self.lidar_dist * np.sin(self.state[4])
    scan_pose = np.array([scan_x, scan_y, self.state[4]])
    scan_simulator = base_classes.RaceCar.scan_simulator
    if scan_simulator is None:
        raise RuntimeError("RaceCar scan simulator is not initialized.")
    return scan_simulator.scan(scan_pose, self.scan_rng)


# Captured at patch-apply time: whatever reset/update_pose was bound at the
# moment of patching, typically actuator_patch's versions. Apply order matters:
# this patch applies after actuator_patch so its update_pose takes precedence.
_ORIGINAL_RACECAR_RESET = None
_ORIGINAL_RACECAR_UPDATE_POSE = None
_FRICTION_PATCH_APPLIED = False


def _racecar_reset_with_tire_state(self, pose):
    """reset override: run the captured original reset, then zero tire state."""
    _ORIGINAL_RACECAR_RESET(self, pose) #type:ignore
    self._F_yf = 0.0
    self._F_yr = 0.0
    self._v_wheel = 0.0


def apply_friction_circle_patch():
    """
    Install the Pacejka/AWD/combined-slip single-track model onto RaceCar.

    Apply AFTER apply_custom_actuator_patch() so this update_pose override wins
    (gym_bridge wires it in that order).
    """
    global _FRICTION_PATCH_APPLIED, _ORIGINAL_RACECAR_RESET, _ORIGINAL_RACECAR_UPDATE_POSE
    if _FRICTION_PATCH_APPLIED:
        return

    # Capture the currently-bound methods before overwriting them; the reset
    # override chains to the captured original.
    _ORIGINAL_RACECAR_RESET = base_classes.RaceCar.reset
    _ORIGINAL_RACECAR_UPDATE_POSE = base_classes.RaceCar.update_pose

    base_classes.RaceCar.reset = _racecar_reset_with_tire_state
    base_classes.RaceCar.update_pose = _racecar_update_pose_lagged_sat

    _FRICTION_PATCH_APPLIED = True