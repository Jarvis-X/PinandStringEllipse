from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class RefConfig:
    """Parameters for the ellipsoid-based reference trajectory."""
    # Cable lengths (used by both reference styles)
    l12: float = 6.2
    l34: float = 5.8

    # Hitch oscillation magnitudes per axis (x, y, z)
    hitch_mag: Tuple[float, float, float] = (0.6, 0.5, 0.0)
    hitch_freq: Tuple[float, float, float] = (0.4, 0.4, 0.0)

    # Major-axis-length oscillation
    d12_mag: float = 0.2
    d12_freq: float = 0.5
    d34_mag: float = 0.3
    d34_freq: float = 0.8

    # Yaw rate of the ellipsoid normal vectors
    yaw_dot_ref: float = 0.3

    # Random perturbation magnitudes for normal vectors (drawn at config build time)
    n12_rand: float = 0.0
    n34_rand: float = 0.0

    # Derived offsets (auto-filled in __post_init__ if 0)
    d12_offset: float = field(default=0.0)
    d34_offset: float = field(default=0.0)

    def __post_init__(self):
        if self.d12_offset == 0.0:
            self.d12_offset = self.l12 / np.sqrt(2) - self.n12_rand
        if self.d34_offset == 0.0:
            self.d34_offset = self.l34 / np.sqrt(2) - (self.n12_rand + self.n34_rand)


def robot_traj(t, l12, l34):
    """Robot-side trajectory: returns (p_ref, v_ref, p_i_ref, v_i_ref) in 3D."""
    MAG1, MAG2 = 0.5, 0.6
    FREQ1, FREQ2 = 0.35, 0.75
    p_ref = np.array([0.0, 0.0, 0.25]) + np.array([MAG1 * np.cos(FREQ1 * t + 1.0), MAG2 * np.sin(FREQ2 * t), 0.0])
    v_ref = np.array([-MAG1 * FREQ1 * np.sin(FREQ1 * t + 1.0), MAG2 * FREQ2 * np.cos(FREQ2 * t), 0.0])

    delta_p_i = np.array([
        [-l12 * np.sqrt(2) / 4, -l12 * np.sqrt(2) / 4, 0.0],
        [-l12 * np.sqrt(2) / 4,  l12 * np.sqrt(2) / 4, 0.0],
        [ l34 * np.sqrt(2) / 4,  l34 * np.sqrt(2) / 4, 0.0],
        [ l34 * np.sqrt(2) / 4, -l34 * np.sqrt(2) / 4, 0.0],
    ])

    yaw_dot_ref = 0.1
    yaw_ref = yaw_dot_ref * t
    pitch_ref = 0.0

    cy, sy = np.cos(yaw_ref), np.sin(yaw_ref)
    cp_, sp = np.cos(pitch_ref), np.sin(pitch_ref)

    R_ref = np.array([
        [cy * cp_, -sy, cy * sp],
        [sy * cp_,  cy, sy * sp],
        [-sp,        0,     cp_],
    ])
    rotated_delta_p_i = (R_ref @ delta_p_i.T).T
    p_i_ref = p_ref + rotated_delta_p_i

    R_dot_ref = yaw_dot_ref * np.array([
        [-sy * cp_, -cy, -sy * sp],
        [ cy * cp_, -sy,  cy * sp],
        [        0,   0,        0],
    ])
    v_i_ref = (R_dot_ref @ delta_p_i.T).T + v_ref
    return p_ref, v_ref, p_i_ref, v_i_ref


def robot_traj_ellipsoids(t, cfg: RefConfig):
    """Adapter: derives ellipsoid normals/distances from robot-side trajectory."""
    p_ref, v_ref, p_i_ref, v_i_ref = robot_traj(t, cfg.l12, cfg.l34)
    r_ref = np.array([p_ref - p_i_ref[i] for i in range(4)])
    r_dot_ref = np.array([v_ref - v_i_ref[i] for i in range(4)])
    r_hat_ref = np.array([r_ref[i] / np.linalg.norm(r_ref[i]) for i in range(4)])
    r_mag_ref = np.linalg.norm(r_ref, axis=1)

    n12_ref = r_hat_ref[0] + r_hat_ref[1]
    n34_ref = r_hat_ref[2] + r_hat_ref[3]
    d12_ref = np.linalg.norm(p_i_ref[0] - p_i_ref[1])
    d34_ref = np.linalg.norm(p_i_ref[2] - p_i_ref[3])

    r_hat_dot = np.array([(
        1.0 / r_mag_ref[i] * r_dot_ref[i] @ (np.eye(3) - np.outer(r_hat_ref[i], r_hat_ref[i]))
    ).flatten() for i in range(4)])

    v_n12_ref = r_hat_dot[0] + r_hat_dot[1]
    v_n34_ref = r_hat_dot[2] + r_hat_dot[3]
    v_d12_ref = (v_i_ref[0] - v_i_ref[1]) @ (p_i_ref[0] - p_i_ref[1]) / d12_ref
    v_d34_ref = (v_i_ref[2] - v_i_ref[3]) @ (p_i_ref[2] - p_i_ref[3]) / d34_ref

    return (p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref,
            v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref)


def get_robot_refs_from_ellipsoids(p_ref, v_ref, n12_ref, v_n12_ref, n34_ref, v_n34_ref,
                                   d12_ref, v_d12_ref, d34_ref, v_d34_ref):
    """
    Eq. (18) of the paper: derive robot reference positions/velocities from a desired
    hitch position, ellipsoid normal vectors, and major-axis lengths.
    """
    p_i_ref = np.zeros((4, 3))
    v_i_ref = np.zeros((4, 3))

    refs = [
        (d12_ref, v_d12_ref, n12_ref, v_n12_ref),
        (d12_ref, v_d12_ref, n12_ref, v_n12_ref),
        (d34_ref, v_d34_ref, n34_ref, v_n34_ref),
        (d34_ref, v_d34_ref, n34_ref, v_n34_ref),
    ]
    kappa_ref = np.array([0., 0., 1.])
    eps = 1e-9

    for i in range(4):
        d_ref, v_d_ref, n_ref, v_n_ref = refs[i]

        n_mag = np.linalg.norm(n_ref)
        n_mag_sq = n_mag**2
        denom = np.sqrt(max(4.0 - n_mag_sq, eps))
        n_hat = n_ref / (n_mag + eps)
        cross_term = np.cross(kappa_ref, n_hat)
        sign = (-1)**(i)
        offset = (d_ref / 2.0) * (-n_ref / denom - sign * cross_term)
        p_i_ref[i] = p_ref + offset

        n_mag_dot = (n_ref @ v_n_ref) / (n_mag + eps)
        denom_dot = -(n_mag * n_mag_dot) / (denom + eps)
        term1_dot = (-v_n_ref * denom - (-n_ref) * denom_dot) / (denom**2 + eps)
        n_hat_dot = (v_n_ref - n_hat * (n_hat @ v_n_ref)) / (n_mag + eps)
        cross_term_dot = np.cross(kappa_ref, n_hat_dot)

        offset_dot = (v_d_ref / 2.0) * (-n_ref / denom + sign * cross_term) + \
                     (d_ref / 2.0) * (term1_dot + sign * cross_term_dot)
        v_i_ref[i] = v_ref + offset_dot

    return p_i_ref, v_i_ref


def robot_traj_from_ellipsoids(t, cfg: RefConfig):
    """Ellipsoid-side trajectory: hitch oscillates, ellipsoid axes oscillate, normals rotate."""
    HM1, HM2, HM3 = cfg.hitch_mag
    HF1, HF2, HF3 = cfg.hitch_freq

    p_ref = np.array([
        HM1 * np.cos(HF1 * t),
        HM2 * np.sin(HF2 * t - 1.0),
        HM3 * np.sin(HF3 * t + 1.0),
    ])
    v_ref = np.array([
        -HM1 * HF1 * np.sin(HF1 * t),
        HM2 * HF2 * np.cos(HF2 * t - 1.0),
        HM3 * HF3 * np.cos(HF3 * t + 1.0),
    ])

    d12_ref = cfg.d12_offset + cfg.d12_mag * np.sin(cfg.d12_freq * t)
    v_d12_ref = cfg.d12_mag * cfg.d12_freq * np.cos(cfg.d12_freq * t)
    d34_ref = cfg.d34_offset + cfg.d34_mag * np.sin(cfg.d34_freq * t)
    v_d34_ref = cfg.d34_mag * cfg.d34_freq * np.cos(cfg.d34_freq * t)

    angle = cfg.yaw_dot_ref * t
    angle_dot = cfg.yaw_dot_ref

    n12_ref = (1.414 + cfg.n12_rand) * np.array([np.cos(angle), np.sin(angle), 0.])
    n34_ref = -(1 + cfg.n34_rand) * n12_ref
    v_n12_ref = angle_dot * np.array([-np.sin(angle), np.cos(angle), 0.])
    v_n34_ref = -v_n12_ref

    p_i_ref, v_i_ref = get_robot_refs_from_ellipsoids(
        p_ref, v_ref, n12_ref, v_n12_ref, n34_ref, v_n34_ref,
        d12_ref, v_d12_ref, d34_ref, v_d34_ref,
    )

    return (p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref,
            v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref)


def make_ref_func(controller_type: str, cfg: RefConfig):
    """Return a closure t -> ref_tuple for the chosen controller type."""
    if controller_type == 'clf_cbf':
        return lambda t: robot_traj_from_ellipsoids(t, cfg)
    if controller_type == 'ellipsoids_clf_cbf':
        return lambda t: robot_traj_ellipsoids(t, cfg)
    raise ValueError(f"Unknown controller_type: {controller_type}")
