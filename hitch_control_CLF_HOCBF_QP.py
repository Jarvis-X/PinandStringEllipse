"""
Entry point for the cable-suspended hitch simulator.

Choose between a single run that produces an animation video, or a batch run that
produces statistical plots, via the MODE constant below. All other knobs are also
plain constants at the top of this file.
"""

import os
import numpy as np
import cvxpy as cp

from cable_robot_system import CableRobotSystem
from references import RefConfig, make_ref_func
from visualization import animate, plot_single_run
from batch_eval import run_batch, plot_batch_statistics


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
MODE = 'single'           # 'single' (one run + video) or 'batch' (N runs + stats plots)
NUM_TRIALS = 3            # only used when MODE == 'batch'
OUTPUT_DIR = 'output'

# ---------------------------------------------------------------------------
# Simulation / solver parameters
# ---------------------------------------------------------------------------
SOLVER = cp.OSQP
L12 = 6.2
L34 = 5.8
N = 3                     # spatial dimension (2 or 3)
DT = 0.005
STEPS = 5000

# Physical parameters
M_HITCH = 0.005           # virtual mass at the hitch
M_ROBOT = 0.35            # mass of each robot
C_D = 0.2                 # damping coefficient at the hitch (unknown to controller)

CONTROLLER = 'clf_cbf'    # 'clf_cbf' (robot-centered CLF) or 'ellipsoids_clf_cbf'

# ---------------------------------------------------------------------------
# Reference trajectory parameters
# ---------------------------------------------------------------------------
REF_CFG = RefConfig(
    l12=L12,
    l34=L34,
    hitch_mag=(0.6, 0.5, 0.0),
    hitch_freq=(0.4, 0.4, 0.0),
    d12_mag=0.2,
    d12_freq=0.5,
    d34_mag=0.3,
    d34_freq=0.8,
    yaw_dot_ref=0.3,
    n12_rand=float(np.random.normal(0, 0.01)),
    n34_rand=float(np.random.normal(0, 0.01)),
)

# ---------------------------------------------------------------------------
# CLF-HOCBF-QP gains and constraints
# ---------------------------------------------------------------------------
CLF_PARAMS = {
    'K_p':       np.diag([5.0] * N),
    'K_n':       np.diag([0.5] * N),
    'k_d':       0.2,
    'K_p_cas':   np.diag([1.0] * N),
    'K_n_cas':   np.diag([0.5] * N),
    'k_d_cas':   0.4,
    'gamma':     DT * 200,
    'alpha':     1e6,
    'beta':      10.0,
    'lambda':    100.0,
    't_min':     0.1,
    'u_max':     20.0,
    'Kp_robot':  np.diag([20.0] * N),
    'Kv_robot':  np.diag([20.0] * N),
}


# ---------------------------------------------------------------------------
# Sim factory: builds a fresh simulator with randomized initial conditions.
# Used directly in single mode and once per trial in batch mode.
# ---------------------------------------------------------------------------
def build_sim():
    shift = np.random.normal(0, 1.0, (3,))
    p_i0 = np.array([
        [-1.8, -2.0, -0.25] + shift,
        [-2.0,  2.3,  0.25] + shift,
        [ 2.0,  2.2, -0.35] + shift,
        [ 2.1, -1.9,  0.20] + shift,
    ])[:, :N] + np.random.normal(0, 0.15, (4, 3))[:, :N]

    v_i0 = np.random.normal(0, 0.05, (4, 3))[:, :N]
    m_i = np.ones(4) * M_ROBOT

    sim = CableRobotSystem(
        p_i0=p_i0, v_i0=v_i0,
        l12=L12, l34=L34,
        m=M_HITCH, m_i=m_i,
        dt=DT, c_d=C_D,
        solver=SOLVER,
    )
    sim.f_ext = np.zeros(N)
    sim.clf_params = CLF_PARAMS
    return sim


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
def run_single():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = build_sim()
    ref_func = make_ref_func(CONTROLLER, REF_CFG)
    try:
        sim.run(STEPS, ref_func, controller_type=CONTROLLER, verbose=False)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    video_path = os.path.join(OUTPUT_DIR, f"hitch_{N}D.mp4")
    animate(sim, dt=DT, save_path=video_path, frame_skip=19)
    plot_single_run(sim, dt=DT, output_dir=OUTPUT_DIR, prefix=f"single_{N}D")


def run_batch_mode():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ref_func = make_ref_func(CONTROLLER, REF_CFG)
    V_hists, Err_hists = run_batch(
        num_trials=NUM_TRIALS,
        sim_factory=build_sim,
        ref_func=ref_func,
        controller_type=CONTROLLER,
        steps=STEPS,
        verbose=False,
    )
    plot_batch_statistics(
        V_hists, Err_hists,
        dt=DT,
        output_dir=OUTPUT_DIR,
        prefix=f"batch_{N}D",
        title_suffix=f"{NUM_TRIALS} trials",
    )


def main():
    if MODE == 'single':
        run_single()
    elif MODE == 'batch':
        run_batch_mode()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'single' or 'batch'.")


if __name__ == "__main__":
    main()
