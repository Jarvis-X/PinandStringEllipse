"""
demo_cable_length.py

Stand-alone 2-D demonstration that the cable-length sums are constant:

    l_12 = ||p - p1|| + ||p - p2||  =  L12  (constant)
    l_34 = ||p - p3|| + ||p - p4||  =  L34  (constant)

The hitch point p is the intersection of two ellipses whose foci are the four
robot positions.  Robots follow prescribed trajectories; the hitch position is
found at every step by minimising the cable-constraint residuals.

Output: output/cable_constant_length.mp4  (falls back to .gif if ffmpeg absent)
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

from visualization import _pick_writer

matplotlib.rc('pdf', fonttype=42)
plt.rcParams["font.family"] = "serif"
plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)

# ── Parameters ─────────────────────────────────────────────────────────────────
# Cable lengths calibrated so the intersection (hitch) sits inside the robot
# cluster: sum12 at centroid ≈ 4.47, sum34 at centroid ≈ 4.22.
L12 = 4.5      # total length of cable pair 1-2
L34 = 4.3      # total length of cable pair 3-4
DT = 0.04      # simulation time step (s)
T_END = 16.0   # total simulation time (s)
FPS = 25       # animation frame rate
FRAME_SKIP = 2 # render every FRAME_SKIP-th simulation step
OUTPUT_DIR = "output"


# ── Robot trajectories (prescribed) ────────────────────────────────────────────
def robot_positions(t):
    """
    Pair 1-2 on the left (x ≈ -2), pair 3-4 on the right (x ≈ +1.8).
    Both pairs oriented mostly vertically so the ellipse intersection falls
    between them, near the centroid of all four robots.
    """
    # ── Cable pair 1-2: left cluster ──────────────────────────────────────
    d12     = 2.0 + 0.6 * np.sin(0.40 * t)        # d12 ∈ [1.4, 2.6] << L12
    angle12 = np.pi / 2 + 0.15 * t                 # mostly vertical, rotating
    c12     = np.array([-2.0 + 0.25 * np.sin(0.15 * t),
                         0.0  + 0.30 * np.sin(0.20 * t)])
    half12  = (d12 / 2) * np.array([np.cos(angle12), np.sin(angle12)])
    p1 = c12 + half12
    p2 = c12 - half12

    # ── Cable pair 3-4: right cluster ─────────────────────────────────────
    d34     = 1.8 + 0.4 * np.cos(0.35 * t)        # d34 ∈ [1.4, 2.2] << L34
    angle34 = np.pi / 2 - 0.15 * t                 # mostly vertical, rotating
    c34     = np.array([1.80 + 0.25 * np.sin(0.18 * t),
                        0.00 + 0.30 * np.sin(0.25 * t)])
    half34  = (d34 / 2) * np.array([np.cos(angle34), np.sin(angle34)])
    p3 = c34 + half34
    p4 = c34 - half34

    return np.array([p1, p2, p3, p4])


# ── Hitch solver ───────────────────────────────────────────────────────────────
def solve_hitch(p_i, l12, l34, guess=None):
    """Minimise cable-length constraint residuals to find the hitch position."""
    def loss(p):
        e1 = np.linalg.norm(p - p_i[0]) + np.linalg.norm(p - p_i[1]) - l12
        e2 = np.linalg.norm(p - p_i[2]) + np.linalg.norm(p - p_i[3]) - l34
        return e1 ** 2 + e2 ** 2

    if guess is None:
        # Bias upward slightly to select the consistent solution branch
        guess = np.mean(p_i, axis=0) + np.array([0.0, 0.3])
    res = minimize(loss, guess, method='BFGS')
    return res.x, res.fun


# ── Simulation ─────────────────────────────────────────────────────────────────
def run_simulation():
    steps    = int(T_END / DT)
    t_hist   = np.arange(steps) * DT
    p_i_hist = np.zeros((steps, 4, 2))
    p_hist   = np.zeros((steps, 2))

    p = None
    for k, t in enumerate(t_hist):
        p_i         = robot_positions(t)
        p_i_hist[k] = p_i
        p, _        = solve_hitch(p_i, L12, L34, guess=p)
        p_hist[k]   = p

    return t_hist, p_i_hist, p_hist


# ── Animation ──────────────────────────────────────────────────────────────────
def animate_demo(t_hist, p_i_hist, p_hist,
                 save_path="output/cable_constant_length.mp4"):

    RC  = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']  # robot colours
    C12 = '#e07b00'   # orange – cable pair 1-2
    C34 = '#7b00e0'   # purple – cable pair 3-4

    # Pre-compute per-segment lengths and sums
    segs  = [np.linalg.norm(p_hist - p_i_hist[:, i], axis=1) for i in range(4)]
    sum12 = segs[0] + segs[1]
    sum34 = segs[2] + segs[3]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[2.8, 1.0],
                            hspace=0.44, wspace=0.28)
    ax    = fig.add_subplot(gs[0, :])   # main 2-D spatial view (full width)
    ax_12 = fig.add_subplot(gs[1, 0])  # time series for cable pair 1-2
    ax_34 = fig.add_subplot(gs[1, 1])  # time series for cable pair 3-4

    # Spatial extent
    all_pts = np.vstack([p_i_hist.reshape(-1, 2), p_hist])
    pad = 0.9
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x  (m)')
    ax.set_ylabel('y  (m)')
    ax.set_title(
        r'$\ell_{12} = \|p-p_1\|+\|p-p_2\| = \mathrm{const}$   and   '
        r'$\ell_{34} = \|p-p_3\|+\|p-p_4\| = \mathrm{const}$',
        fontsize=13, pad=6)
    ax.grid(True, linestyle='--', alpha=0.35)

    # Bottom time-series axes: y-range covers both individual segments and sum
    seg_all_min = min(s.min() for s in segs)
    y_lo = max(0.0, seg_all_min - 0.3)
    y_hi_12 = L12 + 0.3
    y_hi_34 = L34 + 0.3

    for axt, lval, colour, i1, i2, pair, y_hi in [
        (ax_12, L12, C12, '1', '2', '12', y_hi_12),
        (ax_34, L34, C34, '3', '4', '34', y_hi_34),
    ]:
        axt.set_xlim(t_hist[0], t_hist[-1])
        axt.set_ylim(y_lo, y_hi)
        axt.axhline(lval, color=colour, lw=1.5, ls='--', alpha=0.7,
                    label=rf'$\ell_{{{pair}}} = {lval}$')
        axt.set_xlabel('Time  (s)')
        axt.set_ylabel('Length  (m)')
        axt.set_title(
            rf'Cable {i1}–{i2}: $\|p\!-\!p_{i1}\| + \|p\!-\!p_{i2}\|$',
            fontsize=12)
        axt.grid(True, linestyle='--', alpha=0.35)

    # ── Main-plot artists ──────────────────────────────────────────────────
    hitch_dot, = ax.plot([], [], 'ko', ms=10, zorder=12, label='Hitch  $p$')
    rdots = [ax.plot([], [], 'o', color=RC[i], ms=11, zorder=8,
                     label=f'$p_{i+1}$')[0] for i in range(4)]
    clines = [ax.plot([], [], '-',
                      color=C12 if i < 2 else C34,
                      lw=2.5, alpha=0.85, solid_capstyle='round')[0]
              for i in range(4)]
    # Length labels: mid-point of each cable segment
    stexts = [ax.text(0, 0, '', fontsize=10, ha='center', va='center',
                      color=C12 if i < 2 else C34, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                alpha=0.85, ec='none'))
              for i in range(4)]
    # Info box: live numeric readout of sums
    info_box = ax.text(
        0.02, 0.97, '', transform=ax.transAxes, fontsize=11,
        va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.45', fc='lightyellow',
                  alpha=0.93, ec='goldenrod'))
    ax.legend(loc='upper right', fontsize=10, ncol=2)

    # ── Time-series artists ────────────────────────────────────────────────
    # Sum traces (labelled in legend)
    ls12, = ax_12.plot([], [], '-',  color=C12,   lw=2.2,
                       label=r'$\ell_1 + \ell_2$')
    ls34, = ax_34.plot([], [], '-',  color=C34,   lw=2.2,
                       label=r'$\ell_3 + \ell_4$')

    # Individual segment traces (drawn but NOT in legend)
    lp1, = ax_12.plot([], [], ':', color=RC[0], lw=1.5, label='_nolegend_')
    lp2, = ax_12.plot([], [], ':', color=RC[1], lw=1.5, label='_nolegend_')
    lp3, = ax_34.plot([], [], ':', color=RC[2], lw=1.5, label='_nolegend_')
    lp4, = ax_34.plot([], [], ':', color=RC[3], lw=1.5, label='_nolegend_')

    # Legend: only sum trace + dashed reference
    ax_12.legend(fontsize=10, loc='upper right')
    ax_34.legend(fontsize=10, loc='upper right')

    # Vertical time cursor spanning the full y-range
    t0 = t_hist[0]
    cur12, = ax_12.plot([t0, t0], [y_lo, y_hi_12], 'k-', lw=1.2, alpha=0.45)
    cur34, = ax_34.plot([t0, t0], [y_lo, y_hi_34], 'k-', lw=1.2, alpha=0.45)

    frames = list(range(0, len(t_hist), FRAME_SKIP))

    def update(fi):
        k   = frames[fi]
        p   = p_hist[k]
        p_i = p_i_hist[k]
        t   = t_hist[k]

        # Hitch and robots
        hitch_dot.set_data([p[0]], [p[1]])
        for i in range(4):
            rdots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
            clines[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])
            mid = (p + p_i[i]) / 2
            stexts[i].set_position(mid)
            stexts[i].set_text(f'{segs[i][k]:.2f}')

        # Live info box
        s1, s2 = segs[0][k], segs[1][k]
        s3, s4 = segs[2][k], segs[3][k]
        info_box.set_text(
            f'l12 = {s1:.3f} + {s2:.3f} = {s1+s2:.3f}  (const = {L12})\n'
            f'l34 = {s3:.3f} + {s4:.3f} = {s3+s4:.3f}  (const = {L34})'
        )

        # Time-series traces
        idx = k + 1
        ls12.set_data(t_hist[:idx], sum12[:idx])
        lp1.set_data( t_hist[:idx], segs[0][:idx])
        lp2.set_data( t_hist[:idx], segs[1][:idx])
        ls34.set_data(t_hist[:idx], sum34[:idx])
        lp3.set_data( t_hist[:idx], segs[2][:idx])
        lp4.set_data( t_hist[:idx], segs[3][:idx])

        # Advance time cursor
        cur12.set_xdata([t, t])
        cur34.set_xdata([t, t])

        return ([hitch_dot] + rdots + clines + stexts
                + [info_box, ls12, lp1, lp2, ls34, lp3, lp4, cur12, cur34])

    ani = FuncAnimation(fig, update, frames=len(frames),
                        blit=False, interval=int(1000 / FPS))

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer, save_path = _pick_writer(save_path, FPS)
    print(f"Saving animation -> {save_path} ...")
    ani.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def main():
    print("Simulating ...")
    t_hist, p_i_hist, p_hist = run_simulation()
    save_path = os.path.join(OUTPUT_DIR, "cable_constant_length.mp4")
    animate_demo(t_hist, p_i_hist, p_hist, save_path=save_path)


if __name__ == "__main__":
    main()
