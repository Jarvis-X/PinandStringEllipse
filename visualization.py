import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

matplotlib.rc('pdf', fonttype=42)
plt.rcParams["font.family"] = "serif"
plt.rc('axes', titlesize=18)
plt.rc("axes", labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)


def _unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.zeros_like(v)


def _ellipse_points(f1, f2, major_axis_length):
    center = (f1 + f2) / 2
    dist = np.linalg.norm(f1 - f2)
    if dist >= major_axis_length:
        return None
    a = major_axis_length / 2.0
    c = dist / 2.0
    b = np.sqrt(max(a**2 - c**2, 1e-9))
    angle = np.arctan2(f2[1] - f1[1], f2[0] - f1[0])
    t = np.linspace(0, 2 * np.pi, 100)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return R @ np.vstack((a * np.cos(t), b * np.sin(t))) + center[:, np.newaxis]


def _ellipsoid_points(f1, f2, major_axis_length):
    center = (f1 + f2) / 2
    dist = np.linalg.norm(f1 - f2)
    if dist >= major_axis_length:
        return None, None, None
    a = major_axis_length / 2.0
    c = dist / 2.0
    b = np.sqrt(max(a**2 - c**2, 1e-9))
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = b * np.outer(np.ones_like(u), np.cos(v))
    f1_to_f2 = _unit_vector(f2 - f1)
    if np.allclose(f1_to_f2, [1, 0, 0]):
        rot_mat = np.identity(3)
    else:
        v_axis = np.cross([1, 0, 0], f1_to_f2)
        s = np.linalg.norm(v_axis)
        c_angle = np.dot([1, 0, 0], f1_to_f2)
        vx = np.array([[0, -v_axis[2], v_axis[1]],
                       [v_axis[2], 0, -v_axis[0]],
                       [-v_axis[1], v_axis[0], 0]])
        rot_mat = np.identity(3) + vx + vx @ vx * ((1 - c_angle) / (s**2))
    points = np.stack([x, y, z], axis=-1) @ rot_mat.T + center
    return points[..., 0], points[..., 1], points[..., 2]


def _pick_writer(save_path, fps):
    """Choose a matplotlib writer based on the file extension. Returns (writer, resolved_path)."""
    ext = os.path.splitext(save_path)[1].lower()
    if ext in ('.mp4', '.avi', '.mov'):
        try:
            return FFMpegWriter(fps=fps, bitrate=2000), save_path
        except Exception as e:
            print(f"FFMpegWriter unavailable ({e}); falling back to .gif")
            save_path = os.path.splitext(save_path)[0] + '.gif'
            return PillowWriter(fps=fps), save_path
    if ext == '.gif':
        return PillowWriter(fps=fps), save_path
    save_path = save_path + '.gif'
    return PillowWriter(fps=fps), save_path


def animate(sim, dt, save_path="cable_robot_animation.mp4", frame_skip=19, fps=30):
    """Render the simulation history to a video file. Headless (no plt.show)."""
    p_hist = np.array(sim.history["p"])
    p_ref_hist = np.array(sim.history["p_ref"])
    p_i_hist = np.array(sim.history["p_i"])
    p_i_ref_hist = np.array(sim.history["p_i_ref"])
    tensions_hist = np.array(sim.history["tensions"])
    u_hist = np.array(sim.history["u"])

    n12_ref_hist = np.array(sim.history.get("n12_ref", []))
    n34_ref_hist = np.array(sim.history.get("n34_ref", []))
    n12_hist = np.array(sim.history.get("n12", []))
    n34_hist = np.array(sim.history.get("n34", []))
    d12_ref_hist = np.array(sim.history.get("d12_ref", []))
    d34_ref_hist = np.array(sim.history.get("d34_ref", []))
    d12_hist = np.array(sim.history.get("d12", []))
    d34_hist = np.array(sim.history.get("d34", []))

    fig = plt.figure(figsize=(12, 12))
    step = frame_skip + 1
    animation_frames = list(range(0, len(p_hist), step))
    robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
    all_points = np.vstack([p_hist, p_i_hist.reshape(-1, sim.n), p_ref_hist, p_i_ref_hist.reshape(-1, sim.n)])

    if sim.n == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("3D Cable Robot Simulation")
        ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
        ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
        ax.set_zlim(all_points[:, 2].min() - 1, all_points[:, 2].max() + 1)

        hitch_point, = ax.plot([], [], [], 'ko', ms=8, zorder=10, label='Hitch Point')
        robot_dots = [ax.plot([], [], [], 'o', color=c, ms=10, label='robot position')[0] for c in robot_colors]
        hitch_ref, = ax.plot([], [], [], 'gx', ms=10, mew=3, label='hitch reference')
        robot_ref = [ax.plot([], [], [], 'x', color=c, ms=10, label='robot reference')[0] for c in robot_colors]
        cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
        input_arrows = [ax.quiver([], [], [], [], [], [], color=c, length=0.5, normalize=True, arrow_length_ratio=0.1) for c in robot_colors]
        normal_ref1_arrow = normal_ref2_arrow = None
        wireframe1 = wireframe2 = None
        n12_arrow = n34_arrow = None

        dist_ref_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                                  verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

        def update_3d(frame_index):
            nonlocal wireframe1, wireframe2, input_arrows, normal_ref1_arrow, normal_ref2_arrow, n12_arrow, n34_arrow
            p = p_hist[frame_index]; p_i = p_i_hist[frame_index]; u = u_hist[frame_index]
            p_ref = p_ref_hist[frame_index]; p_i_ref = p_i_ref_hist[frame_index]
            hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
            hitch_ref.set_data_3d([p_ref[0]], [p_ref[1]], [p_ref[2]])
            for i in range(4):
                robot_dots[i].set_data_3d([p_i[i, 0]], [p_i[i, 1]], [p_i[i, 2]])
                cables[i].set_data_3d([p[0], p_i[i, 0]], [p[1], p_i[i, 1]], [p[2], p_i[i, 2]])
                robot_ref[i].set_data_3d([p_i_ref[i, 0]], [p_i_ref[i, 1]], [p_i_ref[i, 2]])
                if input_arrows[i] is not None:
                    input_arrows[i].remove()
                input_arrows[i] = ax.quiver(p_i[i, 0], p_i[i, 1], p_i[i, 2],
                                            u[i, 0], u[i, 1], u[i, 2],
                                            color=robot_colors[i], length=np.linalg.norm(u[i]) * 0.1,
                                            normalize=False, arrow_length_ratio=0.3)

            n12_ref = n12_ref_hist[frame_index]; n34_ref = n34_ref_hist[frame_index]
            n12 = n12_hist[frame_index]; n34 = n34_hist[frame_index]
            if normal_ref1_arrow:
                normal_ref1_arrow.remove()
            if normal_ref2_arrow:
                normal_ref2_arrow.remove()
            normal_ref1_arrow = ax.quiver(p_ref[0], p_ref[1], p_ref[2], n12_ref[0], n12_ref[1], n12_ref[2],
                                          color='c', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2)
            normal_ref2_arrow = ax.quiver(p_ref[0], p_ref[1], p_ref[2], n34_ref[0], n34_ref[1], n34_ref[2],
                                          color='m', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2)
            if n12_arrow:
                n12_arrow.remove()
            if n34_arrow:
                n34_arrow.remove()
            n12_arrow = ax.quiver(p[0], p[1], p[2], n12[0], n12[1], n12[2],
                                  color='c', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2, linestyle='dashed')
            n34_arrow = ax.quiver(p[0], p[1], p[2], n34[0], n34[1], n34[2],
                                  color='m', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2, linestyle='dashed')

            if len(d12_ref_hist) > 0:
                text_str = (f'$d_{{12,ref}}$: {d12_ref_hist[frame_index]:.2f}, '
                            f'$d_{{34,ref}}$: {d34_ref_hist[frame_index]:.2f}\n'
                            f'$d_{{12}}$: {d12_hist[frame_index]:.2f}, '
                            f'$d_{{34}}$: {d34_hist[frame_index]:.2f}')
                dist_ref_text.set_text(text_str)

            if wireframe1:
                wireframe1.remove()
            if wireframe2:
                wireframe2.remove()
            x1, y1, z1 = _ellipsoid_points(p_i[0], p_i[1], sim.l12)
            if x1 is not None:
                wireframe1 = ax.plot_wireframe(x1, y1, z1, color='c', alpha=0.1)
            x2, y2, z2 = _ellipsoid_points(p_i[2], p_i[3], sim.l34)
            if x2 is not None:
                wireframe2 = ax.plot_wireframe(x2, y2, z2, color='m', alpha=0.1)
            return [hitch_point, *robot_dots, *cables, *input_arrows]

        ani = FuncAnimation(fig, update_3d, frames=animation_frames, blit=False, interval=50)

    elif sim.n == 2:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title("2D Cable Robot Simulation")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
        ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)

        hitch_point, = ax.plot([], [], 'ko', ms=8, zorder=10, label='Hitch Point')
        robot_dots = [ax.plot([], [], 'o', color=c, ms=10, label='robot position')[0] for c in robot_colors]
        hitch_ref, = ax.plot([], [], 'gx', ms=10, mew=3, label='hitch reference')
        robot_ref = [ax.plot([], [], 'x', color=c, ms=10, label='robot reference')[0] for c in robot_colors]
        cables = [ax.plot([], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
        tension_texts = [ax.text(0, 0, '', fontsize=9, ha='center', va='center', backgroundcolor=(1, 1, 1, 0.7)) for _ in range(4)]
        input_arrows = [ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc=c, ec=c, length_includes_head=True) for c in robot_colors]
        ellipse1_line, = ax.plot([], [], 'c--', lw=1, label='Constraint Ellipse 1-2')
        ellipse2_line, = ax.plot([], [], 'm--', lw=1, label='Constraint Ellipse 3-4')
        normal_ref1_arrow = normal_ref2_arrow = None
        n12_arrow = n34_arrow = None
        dist_ref_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        ax.legend()

        def update_2d(frame_index):
            nonlocal normal_ref1_arrow, normal_ref2_arrow, n12_arrow, n34_arrow
            p = p_hist[frame_index]; p_i = p_i_hist[frame_index]; u = u_hist[frame_index]
            p_ref = p_ref_hist[frame_index]; p_i_ref = p_i_ref_hist[frame_index]
            t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
            hitch_point.set_data([p[0]], [p[1]])
            hitch_ref.set_data([p_ref[0]], [p_ref[1]])
            for i in range(4):
                robot_dots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
                cables[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])
                robot_ref[i].set_data([p_i_ref[i, 0]], [p_i_ref[i, 1]])
                mid_point = (p + p_i[i]) / 2
                tension_texts[i].set_position((mid_point[0], mid_point[1]))
                tension_texts[i].set_text(f"{t_arr[i]:.2f} N")
                input_arrows[i].remove()
                input_arrows[i] = ax.arrow(p_i[i, 0], p_i[i, 1], u[i, 0] * 0.1, u[i, 1] * 0.1,
                                           head_width=0.1, head_length=0.1,
                                           fc=robot_colors[i], ec=robot_colors[i],
                                           length_includes_head=False)

            n12_ref = n12_ref_hist[frame_index]; n34_ref = n34_ref_hist[frame_index]
            n12 = n12_hist[frame_index]; n34 = n34_hist[frame_index]
            if normal_ref1_arrow:
                normal_ref1_arrow.remove()
            if normal_ref2_arrow:
                normal_ref2_arrow.remove()
            normal_ref1_arrow = ax.arrow(p_ref[0], p_ref[1], n12_ref[0], n12_ref[1],
                                         head_width=0.1, head_length=0.1, fc='c', ec='c', lw=2,
                                         length_includes_head=False)
            normal_ref2_arrow = ax.arrow(p_ref[0], p_ref[1], n34_ref[0], n34_ref[1],
                                         head_width=0.1, head_length=0.1, fc='m', ec='m', lw=2,
                                         length_includes_head=False)
            if n12_arrow:
                n12_arrow.remove()
            if n34_arrow:
                n34_arrow.remove()
            n12_arrow = ax.arrow(p[0], p[1], n12[0], n12[1],
                                 head_width=0.1, head_length=0.1, fc='c', ec='c', lw=2,
                                 length_includes_head=False, linestyle='dashed')
            n34_arrow = ax.arrow(p[0], p[1], n34[0], n34[1],
                                 head_width=0.1, head_length=0.1, fc='m', ec='m', lw=2,
                                 length_includes_head=False, linestyle='dashed')

            if len(d12_ref_hist) > 0:
                text_str = (f'$d_{{12,ref}}$: {d12_ref_hist[frame_index]:.2f}, '
                            f'$d_{{34,ref}}$: {d34_ref_hist[frame_index]:.2f}\n'
                            f'$d_{{12}}$: {d12_hist[frame_index]:.2f}, '
                            f'$d_{{34}}$: {d34_hist[frame_index]:.2f}')
                dist_ref_text.set_text(text_str)

            ellipse1_pts = _ellipse_points(p_i[0], p_i[1], sim.l12)
            if ellipse1_pts is not None:
                ellipse1_line.set_data(ellipse1_pts[0], ellipse1_pts[1])
            ellipse2_pts = _ellipse_points(p_i[2], p_i[3], sim.l34)
            if ellipse2_pts is not None:
                ellipse2_line.set_data(ellipse2_pts[0], ellipse2_pts[1])
            return [hitch_point, *robot_dots, *cables, *tension_texts, *input_arrows, ellipse1_line, ellipse2_line]

        ani = FuncAnimation(fig, update_2d, frames=animation_frames, blit=False, interval=20)

    else:
        plt.close(fig)
        print(f"Animation is not supported for n={sim.n}.")
        return None

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer, save_path = _pick_writer(save_path, fps)
    print(f"Saving animation to {save_path} ...")
    ani.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Animation saved: {save_path}")
    return save_path


def plot_single_run(sim, dt, output_dir=".", prefix="single_run"):
    """Save Lyapunov function and error-sum plots for a single simulation."""
    os.makedirs(output_dir, exist_ok=True)
    V_hist = np.array(sim.history["V"])
    Err_hist = np.array(sim.history["error_mag_sum"])
    paths = []

    fig_v, ax_v = plt.subplots(figsize=(12, 8))
    ax_v.set_title("Lyapunov function Over Time")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("V")
    ax_v.plot(np.arange(len(V_hist)) * dt, V_hist, 'b-')
    ax_v.grid(True, linestyle='--', alpha=0.6)
    v_path = os.path.join(output_dir, f"{prefix}_V.pdf")
    fig_v.savefig(v_path)
    plt.close(fig_v)
    paths.append(v_path)

    fig_e, ax_e = plt.subplots(figsize=(12, 8))
    ax_e.set_title("Error sum Over Time")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel("Error Sum")
    ax_e.plot(np.arange(len(Err_hist)) * dt, Err_hist, 'b-')
    ax_e.grid(True, linestyle='--', alpha=0.6)
    e_path = os.path.join(output_dir, f"{prefix}_Err.pdf")
    fig_e.savefig(e_path)
    plt.close(fig_e)
    paths.append(e_path)

    return paths
