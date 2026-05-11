import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('pdf', fonttype=42)
plt.rcParams["font.family"] = "serif"
plt.rc('axes', titlesize=18)
plt.rc("axes", labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)


def run_batch(num_trials, sim_factory, ref_func, controller_type, steps, verbose=False, progress=True):
    """
    Run num_trials independent simulations and collect Lyapunov / error-sum traces.

    sim_factory: callable () -> CableRobotSystem (must produce a fresh sim each call;
                 randomness in initialization should live inside the factory).
    ref_func:    callable t -> ref_tuple, shared across trials.
    """
    V_histories = []
    Err_histories = []

    for trial in range(num_trials):
        if progress:
            print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        sim = sim_factory()
        try:
            sim.run(steps, ref_func, controller_type=controller_type, verbose=verbose, progress=progress)
        except KeyboardInterrupt:
            print("\nBatch interrupted by user.")
            break

        V_hist = np.array(sim.history["V"]).copy()
        Err_hist = np.array(sim.history["error_mag_sum"]).copy()
        if len(V_hist) > 0:
            V_histories.append(V_hist)
        if len(Err_hist) > 0:
            Err_histories.append(Err_hist)

    return V_histories, Err_histories


def _stack_min_length(traces):
    """Trim to the shortest trace length so we can stack into a 2D array."""
    if not traces:
        return None
    min_len = min(len(t) for t in traces)
    return np.stack([t[:min_len] for t in traces], axis=0)


def plot_batch_statistics(V_histories, Err_histories, dt, output_dir=".", prefix="batch", title_suffix=""):
    """Produce mean / ±std / min-max shaded plots for V and error-sum across trials."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    V_arr = _stack_min_length(V_histories)
    Err_arr = _stack_min_length(Err_histories)

    if V_arr is not None:
        v_mean = V_arr.mean(axis=0)
        v_std = V_arr.std(axis=0)
        v_min = V_arr.min(axis=0)
        v_max = V_arr.max(axis=0)
        t_axis = np.arange(len(v_mean)) * dt

        fig, ax = plt.subplots(figsize=(12, 6))
        title = "Lyapunov function Over Time" + (f" ({title_suffix})" if title_suffix else "")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V")
        upper = v_mean.max() + v_std.max() * 2.0 if v_std.max() > 0 else v_mean.max() * 1.1 + 1
        ax.set_ylim([0, upper])
        ax.set_xlim([0, t_axis[-1]])
        ax.plot(t_axis, v_mean, 'b-', label='Mean V')
        ax.fill_between(t_axis, v_mean - v_std, v_mean + v_std, color='b', alpha=0.2, label='±1 Std Dev')
        ax.fill_between(t_axis, v_min, v_max, color='b', alpha=0.1, label='Min-Max Range')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=18)
        v_path = os.path.join(output_dir, f"{prefix}_V.pdf")
        fig.savefig(v_path, bbox_inches='tight')
        plt.close(fig)
        paths.append(v_path)

    if Err_arr is not None:
        e_mean = Err_arr.mean(axis=0)
        e_std = Err_arr.std(axis=0)
        e_min = Err_arr.min(axis=0)
        e_max = Err_arr.max(axis=0)
        t_axis = np.arange(len(e_mean)) * dt

        fig, ax = plt.subplots(figsize=(12, 6))
        title = "Error sum Over Time" + (f" ({title_suffix})" if title_suffix else "")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error Sum")
        upper = e_mean.max() + e_std.max() * 2.0 if e_std.max() > 0 else e_mean.max() * 1.1 + 1
        ax.set_ylim([0, upper])
        ax.set_xlim([0, t_axis[-1]])
        ax.plot(t_axis, e_mean, 'r-', label='Mean Error Sum')
        ax.fill_between(t_axis, e_mean - e_std, e_mean + e_std, color='r', alpha=0.2, label='±1 Std Dev')
        ax.fill_between(t_axis, e_min, e_max, color='r', alpha=0.1, label='Min-Max Range')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=18)
        e_path = os.path.join(output_dir, f"{prefix}_Err.pdf")
        fig.savefig(e_path, bbox_inches='tight')
        plt.close(fig)
        paths.append(e_path)

    return paths
