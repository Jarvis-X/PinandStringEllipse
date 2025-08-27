import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import casadi as ca
import do_mpc

def _safe_norm(v):
    return ca.sqrt(ca.sum1(v**2) + 1e-12)

class CableRobotSystem:
    """
    Simulates the dynamics of a cable-driven robot system based on the paper
    "Aerial cable hitch control based on ellipsoid manipulation".
    """
    def __init__(self, p_i0, v_i0, l12, l34, m, m_i, dt, c_d=0.0):
        self.n = p_i0.shape[1]
        self.dt = dt
        self.l12 = l12
        self.l34 = l34
        self.m = m
        self.m_i = m_i
        self.c_d = c_d

        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.u = np.zeros((4, self.n))
        self.f_ext = np.zeros(self.n)

        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess)
        self.v = np.zeros(self.n)

        self.history = {
            "p": [self.p.copy()],
            "p_i": [self.p_i.copy()],
            "tensions": [],
            "u": [self.u.copy()]
        }

    def _constraint_loss(self, p):
        err1 = np.linalg.norm(p - self.p_i[0]) + np.linalg.norm(p - self.p_i[1]) - self.l12
        err2 = np.linalg.norm(p - self.p_i[2]) + np.linalg.norm(p - self.p_i[3]) - self.l34
        return err1**2 + err2**2

    def _solve_initial_p(self, guess):
        res = minimize(self._constraint_loss, guess, method='BFGS')
        return res.x

    def _unit_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _compute_dynamics(self):
        r = np.array([self.p - self.p_i[i] for i in range(4)])
        r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        r_hat = np.array([self._unit_vector(r[i]) for i in range(4)])
        r_mag = np.linalg.norm(r, axis=1)

        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(self.n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])
        alpha = self.m * np.array([1/self.m_i[0] + 1/self.m_i[1], 1/self.m_i[2] + 1/self.m_i[3]])

        def c_term(i):
            return (1/self.m_i[i]) * r_hat[i] @ self.u[i] - r_hat_dot[i] @ r_dot[i]
        
        c1, c2 = c_term(0) + c_term(1), c_term(2) + c_term(3)

        M = np.array([
            [np.linalg.norm(r_hat[0] + r_hat[1])**2, (r_hat[0] + r_hat[1]) @ (r_hat[2] + r_hat[3])],
            [(r_hat[2] + r_hat[3]) @ (r_hat[0] + r_hat[1]), np.linalg.norm(r_hat[2] + r_hat[3])**2]
        ])
        M += np.diag(alpha)
        
        RHS = -self.m * np.array([c1, c2])
        damping_correction = np.array([(r_hat[0] + r_hat[1]) @ (self.c_d * self.v), (r_hat[2] + r_hat[3]) @ (self.c_d * self.v)])
        RHS -= damping_correction
        RHS += np.array([(r_hat[0] + r_hat[1]) @ self.f_ext, (r_hat[2] + r_hat[3]) @ self.f_ext])

        try:
            tensions = np.linalg.solve(M, RHS)
        except np.linalg.LinAlgError:
            tensions = np.zeros(2)

        t12, t34 = max(0, tensions[0]), max(0, tensions[1])
        t_per_segment = np.array([t12, t12, t34, t34])

        tension_force = -t12 * (r_hat[0] + r_hat[1]) - t34 * (r_hat[2] + r_hat[3])
        damping_force = -self.c_d * self.v + self.f_ext
        a = (tension_force + damping_force) / self.m
        a_i = np.array([(t_per_segment[i] * r_hat[i] + self.u[i]) / self.m_i[i] for i in range(4)])
        return a, a_i, t_per_segment

    def euler_integrate(self):
        a, a_i, tensions = self._compute_dynamics()
        v_unprojected = self.v + self.dt * a
        p_unprojected = self.p + self.dt * v_unprojected
        v_i = self.v_i + self.dt * a_i
        p_i = self.p_i + self.dt * v_i
        return v_unprojected, p_unprojected, v_i, p_i, tensions

    def step(self):
        v_unprojected, p_unprojected, v_i, p_i, tensions = self.euler_integrate()
        self.v, self.v_i, self.p_i = v_unprojected, v_i, p_i
        p_corrected = self._solve_initial_p(p_unprojected)
        correction = (p_corrected - p_unprojected) / self.dt
        self.v += 0.5 * correction
        self.p = p_corrected
        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())

    def build_mpc(
            self, 
            p_ref_val, 
            d12_ref_val, 
            d34_ref_val, 
            n_ref_val, 
            N=10, 
            dt_mpc=None, 
            Qp=10.0, 
            Qshape=1.0, 
            Qcot=1.0, 
            Ru=1e-3):
        if ca is None or do_mpc is None: raise ImportError("casadi and do-mpc must be installed.")
        if dt_mpc is None: dt_mpc = self.dt

        n = self.n
        model = do_mpc.model.Model('discrete')

        # --- State variables (unified vector) ---
        nx_total = n + 4*n + n + 4*n
        p = model.set_variable('_x', 'p', (n, 1))
        p_i = model.set_variable('_x', 'p_i', (n*4, 1))
        v = model.set_variable('_x', 'v', (n, 1))
        v_i = model.set_variable('_x', 'v_i', (n*4, 1))
        # x = model.set_variable('_x', 'x', (nx_total, 1))
        
        # idx = 0
        # p = x[idx:idx+n]; idx += n
        # p_i = x[idx:idx+4*n]; idx += 4*n
        # v = x[idx:idx+n]; idx += n
        # v_i = x[idx:idx+4*n]; idx += 4*n
        
        p_i_reshaped = ca.reshape(p_i, (n, 4))
        v_i_reshaped = ca.reshape(v_i, (n, 4))

        # --- Control inputs ---
        u = model.set_variable('_u', 'u', (n*4, 1))
        u_reshaped = ca.reshape(u, (n, 4))

        # --- Time-varying parameters ---
        p_ref = model.set_variable('_tvp', 'p_ref', (n, 1))
        d12_ref = model.set_variable('_tvp', 'd12_ref', (1, 1))
        d34_ref = model.set_variable('_tvp', 'd34_ref', (1, 1))
        n_ref = model.set_variable('_tvp', 'n_ref', (n, 1))

        # --- Dynamics ---
        r = [p - p_i_reshaped[:, i] for i in range(4)]
        r_mag = [_safe_norm(ri) for ri in r]
        r_hat = [r[i]/r_mag[i] for i in range(4)]
        r_dot = [v - v_i_reshaped[:, i] for i in range(4)]
        r_hat_dot = [(1/r_mag[i] * r_dot[i].T @ (ca.SX.eye(n) - r_hat[i] @ r_hat[i].T)).T for i in range(4)]

        n1, n2 = r_hat[0] + r_hat[1], r_hat[2] + r_hat[3]
        M11 = n1.T @ n1 + self.m*(1/self.m_i[0] + 1/self.m_i[1])
        M22 = n2.T @ n2 + self.m*(1/self.m_i[2] + 1/self.m_i[3])
        M12 = n1.T @ n2
        Mmat = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M12, M22))

        term1 = (r_hat[0].T @ u_reshaped[:, 0])/self.m_i[0] + (r_hat[1].T @ u_reshaped[:, 1])/self.m_i[1]
        term2 = (r_hat[2].T @ u_reshaped[:, 2])/self.m_i[2] + (r_hat[3].T @ u_reshaped[:, 3])/self.m_i[3]
        v0_1 = -self.m*(term1 - r_hat_dot[0].T@r_dot[0] - r_hat_dot[1].T@r_dot[1]) - self.c_d*n1.T@v + n1.T@self.f_ext
        v0_2 = -self.m*(term2 - r_hat_dot[2].T@r_dot[2] - r_hat_dot[3].T@r_dot[3]) - self.c_d*n2.T@v + n2.T@self.f_ext
        v0 = ca.vertcat(v0_1, v0_2)

        t_var = ca.solve(Mmat, v0)
        t12, t34 = t_var[0], t_var[1]

        tension_force = -(t12*n1 + t34*n2)
        a = (tension_force - self.c_d*v + self.f_ext)/self.m
        a_i = ca.SX(n, 4)
        for i in range(4): a_i[:, i] = ((ca.if_else(i < 2, t12, t34))*r_hat[i] + u_reshaped[:, i])/self.m_i[i]

        p_next = p + dt_mpc*v
        v_next = v + dt_mpc*a
        p_i_next = p_i_reshaped + dt_mpc*v_i_reshaped
        v_i_next = v_i_reshaped + dt_mpc*a_i

        # x_next = ca.vertcat(p_next, ca.reshape(p_i_next, (n*4, 1)), v_next, ca.reshape(v_i_next, (n*4, 1)))
        # model.set_rhs('x', x_next)
        model.set_rhs('p', p_next)
        model.set_rhs('p_i', ca.reshape(p_i_next, (n*4, 1)))
        model.set_rhs('v', v_next)
        model.set_rhs('v_i', ca.reshape(v_i_next, (n*4, 1)))
        model.setup()
        
        # model setup is O.K.
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
        'n_horizon': 10,  # Prediction horizon
        't_step': self.dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 0.001,
            'ipopt.max_iter': 100,
            'ipopt.print_level': 2, 
            # 'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }
        mpc.set_param(**setup_mpc)


        d12 = _safe_norm(p_i_reshaped[:, 0] - p_i_reshaped[:, 1])
        d34 = _safe_norm(p_i_reshaped[:, 2] - p_i_reshaped[:, 3])
        n2_norm2 = n2.T @ n2 + 1e-12
        proj = (n1.T @ n2 / n2_norm2) * n2
        e_c = _safe_norm(n1 - proj)

        lterm = Qp*((p - p_ref).T@(p - p_ref)) + 0.0*Qshape*(d12 - d12_ref)**2 + 0.0*Qshape*(d34 - d34_ref)**2 + 0.0*Qcot*e_c**2
        mterm = Qp*((p - p_ref).T@(p - p_ref)) + 0.0*Qshape*(d12 - d12_ref)**2 + 0.0*Qshape*(d34 - d34_ref)**2
        print(lterm)
        print(mterm)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=Ru)

        mpc.bounds['lower','_u','u'] = -1*np.ones((n*4, 1))
        mpc.bounds['upper','_u','u'] = 1*np.ones((n*4, 1))

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp', 0, 'p_ref'] = p_ref_val
            tvp_template['_tvp', 0, 'd12_ref'] = d12_ref_val
            tvp_template['_tvp', 0, 'd34_ref'] = d34_ref_val
            tvp_template['_tvp', 0, 'n_ref'] = n_ref_val
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        print("MPC built successfully.")
        
        self._do_mpc = {'mpc': mpc}
        return mpc

    def controller_mpc_do_mpc(self, mpc):
        x0 = np.vstack([self.p, self.p_i, self.v, self.v_i]).reshape(-1, 1)
        try:
            u_opt = mpc.make_step(x0)
            return np.asarray(u_opt).reshape(self.n, 4).T
        except Exception as e:
            print(f"MPC solve failed; fallback: {e}")
            return np.zeros((4, self.n))

    def run(self, steps):
        p_ref_val = np.zeros((self.n, 1))
        d12_ref_val = np.array([[self.l12 * 3/4]])
        d34_ref_val = np.array([[self.l34 * 3/4]])
        n_ref_val = np.array([1.0, 0.0, 0.0])[:self.n].reshape(self.n, 1)
        
        mpc = self.build_mpc(p_ref_val, d12_ref_val, d34_ref_val, n_ref_val)
        mpc.set_initial_guess()

        for i in range(steps):
            print(f"Step {i+1}/{steps}")
            u_cmd = self.controller_mpc_do_mpc(mpc)
            self.u[:] = u_cmd
            self.step()

    def _get_ellipse_points(self, f1, f2, major_axis_length):
        center = (f1 + f2) / 2
        dist = np.linalg.norm(f1 - f2)
        if dist >= major_axis_length: return None
        a = major_axis_length / 2.0
        c = dist / 2.0
        b = np.sqrt(max(a**2 - c**2, 1e-9))
        angle = np.arctan2(f2[1] - f1[1], f2[0] - f1[0])
        t = np.linspace(0, 2 * np.pi, 100)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = R @ np.vstack((a * np.cos(t), b * np.sin(t))) + center[:, np.newaxis]
        return points

    def _get_ellipsoid_points(self, f1, f2, major_axis_length):
        center = (f1 + f2) / 2
        dist = np.linalg.norm(f1 - f2)
        if dist >= major_axis_length: return None, None, None
        a = major_axis_length / 2.0
        c = dist / 2.0
        b = np.sqrt(max(a**2 - c**2, 1e-9))
        u, v = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
        x, y, z = a*np.outer(np.cos(u), np.sin(v)), b*np.outer(np.sin(u), np.sin(v)), b*np.outer(np.ones_like(u), np.cos(v))
        f1_to_f2 = self._unit_vector(f2 - f1)
        if np.allclose(f1_to_f2, [1, 0, 0]): rot_mat = np.identity(3)
        else:
            v_axis = np.cross([1, 0, 0], f1_to_f2)
            s, c_angle = np.linalg.norm(v_axis), np.dot([1, 0, 0], f1_to_f2)
            vx = np.array([[0, -v_axis[2], v_axis[1]], [v_axis[2], 0, -v_axis[0]], [-v_axis[1], v_axis[0], 0]])
            rot_mat = np.identity(3) + vx + vx@vx*((1-c_angle)/(s**2))
        points = np.stack([x, y, z], axis=-1) @ rot_mat.T + center
        return points[..., 0], points[..., 1], points[..., 2]

    def animate(self, frame_skip=0):
        p_hist, p_i_hist, tensions_hist, u_hist = (np.array(self.history[k]) for k in ["p", "p_i", "tensions", "u"])
        fig = plt.figure(figsize=(12, 12))
        step = frame_skip + 1
        animation_frames = range(0, len(p_hist), step)

        if self.n == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title("3D Cable Robot Simulation")
            all_points = np.vstack(p_i_hist)
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
            ax.set_zlim(all_points[:,2].min()-1, all_points[:,2].max()+1)
            hitch_point, = ax.plot([],[],[], 'ko', ms=8, zorder=10, label='Hitch')
            robot_dots = [ax.plot([],[],[], 'o', color=c, ms=10)[0] for c in ['r','g','b','m']]
            cables = [ax.plot([],[],[], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['r','r','b','b']]
            force_arrows = [ax.plot([],[],[], '-', color=c, lw=2)[0] for c in ['r','g','b','m']]
            input_texts = [ax.text(0,0,0,'',fontsize=8,color=c) for c in ['r','g','b','m']]
            wireframe1_lines, wireframe2_lines = [], []
            def update_3d(frame_index):
                nonlocal wireframe1_lines, wireframe2_lines
                p, p_i, u_arr = p_hist[frame_index], p_i_hist[frame_index], u_hist[frame_index]
                hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
                for i in range(4):
                    robot_dots[i].set_data_3d([p_i[i,0]], [p_i[i,1]], [p_i[i,2]])
                    cables[i].set_data_3d([p[0],p_i[i,0]],[p[1],p_i[i,1]],[p[2],p_i[i,2]])
                    start, end = p_i[i], p_i[i] + 0.5*u_arr[i]
                    force_arrows[i].set_data_3d([start[0],end[0]], [start[1],end[1]], [start[2],end[2]])
                    input_texts[i].set_position((p_i[i,0], p_i[i,1])); input_texts[i].set_3d_properties(p_i[i,2])
                    input_texts[i].set_text(f"u={np.linalg.norm(u_arr[i]):.2f}")
                for wf in wireframe1_lines + wireframe2_lines: wf.remove()
                wireframe1_lines.clear(); wireframe2_lines.clear()
                x1,y1,z1 = self._get_ellipsoid_points(p_i[0], p_i[1], self.l12)
                if x1 is not None: wireframe1_lines.append(ax.plot_wireframe(x1,y1,z1,color='c',alpha=0.2))
                x2,y2,z2 = self._get_ellipsoid_points(p_i[2], p_i[3], self.l34)
                if x2 is not None: wireframe2_lines.append(ax.plot_wireframe(x2,y2,z2,color='m',alpha=0.2))
                return [hitch_point,*robot_dots,*cables,*force_arrows,*input_texts]
            ani = FuncAnimation(fig, update_3d, frames=animation_frames, blit=False, interval=50)
        elif self.n == 2:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal'); ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title("2D Cable Robot Simulation")
            ax.grid(True, linestyle='--', alpha=0.6)
            all_points = np.vstack(p_i_hist)
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
            hitch_point, = ax.plot([],[], 'ko', ms=8, zorder=10, label='Hitch')
            robot_dots = [ax.plot([],[], 'o', color=c, ms=10)[0] for c in ['r','g','b','m']]
            cables = [ax.plot([],[], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['r','r','b','b']]
            force_arrows = [ax.add_patch(FancyArrowPatch((0,0),(0,0), color=c, arrowstyle='->', mutation_scale=20, lw=1.5)) for c in ['r','g','b','m']]
            tension_texts = [ax.text(0,0,'',fontsize=9,ha='center',va='center',backgroundcolor=(1,1,1,0.7)) for _ in range(4)]
            input_texts = [ax.text(0,0,'',fontsize=8,ha='left',va='bottom',color=c) for c in ['r','g','b','m']]
            ellipse1_line, = ax.plot([],[], 'c--', lw=1, label='Ellipse 1-2')
            ellipse2_line, = ax.plot([],[], 'm--', lw=1, label='Ellipse 3-4')
            def update_2d(frame_index):
                p, p_i = p_hist[frame_index], p_i_hist[frame_index]
                t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
                u_arr = u_hist[frame_index] if frame_index < len(u_hist) else np.zeros((4,self.n))
                hitch_point.set_data([p[0]], [p[1]])
                for i in range(4):
                    robot_dots[i].set_data([p_i[i,0]], [p_i[i,1]])
                    cables[i].set_data([p[0],p_i[i,0]], [p[1],p_i[i,1]])
                    mid = (p + p_i[i])/2
                    tension_texts[i].set_position((mid[0], mid[1])); tension_texts[i].set_text(f"{t_arr[i]:.2f} N")
                    start, end = p_i[i], p_i[i] + 0.5*u_arr[i]
                    force_arrows[i].set_positions(start, end)
                    input_texts[i].set_position((p_i[i,0], p_i[i,1]))
                    input_texts[i].set_text(f"u={np.linalg.norm(u_arr[i]):.2f}")
                e1_pts = self._get_ellipse_points(p_i[0], p_i[1], self.l12)
                if e1_pts is not None: ellipse1_line.set_data(e1_pts[0], e1_pts[1])
                e2_pts = self._get_ellipse_points(p_i[2], p_i[3], self.l34)
                if e2_pts is not None: ellipse2_line.set_data(e2_pts[0], e2_pts[1])
                return [hitch_point, *robot_dots, *cables, *tension_texts, ellipse1_line, ellipse2_line, *force_arrows, *input_texts]
            ani = FuncAnimation(fig, update_2d, frames=animation_frames, blit=True, interval=2)
        else:
            print(f"Animation not supported for n={self.n}.")
            return
        plt.show()

def main():
    n = 2
    dt = 0.001
    steps = 2000

    p_i0 = np.array([
        [-2.0, -2.5, 0.0],
        [-1.5,  2.0, 0.5],
        [ 1.5,  2.0, -0.5],
        [ 2.0, -2.5, 0.0]
    ])[:, :n]

    v_i0 = np.zeros((4, n))
    m = 0.1
    m_i = np.ones(4) * 0.5
    l12 = 6.5
    l34 = 7.5
    c_d = 0.1

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)
    sim.f_ext = np.array([0.0, 0.0, 0])[:n]

    sim.run(steps)
    sim.animate(frame_skip=19)

if __name__ == "__main__":
    main()
