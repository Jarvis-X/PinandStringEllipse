import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import casadi as ca
import do_mpc

# smallest_det_M = 1e6
# greatest_damping_force = 0.0
def _safe_norm(v):
    return ca.sqrt(ca.sum1(v**2) + 1e-12)

class CableRobotSystem:
    """
    Simulates the dynamics of a cable-driven robot system based on the paper
    "Aerial cable hitch control based on ellipsoid manipulation".

    The system consists of a central hitch point connected by two cables to four
    robots. The hitch point's position is constrained to the intersection of two
    ellipsoids, where the robot positions act as foci. The dynamics
    are governed by Newton's second law, and the cable tensions are solved using
    the affine expression derived in the paper.
    """
    def __init__(self, p_i0, v_i0, l12, l34, m, m_i, dt, c_d=0.0):
        """
        Initializes the cable robot system.

        Args:
            p_i0 (np.ndarray): Initial positions of the 4 robots (4xN array).
            v_i0 (np.ndarray): Initial velocities of the 4 robots (4xN array).
            l12 (float): The fixed total length of the first cable (connecting R1 and R2).
            l34 (float): The fixed total length of the second cable (connecting R3 and R4).
            m (float): The virtual mass of the hitch point.
            m_i (np.ndarray): The masses of the 4 robots (1x4 array).
            dt (float): The simulation time step.
            c_d (float): Damping coefficient for the hitch point to simulate friction.
        """
        self.n = p_i0.shape[1]
        self.dt = dt
        self.l12 = l12
        self.l34 = l34
        self.m = m
        self.m_i = m_i
        self.c_d = c_d # Damping coefficient

        # Robot states
        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.u = np.zeros((4, self.n)) # External forces applied to robots
        self.f_ext = np.zeros(self.n) # External force on the hitch point

        # Solve for the initial hitch point position
        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess)
        self.v = np.zeros(self.n)

        # Store history for animation
        self.history = {
            "p": [self.p.copy()],
            "p_i": [self.p_i.copy()],
            "tensions": [],
            "u": [self.u.copy()]
        }

    def _constraint_loss(self, p):
        """Loss function representing the geometric ellipsoid constraints."""
        err1 = np.linalg.norm(p - self.p_i[0]) + np.linalg.norm(p - self.p_i[1]) - self.l12
        err2 = np.linalg.norm(p - self.p_i[2]) + np.linalg.norm(p - self.p_i[3]) - self.l34
        return err1**2 + err2**2

    def _solve_initial_p(self, guess):
        """Numerically solves for the initial hitch point position 'p'."""
        res = minimize(self._constraint_loss, guess, method='BFGS')
        return res.x

    def _unit_vector(self, v):
        """Computes the unit vector of a given vector 'v'."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _compute_dynamics(self):
        """
        Computes tensions and accelerations based on the system's current state.
        This method implements the core dynamics described in the paper.
        """
        # Define vectors based on current state (p, v, p_i, v_i)
        # r points from robot i to the hitch point p.
        r = np.array([self.p - self.p_i[i] for i in range(4)])
        r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        r_hat = np.array([self._unit_vector(r[i]) for i in range(4)])
        r_mag = np.linalg.norm(r, axis=1)

        # Derivative of the unit vector r_hat
        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(self.n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])
        
        # Mass-related term for the tension matrix
        alpha = self.m * np.array([1/self.m_i[0] + 1/self.m_i[1], 1/self.m_i[2] + 1/self.m_i[3]])

        # Right-hand side term from the paper's affine tension equation
        def c_term(i):
            return (1/self.m_i[i]) * r_hat[i] @ self.u[i] - r_hat_dot[i] @ r_dot[i]
        
        c1 = c_term(0) + c_term(1)
        c2 = c_term(2) + c_term(3)

        # Build and solve the linear system M*t = RHS for tensions
        M = np.array([
            [np.linalg.norm(r_hat[0] + r_hat[1])**2, (r_hat[0] + r_hat[1]) @ (r_hat[2] + r_hat[3])],
            [(r_hat[2] + r_hat[3]) @ (r_hat[0] + r_hat[1]), np.linalg.norm(r_hat[2] + r_hat[3])**2]
        ])
        M += np.diag(alpha)
        # global smallest_det_M
        # if abs(np.linalg.det(M)) < smallest_det_M:
        #     smallest_det_M = abs(np.linalg.det(M))
        #     print(f"New smallest det(M): {smallest_det_M}")   
        
        # Calculate the RHS, now including the effect of the damping force
        RHS = -self.m * np.array([c1, c2])
        damping_correction = np.array([
            (r_hat[0] + r_hat[1]) @ (self.c_d * self.v),
            (r_hat[2] + r_hat[3]) @ (self.c_d * self.v)
        ])
        RHS -= damping_correction
        RHS += np.array([
            (r_hat[0] + r_hat[1]) @ (self.f_ext),
            (r_hat[2] + r_hat[3]) @ (self.f_ext)
        ])
        # global greatest_damping_force
        # if np.linalg.norm(damping_correction) > greatest_damping_force:
        #     greatest_damping_force = np.linalg.norm(damping_correction)
        #     print(f"New greatest damping force effect: {greatest_damping_force}")

        try:
            tensions = np.linalg.solve(M, RHS)
        except np.linalg.LinAlgError:
            tensions = np.zeros(2)

        # Tensions must be non-negative (cables pull, not push)
        t12, t34 = max(0, tensions[0]), max(0, tensions[1])
        t_per_segment = np.array([t12, t12, t34, t34])

        # Calculate accelerations using Newton's second law (Eq. 6 and 7)
        # Add damping term to the hitch point acceleration
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
        """Advances the simulation by one time step with smooth projection correction."""
        v_unprojected, p_unprojected, v_i, p_i, tensions = self.euler_integrate()

        # --- Integration Step (semi-implicit Euler) ---
        self.v = v_unprojected
        self.v_i = v_i
        self.p_i = p_i


        # --- Projection Step (smooth correction) ---
        p_corrected = self._solve_initial_p(p_unprojected)
        correction = (p_corrected - p_unprojected) / self.dt


        # Apply only partial correction to smooth dynamics
        self.v += 0.5 * correction
        self.p = p_corrected


        # Store results for this step
        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())

    def build_mpc(self,
                N=10,
                dt_mpc=None,
                Qp=10.0,
                Qshape=1.0,
                Qcot=1.0,
                Ru=1e-3,
                t_min=1e-3):
        if ca is None or do_mpc is None:
            raise ImportError("casadi and do-mpc must be installed to use build_mpc().")

        if dt_mpc is None:
            dt_mpc = self.dt

        n = self.n
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # --- Merge all states into one flat vector ---
        nx_total = n*(1+4+1+4) # p, p_i(4), v, v_i(4)
        x = model.set_variable(var_type='_x', var_name='x', shape=(nx_total,1))

        # Slice helper
        def slice_var(start, count):
            return x[start:start+count]

        idx = 0
        p = slice_var(idx,n); idx+=n
        p_i = slice_var(idx,4*n); idx+=4*n
        v = slice_var(idx,n); idx+=n
        v_i = slice_var(idx,4*n); idx+=4*n

        # reshape for convenience
        p_i = ca.reshape(p_i,(n,4))
        v_i = ca.reshape(v_i,(n,4))

        # Controls
        u = model.set_variable(var_type='_u', var_name='u', shape=(n*4,1))
        u = ca.reshape(u,(n,4))

        # Time-varying parameters
        p_ref = model.set_variable(var_type='_tvp', var_name='p_ref', shape=(n,1))
        d12_ref = model.set_variable(var_type='_tvp', var_name='d12_ref', shape=(1,1))
        d34_ref = model.set_variable(var_type='_tvp', var_name='d34_ref', shape=(1,1))
        n_ref = model.set_variable(var_type='_tvp', var_name='n_ref', shape=(n,1))

        # compute dynamics
        r = [p - p_i[:, i] for i in range(4)]
        r_mag = [_safe_norm(r[i]) for i in range(4)]
        r_hat = [r[i]/r_mag[i] for i in range(4)]
        r_dot = [v - v_i[:, i] for i in range(4)]

        r_hat_dot = []
        for i in range(4):
            inner = ca.SX.eye(self.n) - r_hat[i] @ r_hat[i].T
            r_hat_dot.append((1/r_mag[i] * r_dot[i].T @ inner).T)

        n1 = r_hat[0] + r_hat[1]
        n2 = r_hat[2] + r_hat[3]

        M11 = n1.T @ n1 + self.m*(1.0/self.m_i[0] + 1.0/self.m_i[1])
        M22 = n2.T @ n2 + self.m*(1.0/self.m_i[2] + 1.0/self.m_i[3])
        M12 = n1.T @ n2
        Mmat = ca.SX(2,2)
        Mmat[0,0]=M11
        Mmat[1,1]=M22
        Mmat[0,1]=M12
        Mmat[1,0]=M12

        term1 = (r_hat[0].T @ u[:,0]) / self.m_i[0] + (r_hat[1].T @ u[:,1]) / self.m_i[1]
        term2 = (r_hat[2].T @ u[:,2]) / self.m_i[2] + (r_hat[3].T @ u[:,3]) / self.m_i[3]
        v0_1 = - self.m * ( term1 - r_hat_dot[0].T @ r_dot[0] - r_hat_dot[1].T @ r_dot[1])  - self.c_d*n1.T @ v + n1.T @ self.f_ext
        v0_2 = - self.m * ( term2 - r_hat_dot[2].T @ r_dot[2] - r_hat_dot[3].T @ r_dot[3])  - self.c_d*n2.T @ v + n2.T @ self.f_ext
        v0 = ca.vertcat(v0_1,v0_2)

        t_var = ca.solve(Mmat,v0)
        t12 = t_var[0]
        t34 = t_var[1]

        tension_force = -(t12*n1 + t34*n2)
        a = (tension_force - self.c_d*v + self.f_ext)/self.m
        a_i = ca.SX(self.n,4)
        for i in range(4):
            a_i[:,i] = ((ca.if_else(i<2,t12,t34))*r_hat[i] + u[:,i]) / self.m_i[i]

        p_next = p + dt_mpc * v
        v_next = v + dt_mpc * a
        p_i_next = p_i + dt_mpc * v_i
        v_i_next = v_i + dt_mpc * a_i

        # flatten back
        x_next = ca.vertcat(p_next,ca.reshape(p_i_next,(n*4,1)),v_next,ca.reshape(v_i_next,(n*4,1)))
        model.set_rhs('x', x_next)

        model.setup()

        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': N,
            't_step': dt_mpc,
            'store_full_solution': True,
            'open_loop': 0,
        }
        mpc.set_param(**setup_mpc)

        # cost
        d12 = _safe_norm(p_i[:,0]-p_i[:,1])
        d34 = _safe_norm(p_i[:,2]-p_i[:,3])
        n2_norm2 = n2.T @ n2 + 1e-12
        proj = (n1.T @ n2 / n2_norm2) * n2
        e_c = _safe_norm(n1 - proj)

        lterm = Qp*((p-p_ref).T @ (p-p_ref)) + Qshape*(d12 - d12_ref)**2 + Qshape*(d34 - d34_ref)**2 + Qcot*e_c**2
        mterm = Qp*((p-p_ref).T @ (p-p_ref)) + Qshape*(d12 - d12_ref)**2 + Qshape*(d34 - d34_ref)**2

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=Ru)

        mpc.bounds['lower','_u','u'] = -10*np.ones((n*4,1))
        mpc.bounds['upper','_u','u'] = 10*np.ones((n*4,1))

        self._do_mpc={'model':model,'mpc':mpc,'p_ref_var':p_ref,'d12_ref_var':d12_ref,'d34_ref_var':d34_ref,'n_ref_var':n_ref,'dt_mpc':dt_mpc}
        return self._do_mpc



    def controller_mpc_do_mpc(self, p_ref, d12_ref, d34_ref, n_ref=None):
        if ca is None or do_mpc is None:
            raise ImportError("casadi and do-mpc must be installed to use controller_mpc_do_mpc().")
        if not hasattr(self,'_do_mpc'):
            self.build_mpc()

        mpc=self._do_mpc['mpc']
        n=self.n

        # Flatten current state into x0
        x0_val=np.zeros((2*n+8*n,1))
        x0_val[0:n,0]=self.p
        x0_val[n:2*n,0]=self.v
        x0_val[2*n:6*n,0]=self.p_i.reshape(-1)
        x0_val[6*n:10*n,0]=self.v_i.reshape(-1)

        mpc.x0 = x0_val
        mpc.set_initial_guess()

        # Build tvp
        tvp_template = mpc.get_tvp_template()
        tvp_template['_tvp','p_ref'] = np.asarray(p_ref).reshape(n,1)
        tvp_template['_tvp','d12_ref'] = np.array([[d12_ref]])
        tvp_template['_tvp','d34_ref'] = np.array([[d34_ref]])
        tvp_template['_tvp','n_ref'] = np.asarray(n_ref if n_ref is not None else np.zeros(n)).reshape(n,1)
        mpc.set_tvp_fun(lambda t_now: tvp_template)

        try:
            u_opt=mpc.make_step(x0_val)
        except Exception as e:
            print("MPC solve failed; fallback:",e)
            u_cmd,_=self.controller_positive_tension(p_ref)
            return u_cmd

        if isinstance(u_opt,dict):
            u_opt_val=u_opt['u']
        else:
            u_opt_val=u_opt

        u_cmd=np.asarray(u_opt_val).reshape(n,4)
        return u_cmd


    def step_with_controller(self):
        """One simulation step where the robot inputs are produced by controller_qp."""
        # compute control inputs from QP
        p_ref = np.zeros(self.n)
        d12_ref = self.l12*3/4
        d34_ref = self.l34*3/4
        u_cmd = self.controller_mpc_do_mpc(p_ref, d12_ref, d34_ref, n_ref=np.array([1.0, 0.0, 0.0])[:self.n])
        self.u[:] = u_cmd.T
        # proceed with the usual dynamics/integration
        return self.step()

    def run(self, steps):
        """Runs the simulation for a given number of steps."""
        for _ in range(steps):
            self.step_with_controller()
            # self.step()

    def _get_ellipse_points(self, f1, f2, major_axis_length):
        """Calculates points for a 2D ellipse defined by its foci."""
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
        """Calculates wireframe points for an ellipsoid defined by its foci."""
        center = (f1 + f2) / 2
        dist = np.linalg.norm(f1 - f2)
        if dist >= major_axis_length: return None, None, None
        a = major_axis_length / 2.0
        c = dist / 2.0
        b = np.sqrt(max(a**2 - c**2, 1e-9))
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = b * np.outer(np.ones_like(u), np.cos(v))
        f1_to_f2 = self._unit_vector(f2 - f1)
        if np.allclose(f1_to_f2, [1, 0, 0]):
            rot_mat = np.identity(3)
        else:
            v_axis = np.cross([1, 0, 0], f1_to_f2)
            s = np.linalg.norm(v_axis)
            c_angle = np.dot([1, 0, 0], f1_to_f2)
            vx = np.array([[0, -v_axis[2], v_axis[1]], [v_axis[2], 0, -v_axis[0]], [-v_axis[1], v_axis[0], 0]])
            rot_mat = np.identity(3) + vx + vx @ vx * ((1 - c_angle) / (s**2))
        points = np.stack([x, y, z], axis=-1) @ rot_mat.T + center
        return points[..., 0], points[..., 1], points[..., 2]

    def animate(self, frame_skip=0):
        """
        Creates and displays an animation of the simulation for 2D or 3D.
        """
        p_hist = np.array(self.history["p"])
        p_i_hist = np.array(self.history["p_i"])
        tensions_hist = np.array(self.history["tensions"])
        u_hist = np.array(self.history["u"])
        
        fig = plt.figure(figsize=(12, 12))
        step = frame_skip + 1
        animation_frames = range(0, len(p_hist), step)

        if self.n == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title("3D Cable Robot Simulation")
            all_points = np.vstack(self.history['p_i'])
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
            ax.set_zlim(all_points[:,2].min()-1, all_points[:,2].max()+1)


            hitch_point, = ax.plot([], [], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
            robot_dots = [ax.plot([], [], [], 'o', color=c, ms=10)[0] for c in robot_colors]
            cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728','#d62728','#1f77b4','#1f77b4']]
            force_arrows = [ax.plot([], [], [], '-', color=c, lw=2)[0] for c in robot_colors]
            input_texts = [ax.text(0,0,0,'',fontsize=8,color=c) for c in robot_colors]
            wireframe1_lines, wireframe2_lines = [], []


            def update_3d(frame_index):
                nonlocal wireframe1_lines, wireframe2_lines
                p, p_i = p_hist[frame_index], p_i_hist[frame_index]
                u_arr = u_hist[frame_index] if frame_index < len(u_hist) else np.zeros((4,self.n))
                hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
                for i in range(4):
                    robot_dots[i].set_data_3d([p_i[i,0]], [p_i[i,1]], [p_i[i,2]])
                    cables[i].set_data_3d([p[0],p_i[i,0]],[p[1],p_i[i,1]],[p[2],p_i[i,2]])
                    start, end = p_i[i], p_i[i] + 0.5*u_arr[i]
                    force_arrows[i].set_data_3d([start[0],end[0]], [start[1],end[1]], [start[2],end[2]])
                    input_texts[i].set_position((p_i[i,0], p_i[i,1]))
                    input_texts[i].set_3d_properties(p_i[i,2])
                    norm_u = np.linalg.norm(u_arr[i])
                    input_texts[i].set_text(f"u={norm_u:.2f}")
                for wf in wireframe1_lines + wireframe2_lines:
                    wf.remove()
                wireframe1_lines.clear(); wireframe2_lines.clear()
                x1,y1,z1 = self._get_ellipsoid_points(p_i[0], p_i[1], self.l12)
                if x1 is not None:
                    wireframe1_lines.append(ax.plot_wireframe(x1,y1,z1,color='c',alpha=0.2))
                    x2,y2,z2 = self._get_ellipsoid_points(p_i[2], p_i[3], self.l34)
                if x2 is not None:
                    wireframe2_lines.append(ax.plot_wireframe(x2,y2,z2,color='m',alpha=0.2))
                return [hitch_point,*robot_dots,*cables,*force_arrows,*input_texts]
            ani = FuncAnimation(fig, update_3d, frames=animation_frames, blit=False, interval=50)

        elif self.n == 2:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title("2D Cable Robot Simulation")
            ax.grid(True, linestyle='--', alpha=0.6)
            all_points = np.vstack(self.history['p_i'])
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)


            hitch_point, = ax.plot([], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
            robot_dots = [ax.plot([], [], 'o', color=c, ms=10)[0] for c in robot_colors]
            cables = [ax.plot([], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            force_arrows = [ax.add_patch(FancyArrowPatch((0,0),(0,0), color=c, arrowstyle='->', mutation_scale=20, lw=1.5)) for c in robot_colors]
            tension_texts = [ax.text(0,0,'',fontsize=9,ha='center',va='center',backgroundcolor=(1,1,1,0.7)) for _ in range(4)]
            input_texts = [ax.text(0,0,'',fontsize=8,ha='left',va='bottom',color=c) for c in robot_colors]

            def update_2d(frame_index):
                p, p_i = p_hist[frame_index], p_i_hist[frame_index]
                t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
                u_arr = u_hist[frame_index] if frame_index < len(u_hist) else np.zeros((4,self.n))
                hitch_point.set_data([p[0]], [p[1]])
                ellipse1_line, = ax.plot([], [], 'c--', lw=1, label='Constraint Ellipse 1-2')
                ellipse2_line, = ax.plot([], [], 'm--', lw=1, label='Constraint Ellipse 3-4')
                for i in range(4):
                    robot_dots[i].set_data([p_i[i,0]], [p_i[i,1]])
                    cables[i].set_data([p[0], p_i[i,0]], [p[1], p_i[i,1]])
                    mid_point = (p + p_i[i]) / 2
                    tension_texts[i].set_position((mid_point[0], mid_point[1]))
                    tension_texts[i].set_text(f"{t_arr[i]:.2f} N")
                    start, end = p_i[i], p_i[i] + 0.5*u_arr[i]
                    force_arrows[i].set_positions(start, end)
                    input_texts[i].set_position((p_i[i,0], p_i[i,1]))
                    norm_u = np.linalg.norm(u_arr[i])
                    input_texts[i].set_text(f"u={norm_u:.2f}")
                ellipse1_pts = self._get_ellipse_points(p_i[0], p_i[1], self.l12)
                if ellipse1_pts is not None: ellipse1_line.set_data([ellipse1_pts[0]], [ellipse1_pts[1]])
                ellipse2_pts = self._get_ellipse_points(p_i[2], p_i[3], self.l34)
                if ellipse2_pts is not None: ellipse2_line.set_data([ellipse2_pts[0]], [ellipse2_pts[1]])
                return [hitch_point, *robot_dots, *cables, *tension_texts, ellipse1_line, ellipse2_line, *force_arrows]

            ani = FuncAnimation(fig, update_2d, frames=animation_frames, blit=True, interval=2)
        else:
            print(f"Animation is not supported for n={self.n}.")
            return
        
        plt.show()

def main():
    n = 2 # Switch to 3D
    dt = 0.001
    steps = 20000

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
    c_d = 0.1 # Damping coefficient

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)
    sim.build_mpc()

    # sim.u[0] = np.array([-1.0, -1.0, 0.25])[:n]
    # sim.u[1] = np.array([-1.0, 1.0, 0.25])[:n]
    # sim.u[2] = np.array([1.0, 1.0, 0.25])[:n]
    # sim.u[3] = np.array([1.0, -1.0, 0.25])[:n]
    sim.f_ext = np.array([0.0, 0.0, 0])[:n]  # External force on the hitch point

    sim.run(steps)
    sim.animate(frame_skip=19)

if __name__ == "__main__":
    main()
