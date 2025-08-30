import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from casadi import SX, vertcat, horzcat, diag, sqrt, if_else
import do_mpc

def build_discrete_cable_model(n, m, m_i, c_d, dt):
    """
    n : spatial dimension (2 or 3)
    m : virtual mass at hitch (scalar)
    m_i: list/array of 4 robot masses [m1,m2,m3,m4]
    c_d: damping scalar
    dt: discrete time step (seconds)
    """
    model = do_mpc.model.Model('discrete')

    # --- States (positions and velocities) --------------------------------
    p  = model.set_variable(var_type='_x', var_name='p',  shape=(n,1))  # hitch pos
    v  = model.set_variable(var_type='_x', var_name='v',  shape=(n,1))  # hitch vel

    p1 = model.set_variable(var_type='_x', var_name='p1', shape=(n,1))
    v1 = model.set_variable(var_type='_x', var_name='v1', shape=(n,1))

    p2 = model.set_variable(var_type='_x', var_name='p2', shape=(n,1))
    v2 = model.set_variable(var_type='_x', var_name='v2', shape=(n,1))

    p3 = model.set_variable(var_type='_x', var_name='p3', shape=(n,1))
    v3 = model.set_variable(var_type='_x', var_name='v3', shape=(n,1))

    p4 = model.set_variable(var_type='_x', var_name='p4', shape=(n,1))
    v4 = model.set_variable(var_type='_x', var_name='v4', shape=(n,1))

    # --- Inputs -----------------------------------------------------------
    u1 = model.set_variable(var_type='_u', var_name='u1', shape=(n,1))
    u2 = model.set_variable(var_type='_u', var_name='u2', shape=(n,1))
    u3 = model.set_variable(var_type='_u', var_name='u3', shape=(n,1))
    u4 = model.set_variable(var_type='_u', var_name='u4', shape=(n,1))

    # --- Time-varying parameter (reference hitch pos) ---------------------
    p_ref = model.set_variable(var_type='_tvp', var_name='p_ref', shape=(n,1))

    # --- Algebraic variables: tensions (4x1) ------------------------------
    t = model.set_variable(var_type='_z', var_name='t', shape=(4,1))

    # --- Helper (safe unit vector) ---------------------------------------
    def unit_vec(r):
        # returns r / ||r|| with small regularizer
        norm = sqrt(1e-12 + (r.T @ r)[0])
        return r / norm

    # --- Relative vectors and unit vectors -------------------------------
    r1 = p - p1
    r2 = p - p2
    r3 = p - p3
    r4 = p - p4

    r1_dot = v - v1
    r2_dot = v - v2
    r3_dot = v - v3
    r4_dot = v - v4

    r1_hat = unit_vec(r1)
    r2_hat = unit_vec(r2)
    r3_hat = unit_vec(r3)
    r4_hat = unit_vec(r4)

    # r_hat_dot = (I - rhat*rhat^T) @ r_dot / ||r||
    def rhat_dot(r_hat, r_dot, r):
        norm = sqrt(1e-12 + (r.T @ r)[0])
        I = SX.eye(n)
        P = I - r_hat @ r_hat.T
        return (P @ r_dot) / norm

    r1_hat_dot = rhat_dot(r1_hat, r1_dot, r1)
    r2_hat_dot = rhat_dot(r2_hat, r2_dot, r2)
    r3_hat_dot = rhat_dot(r3_hat, r3_dot, r3)
    r4_hat_dot = rhat_dot(r4_hat, r4_dot, r4)

    # Normals (ellipsoid normals)
    n12 = r1_hat + r2_hat  # n x 1
    n34 = r3_hat + r4_hat  # n x 1

    # --- Build M(x) 4x4 (SX) ----------------------------------------------
    M11 = (n12.T @ n12)[0] + m/m_i[0] + m/m_i[1]
    M12 = (n12.T @ n34)[0]
    M21 = (n34.T @ n12)[0]
    M22 = (n34.T @ n34)[0] + m/m_i[2] + m/m_i[3]

    M_mat = vertcat(
        horzcat(M11, 0,    M12, 0),
        horzcat(M21, 0,    M22, 0),
        horzcat(1,   -1,   0,   0),
        horzcat(0,   0,    1,  -1)
    )  # 4x4 SX matrix

    # --- Build v(x,u) (4x1) ----------------------------------------------
    term1 = (r1_hat.T @ u1)[0] / m_i[0]
    term2 = (r2_hat.T @ u2)[0] / m_i[1]
    term3 = (r3_hat.T @ u3)[0] / m_i[2]
    term4 = (r4_hat.T @ u4)[0] / m_i[3]

    v1_expr = -m * ( term1 + term2 - (r1_hat_dot.T @ r1_dot)[0] - (r2_hat_dot.T @ r2_dot)[0] )
    v2_expr = -m * ( term3 + term4 - (r3_hat_dot.T @ r3_dot)[0] - (r4_hat_dot.T @ r4_dot)[0] )

    # damping projection and external force (we keep f_ext = 0 here; you can add as tvp/p)
    v1_expr = v1_expr - c_d*(n12.T @ v)[0]
    v2_expr = v2_expr - c_d*(n34.T @ v)[0]

    v_vec = vertcat(v1_expr, v2_expr, 0, 0)  # 4x1 SX

    # --- Algebraic equation: M_mat * t - v_vec = 0 -----------------------
    model.set_alg('tension_algebraic', M_mat @ t - v_vec)

    # --- Hitch acceleration (corrected eq.8) and robot accelerations -----
    R_mat = horzcat(r1_hat, r2_hat, r3_hat, r4_hat)  # n x 4
    e_d = -c_d * v  # n x 1

    p_ddot = (- (R_mat @ t) + e_d) / m  # n x 1

    p1_ddot = (u1 + t[0]*r1_hat) / m_i[0]
    p2_ddot = (u2 + t[1]*r2_hat) / m_i[1]
    p3_ddot = (u3 + t[2]*r3_hat) / m_i[2]
    p4_ddot = (u4 + t[3]*r4_hat) / m_i[3]

    # --- Discrete Euler update: x_{k+1} = x_k + dt * xdot ----------------
    # (We embed a simple Euler integrator into the discrete-time model)
    model.set_rhs('p',  p + dt * v)
    model.set_rhs('v',  v + dt * p_ddot)

    model.set_rhs('p1', p1 + dt * v1)
    model.set_rhs('v1', v1 + dt * p1_ddot)

    model.set_rhs('p2', p2 + dt * v2)
    model.set_rhs('v2', v2 + dt * p2_ddot)

    model.set_rhs('p3', p3 + dt * v3)
    model.set_rhs('v3', v3 + dt * p3_ddot)

    model.set_rhs('p4', p4 + dt * v4)
    model.set_rhs('v4', v4 + dt * p4_ddot)

    # --- finalize model ---------------------------------------------------
    model.setup()
    return model


def build_mpc(model, dt, n_horizon=20):
    """
    MPC that uses the discrete model (algebraic tensions included).
    - dt is used only for informational purposes here (the model is already discrete).
    """

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': n_horizon,  # Prediction horizon
        't_step': dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 0.001,
            'ipopt.max_iter': 100,
            'ipopt.print_level': 2,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'spral'  # Use a faster linear solver
        }
    }

    # basic MPC params
    mpc.set_param(**setup_mpc)

    # state and input access
    p = model.x['p']
    v = model.x['v']
    p_ref = model.tvp['p_ref']

    u1 = model.u['u1']; u2 = model.u['u2']; u3 = model.u['u3']; u4 = model.u['u4']

    # cost: track hitch p to p_ref and penalize inputs and velocities
    Qp = 10.0
    Qv = 1.0
    Ru = 0.01

    stage_cost = Qp * ((p - p_ref).T @ (p - p_ref))[0] + Qv * (v.T @ v)[0] \
                 + Ru * ((u1.T @ u1)[0] + (u2.T @ u2)[0] + (u3.T @ u3)[0] + (u4.T @ u4)[0])
    term_cost = Qp * ((p - p_ref).T @ (p - p_ref))[0]

    mpc.set_objective(mterm=term_cost, lterm=stage_cost)
    mpc.set_rterm(u1=0, u2=0, u3=0, u4=0)  # we already penalize inputs in lterm

    # enforce algebraic tensions non-negative (t >= 0)
    z = model.z['t']
    mpc.set_nl_cons('t_nonneg', -z, ub=-1e-3)

    # control bounds (example)
    for ui in ['u1','u2','u3','u4']:
        mpc.bounds['lower','_u',ui] = -5.0
        mpc.bounds['upper','_u',ui] =  5.0

    # TVP function: constant reference for now (modify to time-dependent easily)
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        tvp_template['_tvp', model.x['p'].shape[0], 'p_ref'] = np.zeros((model.x['p'].shape[0], 1))
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()
    return mpc

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
            "tensions": [], # Store tensions for efficient and accurate animation
            "u": [self.u.copy()]
        }

        self.symbolic_model = build_discrete_cable_model(self.n, self.m, self.m_i, self.c_d, self.dt)
        self.mpc = build_mpc(self.symbolic_model, self.dt, n_horizon=10)

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

    def _compute_reduced_dynamics(self):
        """
        Computes tensions and accelerations for the reduced system where robots 2 and 3 are fixed.
        This implementation is more efficient as it solves a 2x2 linear system for tensions
        instead of a 4x4 system.
        """
        # Define vectors based on current state (p, v, p_i, v_i)
        # Note: v_i for fixed robots (1 and 2) is always zero.
        r = np.array([self.p - self.p_i[i] for i in range(4)])
        r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        r_hat = np.array([self._unit_vector(r[i]) for i in range(4)])
        r_mag = np.linalg.norm(r, axis=1)

        # Derivative of the unit vector r_hat
        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(self.n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])

        # Right-hand side terms for the tension equation
        # For fixed robots (i=1, 2), u_i is zero and v_i is zero.
        c1 = (1/self.m_i[0]) * r_hat[0] @ self.u[0] - r_hat_dot[0] @ r_dot[0] - r_hat_dot[1] @ r_dot[1]
        c2 = (1/self.m_i[3]) * r_hat[3] @ self.u[3] - r_hat_dot[3] @ r_dot[3] - r_hat_dot[2] @ r_dot[2]
        
        # Build and solve the reduced 2x2 linear system M*t = RHS for tensions
        M = np.array([
            [np.linalg.norm(r_hat[0] + r_hat[1])**2 + self.m/self.m_i[0], (r_hat[0] + r_hat[1]) @ (r_hat[2] + r_hat[3])],
            [(r_hat[2] + r_hat[3]) @ (r_hat[0] + r_hat[1]), np.linalg.norm(r_hat[2] + r_hat[3])**2 + self.m/self.m_i[3]]
        ])

        # Calculate the RHS, including damping and external forces
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

        try:
            tensions = np.linalg.solve(M, RHS)
        except np.linalg.LinAlgError:
            tensions = np.zeros(2)

        # Tensions must be non-negative (cables pull, not push)
        t12, t34 = max(0, tensions[0]), max(0, tensions[1])
        t_per_segment = np.array([t12, t12, t34, t34])

        # Calculate accelerations using Newton's second law
        # Add damping term to the hitch point acceleration
        tension_force = -t12 * (r_hat[0] + r_hat[1]) - t34 * (r_hat[2] + r_hat[3])
        damping_force = -self.c_d * self.v + self.f_ext
        a = (tension_force + damping_force) / self.m
        
        # Accelerations for dynamic robots (0 and 3)
        a_i = np.zeros((4, self.n))
        a_i[0] = (t_per_segment[0] * r_hat[0] + self.u[0]) / self.m_i[0]
        a_i[3] = (t_per_segment[3] * r_hat[3] + self.u[3]) / self.m_i[3]

        return a, a_i, t_per_segment

    def _compute_dynamics(self):
        r = np.array([self.p - self.p_i[i] for i in range(4)])
        r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        r_hat = np.array([self._unit_vector(r[i]) for i in range(4)])
        r_mag = np.linalg.norm(r, axis=1)

        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(self.n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])

        # Build 4x4 M matrix as in the paper
        n12 = r_hat[0] + r_hat[1]
        n34 = r_hat[2] + r_hat[3]
        M = np.array([
            [np.linalg.norm(n12)**2 + self.m/self.m_i[0] + self.m/self.m_i[1], 0, n12 @ n34, 0],
            [n34 @ n12, 0, np.linalg.norm(n34)**2 + self.m/self.m_i[2] + self.m/self.m_i[3], 0],
            [1, -1, 0, 0],
            [0, 0, 1, -1]
        ])

        # Build v vector
        v = np.zeros(4)
        v[0] = -self.m*((r_hat[0] @ self.u[0])/self.m_i[0] + (r_hat[1] @ self.u[1])/self.m_i[1]
                - r_hat_dot[0] @ r_dot[0] - r_hat_dot[1] @ r_dot[1])
        v[1] = -self.m*((r_hat[2] @ self.u[2])/self.m_i[2] + (r_hat[3] @ self.u[3])/self.m_i[3]
                - r_hat_dot[2] @ r_dot[2] - r_hat_dot[3] @ r_dot[3])
        v[0:2] += -self.c_d * np.array([(n12 @ self.v), (n34 @ self.v)])
        v[0:2] += np.array([(n12 @ self.f_ext), (n34 @ self.f_ext)])

        try:
            t = np.linalg.solve(M, v)
        except np.linalg.LinAlgError:
            t = np.zeros(4)

        t = np.maximum(t, 0.0)

        # Hitch acceleration (corrected Eq. 8)
        R = np.column_stack(r_hat)
        e_d = -self.c_d * self.v
        a = (-R @ t + e_d + self.f_ext) / self.m

        # Robot accelerations
        a_i = np.array([(t[i] * r_hat[i] + self.u[i]) / self.m_i[i] for i in range(4)])

        return a, a_i, t

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

    def step_mpc(self):
        # Build full current state dict for do-mpc
        x0_list = []
        x0_list.append(self.p.reshape(-1,1))
        x0_list.append(self.v.reshape(-1,1))
        x0_list.append(self.p_i[0].reshape(-1,1))
        x0_list.append(self.v_i[0].reshape(-1,1))
        x0_list.append(self.p_i[1].reshape(-1,1))
        x0_list.append(self.v_i[1].reshape(-1,1))
        x0_list.append(self.p_i[2].reshape(-1,1))
        x0_list.append(self.v_i[2].reshape(-1,1))
        x0_list.append(self.p_i[3].reshape(-1,1))
        x0_list.append(self.v_i[3].reshape(-1,1))

        x0 = np.vstack(x0_list)  # full column vector

        # Set MPC initial state
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

        # Make MPC step
        u_mpc = self.mpc.make_step(x0).flatten()

        # Update external forces from MPC result
        for i in range(len(self.u)):
            self.u[i] = u_mpc[self.n*i:self.n*(i+1)]

        # Advance physics one step with Euler + constraint projection
        v_unprojected, p_unprojected, v_i, p_i, tensions = self.euler_integrate()
        self.v, self.v_i, self.p_i = v_unprojected, v_i, p_i
        p_corrected = self._solve_initial_p(p_unprojected)
        correction = (p_corrected - p_unprojected) / self.dt
        self.v += 0.5 * correction
        self.p = p_corrected

        # Save to history
        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())

    def run(self, steps):
        """Runs the simulation for a given number of steps."""
        for step_count in range(steps):
            print("{}/{}".format(step_count+1, steps))
            self.step_mpc()

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
            cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            force_arrows = [ax.plot([], [], [], '-', color=c, lw=2)[0] for c in robot_colors]
            wireframe1_lines, wireframe2_lines = [], []

            def update_3d(frame_index):
                nonlocal wireframe1_lines, wireframe2_lines
                p, p_i = p_hist[frame_index], p_i_hist[frame_index]
                hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
                for i in range(4):
                    robot_dots[i].set_data_3d([p_i[i, 0]], [p_i[i, 1]], [p_i[i, 2]])
                    cables[i].set_data_3d([p[0], p_i[i, 0]], [p[1], p_i[i, 1]], [p[2], p_i[i, 2]])
                    start, end = p_i[i], p_i[i] + 0.5 * self.u[i]
                    force_arrows[i].set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
                for wf in wireframe1_lines + wireframe2_lines: wf.remove()
                wireframe1_lines.clear(); wireframe2_lines.clear()
                x1, y1, z1 = self._get_ellipsoid_points(p_i[0], p_i[1], self.l12)
                if x1 is not None: wireframe1_lines.append(ax.plot_wireframe(x1, y1, z1, color='c', alpha=0.2))
                x2, y2, z2 = self._get_ellipsoid_points(p_i[2], p_i[3], self.l34)
                if x2 is not None: wireframe2_lines.append(ax.plot_wireframe(x2, y2, z2, color='m', alpha=0.2))
                return [hitch_point, *robot_dots, *cables, *force_arrows]
            
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
            force_arrows = [ax.add_patch(FancyArrowPatch((0,0), (0,0), color=c, arrowstyle='->', mutation_scale=20, lw=1.5)) for c in robot_colors]
            tension_texts = [ax.text(0, 0, '', fontsize=9, ha='center', va='center', backgroundcolor=(1,1,1,0.7)) for _ in range(4)]
            ellipse1_line, = ax.plot([], [], 'c--', lw=1, label='Constraint Ellipse 1-2')
            ellipse2_line, = ax.plot([], [], 'm--', lw=1, label='Constraint Ellipse 3-4')

            def update_2d(frame_index):
                p, p_i = p_hist[frame_index], p_i_hist[frame_index]
                t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
                hitch_point.set_data([p[0]], [p[1]])
                for i in range(4):
                    robot_dots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
                    cables[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])
                    mid_point = (p + p_i[i]) / 2
                    tension_texts[i].set_position((mid_point[0], mid_point[1])); tension_texts[i].set_text(f"{t_arr[i]:.2f} N")
                    start, end = p_i[i], p_i[i] + 0.5 * self.u[i]
                    force_arrows[i].set_positions(start, end)
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
    dt = 0.01
    steps = 5000

    p_i0 = np.array([
        [-2.0, -2.5, 0.0],
        [-1.5,  2.0, 0.5],
        [ 1.5,  2.0, -0.5],
        [ 2.0, -2.5, 0.0]
    ])[:, :n]

    v_i0 = np.zeros((4, n))
    m = 0.1
    m_i = np.ones(4) * 0.5
    # m_i[1] = np.inf  # Make robot 3 immobile
    # m_i[2] = np.inf  # Make robot 4 immobile
    l12 = 6.5
    l34 = 7.5
    c_d = 0.1 # Damping coefficient

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)

    sim.u[0] = np.array([-1.0, -1.0, 0.25])[:n]
    sim.u[1] = np.array([-1.0, 1.0, 0.25])[:n]
    sim.u[2] = np.array([1.0, 1.0, 0.25])[:n]
    sim.u[3] = np.array([1.0, -1.0, 0.25])[:n]
    sim.f_ext = np.array([0.0, 0.0, -1])[:n]  # External force on the hitch point

    sim.run(steps)
    sim.animate(frame_skip=49)

if __name__ == "__main__":
    main()
