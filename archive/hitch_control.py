import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from casadi import SX, vertcat, horzcat, diag, sqrt, if_else
import do_mpc
import cvxpy as cp # Import the cvxpy library for QP solving
import matplotlib.animation as animation # Import animation for saving

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
    p   = model.set_variable(var_type='_x', var_name='p',  shape=(n,1))  # hitch pos
    v   = model.set_variable(var_type='_x', var_name='v',  shape=(n,1))  # hitch vel

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
        horzcat(0,   0,    1,   -1)
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
    """
    def __init__(self, p_i0, v_i0, l12, l34, m, m_i, dt, c_d=0.0):
        self.n = p_i0.shape[1]
        self.dt = dt
        self.l12 = l12
        self.l34 = l34
        self.m = m
        self.m_i = m_i
        self.c_d = c_d

        # Robot states
        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.u = np.zeros((4, self.n))
        self.f_ext = np.zeros(self.n)
        self.p_ref = np.zeros(self.n)

        # Solve for the initial hitch point position
        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess).x
        self.v = np.zeros(self.n)

        # --- CLF-CBF-QP Controller References ---
        self.n12_ref = np.zeros(self.n)
        self.n34_ref = np.zeros(self.n)
        self.d12_ref = 0.0
        self.d34_ref = 0.0

        # --- CLF-CBF-QP Controller Parameters ---
        self.clf_params = {
            'Wp': 10.0,            # Weight for position error
            'Wv': 5.0,             # Weight for velocity error
            'Wd': 2.0,             # Weight for distance error
            'Wn': 2.0,             # Weight for normal vector error
            'gamma_clf': 5.0,      # CLF convergence rate
            'p_clf': 1e6,          # Penalty weight for CLF relaxation (delta)
            't_min': 0.1,          # Minimum desired cable tension (CBF)
            'u_max': 5.0           # Max control input magnitude
        }

        # History for animation
        self.history = {
            "p": [self.p.copy()],
            "p_i": [self.p_i.copy()],
            "tensions": [],
            "u": [self.u.copy()]
        }

        # MPC-related setup (can be ignored as requested)
        # self.symbolic_model = build_discrete_cable_model(self.n, self.m, self.m_i, self.c_d, self.dt)
        # self.mpc = build_mpc(self.symbolic_model, self.dt, n_horizon=10)

    def _constraint_loss(self, p):
        err1 = np.linalg.norm(p - self.p_i[0]) + np.linalg.norm(p - self.p_i[1]) - self.l12
        err2 = np.linalg.norm(p - self.p_i[2]) + np.linalg.norm(p - self.p_i[3]) - self.l34
        return err1**2 + err2**2

    def _solve_initial_p(self, guess):
        res = minimize(self._constraint_loss, guess, method='BFGS')
        return res

    def _unit_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _compute_dynamics(self):
        r = np.array([self.p - self.p_i[i] for i in range(4)])
        r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        r_hat = np.array([self._unit_vector(r[i]) for i in range(4)])
        r_mag = np.linalg.norm(r, axis=1)
        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(self.n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])

        n12 = r_hat[0] + r_hat[1]
        n34 = r_hat[2] + r_hat[3]
        M = np.array([
            [np.linalg.norm(n12)**2 + self.m/self.m_i[0] + self.m/self.m_i[1], 0, n12 @ n34, 0],
            [n34 @ n12, 0, np.linalg.norm(n34)**2 + self.m/self.m_i[2] + self.m/self.m_i[3], 0],
            [1, -1, 0, 0],
            [0, 0, 1, -1]
        ])

        v_rhs = np.zeros(4)
        v_rhs[0] = -self.m*((r_hat[0] @ self.u[0])/self.m_i[0] + (r_hat[1] @ self.u[1])/self.m_i[1] - r_hat_dot[0] @ r_dot[0] - r_hat_dot[1] @ r_dot[1])
        v_rhs[1] = -self.m*((r_hat[2] @ self.u[2])/self.m_i[2] + (r_hat[3] @ self.u[3])/self.m_i[3] - r_hat_dot[2] @ r_dot[2] - r_hat_dot[3] @ r_dot[3])
        v_rhs[0:2] += -self.c_d * np.array([(n12 @ self.v), (n34 @ self.v)])
        v_rhs[0:2] += np.array([(n12 @ self.f_ext), (n34 @ self.f_ext)])

        try:
            t = np.linalg.solve(M, v_rhs)
        except np.linalg.LinAlgError:
            t = np.zeros(4)
        t = np.maximum(t, 0.0)

        R_mat = np.column_stack(r_hat)
        e_d = -self.c_d * self.v
        a = (-R_mat @ t + e_d + self.f_ext) / self.m
        a_i = np.array([(t[i] * r_hat[i] + self.u[i]) / self.m_i[i] for i in range(4)])
        return a, a_i, t

    def step(self, verbose=False):
        """Advances the simulation by one step using the current self.u"""
        a, a_i, tensions = self._compute_dynamics()
        v_unprojected = self.v + self.dt * a
        p_unprojected = self.p + self.dt * v_unprojected
        self.v_i = self.v_i + self.dt * a_i
        self.p_i = self.p_i + self.dt * self.v_i

        self.v = v_unprojected
        res = self._solve_initial_p(p_unprojected)
        if res.fun > 1e-6:
            # This is expected during dynamic motion, not an error
            pass
        p_corrected = res.x

        correction = (p_corrected - p_unprojected) / self.dt
        self.v += 0.5 * correction
        self.p = p_corrected

        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())
        if verbose:
            d12 = np.linalg.norm(self.p_i[0]-self.p_i[1])
            d34 = np.linalg.norm(self.p_i[2]-self.p_i[3])
            n12 = self._unit_vector(self.p - self.p_i[0]) + self._unit_vector(self.p - self.p_i[1])
            n34 = self._unit_vector(self.p - self.p_i[2]) + self._unit_vector(self.p - self.p_i[3])
            print(f"tensions={tensions},\n d12={d12}, d34={d34},\n n12={n12}, n34={n34}")

    def step_clf_cbf(self):
        """Computes control input u via CLF-CBF-QP and then steps the simulation."""
        # 1. Get current state variables
        p, v, p_i, v_i = self.p, self.v, self.p_i, self.v_i
        n = self.n
        
        # 2. Decompose system dynamics into control-affine form: a = f(x) + g(x)u
        # Tension is linear in u: t(x,u) = t_f(x) + t_g(x)u
        # Acceleration is linear in t: a(x,t) = a_f(x) - a_g(x)t
        # This implies a(x,u) is also control-affine. We build the matrices here.
        
        r = np.array([p - pi for pi in p_i])
        r_dot = np.array([v - vi for vi in v_i])
        r_hat = np.array([self._unit_vector(ri) for ri in r])
        r_mag = np.linalg.norm(r, axis=1)
        r_hat_dot = np.array([(1.0/r_mag[i] * r_dot[i] @ (np.eye(n) - np.outer(r_hat[i], r_hat[i]))).ravel() for i in range(4)])
        n12 = r_hat[0] + r_hat[1]
        n34 = r_hat[2] + r_hat[3]
        
        M = np.array([
            [n12@n12+ self.m/self.m_i[0] + self.m/self.m_i[1], 0, n12 @ n34, 0],
            [n34 @ n12, 0, n34@n34 + self.m/self.m_i[2] + self.m/self.m_i[3], 0],
            [1, -1, 0, 0],
            [0, 0, 1, -1]
        ])
        
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("Warning: Singular M matrix in CLF-CBF QP, skipping control update.")
            self.u = np.zeros_like(self.u) # Fallback control
            self.step()
            return
            
        # Build w
        w = np.zeros(4)
        w[0] = self.m * (r_hat_dot[0] @ r_dot[0] + r_hat_dot[1] @ r_dot[1]) # - self.c_d * (n12 @ v) + (n12 @ self.f_ext)
        w[1] = self.m * (r_hat_dot[2] @ r_dot[2] + r_hat_dot[3] @ r_dot[3]) # - self.c_d * (n34 @ v) + (n34 @ self.f_ext)
        
        # Build C
        C_mat = np.zeros((4, 4 * n))
        C_mat[0, 0:n] = -self.m * r_hat[0] / self.m_i[0]
        C_mat[0, n:2*n] = -self.m * r_hat[1] / self.m_i[1]
        C_mat[1, 2*n:3*n] = -self.m * r_hat[2] / self.m_i[2]
        C_mat[1, 3*n:4*n] = -self.m * r_hat[3] / self.m_i[3]

        M_inv_w = M_inv @ w
        M_inv_C = M_inv @ C_mat
        
        R_mat = np.column_stack(r_hat)
        
        # Finally, the affine dynamics for hitch acceleration: p_ddot = f_a(x) + G_a(x)u
        f_a = (-R_mat @ M_inv_w) / self.m
        G_a = (-R_mat @ M_inv_C) / self.m
        
        # 3. Define CLF based on the paper's formulation from Eq. (10)
        # Unpack gains from parameters
        K_p = self.clf_params['K_p']
        K_d = self.clf_params['K_d']
        K_dist = self.clf_params['K_dist']
        K_norm = self.clf_params['K_norm']
        gamma = self.clf_params['gamma']

        # Define the error vector e = y - y_d
        e_p = p - self.p_ref
        
        d12 = np.linalg.norm(p_i[0] - p_i[1])
        d34 = np.linalg.norm(p_i[2] - p_i[3])
        e_d12 = d12 - self.d12_ref
        e_d34 = d34 - self.d34_ref
        
        e_n12 = n12 - self.n12_ref
        e_n34 = n34 - self.n34_ref
        
        # NOTE: We also include hitch velocity error for damping, which is a practical
        # extension to the paper's CLF to ensure the relative degree is one w.r.t 'u'.
        e_v = v

        # Define the time derivative of the error vector e_dot = y_dot
        e_p_dot = v
        d12_dot = ((p_i[0] - p_i[1]) @ (v_i[0] - v_i[1])) / (d12 + 1e-9)
        d34_dot = ((p_i[2] - p_i[3]) @ (v_i[2] - v_i[3])) / (d34 + 1e-9)
        n12_dot = r_hat_dot[0] + r_hat_dot[1]
        n34_dot = r_hat_dot[2] + r_hat_dot[3]
        # The time derivative of velocity error (e_v_dot) is the hitch acceleration 'p_ddot'
        e_v_dot_f = f_a  # State-dependent part of acceleration
        e_v_dot_g = G_a  # Control-dependent part of acceleration

        # Define the composite Control Lyapunov Function V from Eq. (10)
        # with an added velocity term for QP compatibility.
        V_p = 0.5 * K_p * (e_p @ e_p)
        V_v = 0.5 * K_d * (e_v @ e_v)
        V_dist = 0.5 * K_dist * (e_d12**2 + e_d34**2)
        V_norm = 0.5 * K_norm * (e_n12 @ e_n12 + e_n34 @ e_n34)
        V = V_p + V_v + V_dist + V_norm

        # Calculate Lie Derivatives L_f(V) and L_g(V) where V_dot = L_f(V) + L_g(V)u
        # The control input u only appears in the derivative of the velocity error term (V_v_dot).
        V_p_dot = K_p * (e_p @ e_p_dot)
        V_dist_dot = K_dist * (e_d12 * d12_dot + e_d34 * d34_dot)
        V_norm_dot = K_norm * (e_n12 @ n12_dot + e_n34 @ n34_dot)
        
        L_f_V = V_p_dot + V_dist_dot + V_norm_dot + K_d * (e_v @ e_v_dot_f)
        L_g_V = K_d * e_v.T @ e_v_dot_g # This is a row vector
        
        # 4. Setup and Solve the QP
        u_qp = cp.Variable(4 * n)
        delta_qp = cp.Variable(1)
        
        # Objective function
        p_delta = self.clf_params['p_delta']
        objective = cp.Minimize(cp.sum_squares(u_qp) + p_delta * cp.square(delta_qp))
        
        # Constraints
        constraints = []
        # CLF constraint: L_f_V + L_g_V * u + gamma*V <= delta
        constraints.append(L_f_V + L_g_V @ u_qp + gamma * V <= delta_qp)
        
        # CBF constraint: t(x,u) >= t_min  => t_f + t_g*u >= t_min
        t_min = self.clf_params['t_min']
        constraints.append(t_f + t_g @ u_qp >= t_min)
        
        # Input bounds
        u_max = self.clf_params['u_max']
        constraints.append(cp.norm(u_qp, 'inf') <= u_max)
        
        # Solve the QP
        prob = cp.Problem(objective, constraints)
        try:
            # Use a fast solver like OSQP or ECOS
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if u_qp.value is not None:
                self.u = u_qp.value.reshape(4, n)
            else:
                # If solver fails, apply zero control as a safe fallback
                self.u = np.zeros_like(self.u)
        except cp.error.SolverError:
            self.u = np.zeros_like(self.u)

        # 5. Step the simulation with the computed control
        self.step()


    def step_mpc(self):
        x0_list = [
            self.p.reshape(-1,1), self.v.reshape(-1,1),
            self.p_i[0].reshape(-1,1), self.v_i[0].reshape(-1,1),
            self.p_i[1].reshape(-1,1), self.v_i[1].reshape(-1,1),
            self.p_i[2].reshape(-1,1), self.v_i[2].reshape(-1,1),
            self.p_i[3].reshape(-1,1), self.v_i[3].reshape(-1,1),
        ]
        x0 = np.vstack(x0_list)
        # self.mpc.x0 = x0
        # self.mpc.set_initial_guess()
        # u_mpc = self.mpc.make_step(x0).flatten()
        # self.u = u_mpc.reshape(4, self.n)
        self.step()

    def run(self, steps, controller_type='mpc'):
        """Runs the simulation for a given number of steps."""
        for step_count in range(steps):
            if step_count % 100 == 0:
                print(f"Step {step_count+1}/{steps} using {controller_type.upper()} controller")
            if controller_type == 'mpc':
                # self.step_mpc() # Ignoring MPC as requested
                self.step()
            elif controller_type == 'clf_cbf':
                self.step_clf_cbf()
            else:
                self.step(verbose=(step_count % 100 == 0))

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

    def animate(self, frame_skip=0, save_path="cable_robot_animation.avi"):
        p_hist = np.array(self.history["p"])
        p_i_hist = np.array(self.history["p_i"])
        tensions_hist = np.array(self.history["tensions"])
        u_hist = np.array(self.history["u"])

        fig = plt.figure(figsize=(12, 12))
        step = frame_skip + 1
        animation_frames = range(0, len(p_hist), step)
        robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']

        if self.n == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title("3D Cable Robot Simulation")
            all_points = np.vstack([p_hist, p_i_hist.reshape(-1, self.n)])
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
            ax.set_zlim(all_points[:,2].min()-1, all_points[:,2].max()+1)

            ax.plot([self.p_ref[0]], [self.p_ref[1]], [self.p_ref[2]], 'gx', ms=10, mew=3, label='Reference')

            hitch_point, = ax.plot([], [], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_dots = [ax.plot([], [], [], 'o', color=c, ms=10)[0] for c in robot_colors]
            cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            input_arrows = [ax.quiver([], [], [], [], [], [], color=c, length=0.5, normalize=True, arrow_length_ratio=0.3) for c in robot_colors]
            wireframe1, wireframe2 = None, None

            def update_3d(frame_index):
                nonlocal wireframe1, wireframe2, input_arrows
                p, p_i, u = p_hist[frame_index], p_i_hist[frame_index], u_hist[frame_index]
                hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
                for i in range(4):
                    robot_dots[i].set_data_3d([p_i[i, 0]], [p_i[i, 1]], [p_i[i, 2]])
                    cables[i].set_data_3d([p[0], p_i[i, 0]], [p[1], p_i[i, 1]], [p[2], p_i[i, 2]])

                    # Update input arrows
                    if input_arrows[i] is not None:
                        input_arrows[i].remove()
                    input_arrows[i] = ax.quiver(p_i[i, 0], p_i[i, 1], p_i[i, 2],
                                                u[i, 0], u[i, 1], u[i, 2],
                                                color=robot_colors[i], length=np.linalg.norm(u[i])*0.1, normalize=False, arrow_length_ratio=0.3)

                if wireframe1: wireframe1.remove()
                if wireframe2: wireframe2.remove()
                x1, y1, z1 = self._get_ellipsoid_points(p_i[0], p_i[1], self.l12)
                if x1 is not None: wireframe1 = ax.plot_wireframe(x1, y1, z1, color='c', alpha=0.1)
                x2, y2, z2 = self._get_ellipsoid_points(p_i[2], p_i[3], self.l34)
                if x2 is not None: wireframe2 = ax.plot_wireframe(x2, y2, z2, color='m', alpha=0.1)

                return [hitch_point, *robot_dots, *cables, *input_arrows]

            ani = FuncAnimation(fig, update_3d, frames=animation_frames, blit=False, interval=50)

        elif self.n == 2:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title("2D Cable Robot Simulation")
            ax.grid(True, linestyle='--', alpha=0.6)
            all_points = np.vstack([p_hist, p_i_hist.reshape(-1, self.n)])
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)

            ax.plot(self.p_ref[0], self.p_ref[1], 'gx', ms=10, mew=3, label='Reference')

            hitch_point, = ax.plot([], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_dots = [ax.plot([], [], 'o', color=c, ms=10)[0] for c in robot_colors]
            cables = [ax.plot([], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            tension_texts = [ax.text(0, 0, '', fontsize=9, ha='center', va='center', backgroundcolor=(1,1,1,0.7)) for _ in range(4)]
            input_arrows = [ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc=c, ec=c, length_includes_head=True) for c in robot_colors]
            ellipse1_line, = ax.plot([], [], 'c--', lw=1, label='Constraint Ellipse 1-2')
            ellipse2_line, = ax.plot([], [], 'm--', lw=1, label='Constraint Ellipse 3-4')
            ax.legend()

            def update_2d(frame_index):
                p, p_i, u = p_hist[frame_index], p_i_hist[frame_index], u_hist[frame_index]
                t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
                hitch_point.set_data([p[0]], [p[1]])
                for i in range(4):
                    robot_dots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
                    cables[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])
                    mid_point = (p + p_i[i]) / 2
                    tension_texts[i].set_position((mid_point[0], mid_point[1])); tension_texts[i].set_text(f"{t_arr[i]:.2f} N")

                    # Update input arrows
                    input_arrows[i].remove() # Remove previous arrow
                    input_arrows[i] = ax.arrow(p_i[i, 0], p_i[i, 1], u[i, 0]*0.1, u[i, 1]*0.1, # Scale arrow length for visibility
                                               head_width=0.2, head_length=0.3, fc=robot_colors[i], ec=robot_colors[i],
                                               length_includes_head=True)


                ellipse1_pts = self._get_ellipse_points(p_i[0], p_i[1], self.l12)
                if ellipse1_pts is not None: ellipse1_line.set_data(ellipse1_pts[0], ellipse1_pts[1])
                ellipse2_pts = self._get_ellipse_points(p_i[2], p_i[3], self.l34)
                if ellipse2_pts is not None: ellipse2_line.set_data(ellipse2_pts[0], ellipse2_pts[1])
                return [hitch_point, *robot_dots, *cables, *tension_texts, *input_arrows, ellipse1_line, ellipse2_line]

            ani = FuncAnimation(fig, update_2d, frames=animation_frames, blit=False, interval=20) # Blit is False for arrows

        else:
            print(f"Animation is not supported for n={self.n}.")
            return

        plt.show()
        # Save the animation
        # print(f"Saving animation to {save_path}...")
        # ani.save(save_path, writer='ffmpeg', fps=1/(self.dt*(frame_skip+1)))
        # print("Animation saved.")
        # plt.close(fig) # Close the plot after saving

def main():
    n = 2 # Switch to 3 for 3D
    dt = 0.001
    steps = 10000

    if n == 2:
        p_i0 = np.array([
            [-2.0, -2.5],
            [-1.5,  2.0],
            [ 1.5,  0.5],
            [ 2.0, -1.5]
        ])
    else: # n=3
        p_i0 = np.array([
            [-2.0, -2.5, 0.0],
            [-1.5,  2.0, 0.5],
            [ 1.5,  2.0, -0.5],
            [ 2.0, -2.5, 0.0]
        ])

    v_i0 = np.zeros((4, n))
    m = 0.1
    m_i = np.ones(4) * 0.2
    l12 = 6.5
    l34 = 5.5
    c_d = 0.2 # Damping coefficient

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)

    # Set a non-zero reference for the controller to track
    sim.p_ref = np.array([-0.5, -0.5, 0.25])[:n]

    # Set references for the new objectives
    sim.n12_ref = np.array([1.6, 0.4, 0.0])[:n]
    sim.n34_ref = np.array([-1.6, 0.7, 0.0])[:n]
    sim.d12_ref = 5.0
    sim.d34_ref = 4.5

    # We no longer set a constant u, the controller will find it.
    sim.u[0] = np.array([-1.0, -1.0, 0.25])[:n]
    sim.u[1] = np.array([-1.0, 1.0, 0.25])[:n]
    sim.u[2] = np.array([1.0, 1.0, 0.25])[:n]
    sim.u[3] = np.array([1.0, -1.0, 0.25])[:n]

    # Set a constant external force if desired
    # sim.f_ext = np.array([0.0, 0.0, -1])[:n]

    # --- CHOOSE THE CONTROLLER TO RUN ---
    controller = 'none' # Options: 'mpc', 'clf_cbf', 'none'
    sim.run(steps, controller_type=controller)
    sim.animate(frame_skip=19)

if __name__ == "__main__":
    main()