import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from casadi import SX, vertcat, horzcat, diag, sqrt, if_else
import do_mpc
import cvxpy as cp # Import the cvxpy library for QP solving
import time

# --- Helper classes and functions from the original code ---
# (build_discrete_cable_model and build_mpc are not used by the new controller
# but are kept for completeness if you wish to switch back to MPC)

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
        self.timer = None

        # Robot states
        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.p_i_ref = p_i0.copy() # default reference positions for robots
        self.v_i_ref = np.zeros_like(self.v_i)
        self.u = np.zeros((4, self.n))
        self.f_ext = np.zeros(self.n)
        self.p_ref = np.zeros(self.n)
        self.v_ref = np.zeros(self.n) # Added velocity reference

        # Solve for the initial hitch point position
        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess).x
        self.v = np.zeros(self.n)
        
        # --- CLF-HOCBF-QP Controller References ---
        self.n12_ref = np.zeros(self.n)
        self.n34_ref = np.zeros(self.n)
        self.d12_ref = 0.0
        self.d34_ref = 0.0
        self.v_n12_ref = np.zeros(self.n)
        self.v_n34_ref = np.zeros(self.n)
        self.v_d12_ref = 0.0
        self.v_d34_ref = 0.0

        # --- CLF-HOCBF-QP Controller Parameters (naming from manuscript) ---
        self.clf_params = None

        # Pre-computed dynamics quantities
        self.history = {
            "p_ref": [],
            "p": [],
            "p_i": [],
            "p_i_ref": [],
            "n12": [],
            "n34": [],
            "d12": [],
            "d34": [],
            "n12_ref": [],
            "n34_ref": [],
            "d12_ref": [],
            "d34_ref": [],
            "tensions": [np.nan * np.ones(4)],
            "u": [self.u.copy()],
            "V": [],
            "step_time": [],
            "error_mag_sum": []
        }
        self._pre_compute_dynamics()

        # History for animation
        self.controller_type = None


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
    
    def _pre_compute_dynamics(self):
        """Pre-compute dynamics quantities that are used in both step() and step_clf_cbf()."""
        self._r = np.array([self.p - self.p_i[i] for i in range(4)])
        self._r_dot = np.array([self.v - self.v_i[i] for i in range(4)])
        self._r_hat = np.array([self._unit_vector(self._r[i]) for i in range(4)])
        self._r_mag = np.linalg.norm(self._r, axis=1)
        self._R_mat = np.column_stack(self._r_hat)
        
        self._n12 = self._r_hat[0] + self._r_hat[1]
        self._n34 = self._r_hat[2] + self._r_hat[3]
        self._d12 = np.linalg.norm(self.p_i[0] - self.p_i[1])
        self._d34 = np.linalg.norm(self.p_i[2] - self.p_i[3])
        
        self._M_mat = np.array([
            [self._n12 @ self._n12 + self.m / self.m_i[0] + self.m / self.m_i[1], 0, self._n12 @ self._n34, 0],
            [self._n34 @ self._n12, 0, self._n34 @ self._n34 + self.m / self.m_i[2] + self.m / self.m_i[3], 0],
            [1, -1, 0, 0],
            [0, 0, 1, -1]
        ])

        # This part requires r_hat_dot, so compute it here
        self._r_hat_dot = np.array([(
            1.0/self._r_mag[i] * self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))).flatten() for i in range(4)
        ])

        self._n12_dot = self._r_hat_dot[0] + self._r_hat_dot[1]
        self._n34_dot = self._r_hat_dot[2] + self._r_hat_dot[3]
        self._d12_dot = (self.v_i[0] - self.v_i[1]) @ (self.p_i[0] - self.p_i[1]) / self._d12
        self._d34_dot = (self.v_i[2] - self.v_i[3]) @ (self.p_i[2] - self.p_i[3]) / self._d34
        
        self._C_mat = np.zeros((4, 4 * self.n))
        self._C_mat[0, 0:self.n] = -self.m * self._r_hat[0] / self.m_i[0]
        self._C_mat[0, self.n:2*self.n] = -self.m * self._r_hat[1] / self.m_i[1]
        self._C_mat[1, 2*self.n:3*self.n] = -self.m * self._r_hat[2] / self.m_i[2]
        self._C_mat[1, 3*self.n:4*self.n] = -self.m * self._r_hat[3] / self.m_i[3]

        # w: drift only (no control terms)
        self._w = np.zeros(4)
        self._w[0] = self.m * (self._r_hat_dot[0] @ self._r_dot[0] + self._r_hat_dot[1] @ self._r_dot[1])
        self._w[1] = self.m * (self._r_hat_dot[2] @ self._r_dot[2] + self._r_hat_dot[3] @ self._r_dot[3])

        cond_M = np.linalg.cond(self._M_mat)
        self._M_inv = None
        self.history["p_ref"].append(self.p_ref.copy())
        self.history["p"].append(self.p.copy())
        self.history["p_i_ref"].append(self.p_i_ref.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["n12_ref"].append(self.n12_ref.copy())
        self.history["n34_ref"].append(self.n34_ref.copy())
        self.history["n12"].append(self._n12.copy())
        self.history["n34"].append(self._n34.copy())
        self.history["d12_ref"].append(self.d12_ref)
        self.history["d34_ref"].append(self.d34_ref)
        self.history["d12"].append(self._d12)
        self.history["d34"].append(self._d34)
        if np.isnan(cond_M) or cond_M > 1e8:
            eps = 1e-6
            M_reg = self._M_mat + eps * np.eye(4)
            try:
                self._M_inv = np.linalg.inv(M_reg)
                print(f"Warning: regularized M (cond {cond_M:.3e}). Added eps={eps}.")
            except np.linalg.LinAlgError:
                print("Error: M is singular even after regularization. Setting zero input and stepping.")
                assert False, "M inversion failed"
        else:
            try:
                self._M_inv = np.linalg.inv(self._M_mat)
            except np.linalg.LinAlgError:
                print("Error: M inversion failed unexpectedly. Setting zero input and stepping.")
                assert False, "M inversion failed"

    def _compute_dynamics(self):
        """Computes the current accelerations and tensions based on current state and input."""
        f_d = -self.c_d * self.v # Damping force
        w = self._w.copy()
        w[:2] += (np.vstack((self._n12, self._n34)) @ (f_d + self.f_ext))
        t = self._M_inv @ (self._C_mat @ self.u.flatten() + w)
        a = (-self._R_mat @ t + f_d + self.f_ext) / self.m
        a_i = np.array([(t[i] * self._r_hat[i] + self.u[i]) / self.m_i[i] for i in range(4)])
        return a, a_i, t
    
    def _compute_control_affine_matrices(self):
        # 4. Build control-affine dynamics: h_func (10n) and B_func (10n x 4n)
        # R_mat and J matrices
        # hitch acceleration drift
        a_h = -(1.0 / self.m) * (self._R_mat @ (self._M_inv @ self._w))
        # robot acceleration drift
        # build J (4n x 4) mapping t -> t_i * r_hat_i blocks: use as J @ (M_inv @ w)
        J = np.zeros((4 * self.n, 4))
        for i in range(4):
            J[i * self.n:(i + 1) * self.n, i] = self._r_hat[i]
        a_i_h = (J @ (self._M_inv @ self._w)).reshape((4, self.n))  # shape (4, n)

        # assemble h_func
        self._h_func = np.zeros(10 * self.n)
        # p dot and p_i dot
        self._h_func[0:self.n] = self.v
        for i in range(4):
            self._h_func[(i + 1) * self.n:(i + 2) * self.n] = self.v_i[i]
        # dv/dt (hitch)
        self._h_func[5 * self.n:6 * self.n] = a_h
        # dv_i/dt (robots) flattened in order 6n:10n
        for i in range(4):
            self._h_func[(6 + i) * self.n:(7 + i) * self.n] = a_i_h[i]

        # Build B_func: inputs map to accelerations
        # hitch input matrix
        a_B = -(1.0 / self.m) * (self._R_mat @ (self._M_inv @ self._C_mat))
        # robot input blocks: J_mat + J @ M_inv @ C_mat
        J_mat = np.zeros((4 * self.n, 4 * self.n))
        for i in range(4):
            J_mat[i * self.n:(i + 1) * self.n, i * self.n:(i + 1) * self.n] = (1.0 / self.m_i[i]) * np.eye(self.n)
        a_i_B = J_mat + (J @ (self._M_inv @ self._C_mat)).reshape((4 * self.n, 4 * self.n))  # careful shapes
        # assemble B_func (10n x 4n)
        self._B_func = np.zeros((10 * self.n, 4 * self.n))
        self._B_func[5 * self.n:6 * self.n, :] = a_B
        self._B_func[6 * self.n:10 * self.n, :] = a_i_B

    def step(self, verbose=False):
        if self.timer == None:
            self.timer = time.time()
        else:
            step_time = time.time() - self.timer
            self.history["step_time"].append(step_time)
            self.timer = time.time()
        """Advances the simulation by one step using the current self.u"""
        a, a_i, tensions = self._compute_dynamics()
        v_unprojected = self.v + self.dt * a
        p_unprojected = self.p + self.dt * v_unprojected
        self.v_i = self.v_i + self.dt * a_i
        self.p_i = self.p_i + self.dt * self.v_i

        self.v = v_unprojected
        res = self._solve_initial_p(p_unprojected)
        if res.fun > 1e-3:
            # This is expected during dynamic motion, not an error
            print(f"Warning: High constraint violation {res.fun:.6f} when projecting hitch position.")

        p_corrected = res.x

        correction = (p_corrected - p_unprojected) / self.dt
        self.v += 0.5 * correction
        self.p = p_corrected

        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())
        if verbose:
            d12 = np.linalg.norm(self.p_i[0]-self.p_i[1])
            d34 = np.linalg.norm(self.p_i[2]-self.p_i[3])
            n12 = self._unit_vector(self.p - self.p_i[0]) + self._unit_vector(self.p - self.p_i[1])
            n34 = self._unit_vector(self.p - self.p_i[2]) + self._unit_vector(self.p - self.p_i[3])
            return f"tensions={tensions},\n d12={d12}, d34={d34},\n n12={n12}, n34={n34}"

    def step_clf_cbf(self, verbose=False):
        """Compute control input u via CLF-HOCBF-QP (robot-centered CLF) and then step the sim."""
        self._compute_control_affine_matrices()
        # 1. State & params
        n = self.n
        params = self.clf_params
        # CLF robot gains (use provided or defaults)
        Kp_robot = params['Kp_robot']
        Kv_robot = params['Kv_robot']
        gamma = params['gamma']
        alpha = params['alpha']
        beta = params['beta']
        lam = params['lambda']
        t_min = params['t_min']
        u_max = params['u_max']

        # 5. NEW CLF: robot positions & velocities
        p_i_ref = self.p_i_ref
        e_p_i = p_i_ref - self.p_i  # shape (4, n)
        # self.history["error_mag_sum"].append(np.sum(np.linalg.norm(e_p_i, axis=1)))
        e_p_cas = self.p_ref - self.p
        e_n12_cas = self.n12_ref - self._n12
        e_n34_cas = self.n34_ref - self._n34
        e_d12_cas = self.d12_ref - self._d12
        e_d34_cas = self.d34_ref - self._d34
        self.history["error_mag_sum"].append(
            np.linalg.norm(e_p_cas) + np.linalg.norm(e_n12_cas) + np.linalg.norm(e_n34_cas) + abs(e_d12_cas) + abs(e_d34_cas)
        )

        v_i_ref = self.v_i_ref + (Kp_robot @ e_p_i.T).T
        e_v_i = v_i_ref - self.v_i  # shape (4, n)

        V = 0.0
        for i in range(4):
            V += 0.5 * (e_v_i[i] @ Kv_robot @ e_v_i[i])
        self.history["V"].append(V)

        # gradient partial_V_x (row vector, length 10n)
        partial_V_x = np.zeros(10 * n)
        # robot position blocks: p_i located at indices (i+1)*n:(i+2)*n
        # robot velocity blocks: v_i located at indices (6+i)*n:(7+i)*n
        for i in range(4):
            # We apply the chain rule: ∂V/∂x = (∂e_v_i/∂x)ᵀ (∂V/∂e_v_i)
            # where (∂V/∂e_v_i) = e_v_i[i] @ Kv_robot (this is a row vector)

            # --- Partial derivative with respect to robot position p_i ---
            # The term ∂e_v_i/∂p_i comes from the reference velocity:
            # ∂e_v_i/∂p_i = ∂(v_i_ref - v_i)/∂p_i 
            #             = ∂(Kp_robot @ (p_i_ref - p_i))/∂p_i 
            #             = -Kp_robot
            # So, the final gradient component is: (-Kp_robot)ᵀ * (Kv_robot @ e_v_i[i])
            # Assuming gain matrices are symmetric, this is -e_v_i[i] @ Kv_robot @ Kp_robot
            partial_V_p_i = -e_v_i[i] @ Kv_robot @ Kp_robot
            
            p_i_start, p_i_end = (i + 1) * n, (i + 2) * n
            partial_V_x[p_i_start:p_i_end] = partial_V_p_i

            # --- Partial derivative with respect to robot velocity v_i ---
            # The term ∂e_v_i/∂v_i is simpler:
            # ∂e_v_i/∂v_i = ∂(v_i_ref - v_i)/∂v_i = -I (identity matrix)
            # So, the final gradient component is: (-I)ᵀ * (Kv_robot @ e_v_i[i])
            # which simplifies to -e_v_i[i] @ Kv_robot
            partial_V_v_i = -e_v_i[i] @ Kv_robot
            
            v_i_start, v_i_end = (6 + i) * n, (7 + i) * n
            partial_V_x[v_i_start:v_i_end] = partial_V_v_i

        # Lie derivatives of V
        L_h_V = partial_V_x @ self._h_func           # scalar
        L_B_V = partial_V_x @ self._B_func           # row vector (length 4*n)

        # 6. HOCBFs (same as before) and Lie derivatives
        psi = np.zeros(4)
        L_h_psi = np.zeros(4)
        L_B_psi = np.zeros((4, 4 * n))
        l_cables = [self.l12, self.l12, self.l34, self.l34]

        for i in range(4):
            q_i = l_cables[i] - self._r_mag[i]
            q_i_dot = -self._r_hat[i] @ self._r_dot[i]
            psi[i] = q_i_dot + beta * q_i

            # gradients as row vectors
            dpsi_dp = -beta * self._r_hat[i] - (self._r_dot[i] @ (np.eye(n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dpi = beta * self._r_hat[i] + (self._r_dot[i] @ (np.eye(n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dv = -self._r_hat[i]
            dpsi_dvi = self._r_hat[i]

            partial_psi_x = np.zeros(10 * n)
            partial_psi_x[0:n] = dpsi_dp
            partial_psi_x[(i + 1) * n:(i + 2) * n] = dpsi_dpi
            partial_psi_x[5 * n:6 * n] = dpsi_dv
            partial_psi_x[(i + 6) * n:(i + 7) * n] = dpsi_dvi

            L_h_psi[i] = partial_psi_x @ self._h_func
            L_B_psi[i, :] = partial_psi_x @ self._B_func

        # 7. QP setup with diagnostics
        u_var = cp.Variable((4 * n))
        u_nominal = -t_min * np.concatenate(self._r_hat)
        delta = cp.Variable()

        cost = cp.sum_squares(u_var - u_nominal) + alpha * cp.square(delta)

        constraints = []
        # CLF constraint (soft)
        constraints.append(L_B_V @ u_var + L_h_V + gamma * V <= delta)
        # HOCBF constraints (hard)
        constraints.append(L_h_psi + lam * psi + L_B_psi @ u_var >= 0.0)
        # tension constraint (hard, require >= t_min elementwise)
        constraints.append(self._M_inv @ self._C_mat @ u_var + self._M_inv @ self._w >= t_min)
        # actuator bounds
        constraints.append(u_var <= u_max)
        constraints.append(u_var >= -u_max)

        # Diagnostics at u=0
        if verbose:
            delta0 = 0.0
            clf_residual = L_h_V + gamma * V - delta0          # should be <= 0
            cbf_residual = L_h_psi + lam * psi                 # should be >= 0 elementwise
            tension_residual = self._M_inv @ self._w                       # should be >= t_min elementwise
            delta_test = 0.0

            clf_residual = L_h_V + gamma*V + L_B_V @ u_nominal - delta_test   # ≤ 0 desired
            cbf_residual = L_h_psi + L_B_psi @ u_nominal + lam*psi              # ≥ 0 desired
            tension_residual = self._M_inv @ (self._C_mat @ u_nominal + self._w)                  # ≥ t_min desired
            print(f"=== Constraint residuals at u={u_nominal} ===")
            print(f"CLF residual (<=0 desired): {clf_residual:.6e}")
            print(f"CBF residuals (>=0 desired): {cbf_residual}")
            print(f"Tension residuals (>= {t_min} desired): {tension_residual}")
            print("===================================")

        # Solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            # prob.solve(solver=SOLVER, warm_start=True, parallel="on", qp_iteration_limit=10000) # HIGHS
            prob.solve(solver=SOLVER, warm_start=True, max_iter=100000) # OSQP
            # prob.solve(solver=SOLVER, warm_start=True, iter_limit=100000) # DAQP
        except cp.SolverError:
            print("Warning: QP solver error. Setting u=0 and stepping.")
            self.u = np.zeros((4, n))
            self.step(verbose=True)
            return

        # Apply solution
        if prob.status in ["optimal", "optimal_inaccurate"] and u_var.value is not None:
            self.u = u_var.value.reshape((4, n))
        else:
            print(f"Warning: QP status {prob.status}. Using zero input.")
            self.u = np.zeros((4, n))

        # 8. Step the simulation using the existing step() method
        self.step(verbose=verbose)

    def step_ellipsoids(self, verbose=False):
        """Computes control input u via CLF-HOCBF-QP and then steps the simulation."""
        params = self.clf_params
        # NOTE: K_v is removed as the composite error e_p replaces it.
        # NOTE: New cascaded gain parameters are now used.
        K_p, K_n, k_d = params['K_p'], params['K_n'], params['k_d']
        K_p_cas, K_n_cas, k_d_cas = params['K_p_cas'], params['K_n_cas'], params['k_d_cas']
        gamma = params['gamma']
        alpha = params['alpha']
        beta = params['beta']
        lam = params['lambda']
        t_min = params['t_min']
        u_max = params['u_max']
        self._compute_control_affine_matrices()

        # --- First, compute cascaded position errors ---
        e_p_cas = self.p_ref - self.p
        e_n12_cas = self.n12_ref - self._n12
        e_n34_cas = self.n34_ref - self._n34
        e_d12_cas = self.d12_ref - self._d12
        e_d34_cas = self.d34_ref - self._d34
        self.history["error_mag_sum"].append(
            np.linalg.norm(e_p_cas) + np.linalg.norm(e_n12_cas) + np.linalg.norm(e_n34_cas) + abs(e_d12_cas) + abs(e_d34_cas)
        )

        # --- Then, assemble the composite errors as per Eq. (25) ---
        e_p = self.v_ref + K_p_cas @ e_p_cas - self.v
        e_n12 = self.v_n12_ref + K_n_cas @ e_n12_cas - self._n12_dot
        e_n34 = self.v_n34_ref + K_n_cas @ e_n34_cas - self._n34_dot
        e_d12 = self.v_d12_ref + k_d_cas * e_d12_cas - self._d12_dot
        e_d34 = self.v_d34_ref + k_d_cas * e_d34_cas - self._d34_dot

        # Lyapunov function (scalar) - structure is the same, but uses composite errors
        # Note: The e_v term is removed as its role is now captured by e_p.
        V = 0.5 * (
            e_p @ K_p @ e_p
            + e_n12 @ K_n @ e_n12
            + e_n34 @ K_n @ e_n34
            + k_d * (e_d12**2 + e_d34**2)
        )
        self.history["V"].append(V)
        if verbose:
            print("hitch component of V:", 0.5 * (e_p @ K_p @ e_p))
            print("ellipsoid norm component of V:", 0.5 * (e_n12 @ K_n @ e_n12 + e_n34 @ K_n @ e_n34))
            print("ellipsoid distance component of V:", 0.5 * k_d * (e_d12**2 + e_d34**2))

        # --- Gradient of V wrt state x (row vector length 10n) ---
        partial_V_x = np.zeros(10*self.n)

        # --- Pre-compute Jacobians J_i for the rigorous derivatives ---
        J = [np.zeros((self.n, self.n)) for _ in range(4)]
        for i in range(4):
            r_hat_i, r_dot_i, r_mag_i = self._r_hat[i], self._r_dot[i], self._r_mag[i]
            term1 = np.outer(r_hat_i, r_dot_i) + np.outer(r_dot_i, r_hat_i)
            term2 = (r_hat_i @ r_dot_i) * np.eye(self.n)
            term3 = 3 * (r_hat_i @ r_dot_i) * np.outer(r_hat_i, r_hat_i)
            J[i] = -1 / (r_mag_i**2) * (term1 + term2 - term3)

        # --- ∂V/∂p ---
        I_n = np.eye(self.n)
        P_mat = [(I_n - np.outer(self._r_hat[i], self._r_hat[i])) / self._r_mag[i] for i in range(4)]
        
        dV_dp = -e_p @ K_p @ K_p_cas
        dV_dp -= e_n12 @ K_n @ (K_n_cas @ (P_mat[0] + P_mat[1]) + J[0] + J[1])
        dV_dp -= e_n34 @ K_n @ (K_n_cas @ (P_mat[2] + P_mat[3]) + J[2] + J[3])
        partial_V_x[0:self.n] = dV_dp

        # --- ∂V/∂p_i ---
        r_12, r_34 = self.p_i[0] - self.p_i[1], self.p_i[2] - self.p_i[3]
        r_hat_12, r_hat_34 = r_12 / self._d12, r_34 / self._d34
        v_12, v_34 = self.v_i[0] - self.v_i[1], self.v_i[2] - self.v_i[3]
        
        # d(d_dot)/dp_i terms
        dddot_dp1 = (v_12 @ (I_n - np.outer(r_hat_12, r_hat_12))) / self._d12
        dddot_dp3 = (v_34 @ (I_n - np.outer(r_hat_34, r_hat_34))) / self._d34

        dV_dp1 = e_n12 @ K_n @ (K_n_cas @ P_mat[0] + J[0]) \
                 + k_d * e_d12 * (-k_d_cas * r_hat_12 - dddot_dp1)
        dV_dp2 = e_n12 @ K_n @ (K_n_cas @ P_mat[1] + J[1]) \
                 + k_d * e_d12 * (k_d_cas * r_hat_12 + dddot_dp1)
        dV_dp3 = e_n34 @ K_n @ (K_n_cas @ P_mat[2] + J[2]) \
                 + k_d * e_d34 * (-k_d_cas * r_hat_34 - dddot_dp3)
        dV_dp4 = e_n34 @ K_n @ (K_n_cas @ P_mat[3] + J[3]) \
                 + k_d * e_d34 * (k_d_cas * r_hat_34 + dddot_dp3)
        
        partial_V_x[self.n:2*self.n]   = dV_dp1
        partial_V_x[2*self.n:3*self.n] = dV_dp2
        partial_V_x[3*self.n:4*self.n] = dV_dp3
        partial_V_x[4*self.n:5*self.n] = dV_dp4

        # --- ∂V/∂v (or ∂V/∂p_dot) ---
        dV_dv = -e_p @ K_p \
                - e_n12 @ K_n @ (P_mat[0] + P_mat[1]) \
                - e_n34 @ K_n @ (P_mat[2] + P_mat[3])
        partial_V_x[5*self.n:6*self.n] = dV_dv

        # --- ∂V/∂v_i (or ∂V/∂p_i_dot) ---
        dV_dv1 = e_n12 @ K_n @ P_mat[0] - k_d * e_d12 * r_hat_12
        dV_dv2 = e_n12 @ K_n @ P_mat[1] + k_d * e_d12 * r_hat_12
        dV_dv3 = e_n34 @ K_n @ P_mat[2] - k_d * e_d34 * r_hat_34
        dV_dv4 = e_n34 @ K_n @ P_mat[3] + k_d * e_d34 * r_hat_34

        partial_V_x[6*self.n:7*self.n] = dV_dv1
        partial_V_x[7*self.n:8*self.n] = dV_dv2
        partial_V_x[8*self.n:9*self.n] = dV_dv3
        partial_V_x[9*self.n:10*self.n] = dV_dv4

        # Lie derivatives of V
        L_h_V = partial_V_x @ self._h_func
        L_B_V = partial_V_x @ self._B_func
        # === 6. HOCBFs (psi) and their Lie Derivatives ===
        psi = np.zeros(4)
        L_h_psi = np.zeros(4)
        L_B_psi = np.zeros((4, 4*self.n))

        l_cables = [self.l12, self.l12, self.l34, self.l34]

        for i in range(4):
            q_i = l_cables[i] - self._r_mag[i]
            q_i_dot = -self._r_hat[i] @ self._r_dot[i]
            psi[i] = q_i_dot + beta * q_i

            # Gradients (row vectors)
            dpsi_dp = -beta * self._r_hat[i] - (self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dpi = beta * self._r_hat[i] + (self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dv = -self._r_hat[i]
            dpsi_dvi = self._r_hat[i]

            # Fill into big gradient row (length 10n)
            partial_psi_x = np.zeros(10*self.n)
            partial_psi_x[0:self.n] = dpsi_dp
            partial_psi_x[(i+1)*self.n:(i+2)*self.n] = dpsi_dpi
            partial_psi_x[5*self.n:6*self.n] = dpsi_dv
            partial_psi_x[(i+6)*self.n:(i+7)*self.n] = dpsi_dvi

            # Lie derivatives
            L_h_psi[i] = partial_psi_x @ self._h_func
            L_B_psi[i, :] = partial_psi_x @ self._B_func

            
        # === 7. Setup and Solve the Quadratic Programming problem ===
        u_nominal = -t_min * np.concatenate(self._r_hat)
        u = cp.Variable((4*self.n))
        # delta = cp.Variable()

        cost = cp.sum_squares(u - u_nominal) #+ alpha * cp.square(delta)

        # Constraints
        constraints = [
            # CLF constraint (performance goal): L_hV + L_BV u + gamma*V <= delta
            L_B_V @ u + L_h_V + gamma * V <= 0,#delta,

            # HOCBF constraints (safety rule): L_h(psi) + L_B(psi) u + lambda*psi >= 0
            L_h_psi + lam * psi + L_B_psi @ u >= 0.0,

            # Positive tension constraint (safety rule): t = M_inv(Cu+w) >= t_min
            self._M_inv @ self._C_mat @ u + self._M_inv @ self._w >= t_min,

            # Actuator limits
            u <= u_max,
            u >= -u_max
        ]

        # --- Debug: print residuals at u=0 (and delta=0) ---
        if verbose:
            delta_test = 0.0
            clf_residual = L_h_V + gamma*V + L_B_V @ u_nominal - delta_test   # ≤ 0 desired
            cbf_residual = L_h_psi + L_B_psi @ u_nominal + lam*psi              # ≥ 0 desired
            tension_residual = self._M_inv @ (self._C_mat @ u_nominal + self._w)                  # ≥ t_min desired
            print(f"=== Constraint residuals at u={u_nominal} ===")
            print(f"CLF residual (should be ≤ 0): {clf_residual:.4e}")
            print(f"CBF residuals (should be ≥ 0): {cbf_residual}")
            print(f"Tension residuals (should be ≥ {t_min}): {tension_residual}")
            print("===================================")

        # --- Solve QP ---
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            # prob.solve(solver=SOLVER, warm_start=True, parallel="on", qp_iteration_limit=10000)
            prob.solve(solver=SOLVER, warm_start=True, max_iter=100000)
            # prob.solve(solver=SOLVER, warm_start=True, iter_limit=100000)
        except cp.error.SolverError:
            print(f"Warning: QP solver failed: {prob.status}. Setting u to zero.")
            self.u = np.zeros((4, self.n))
            self.step(verbose=True)
            return

        # 8. Update control input and step the simulation
        if prob.status in ["optimal", "optimal_inaccurate"]:
            self.u = u.value.reshape((4, self.n)) if u.value is not None else np.zeros_like(self.u)
        else:
            print(f"Warning: QP solver failed with status '{prob.status}'. Using zero input.")
            self.u = np.zeros((4, self.n))

        self.step(verbose=verbose)


    def run(self, steps, ref_trajectory, controller_type='clf_cbf',verbose=True):
        """Runs the simulation for a given number of steps."""
        outer_timer = time.time()
        self.controller_type = controller_type
        ref_func = ref_trajectory[controller_type]
        for step_count in range(steps):
            string = ""
            self._pre_compute_dynamics()
            p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref, v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref = ref_func(step_count * self.dt)
            self.p_ref, self.p_i_ref, self.n12_ref, self.n34_ref, self.d12_ref, self.d34_ref = p_ref[:N], p_i_ref[:, :N], n12_ref[:N], n34_ref[:N], d12_ref, d34_ref 
            self.v_ref, self.v_i_ref, self.v_n12_ref, self.v_n34_ref, self.v_d12_ref, self.v_d34_ref = v_ref[:N], v_i_ref[:, :N], v_n12_ref[:N], v_n34_ref[:N], v_d12_ref, v_d34_ref
            
            if self.controller_type == 'clf_cbf':
                self.step_clf_cbf(verbose=verbose)
            elif self.controller_type == 'ellipsoids_clf_cbf':
                self.step_ellipsoids(verbose=verbose)
            else: # Fallback to a passive step
                string = self.step(verbose=verbose)
            if step_count % 100 == 0:
                # --- Progression Bar ---
                progress = (step_count + 1) / steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                # Ensure the custom 'string' doesn't make the line too long
                max_info_len = 40 
                info_string = (string[:max_info_len-3] + '...') if len(string) > max_info_len else string
                
                # Use carriage return '\r' to return to the start of the line
                print(f'\rProgress: |{bar}| {progress:.1%} ({step_count+1}/{steps}) | {info_string.ljust(max_info_len)}', end="")
                # print( f'Progress: {step_count+1}/{steps}' )

        print(f"\nSimulation completed in {time.time() - outer_timer:.2f} seconds.")


    # --- Animation functions ---
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
        p_ref_hist = np.array(self.history["p_ref"])
        p_i_hist = np.array(self.history["p_i"])
        p_i_ref_hist = np.array(self.history["p_i_ref"])
        tensions_hist = np.array(self.history["tensions"])
        u_hist = np.array(self.history["u"])
        V_hist = np.array(self.history["V"])
        Err_hist = np.array(self.history["error_mag_sum"])
        step_time_hist = np.array(self.history["step_time"])

        # History for normal vectors and desired distances
        n12_ref_hist = np.array(self.history.get("n12_ref", []))
        n34_ref_hist = np.array(self.history.get("n34_ref", []))
        n12_hist = np.array(self.history.get("n12", []))
        n34_hist = np.array(self.history.get("n34", []))
        d12_ref_hist = np.array(self.history.get("d12_ref", []))
        d34_ref_hist = np.array(self.history.get("d34_ref", []))
        d12_hist = np.array(self.history.get("d12", []))
        d34_hist = np.array(self.history.get("d34", []))

        fig = plt.figure(figsize=(12, 12))
        step = frame_skip + 1
        animation_frames = range(0, len(p_hist), step)
        robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
        all_points = np.vstack([p_hist, p_i_hist.reshape(-1, self.n), p_ref_hist, p_i_ref_hist.reshape(-1, self.n)])

        if self.n == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title("3D Cable Robot Simulation")
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
            ax.set_zlim(all_points[:,2].min()-1, all_points[:,2].max()+1)

            hitch_point, = ax.plot([], [], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_dots = [ax.plot([], [], [], 'o', color=c, ms=10, label='robot position')[0] for c in robot_colors]
            hitch_ref, = ax.plot([], [], [], 'gx', ms=10, mew=3, label='hitch reference')
            robot_ref = [ax.plot([], [], [], 'x', color=c, ms=10, label='robot reference')[0] for c in robot_colors]
            cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            input_arrows = [ax.quiver([], [], [], [], [], [], color=c, length=0.5, normalize=True, arrow_length_ratio=0.1) for c in robot_colors]
            normal_ref1_arrow, normal_ref2_arrow = None, None
            wireframe1, wireframe2 = None, None
            
            # ADDED: Text object for displaying d_ref values
            dist_ref_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                                      verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
            normal_ref1_arrow, normal_ref2_arrow = None, None
            n12_arrow, n34_arrow = None, None  # Add arrows for n12 and n34

            def update_3d(frame_index):
                nonlocal wireframe1, wireframe2, input_arrows, normal_ref1_arrow, normal_ref2_arrow, n12_arrow, n34_arrow
                # ... (rest of the variable assignments)
                p, p_i, u = p_hist[frame_index], p_i_hist[frame_index], u_hist[frame_index]
                p_ref, p_i_ref = p_ref_hist[frame_index], p_i_ref_hist[frame_index]
                hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
                hitch_ref.set_data_3d([p_ref[0]], [p_ref[1]], [p_ref[2]])
                for i in range(4):
                    # ... (robot, cable, and input arrow updates)
                    robot_dots[i].set_data_3d([p_i[i, 0]], [p_i[i, 1]], [p_i[i, 2]])
                    cables[i].set_data_3d([p[0], p_i[i, 0]], [p[1], p_i[i, 1]], [p[2], p_i[i, 2]])
                    robot_ref[i].set_data_3d([p_i_ref[i, 0]], [p_i_ref[i, 1]], [p_i_ref[i, 2]])
                    if input_arrows[i] is not None:
                        input_arrows[i].remove()
                    input_arrows[i] = ax.quiver(p_i[i, 0], p_i[i, 1], p_i[i, 2],
                                                u[i, 0], u[i, 1], u[i, 2],
                                                color=robot_colors[i], length=np.linalg.norm(u[i])*0.1, normalize=False, arrow_length_ratio=0.3)

                # ... (normal vector arrow updates)
                n12_ref, n34_ref = n12_ref_hist[frame_index], n34_ref_hist[frame_index]
                n12, n34 = n12_hist[frame_index], n34_hist[frame_index]

                if normal_ref1_arrow: normal_ref1_arrow.remove()
                if normal_ref2_arrow: normal_ref2_arrow.remove()
                normal_ref1_arrow = ax.quiver(p_ref[0], p_ref[1], p_ref[2],
                                                n12_ref[0], n12_ref[1], n12_ref[2],
                                                color='c', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2)
                normal_ref2_arrow = ax.quiver(p_ref[0], p_ref[1], p_ref[2],
                                                n34_ref[0], n34_ref[1], n34_ref[2],
                                                color='m', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2)
                if n12_arrow: n12_arrow.remove()
                if n34_arrow: n34_arrow.remove()
                n12_arrow = ax.quiver(p[0], p[1], p[2],
                                    n12[0], n12[1], n12[2],
                                    color='c', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2, linestyle='dashed')
                n34_arrow = ax.quiver(p[0], p[1], p[2],
                                    n34[0], n34[1], n34[2],
                                    color='m', length=0.8, normalize=True, arrow_length_ratio=0.3, lw=2, linestyle='dashed')
                
                # ADDED: Update text with d_ref values
                if len(d12_ref_hist) > 0:
                    d12_ref = d12_ref_hist[frame_index]
                    d34_ref = d34_ref_hist[frame_index]
                    d12 = d12_hist[frame_index]
                    d34 = d34_hist[frame_index]
                    text_str = f'$d_{{12,ref}}$: {d12_ref:.2f}, $d_{{34,ref}}$: {d34_ref:.2f}\n$d_{{12}}$: {d12:.2f}, $d_{{34}}$: {d34:.2f}'
                    dist_ref_text.set_text(text_str)

                # ... (wireframe updates)
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
            ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
            ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)

            hitch_point, = ax.plot([], [], 'ko', ms=8, zorder=10, label='Hitch Point')
            robot_dots = [ax.plot([], [], 'o', color=c, ms=10, label='robot position')[0] for c in robot_colors]
            hitch_ref, = ax.plot([], [], 'gx', ms=10, mew=3, label='hitch reference')
            robot_ref = [ax.plot([], [], 'x', color=c, ms=10, label='robot reference')[0] for c in robot_colors]
            cables = [ax.plot([], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
            tension_texts = [ax.text(0, 0, '', fontsize=9, ha='center', va='center', backgroundcolor=(1,1,1,0.7)) for _ in range(4)]
            input_arrows = [ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.3, fc=c, ec=c, length_includes_head=True) for c in robot_colors]
            ellipse1_line, = ax.plot([], [], 'c--', lw=1, label='Constraint Ellipse 1-2')
            ellipse2_line, = ax.plot([], [], 'm--', lw=1, label='Constraint Ellipse 3-4')
            normal_ref1_arrow, normal_ref2_arrow = None, None
            n12_arrow, n34_arrow = None, None  # Add arrows for n12 and n34
            
            # ADDED: Text object for displaying d_ref values
            dist_ref_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
            ax.legend()

            def update_2d(frame_index):
                nonlocal normal_ref1_arrow, normal_ref2_arrow, n12_arrow, n34_arrow
                # ... (rest of the variable assignments)
                p, p_i, u = p_hist[frame_index], p_i_hist[frame_index], u_hist[frame_index]
                p_ref, p_i_ref = p_ref_hist[frame_index], p_i_ref_hist[frame_index]
                t_arr = tensions_hist[frame_index] if frame_index < len(tensions_hist) else np.zeros(4)
                hitch_point.set_data([p[0]], [p[1]])
                hitch_ref.set_data([p_ref[0]], [p_ref[1]])
                for i in range(4):
                    # ... (robot, cable, and input arrow updates)
                    robot_dots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
                    cables[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])
                    robot_ref[i].set_data([p_i_ref[i, 0]], [p_i_ref[i, 1]])
                    mid_point = (p + p_i[i]) / 2
                    tension_texts[i].set_position((mid_point[0], mid_point[1])); tension_texts[i].set_text(f"{t_arr[i]:.2f} N")
                    input_arrows[i].remove()
                    input_arrows[i] = ax.arrow(p_i[i, 0], p_i[i, 1], u[i, 0]*0.1, u[i, 1]*0.1,
                                               head_width=0.1, head_length=0.1, fc=robot_colors[i], ec=robot_colors[i],
                                               length_includes_head=False)
                
                # ... (normal vector arrow updates)
                n12_ref, n34_ref = n12_ref_hist[frame_index], n34_ref_hist[frame_index]
                n12, n34 = n12_hist[frame_index], n34_hist[frame_index]
                if normal_ref1_arrow: normal_ref1_arrow.remove()
                if normal_ref2_arrow: normal_ref2_arrow.remove()
                normal_ref1_arrow = ax.arrow(p_ref[0], p_ref[1], n12_ref[0], n12_ref[1],
                                                head_width=0.1, head_length=0.1, fc='c', ec='c', lw=2, length_includes_head=False)
                normal_ref2_arrow = ax.arrow(p_ref[0], p_ref[1], n34_ref[0], n34_ref[1],
                                                head_width=0.1, head_length=0.1, fc='m', ec='m', lw=2, length_includes_head=False)
                # --- Plot n12 and n34 at each frame ---
                if n12_arrow: n12_arrow.remove()
                if n34_arrow: n34_arrow.remove()
                n12_arrow = ax.arrow(p[0], p[1], n12[0], n12[1],
                                    head_width=0.1, head_length=0.1, fc='c', ec='c', lw=2, length_includes_head=False, linestyle='dashed')
                n34_arrow = ax.arrow(p[0], p[1], n34[0], n34[1],
                                    head_width=0.1, head_length=0.1, fc='m', ec='m', lw=2, length_includes_head=False, linestyle='dashed')
                
                # ADDED: Update text with d_ref values
                if len(d12_ref_hist) > 0:
                    d12_ref = d12_ref_hist[frame_index]
                    d34_ref = d34_ref_hist[frame_index]
                    d12 = d12_hist[frame_index]
                    d34 = d34_hist[frame_index]
                    text_str = f'$d_{{12,ref}}$: {d12_ref:.2f}, $d_{{34,ref}}$: {d34_ref:.2f}\n$d_{{12}}$: {d12:.2f}, $d_{{34}}$: {d34:.2f}'
                    dist_ref_text.set_text(text_str)

                # ... (ellipse updates)
                ellipse1_pts = self._get_ellipse_points(p_i[0], p_i[1], self.l12)
                if ellipse1_pts is not None: ellipse1_line.set_data(ellipse1_pts[0], ellipse1_pts[1])
                ellipse2_pts = self._get_ellipse_points(p_i[2], p_i[3], self.l34)
                if ellipse2_pts is not None: ellipse2_line.set_data(ellipse2_pts[0], ellipse2_pts[1])
                return [hitch_point, *robot_dots, *cables, *tension_texts, *input_arrows, ellipse1_line, ellipse2_line]

            ani = FuncAnimation(fig, update_2d, frames=animation_frames, blit=False, interval=20)

        else:
            print(f"Animation is not supported for n={self.n}.")
            return

        plt.show()

        # --- Plot for Lyapunov function ---
        fig_v, ax_v = plt.subplots(figsize=(12, 8))
        ax_v.set_title("Lyapunov function Over Time")
        ax_v.set_xlabel("Time (s))")
        ax_v.set_ylabel("V")
        ax_v.plot(np.array(range(len(V_hist)))*DT, V_hist, 'b-')
        ax_v.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        fig_e, ax_e = plt.subplots(figsize=(12, 8))
        ax_e.set_title("Error sum Over Time")
        ax_e.set_xlabel("Time (s))")
        ax_e.set_ylabel("Error Sum")
        ax_e.plot(np.array(range(len(Err_hist)))*DT, Err_hist, 'b-')
        ax_e.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # # --- Plot for step time ---
        # fig_t, ax_t = plt.subplots(figsize=(12, 8))
        # ax_t.set_title("Step Time")
        # ax_t.set_xlabel("Time Step")
        # ax_t.set_ylabel("Computation Time (s)")
        # ax_t.plot(step_time_hist, 'b.')
        # ax_t.grid(True, linestyle='--', alpha=0.6)
        # plt.show()


def robot_traj(t):
    """
    Computes the desired position and velocity for four robots
    following a rotating trajectory around a central point.
    """
    # --- Position Calculation ---
    MAG1, MAG2 = 0.5, 0.6  # Mags for x and y oscillations
    FREQ1, FREQ2 = 0.35, 0.75  # Frequencies for x and y oscillations
    p_ref = np.array([0.0, 0.0, 0.25]) + np.array([MAG1*np.cos(FREQ1*t + 1.0), MAG2*np.sin(FREQ2*t), 0.0])
    v_ref = np.array([0.0, 0.0, 0.0]) + np.array([-MAG1*FREQ1*np.sin(FREQ1*t + 1.0), MAG2*FREQ2*np.cos(FREQ2*t), 0.0])
    delta_p_i = np.array([
        [-L12*np.sqrt(2)/4, -L12*np.sqrt(2)/4, 0.0],
        [-L12*np.sqrt(2)/4,  L12*np.sqrt(2)/4, 0.0],
        [ L34*np.sqrt(2)/4,  L34*np.sqrt(2)/4, 0.0],
        [ L34*np.sqrt(2)/4, -L34*np.sqrt(2)/4, 0.0]
    ])

    # Angular position (yaw) and its time derivative (angular velocity)
    yaw_dot_ref = 0.1
    yaw_ref = yaw_dot_ref * t
    
    # Pitch is constant, so its derivative is zero
    pitch_ref = 0.0

    # Shorthand for sine and cosine values
    cy, sy = np.cos(yaw_ref), np.sin(yaw_ref)
    cp, sp = np.cos(pitch_ref), np.sin(pitch_ref)
    
    # Rotation matrix for position
    R_ref = np.array([
        [cy*cp, -sy, cy*sp],
        [sy*cp,  cy, sy*sp],
        [-sp,    0,   cp]
    ])
    
    # Calculate desired position
    rotated_delta_p_i = (R_ref @ delta_p_i.T).T
    p_i_ref = p_ref + rotated_delta_p_i

    # --- Velocity Calculation ---
    # v_i(t) = d/dt(p_ref + R_ref(t) * delta_p_i) = R_dot_ref(t) * delta_p_i
    
    # Time derivative of the rotation matrix (R_dot_ref)
    # This is found using the chain rule: dR/dt = (dR/dyaw) * (dyaw/dt)
    R_dot_ref = yaw_dot_ref * np.array([
        [-sy*cp, -cy, -sy*sp],
        [ cy*cp, -sy,  cy*sp],
        [0,       0,     0]
    ])
    
    # Calculate desired velocity
    v_i_ref = (R_dot_ref @ delta_p_i.T).T + v_ref

    # Return position and velocity, sliced to the specified dimension N
    return p_ref, v_ref, p_i_ref, v_i_ref

def robot_traj_ellipsoids(t):
    p_ref, v_ref, p_i_ref_full, v_i_ref_full = robot_traj(t)
    r_ref = np.array([p_ref - p_i_ref_full[i] for i in range(4)])
    r_dot_ref = np.array([v_ref - v_i_ref_full[i] for i in range(4)])
    r_hat_ref = np.array([r_ref[i]/np.linalg.norm(r_ref[i]) for i in range(4)])
    r_mag_ref = np.linalg.norm(r_ref, axis=1)
    
    n12_ref = r_hat_ref[0] + r_hat_ref[1]
    n34_ref = r_hat_ref[2] + r_hat_ref[3]
    d12_ref = np.linalg.norm(p_i_ref_full[0] - p_i_ref_full[1])
    d34_ref = np.linalg.norm(p_i_ref_full[2] - p_i_ref_full[3])

    # This part requires r_hat_dot, so compute it here
    r_hat_dot = np.array([(
        1.0/r_mag_ref[i] * r_dot_ref[i] @ (np.eye(3) - np.outer(r_hat_ref[i], r_hat_ref[i]))).flatten() for i in range(4)
    ])

    v_n12_ref = r_hat_dot[0] + r_hat_dot[1]
    v_n34_ref = r_hat_dot[2] + r_hat_dot[3]
    v_d12_ref = (v_i_ref_full[0] - v_i_ref_full[1]) @ (p_i_ref_full[0] - p_i_ref_full[1]) / d12_ref
    v_d34_ref = (v_i_ref_full[2] - v_i_ref_full[3]) @ (p_i_ref_full[2] - p_i_ref_full[3]) / d34_ref
    # print("Reference normal vector velocities:", v_n12_ref, v_n34_ref)
    # print("Reference distance velocities:", v_d12_ref, v_d12_ref)

    return p_ref, p_i_ref_full, n12_ref, n34_ref, d12_ref, d34_ref, v_ref, v_i_ref_full, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref


def get_robot_refs_from_ellipsoids(p_ref, v_ref, n12_ref, v_n12_ref, n34_ref, v_n34_ref, d12_ref, v_d12_ref, d34_ref, v_d34_ref):
    """
    Computes desired robot positions and velocities based on ellipsoid properties.

    This function implements Equation (18) and its time derivative to determine
    the reference positions and velocities of four robots that would create
    the specified ellipsoid geometry.

    Args:
        p_ref (np.ndarray): Desired hitch position (3,).
        v_ref (np.ndarray): Desired hitch velocity (3,).
        n12_ref (np.ndarray): Desired normal vector for ellipsoid 12 (3,).
        v_n12_ref (np.ndarray): Desired velocity of normal vector 12 (3,).
        n34_ref (np.ndarray): Desired normal vector for ellipsoid 34 (3,).
        v_n34_ref (np.ndarray): Desired velocity of normal vector 34 (3,).
        d12_ref (float): Desired major axis length for ellipsoid 12.
        v_d12_ref (float): Desired velocity of major axis length 12.
        d34_ref (float): Desired major axis length for ellipsoid 34.
        v_d34_ref (float): Desired velocity of major axis length 34.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - p_i_ref (np.ndarray): Desired positions for the 4 robots (4, 3).
            - v_i_ref (np.ndarray): Desired velocities for the 4 robots (4, 3).
    """
    p_i_ref = np.zeros((4, 3))
    v_i_ref = np.zeros((4, 3))

    refs = [
        (d12_ref, v_d12_ref, n12_ref, v_n12_ref),
        (d12_ref, v_d12_ref, n12_ref, v_n12_ref),
        (d34_ref, v_d34_ref, n34_ref, v_n34_ref),
        (d34_ref, v_d34_ref, n34_ref, v_n34_ref)
    ]

    # This vector defines the plane in which the robots are arranged.
    # It is an interpretation of the term kappa_i in Equation (18),
    # assumed to be a constant unit vector for a coplanar configuration.
    kappa_ref = np.array([0., 0., 1.])

    for i in range(4):
        d_ref, v_d_ref, n_ref, v_n_ref = refs[i]
        
        # --- Position Calculation ---
        n_mag = np.linalg.norm(n_ref)
        # Add epsilon for numerical stability, preventing division by zero or sqrt of negative
        eps = 1e-9
        n_mag_sq = n_mag**2
        
        # Denominator term from Eq. (18)
        denom = np.sqrt(max(4.0 - n_mag_sq, eps))
        
        # Normalized normal vector
        n_hat = n_ref / (n_mag + eps)
        
        # The cross product term orients the major axis in the plane
        cross_term = np.cross(kappa_ref, n_hat)
        
        # The (-1)^r term separates the two robots on the same cable
        sign = (-1)**(i) # for i=0,1 gives 1,-1; for i=2,3 gives 1,-1

        # Full offset vector from Eq. (18)
        offset = (d_ref / 2.0) * (-n_ref / denom + sign * cross_term)
        p_i_ref[i] = p_ref + offset

        # --- Velocity Calculation (Time Derivative of the above) ---
        # This requires applying product and chain rules extensively.
        
        # Derivative of the magnitude: d/dt(||n||) = (n . n_dot) / ||n||
        n_mag_dot = (n_ref @ v_n_ref) / (n_mag + eps)
        
        # Derivative of the denominator: d/dt(sqrt(4-||n||^2))
        denom_dot = - (n_mag * n_mag_dot) / (denom + eps)

        # Derivative of the first term: d/dt(-n/denom) using quotient rule
        term1_dot = (-v_n_ref * denom - (-n_ref) * denom_dot) / (denom**2 + eps)
        
        # Derivative of the unit normal vector: d/dt(n_hat)
        n_hat_dot = (v_n_ref - n_hat * (n_hat @ v_n_ref)) / (n_mag + eps)
        
        # Derivative of the cross product term
        cross_term_dot = np.cross(kappa_ref, n_hat_dot)

        # Assemble the velocity of the offset using the product rule
        offset_dot = (v_d_ref / 2.0) * (-n_ref / denom + sign * cross_term) + \
                     (d_ref / 2.0) * (term1_dot + sign * cross_term_dot)
        
        v_i_ref[i] = v_ref + offset_dot

    return p_i_ref, v_i_ref


def robot_traj_from_ellipsoids(t):
    """
    Generates a sample trajectory for the ellipsoid properties and computes
    the corresponding robot reference positions and velocities.
    """
    # --- Generate Reference Trajectory for Ellipsoid Properties ---
    # 1. Hitch Point Trajectory (e.g., oscillating)
    HITCH_MAG1, HITCH_MAG2, HITCH_MAG3 = 0.6, 0.5, 0.0
    HITCH_FREQ1, HITCH_FREQ2, HITCH_FREQ3 = 0.4, 0.4, 0.0
    D12_OFFSET = 3.8 # Average length
    D12_MAG = 0.2    # Oscillation amplitude
    D12_FREQ = 0.5   # Oscillation frequency (rad/s)

    # Parameters for d34 (major axis length of ellipsoid 3-4)
    D34_OFFSET = 3.5 # Average length
    D34_MAG = 0.3    # Oscillation amplitude
    D34_FREQ = 0.8   # Oscillation frequency (rad/s)
    
    # Constant angular velocity for the normal vectors' rotation
    YAW_DOT_REF = 0.3 # rad/s

    # --- Generate Reference Trajectory for Ellipsoid Properties ---
    
    # 1. Hitch Point Trajectory (e.g., oscillating)
    p_ref = np.array([HITCH_MAG1 * np.cos(HITCH_FREQ1 * t), 
                      HITCH_MAG2 * np.sin(HITCH_FREQ2 * t - 1.0), 
                      HITCH_MAG3 * np.sin(HITCH_FREQ3 * t + 1.0)])
    v_ref = np.array([-HITCH_MAG1 * HITCH_FREQ1 * np.sin(HITCH_FREQ1 * t), 
                      HITCH_MAG2 * HITCH_FREQ2 * np.cos(HITCH_FREQ2 * t - 1.0), 
                      HITCH_MAG3 * HITCH_FREQ3 * np.cos(HITCH_FREQ3 * t + 1.0)])

    # 2. Ellipsoid Major Axis Lengths (now both oscillating)
    d12_ref = D12_OFFSET + D12_MAG * np.sin(D12_FREQ * t)
    v_d12_ref = D12_MAG * D12_FREQ * np.cos(D12_FREQ * t)
    
    d34_ref = D34_OFFSET + D34_MAG * np.sin(D34_FREQ * t)
    v_d34_ref = D34_MAG * D34_FREQ * np.cos(D34_FREQ * t)
    
    # 3. Ellipsoid Normal Vectors (rotating at constant velocity)
    angle = YAW_DOT_REF * t
    angle_dot = YAW_DOT_REF # The constant velocity

    n12_ref = np.array([1.4*np.cos(angle), 1.4*np.sin(angle), 0.])
    # For equilibrium, n34 should be collinear and opposite to n12
    n34_ref = -n12_ref

    # Time derivatives of the normal vectors
    v_n12_ref = angle_dot * np.array([-np.sin(angle), np.cos(angle), 0.])
    v_n34_ref = -v_n12_ref

    # --- Compute Robot References from Ellipsoid Properties ---
    p_i_ref, v_i_ref = get_robot_refs_from_ellipsoids(
        p_ref, v_ref, n12_ref, v_n12_ref, n34_ref, v_n34_ref,
        d12_ref, v_d12_ref, d34_ref, v_d34_ref
    )

    return p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref, \
           v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref


def main():
    n = N # Switch to 3 for 3D
    dt = DT
    steps = 5000

    p_i0 = np.array([
        [-1.5, -1.8, -0.25],
        [-2.0,  2.3, 0.25],
        [ 2.0,  2.2, -0.35],
        [ 1.5, -1.9, 0.20]
    ])[:, :n]

    v_i0 = np.random.normal(0, 0.05, (4, 3))[:, :n]
    # v_i0[:, 1] = -0.2 # Give a small initial upward velocity to avoid singularity
    m = 0.001
    m_i = np.ones(4) * 0.1
    l12 = L12
    l34 = L34
    c_d = 0.2 # Damping coefficient
    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)

    # Set references for the controller to track
    p_ref = np.array([0.0, 0.0, 0.25])
    delta_p_i = np.array([[-l12*np.sqrt(2)/4, -l12*np.sqrt(2)/4, 0.0],
                        [-l12*np.sqrt(2)/4,  l12*np.sqrt(2)/4, 0.0],
                        [ l34*np.sqrt(2)/4,  l34*np.sqrt(2)/4, 0.0],
                        [ l34*np.sqrt(2)/4, -l34*np.sqrt(2)/4, 0.0]])

    yaw_ref = 0.0 # radians
    pitch_ref = 0.0 # radians
    R_ref = np.array([
        [np.cos(yaw_ref)*np.cos(pitch_ref), -np.sin(yaw_ref), np.cos(yaw_ref)*np.sin(pitch_ref)],
        [np.sin(yaw_ref)*np.cos(pitch_ref),  np.cos(yaw_ref), np.sin(yaw_ref)*np.sin(pitch_ref)],
        [-np.sin(pitch_ref),                 0,               np.cos(pitch_ref)]
    ])
    rotated_delta_p_i = (R_ref @ delta_p_i.T).T
    sim.p_ref = p_ref[:n] 
    sim.p_i_ref = sim.p_ref + rotated_delta_p_i[:, :n]
    
    sim.n12_ref = -sim._unit_vector(rotated_delta_p_i[0, :n]) - sim._unit_vector(rotated_delta_p_i[1, :n])
    sim.n34_ref = -sim._unit_vector(rotated_delta_p_i[2, :n]) - sim._unit_vector(rotated_delta_p_i[3, :n])
    sim.d12_ref = np.sqrt(2) * l12/2
    sim.d34_ref = np.sqrt(2) * l34/2
    
    sim.v_i_ref = np.zeros((4, n))
    sim.f_ext = np.array([0.0, 0.0, -2 * m])[:n] # Gravity force on the payload
    sim.clf_params = {
        'K_p': np.diag([5.0] * n),      # Proportional gain for hitch error
        'K_n': np.diag([0.5] * n),       # Gain for normal vector error
        'k_d': 0.2,                           # Gain for robot distance error (scalar)
        'K_p_cas': np.diag([0.5] * n),  # CLF gain for hitch position cascade error 
        'K_n_cas': np.diag([0.5] * n),  # CLF gain for normal vector cascade error
        'k_d_cas': 0.2,                         # CLF gain for robot distance cascade error (scalar)
        'gamma': DT*100,                        # CLF exponential convergence rate
        'alpha': 1e6,                         # Penalty weight for CLF relaxation (delta)
        'beta': 10.0,                          # HOCBF parameter for q_i
        'lambda': 100.0,                        # HOCBF parameter for psi_i
        't_min': 0.1,                         # Minimum desired cable tension (CBF)
        'u_max': 100.0,                         # Max control input magnitude
        'Kp_robot': np.diag([10.0] * n),  # Robot position CLF gain
        'Kv_robot': np.diag([20.0] * n),  # Robot velocity CLF gain
    }

    ref_trajectory = {'clf_cbf': robot_traj_from_ellipsoids,
                      'ellipsoids_clf_cbf': robot_traj_ellipsoids}

    # --- CHOOSE THE CONTROLLER TO RUN ---
    controller = 'clf_cbf' # Options: 'mpc', 'clf_cbf', 'ellipsoids'
    try:
        sim.run(steps, ref_trajectory, controller_type=controller,verbose=False)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    sim.animate(frame_skip=19)

if __name__ == "__main__":
    SOLVER = cp.OSQP
    L12 = 5.8
    L34 = 5.5
    N = 3
    DT = 0.01
    main()

