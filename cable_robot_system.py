import time
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


class CableRobotSystem:
    """
    Simulates the dynamics of a cable-driven robot system based on the paper
    "From Ellipsoids to Midair Control of Dynamic Hitches".

    Pure dynamics + CLF-HOCBF-QP control. No plotting; visualization lives in
    visualization.py and batch aggregation in batch_eval.py.
    """

    def __init__(self, p_i0, v_i0, l12, l34, m, m_i, dt, c_d=0.0, solver=cp.OSQP):
        self.n = p_i0.shape[1]
        self.dt = dt
        self.l12 = l12
        self.l34 = l34
        self.m = m
        self.m_i = m_i
        self.c_d = c_d
        self.solver = solver
        self.timer = None

        # Robot states
        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.p_i_ref = p_i0.copy()
        self.v_i_ref = np.zeros_like(self.v_i)
        self.u = np.zeros((4, self.n))
        self.f_ext = np.zeros(self.n)
        self.p_ref = np.zeros(self.n)
        self.v_ref = np.zeros(self.n)

        # Solve for the initial hitch point position
        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess).x
        self.v = np.zeros(self.n)

        # Configuration references
        self.n12_ref = np.zeros(self.n)
        self.n34_ref = np.zeros(self.n)
        self.d12_ref = 0.0
        self.d34_ref = 0.0
        self.v_n12_ref = np.zeros(self.n)
        self.v_n34_ref = np.zeros(self.n)
        self.v_d12_ref = 0.0
        self.v_d34_ref = 0.0

        # CLF-HOCBF-QP gains/parameters set by caller
        self.clf_params = None

        self.history = {
            "p_ref": [], "p": [],
            "p_i": [], "p_i_ref": [],
            "n12": [], "n34": [],
            "d12": [], "d34": [],
            "n12_ref": [], "n34_ref": [],
            "d12_ref": [], "d34_ref": [],
            "tensions": [np.nan * np.ones(4)],
            "u": [self.u.copy()],
            "V": [],
            "step_time": [],
            "error_mag_sum": []
        }
        self._pre_compute_dynamics()
        self.controller_type = None

    def _constraint_loss(self, p):
        err1 = np.linalg.norm(p - self.p_i[0]) + np.linalg.norm(p - self.p_i[1]) - self.l12
        err2 = np.linalg.norm(p - self.p_i[2]) + np.linalg.norm(p - self.p_i[3]) - self.l34
        return err1**2 + err2**2

    def _solve_initial_p(self, guess):
        return minimize(self._constraint_loss, guess, method='BFGS')

    @staticmethod
    def _unit_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else np.zeros_like(v)

    def _pre_compute_dynamics(self):
        """Pre-compute dynamics quantities used in step() and step_clf_cbf()."""
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

        self._r_hat_dot = np.array([(
            1.0 / self._r_mag[i] * self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))
        ).flatten() for i in range(4)])

        self._n12_dot = self._r_hat_dot[0] + self._r_hat_dot[1]
        self._n34_dot = self._r_hat_dot[2] + self._r_hat_dot[3]
        self._d12_dot = (self.v_i[0] - self.v_i[1]) @ (self.p_i[0] - self.p_i[1]) / self._d12
        self._d34_dot = (self.v_i[2] - self.v_i[3]) @ (self.p_i[2] - self.p_i[3]) / self._d34

        self._C_mat = np.zeros((4, 4 * self.n))
        self._C_mat[0, 0:self.n] = -self.m * self._r_hat[0] / self.m_i[0]
        self._C_mat[0, self.n:2 * self.n] = -self.m * self._r_hat[1] / self.m_i[1]
        self._C_mat[1, 2 * self.n:3 * self.n] = -self.m * self._r_hat[2] / self.m_i[2]
        self._C_mat[1, 3 * self.n:4 * self.n] = -self.m * self._r_hat[3] / self.m_i[3]

        # Drift only (no control terms, no damping/external)
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
                assert False, "M inversion failed"
        else:
            try:
                self._M_inv = np.linalg.inv(self._M_mat)
            except np.linalg.LinAlgError:
                assert False, "M inversion failed"

    def _compute_dynamics(self):
        f_d = -self.c_d * self.v
        w = self._w.copy()
        w[:2] += (np.vstack((self._n12, self._n34)) @ (f_d + self.f_ext))
        t = self._M_inv @ (self._C_mat @ self.u.flatten() + w)
        a = (-self._R_mat @ t + f_d + self.f_ext) / self.m
        a_i = np.array([(t[i] * self._r_hat[i] + self.u[i]) / self.m_i[i] for i in range(4)])
        return a, a_i, t

    def _compute_control_affine_matrices(self):
        a_h = -(1.0 / self.m) * (self._R_mat @ (self._M_inv @ self._w))
        J = np.zeros((4 * self.n, 4))
        for i in range(4):
            J[i * self.n:(i + 1) * self.n, i] = self._r_hat[i]
        a_i_h = (J @ (self._M_inv @ self._w)).reshape((4, self.n))

        self._h_func = np.zeros(10 * self.n)
        self._h_func[0:self.n] = self.v
        for i in range(4):
            self._h_func[(i + 1) * self.n:(i + 2) * self.n] = self.v_i[i]
        self._h_func[5 * self.n:6 * self.n] = a_h
        for i in range(4):
            self._h_func[(6 + i) * self.n:(7 + i) * self.n] = a_i_h[i]

        a_B = -(1.0 / self.m) * (self._R_mat @ (self._M_inv @ self._C_mat))
        J_mat = np.zeros((4 * self.n, 4 * self.n))
        for i in range(4):
            J_mat[i * self.n:(i + 1) * self.n, i * self.n:(i + 1) * self.n] = (1.0 / self.m_i[i]) * np.eye(self.n)
        a_i_B = J_mat + (J @ (self._M_inv @ self._C_mat)).reshape((4 * self.n, 4 * self.n))
        self._B_func = np.zeros((10 * self.n, 4 * self.n))
        self._B_func[5 * self.n:6 * self.n, :] = a_B
        self._B_func[6 * self.n:10 * self.n, :] = a_i_B

    def step(self, verbose=False):
        if self.timer is None:
            self.timer = time.time()
        else:
            self.history["step_time"].append(time.time() - self.timer)
            self.timer = time.time()

        a, a_i, tensions = self._compute_dynamics()
        v_unprojected = self.v + self.dt * a
        p_unprojected = self.p + self.dt * v_unprojected
        self.v_i = self.v_i + self.dt * a_i
        self.p_i = self.p_i + self.dt * self.v_i

        self.v = v_unprojected
        res = self._solve_initial_p(p_unprojected)
        if res.fun > 1e-3:
            print(f"Warning: High constraint violation {res.fun:.6f} when projecting hitch position.")
            for i in range(4):
                self.p_i[i, :] = 0.99 * self.p_i[i, :] + 0.01 * self.p

        p_corrected = res.x
        correction = (p_corrected - p_unprojected) / self.dt
        self.v += 0.5 * correction
        self.p = p_corrected

        self.history["tensions"].append(tensions.copy())
        self.history["u"].append(self.u.copy())
        if verbose:
            d12 = np.linalg.norm(self.p_i[0] - self.p_i[1])
            d34 = np.linalg.norm(self.p_i[2] - self.p_i[3])
            n12 = self._unit_vector(self.p - self.p_i[0]) + self._unit_vector(self.p - self.p_i[1])
            n34 = self._unit_vector(self.p - self.p_i[2]) + self._unit_vector(self.p - self.p_i[3])
            return f"tensions={tensions},\n d12={d12}, d34={d34},\n n12={n12}, n34={n34}"

    def step_clf_cbf(self, verbose=False):
        """Robot-centered CLF, HOCBF on cable length, tension lower bound."""
        self._compute_control_affine_matrices()
        n = self.n
        p = self.clf_params
        Kp_robot, Kv_robot = p['Kp_robot'], p['Kv_robot']
        gamma, alpha, beta = p['gamma'], p['alpha'], p['beta']
        lam, t_min, u_max = p['lambda'], p['t_min'], p['u_max']

        e_p_i = self.p_i_ref - self.p_i
        e_p_cas = self.p_ref - self.p
        e_n12_cas = self.n12_ref - self._n12
        e_n34_cas = self.n34_ref - self._n34
        e_d12_cas = self.d12_ref - self._d12
        e_d34_cas = self.d34_ref - self._d34
        self.history["error_mag_sum"].append(
            np.linalg.norm(e_p_cas) + np.linalg.norm(e_n12_cas)
            + np.linalg.norm(e_n34_cas) + abs(e_d12_cas) + abs(e_d34_cas)
        )

        v_i_ref = self.v_i_ref + (Kp_robot @ e_p_i.T).T
        e_v_i = v_i_ref - self.v_i

        V = 0.0
        for i in range(4):
            V += 0.5 * (e_v_i[i] @ Kv_robot @ e_v_i[i])
        self.history["V"].append(V)

        partial_V_x = np.zeros(10 * n)
        for i in range(4):
            partial_V_p_i = -e_v_i[i] @ Kv_robot @ Kp_robot
            partial_V_x[(i + 1) * n:(i + 2) * n] = partial_V_p_i
            partial_V_v_i = -e_v_i[i] @ Kv_robot
            partial_V_x[(6 + i) * n:(7 + i) * n] = partial_V_v_i

        L_h_V = partial_V_x @ self._h_func
        L_B_V = partial_V_x @ self._B_func

        psi = np.zeros(4)
        L_h_psi = np.zeros(4)
        L_B_psi = np.zeros((4, 4 * n))
        l_cables = [self.l12, self.l12, self.l34, self.l34]

        for i in range(4):
            q_i = l_cables[i] - self._r_mag[i]
            q_i_dot = -self._r_hat[i] @ self._r_dot[i]
            psi[i] = q_i_dot + beta * q_i

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

        u_var = cp.Variable((4 * n))
        u_nominal = -t_min * np.concatenate(self._r_hat)
        delta = cp.Variable()

        cost = cp.sum_squares(u_var - u_nominal) + alpha * cp.square(delta)
        constraints = [
            L_B_V @ u_var + L_h_V + gamma * V <= delta,
            L_h_psi + lam * psi + L_B_psi @ u_var >= 0.0,
            self._M_inv @ self._C_mat @ u_var + self._M_inv @ self._w >= t_min,
            u_var <= u_max,
            u_var >= -u_max,
        ]

        if verbose:
            delta_test = 0.0
            clf_residual = L_h_V + gamma * V + L_B_V @ u_nominal - delta_test
            cbf_residual = L_h_psi + L_B_psi @ u_nominal + lam * psi
            tension_residual = self._M_inv @ (self._C_mat @ u_nominal + self._w)
            print(f"=== Constraint residuals at u={u_nominal} ===")
            print(f"CLF residual (<=0 desired): {clf_residual:.6e}")
            print(f"CBF residuals (>=0 desired): {cbf_residual}")
            print(f"Tension residuals (>= {t_min} desired): {tension_residual}")
            print("===================================")

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=self.solver, warm_start=True, max_iter=100000)
        except cp.SolverError:
            print("Warning: QP solver error. Setting u=0 and stepping.")
            self.u = np.zeros((4, n))
            self.step(verbose=True)
            return

        if prob.status in ["optimal", "optimal_inaccurate"] and u_var.value is not None:
            self.u = u_var.value.reshape((4, n))
        else:
            print(f"Warning: QP status {prob.status}. Using zero input.")
            self.u = np.zeros((4, n))

        self.step(verbose=verbose)

    def step_ellipsoids(self, verbose=False):
        """Ellipsoid-centered CLF (composite cascade error), HOCBF on cable length."""
        params = self.clf_params
        K_p, K_n, k_d = params['K_p'], params['K_n'], params['k_d']
        K_p_cas, K_n_cas, k_d_cas = params['K_p_cas'], params['K_n_cas'], params['k_d_cas']
        gamma, alpha, beta = params['gamma'], params['alpha'], params['beta']
        lam, t_min, u_max = params['lambda'], params['t_min'], params['u_max']
        self._compute_control_affine_matrices()

        e_p_cas = self.p_ref - self.p
        e_n12_cas = self.n12_ref - self._n12
        e_n34_cas = self.n34_ref - self._n34
        e_d12_cas = self.d12_ref - self._d12
        e_d34_cas = self.d34_ref - self._d34
        self.history["error_mag_sum"].append(
            np.linalg.norm(e_p_cas) + np.linalg.norm(e_n12_cas)
            + np.linalg.norm(e_n34_cas) + abs(e_d12_cas) + abs(e_d34_cas)
        )

        e_p = self.v_ref + K_p_cas @ e_p_cas - self.v
        e_n12 = self.v_n12_ref + K_n_cas @ e_n12_cas - self._n12_dot
        e_n34 = self.v_n34_ref + K_n_cas @ e_n34_cas - self._n34_dot
        e_d12 = self.v_d12_ref + k_d_cas * e_d12_cas - self._d12_dot
        e_d34 = self.v_d34_ref + k_d_cas * e_d34_cas - self._d34_dot

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

        partial_V_x = np.zeros(10 * self.n)
        J = [np.zeros((self.n, self.n)) for _ in range(4)]
        for i in range(4):
            r_hat_i, r_dot_i, r_mag_i = self._r_hat[i], self._r_dot[i], self._r_mag[i]
            term1 = np.outer(r_hat_i, r_dot_i) + np.outer(r_dot_i, r_hat_i)
            term2 = (r_hat_i @ r_dot_i) * np.eye(self.n)
            term3 = 3 * (r_hat_i @ r_dot_i) * np.outer(r_hat_i, r_hat_i)
            J[i] = -1 / (r_mag_i**2) * (term1 + term2 - term3)

        I_n = np.eye(self.n)
        P_mat = [(I_n - np.outer(self._r_hat[i], self._r_hat[i])) / self._r_mag[i] for i in range(4)]

        dV_dp = -e_p @ K_p @ K_p_cas
        dV_dp -= e_n12 @ K_n @ (K_n_cas @ (P_mat[0] + P_mat[1]) + J[0] + J[1])
        dV_dp -= e_n34 @ K_n @ (K_n_cas @ (P_mat[2] + P_mat[3]) + J[2] + J[3])
        partial_V_x[0:self.n] = dV_dp

        r_12, r_34 = self.p_i[0] - self.p_i[1], self.p_i[2] - self.p_i[3]
        r_hat_12, r_hat_34 = r_12 / self._d12, r_34 / self._d34
        v_12, v_34 = self.v_i[0] - self.v_i[1], self.v_i[2] - self.v_i[3]
        dddot_dp1 = (v_12 @ (I_n - np.outer(r_hat_12, r_hat_12))) / self._d12
        dddot_dp3 = (v_34 @ (I_n - np.outer(r_hat_34, r_hat_34))) / self._d34

        dV_dp1 = e_n12 @ K_n @ (K_n_cas @ P_mat[0] + J[0]) + k_d * e_d12 * (-k_d_cas * r_hat_12 - dddot_dp1)
        dV_dp2 = e_n12 @ K_n @ (K_n_cas @ P_mat[1] + J[1]) + k_d * e_d12 * (k_d_cas * r_hat_12 + dddot_dp1)
        dV_dp3 = e_n34 @ K_n @ (K_n_cas @ P_mat[2] + J[2]) + k_d * e_d34 * (-k_d_cas * r_hat_34 - dddot_dp3)
        dV_dp4 = e_n34 @ K_n @ (K_n_cas @ P_mat[3] + J[3]) + k_d * e_d34 * (k_d_cas * r_hat_34 + dddot_dp3)

        partial_V_x[self.n:2 * self.n] = dV_dp1
        partial_V_x[2 * self.n:3 * self.n] = dV_dp2
        partial_V_x[3 * self.n:4 * self.n] = dV_dp3
        partial_V_x[4 * self.n:5 * self.n] = dV_dp4

        dV_dv = -e_p @ K_p \
                - e_n12 @ K_n @ (P_mat[0] + P_mat[1]) \
                - e_n34 @ K_n @ (P_mat[2] + P_mat[3])
        partial_V_x[5 * self.n:6 * self.n] = dV_dv

        dV_dv1 = e_n12 @ K_n @ P_mat[0] - k_d * e_d12 * r_hat_12
        dV_dv2 = e_n12 @ K_n @ P_mat[1] + k_d * e_d12 * r_hat_12
        dV_dv3 = e_n34 @ K_n @ P_mat[2] - k_d * e_d34 * r_hat_34
        dV_dv4 = e_n34 @ K_n @ P_mat[3] + k_d * e_d34 * r_hat_34

        partial_V_x[6 * self.n:7 * self.n] = dV_dv1
        partial_V_x[7 * self.n:8 * self.n] = dV_dv2
        partial_V_x[8 * self.n:9 * self.n] = dV_dv3
        partial_V_x[9 * self.n:10 * self.n] = dV_dv4

        L_h_V = partial_V_x @ self._h_func
        L_B_V = partial_V_x @ self._B_func

        psi = np.zeros(4)
        L_h_psi = np.zeros(4)
        L_B_psi = np.zeros((4, 4 * self.n))
        l_cables = [self.l12, self.l12, self.l34, self.l34]
        for i in range(4):
            q_i = l_cables[i] - self._r_mag[i]
            q_i_dot = -self._r_hat[i] @ self._r_dot[i]
            psi[i] = q_i_dot + beta * q_i

            dpsi_dp = -beta * self._r_hat[i] - (self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dpi = beta * self._r_hat[i] + (self._r_dot[i] @ (np.eye(self.n) - np.outer(self._r_hat[i], self._r_hat[i]))) / self._r_mag[i]
            dpsi_dv = -self._r_hat[i]
            dpsi_dvi = self._r_hat[i]

            partial_psi_x = np.zeros(10 * self.n)
            partial_psi_x[0:self.n] = dpsi_dp
            partial_psi_x[(i + 1) * self.n:(i + 2) * self.n] = dpsi_dpi
            partial_psi_x[5 * self.n:6 * self.n] = dpsi_dv
            partial_psi_x[(i + 6) * self.n:(i + 7) * self.n] = dpsi_dvi

            L_h_psi[i] = partial_psi_x @ self._h_func
            L_B_psi[i, :] = partial_psi_x @ self._B_func

        u_nominal = -t_min * np.concatenate(self._r_hat)
        u = cp.Variable((4 * self.n))
        cost = cp.sum_squares(u - u_nominal)
        constraints = [
            L_B_V @ u + L_h_V + gamma * V <= 0,
            L_h_psi + lam * psi + L_B_psi @ u >= 0.0,
            self._M_inv @ self._C_mat @ u + self._M_inv @ self._w >= t_min,
            u <= u_max,
            u >= -u_max,
        ]

        if verbose:
            delta_test = 0.0
            clf_residual = L_h_V + gamma * V + L_B_V @ u_nominal - delta_test
            cbf_residual = L_h_psi + L_B_psi @ u_nominal + lam * psi
            tension_residual = self._M_inv @ (self._C_mat @ u_nominal + self._w)
            print(f"=== Constraint residuals at u={u_nominal} ===")
            print(f"CLF residual (should be <= 0): {clf_residual:.4e}")
            print(f"CBF residuals (should be >= 0): {cbf_residual}")
            print(f"Tension residuals (should be >= {t_min}): {tension_residual}")
            print("===================================")

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=self.solver, warm_start=True, max_iter=100000)
        except cp.error.SolverError:
            print(f"Warning: QP solver failed: {prob.status}. Setting u to zero.")
            self.u = np.zeros((4, self.n))
            self.step(verbose=True)
            return

        if prob.status in ["optimal", "optimal_inaccurate"]:
            self.u = u.value.reshape((4, self.n)) if u.value is not None else np.zeros_like(self.u)
        else:
            print(f"Warning: QP solver failed with status '{prob.status}'. Using zero input.")
            self.u = np.zeros((4, self.n))

        self.step(verbose=verbose)

    def run(self, steps, ref_func, controller_type='clf_cbf', verbose=True, progress=True):
        """
        Run the simulation for a given number of steps.

        ref_func: callable t -> (p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref,
                                 v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref)
        controller_type: 'clf_cbf' (robot-centered) or 'ellipsoids_clf_cbf' (ellipsoid-centered).
        """
        outer_timer = time.time()
        self.controller_type = controller_type
        for step_count in range(steps):
            string = ""
            self._pre_compute_dynamics()
            (p_ref, p_i_ref, n12_ref, n34_ref, d12_ref, d34_ref,
             v_ref, v_i_ref, v_n12_ref, v_n34_ref, v_d12_ref, v_d34_ref) = ref_func(step_count * self.dt)
            n = self.n
            self.p_ref, self.p_i_ref = p_ref[:n], p_i_ref[:, :n]
            self.n12_ref, self.n34_ref = n12_ref[:n], n34_ref[:n]
            self.d12_ref, self.d34_ref = d12_ref, d34_ref
            self.v_ref, self.v_i_ref = v_ref[:n], v_i_ref[:, :n]
            self.v_n12_ref, self.v_n34_ref = v_n12_ref[:n], v_n34_ref[:n]
            self.v_d12_ref, self.v_d34_ref = v_d12_ref, v_d34_ref

            if controller_type == 'clf_cbf':
                self.step_clf_cbf(verbose=verbose)
            elif controller_type == 'ellipsoids_clf_cbf':
                self.step_ellipsoids(verbose=verbose)
            else:
                string = self.step(verbose=verbose) or ""

            if progress and step_count % 100 == 0:
                pct = (step_count + 1) / steps
                bar_length = 40
                filled_length = int(bar_length * pct)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                max_info_len = 40
                info_string = (string[:max_info_len - 3] + '...') if len(string) > max_info_len else string
                print(f'\rProgress: |{bar}| {pct:.1%} ({step_count + 1}/{steps}) | {info_string.ljust(max_info_len)}', end="")

        if progress:
            print(f"\nSimulation completed in {time.time() - outer_timer:.2f} seconds.")
