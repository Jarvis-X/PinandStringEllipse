import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, FancyArrowPatch

class CableRobotSystem:
    def __init__(self, p_i0, v_i0, l12, l34, m, m_i, dt):
        self.n = p_i0.shape[1]
        self.dt = dt
        self.l12 = l12
        self.l34 = l34
        self.m = m
        self.m_i = m_i

        self.p_i = p_i0.copy()
        self.v_i = v_i0.copy()
        self.f_i_ext = np.zeros((4, self.n))

        p0_guess = np.mean(p_i0, axis=0)
        self.p = self._solve_initial_p(p0_guess)
        self.v = np.zeros(self.n)

        self.history = {"p": [self.p.copy()], "p_i": [self.p_i.copy()]}

    def _constraint_loss(self, p):
        r1 = np.linalg.norm(p - self.p_i[0]) + np.linalg.norm(p - self.p_i[1]) - self.l12
        r2 = np.linalg.norm(p - self.p_i[2]) + np.linalg.norm(p - self.p_i[3]) - self.l34
        return r1**2 + r2**2

    def _solve_initial_p(self, guess):
        res = minimize(self._constraint_loss, guess, method='BFGS')
        return res.x

    def _unit_vector(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else np.zeros_like(v)

    def _compute_tensions(self):
        u = np.array([self._unit_vector(self.p - self.p_i[i]) for i in range(4)])
        r = np.linalg.norm(self.p - self.p_i, axis=1)
        alpha = np.array([1/self.m_i[0] + 1/self.m_i[1], 1/self.m_i[2] + 1/self.m_i[3]])

        def c_term(i):
            return (
                (1/self.m_i[i]) * u[i] @ self.f_i_ext[i] -
                (1/r[i]) * (self.v - self.v_i[i]) @ (np.eye(self.n) - np.outer(u[i], u[i])) @ (self.v - self.v_i[i])
            )

        c1 = c_term(0) + c_term(1)
        c2 = c_term(2) + c_term(3)

        M = np.array([
            [np.linalg.norm(u[0] + u[1])**2, (u[0] + u[1]) @ (u[2] + u[3])],
            [(u[2] + u[3]) @ (u[0] + u[1]), np.linalg.norm(u[2] + u[3])**2]
        ])

        RHS = -self.m * np.array([c1, c2])
        M += self.m * np.diag(alpha)

        tensions = np.linalg.solve(M, RHS)
        return u, tensions
    
    
    def compute_accelerations(self):
        u = np.array([self._unit_vector(self.p - self.p_i[i]) for i in range(4)])
        r = np.linalg.norm(self.p - self.p_i, axis=1)

        def constraint_ddot(i, j):
            ri = self.p - self.p_i[i]
            rj = self.p - self.p_i[j]
            vi = self.v - self.v_i[i]
            vj = self.v - self.v_i[j]

            ui = self._unit_vector(ri)
            uj = self._unit_vector(rj)

            term1 = (vi @ ui)**2 / np.linalg.norm(ri)
            term2 = (vj @ uj)**2 / np.linalg.norm(rj)
            return - (term1 + term2)

        c1_ddot = constraint_ddot(0, 1)
        c2_ddot = constraint_ddot(2, 3)

        alpha = np.array([1/self.m_i[0] + 1/self.m_i[1], 1/self.m_i[2] + 1/self.m_i[3]])

        def c_term(i):
            return (
                (1/self.m_i[i]) * u[i] @ self.f_i_ext[i] -
                (1/r[i]) * (self.v - self.v_i[i]) @ (np.eye(self.n) - np.outer(u[i], u[i])) @ (self.v - self.v_i[i])
            )

        c1 = c_term(0) + c_term(1)
        c2 = c_term(2) + c_term(3)

        M = np.array([
            [np.linalg.norm(u[0] + u[1])**2, (u[0] + u[1]) @ (u[2] + u[3])],
            [(u[2] + u[3]) @ (u[0] + u[1]), np.linalg.norm(u[2] + u[3])**2]
        ])

        RHS = -self.m * np.array([c1, c2]) + np.array([c1_ddot, c2_ddot])
        M += self.m * np.diag(alpha)

        tensions = np.linalg.solve(M, RHS)
        t12, t34 = tensions
        t = np.array([t12, t12, t34, t34])

        a = (-t12 * (u[0] + u[1]) - t34 * (u[2] + u[3])) / self.m
        a_i = np.array([(t[i] * u[i] + self.f_i_ext[i]) / self.m_i[i] for i in range(4)])

        return a, a_i

    def step(self):
        u, tensions = self._compute_tensions()
        t12, t34 = tensions
        t = np.array([t12, t12, t34, t34])

        a = (-t12 * (u[0] + u[1]) - t34 * (u[2] + u[3])) / self.m
        a_i = np.array([(t[i] * u[i] + self.f_i_ext[i]) / self.m_i[i] for i in range(4)])

        self.v += self.dt * a
        self.p += self.dt * self.v

        self.v_i += self.dt * a_i
        self.p_i += self.dt * self.v_i

        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def _ellipse_points_from_foci(self, f1, f2, major_axis_length, num_points=100):
        c = np.linalg.norm(f2 - f1) / 2
        a = major_axis_length / 2
        b = np.sqrt(max(a**2 - c**2, 0))
        center = (f1 + f2) / 2
        angle = np.arctan2(f2[1] - f1[1], f2[0] - f1[0])

        t = np.linspace(0, 2*np.pi, num_points)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)

        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        ellipse_points = np.dot(R, np.vstack((ellipse_x, ellipse_y)))
        ellipse_points[0, :] += center[0]
        ellipse_points[1, :] += center[1]
        return ellipse_points

    def animate(self):
        p_hist = np.array(self.history["p"])
        p_i_hist = np.array(self.history["p_i"])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title("Cable Robot Simulation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        hitch_point, = ax.plot([], [], 'ko', label='Hitch Point')
        robot_colors = ['r', 'g', 'b', 'm']
        robot_labels = ['R1', 'R2', 'R3', 'R4']
        robot_dots = [ax.plot([], [], 'o', color=c, label=lab)[0] for c, lab in zip(robot_colors, robot_labels)]
        cables = [ax.plot([], [], '-', color=c, label=f'C{i+1}')[0] for i, c in enumerate(['darkred', 'darkred', 'navy', 'navy'])]

        force_arrows = [FancyArrowPatch((0,0), (0,0), color=robot_colors[i], arrowstyle='->', mutation_scale=15) for i in range(4)]
        for arrow in force_arrows:
            ax.add_patch(arrow)

        tension_texts = [ax.text(0, 0, '', color='black', fontsize=9, backgroundcolor='white') for _ in range(4)]

        def init():
            return [hitch_point, *robot_dots, *cables, *tension_texts, *force_arrows]

        def update(frame):
            p = p_hist[frame]
            p_i = p_i_hist[frame]

            hitch_point.set_data([p[0]], [p[1]])
            for i in range(4):
                robot_dots[i].set_data([p_i[i, 0]], [p_i[i, 1]])
                cables[i].set_data([p[0], p_i[i, 0]], [p[1], p_i[i, 1]])

            u = np.array([self._unit_vector(p - p_i[i]) for i in range(4)])
            alpha = np.array([1/self.m_i[0] + 1/self.m_i[1], 1/self.m_i[2] + 1/self.m_i[3]])
            def c_term(i):
                return (1/self.m_i[i]) * u[i] @ self.f_i_ext[i]
            c1 = c_term(0) + c_term(1)
            c2 = c_term(2) + c_term(3)
            M = np.array([
                [np.linalg.norm(u[0] + u[1])**2, (u[0] + u[1]) @ (u[2] + u[3])],
                [(u[2] + u[3]) @ (u[0] + u[1]), np.linalg.norm(u[2] + u[3])**2]
            ])
            RHS = -self.m * np.array([c1, c2])
            M += self.m * np.diag(alpha)
            tensions = np.linalg.solve(M, RHS)
            t12, t34 = tensions
            t_arr = np.array([t12, t12, t34, t34])

            for i in range(4):
                mid_x = (p[0] + p_i[i, 0]) / 2
                mid_y = (p[1] + p_i[i, 1]) / 2
                tension_texts[i].set_position((mid_x, mid_y))
                tension_texts[i].set_text(f"{t_arr[i]:.2f}")

            ellipse1_pts = self._ellipse_points_from_foci(p_i[0], p_i[1], self.l12)
            ellipse2_pts = self._ellipse_points_from_foci(p_i[2], p_i[3], self.l34)

            if hasattr(self, '_ellipse_lines'):
                for line in self._ellipse_lines:
                    line.remove()
            self._ellipse_lines = [
                ax.plot(ellipse1_pts[0], ellipse1_pts[1], 'c--', label='Ellipse 1')[0],
                ax.plot(ellipse2_pts[0], ellipse2_pts[1], 'm--', label='Ellipse 2')[0]
            ]

            for i in range(4):
                start = p_i[i]
                end = start + 0.5 * self.f_i_ext[i]
                force_arrows[i].set_positions(start, end)

            return [hitch_point, *robot_dots, *cables, *tension_texts, *force_arrows, *self._ellipse_lines]

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        ani = FuncAnimation(fig, update, frames=len(p_hist), init_func=init,
                            blit=True, interval=0.1, repeat=False)
        plt.show()

def main():
    n = 2
    dt = 0.001
    steps = 5000

    p_i0 = np.array([
        [-1.5, -2.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [1.0, -2.0]
    ])

    v_i0 = np.zeros((4, n))
    m = 0.001
    m_i = np.ones(4) * 0.2
    l12 = 4.5
    l34 = 3.5

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt)

    sim.f_i_ext[0] = np.array([-1.0, -1.0])
    sim.f_i_ext[1] = np.array([-1.0, 1.0])
    sim.f_i_ext[2] = np.array([1.0, 1.0])
    sim.f_i_ext[3] = np.array([1.0, -1.0])

    sim.run(steps)
    sim.animate()

if __name__ == "__main__":
    main()
