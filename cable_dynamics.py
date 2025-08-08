import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

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
            "tensions": [] # Store tensions for efficient and accurate animation
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

    def step(self):
        """Advances the simulation by one time step."""
        a, a_i, tensions = self._compute_dynamics()

        # --- Integration Step ---
        # Update velocities and positions using Euler integration
        self.v += self.dt * a
        p_unprojected = self.p + self.dt * self.v
        
        self.v_i += self.dt * a_i
        self.p_i += self.dt * self.v_i
        
        # --- Projection Step ---
        # Project the hitch point back onto the constraint manifold to correct drift
        p_corrected = self._solve_initial_p(p_unprojected)
        
        # Update velocity based on the correction to maintain physical consistency
        self.v += (p_corrected - p_unprojected) / self.dt
        self.p = p_corrected

        # Store results for this step
        self.history["p"].append(self.p.copy())
        self.history["p_i"].append(self.p_i.copy())
        self.history["tensions"].append(tensions.copy())

    def run(self, steps):
        """Runs the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()

    def _get_ellipsoid_points(self, f1, f2, major_axis_length):
        """Calculates wireframe points for an ellipsoid defined by its foci."""
        if self.n != 3: return None, None, None
        
        center = (f1 + f2) / 2
        dist = np.linalg.norm(f1 - f2)
        if dist >= major_axis_length: return None, None, None

        a = major_axis_length / 2.0
        c = dist / 2.0
        b = np.sqrt(a**2 - c**2)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = b * np.outer(np.ones_like(u), np.cos(v))

        # Rotation logic
        f1_to_f2 = self._unit_vector(f2 - f1)
        if np.allclose(f1_to_f2, [1, 0, 0]):
            rot_mat = np.identity(3)
        else:
            v_axis = np.cross([1, 0, 0], f1_to_f2)
            s = np.linalg.norm(v_axis)
            c = np.dot([1, 0, 0], f1_to_f2)
            vx = np.array([[0, -v_axis[2], v_axis[1]], [v_axis[2], 0, -v_axis[0]], [-v_axis[1], v_axis[0], 0]])
            rot_mat = np.identity(3) + vx + vx @ vx * ((1 - c) / (s**2))

        points = np.stack([x, y, z], axis=-1)
        points = points @ rot_mat.T + center
        
        return points[..., 0], points[..., 1], points[..., 2]

    def animate(self, frame_skip=0):
        """
        Creates and displays an animation of the simulation.

        Args:
            frame_skip (int): The number of frames to skip between each rendered frame.
        """
        if self.n != 3:
            print("Animation is only supported for n=3.")
            return

        p_hist = np.array(self.history["p"])
        p_i_hist = np.array(self.history["p_i"])
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D Cable Robot Simulation")

        # Set plot limits
        all_points = np.vstack(self.history['p_i'])
        ax.set_xlim(all_points[:,0].min()-1, all_points[:,0].max()+1)
        ax.set_ylim(all_points[:,1].min()-1, all_points[:,1].max()+1)
        ax.set_zlim(all_points[:,2].min()-1, all_points[:,2].max()+1)

        hitch_point, = ax.plot([], [], [], 'ko', ms=8, zorder=10, label='Hitch Point')
        robot_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
        robot_dots = [ax.plot([], [], [], 'o', color=c, ms=10)[0] for c in robot_colors]
        cables = [ax.plot([], [], [], '-', color=c, lw=1.5, alpha=0.8)[0] for c in ['#d62728', '#d62728', '#1f77b4', '#1f77b4']]
        force_arrows = [ax.plot([], [], [], '-', color=c, lw=2)[0] for c in robot_colors]
        
        # Use lists for wireframes to handle removal
        wireframe1_lines = []
        wireframe2_lines = []

        step = frame_skip + 1
        animation_frames = range(0, len(p_hist), step)

        def update(frame_index):
            nonlocal wireframe1_lines, wireframe2_lines
            p = p_hist[frame_index]
            p_i = p_i_hist[frame_index]

            hitch_point.set_data_3d([p[0]], [p[1]], [p[2]])
            for i in range(4):
                robot_dots[i].set_data_3d([p_i[i, 0]], [p_i[i, 1]], [p_i[i, 2]])
                cables[i].set_data_3d([p[0], p_i[i, 0]], [p[1], p_i[i, 1]], [p[2], p_i[i, 2]])
                
                start = p_i[i]
                end = start + 0.5 * self.u[i]
                force_arrows[i].set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

            # Remove old wireframes
            for wf in wireframe1_lines: wf.remove()
            for wf in wireframe2_lines: wf.remove()
            wireframe1_lines.clear()
            wireframe2_lines.clear()
            
            # Draw new wireframes
            x1, y1, z1 = self._get_ellipsoid_points(p_i[0], p_i[1], self.l12)
            if x1 is not None:
                wireframe1_lines.append(ax.plot_wireframe(x1, y1, z1, color='c', alpha=0.3))
            
            x2, y2, z2 = self._get_ellipsoid_points(p_i[2], p_i[3], self.l34)
            if x2 is not None:
                wireframe2_lines.append(ax.plot_wireframe(x2, y2, z2, color='m', alpha=0.3))

            return [hitch_point, *robot_dots, *cables, *force_arrows]

        ani = FuncAnimation(fig, update, frames=animation_frames, blit=False, interval=50)
        plt.show()

def main():
    n = 3 # Switch to 3D
    dt = 0.005
    steps = 1000

    p_i0 = np.array([
        [-2.0, -2.5, 0.0],
        [-1.5,  2.0, 0.5],
        [ 1.5,  2.0, -0.5],
        [ 2.0, -2.5, 0.0]
    ])

    v_i0 = np.zeros((4, n))
    m = 0.1
    m_i = np.ones(4) * 0.5
    l12 = 6.5
    l34 = 6.5
    c_d = 0.0 # Damping coefficient

    sim = CableRobotSystem(p_i0, v_i0, l12, l34, m, m_i, dt, c_d)

    sim.u[0] = np.array([-1.0, -1.0, 0.3])
    sim.u[1] = np.array([-1.0, 1.0, -0.1])
    sim.u[2] = np.array([1.0, 1.0, 0.3])
    sim.u[3] = np.array([1.0, -1.0, 0.0])
    sim.f_ext = np.array([0.0, 0.0, -0.5])  # External force on the hitch point

    sim.run(steps)
    sim.animate(frame_skip=9)

if __name__ == "__main__":
    main()
