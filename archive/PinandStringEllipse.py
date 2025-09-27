import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
import time
from matplotlib.animation import FuncAnimation

class Ellipse:
    """
    Represents an ellipse using the dual-foci (pin-and-string) definition.
    The class stores the definition of the ellipse but does not compute
    its equation upon initialization.
    """
    def __init__(self, p1: np.ndarray, p2: np.ndarray, d: float):
        """
        Initializes an Ellipse.

        Args:
            p1 (np.ndarray): The first focus point, e.g., np.array([x, y]).
            p2 (np.ndarray): The second focus point, e.g., np.array([x, y]).
            d (float): The major axis length (the constant "string" length).
                       This must be greater than the distance between the foci.
        """
        # Validate that the major axis length is physically possible
        focal_distance = np.linalg.norm(p1 - p2)
        if d <= focal_distance:
            raise ValueError(
                f"Major axis length ({d}) must be greater than the "
                f"focal distance ({focal_distance:.4f})."
            )
        
        self.p1 = p1
        self.p2 = p2
        self.d = d # Major axis length

    def get_implicit_equation(self) -> Callable[[np.ndarray], float]:
        """
        Computes and returns the implicit equation of the ellipse.
        """
        return lambda C: float(np.linalg.norm(C - self.p1) + np.linalg.norm(C - self.p2) - self.d)

class Hitch:
    """
    Manages the interaction between two Ellipse objects, finding and updating
    their intersection points.
    """
    def __init__(self, e1: Ellipse, e2: Ellipse):
        self.e1 = e1
        self.e2 = e2
        self.update_times = [] # For time profiling
        
        start_time = time.perf_counter()
        self.intersections = self._calculate_intersections()
        end_time = time.perf_counter()
        print(f"Initial calculation took: {(end_time - start_time) * 1000:.4f} ms")

    def _calculate_intersections(self, initial_guesses_list: Optional[List[np.ndarray]] = None, max_iterations=200) -> np.ndarray:
        eq1 = self.e1.get_implicit_equation()
        eq2 = self.e2.get_implicit_equation()

        def cost_function(p):
            return eq1(p)**2 + eq2(p)**2

        def gradient(p):
            epsilon = 1e-8
            d_p_p1 = np.linalg.norm(p - self.e1.p1)
            d_p_p2 = np.linalg.norm(p - self.e1.p2)
            grad_e1 = (p - self.e1.p1) / (d_p_p1 + epsilon) + (p - self.e1.p2) / (d_p_p2 + epsilon)
            d_p_p3 = np.linalg.norm(p - self.e2.p1)
            d_p_p4 = np.linalg.norm(p - self.e2.p2)
            grad_e2 = (p - self.e2.p1) / (d_p_p3 + epsilon) + (p - self.e2.p2) / (d_p_p4 + epsilon)
            return 2 * eq1(p) * grad_e1 + 2 * eq2(p) * grad_e2

        if initial_guesses_list and len(initial_guesses_list) > 0:
            guess = np.mean(np.array(initial_guesses_list), axis=0)
        else:
            center1 = (self.e1.p1 + self.e1.p2) / 2
            center2 = (self.e2.p1 + self.e2.p2) / 2
            guess = (center1 + center2) / 2

        learning_rate = 0.1
        convergence_threshold = 1e-7
        p = np.copy(guess)
        for _ in range(max_iterations):
            grad = gradient(p)
            p = p - learning_rate * grad
            if cost_function(p) < convergence_threshold:
                break
        return np.array([p])

    def update(self, p1_new: Optional[np.ndarray] = None, p2_new: Optional[np.ndarray] = None, p3_new: Optional[np.ndarray] = None, p4_new: Optional[np.ndarray] = None):
        if p1_new is not None: self.e1.p1 = p1_new
        if p2_new is not None: self.e1.p2 = p2_new
        if p3_new is not None: self.e2.p1 = p3_new
        if p4_new is not None: self.e2.p2 = p4_new

        focal_dist1 = np.linalg.norm(self.e1.p1 - self.e1.p2)
        if self.e1.d <= focal_dist1:
            raise ValueError(f"Update failed for Ellipse 1: Major axis length ({self.e1.d}) is not greater than new focal distance ({focal_dist1:.4f}).")
        focal_dist2 = np.linalg.norm(self.e2.p1 - self.e2.p2)
        if self.e2.d <= focal_dist2:
            raise ValueError(f"Update failed for Ellipse 2: Major axis length ({self.e2.d}) is not greater than new focal distance ({focal_dist2:.4f}).")

        start_time = time.perf_counter()
        self.intersections = self._calculate_intersections(initial_guesses_list=[pt for pt in self.intersections])
        end_time = time.perf_counter()
        self.update_times.append((end_time - start_time) * 1000)

class DynamicHitch(Hitch):
    """
    Extends Hitch to solve the fully coupled dynamics of the system,
    calculating accelerations and tensions based on masses and external forces.
    """
    def __init__(self, e1: Ellipse, e2: Ellipse, foci_vels: List[np.ndarray], mass: float, foci_masses: List[float]):
        self.foci_vels = foci_vels
        self.foci_accels = [np.zeros(2) for _ in range(4)]
        self.mass = mass
        self.foci_masses = foci_masses
        self.hitch_vel = np.zeros(2)
        self.hitch_accel = np.zeros(2)
        self.tensions = np.zeros(4)
        super().__init__(e1, e2)
        self.calculate_dynamics(external_forces=[np.zeros(2) for _ in range(4)])

    def calculate_dynamics(self, external_forces: List[np.ndarray]):
        """
        Calculates the coupled accelerations and tensions for the entire system
        by solving a linear system derived from kinematic and dynamic constraints.
        """
        p = self.intersections[0]
        p1, p2, p3, p4 = self.e1.p1, self.e1.p2, self.e2.p1, self.e2.p2
        v1, v2, v3, v4 = self.foci_vels
        m0, m1, m2, m3, m4 = self.mass, *self.foci_masses
        F1, F2, F3, F4 = external_forces
        epsilon = 1e-9

        # --- Unit vectors and distances ---
        r1, r2, r3, r4 = p - p1, p - p2, p - p3, p - p4
        d1, d2, d3, d4 = [np.linalg.norm(r) for r in [r1, r2, r3, r4]]
        u1, u2, u3, u4 = r1/(d1+epsilon), r2/(d2+epsilon), r3/(d3+epsilon), r4/(d4+epsilon)

        # --- Jacobian-like vectors ---
        J1 = u1 + u2
        J2 = u3 + u4
        
        # --- Velocity Calculation (unchanged) ---
        J_matrix = np.array([J1, J2])
        try:
            J_inv = np.linalg.inv(J_matrix)
            B = np.array([np.dot(u1, v1) + np.dot(u2, v2), np.dot(u3, v3) + np.dot(u4, v4)])
            self.hitch_vel = J_inv @ B
        except np.linalg.LinAlgError:
            print("Warning: Jacobian is singular. Dynamics calculation skipped.")
            self.hitch_vel = np.zeros(2); self.hitch_accel = np.zeros(2)
            self.foci_accels = [np.zeros(2) for _ in range(4)]; self.tensions = np.zeros(4)
            return

        # --- Construct the 2x2 Linear System for Tensions: M * T = RHS ---
        vp = self.hitch_vel
        vr1, vr2, vr3, vr4 = vp - v1, vp - v2, vp - v3, vp - v4

        # Right-Hand Side (RHS) of the linear system
        V_C1 = (np.dot(vr1, vr1) - np.dot(u1, vr1)**2) / (d1 + epsilon) + (np.dot(vr2, vr2) - np.dot(u2, vr2)**2) / (d2 + epsilon)
        V_C2 = (np.dot(vr3, vr3) - np.dot(u3, vr3)**2) / (d3 + epsilon) + (np.dot(vr4, vr4) - np.dot(u4, vr4)**2) / (d4 + epsilon)
        
        RHS1 = V_C1 - (np.dot(J1, F1)/m1 + np.dot(J1, F2)/m2 - np.dot(u1, F1)/m1 - np.dot(u2, F2)/m2)
        RHS2 = V_C2 - (np.dot(J2, F3)/m3 + np.dot(J2, F4)/m4 - np.dot(u3, F3)/m3 - np.dot(u4, F4)/m4)
        RHS = np.array([RHS1, RHS2])

        # Matrix M of the linear system
        M11 = np.dot(J1, J1)/m0 + 1/m1 + 1/m2 - (np.dot(J1, u1)/m1 + np.dot(J1, u2)/m2)
        M12 = np.dot(J1, J2)/m0
        M21 = np.dot(J2, J1)/m0
        M22 = np.dot(J2, J2)/m0 + 1/m3 + 1/m4 - (np.dot(J2, u3)/m3 + np.dot(J2, u4)/m4)
        M = np.array([[M11, M12], [M21, M22]])

        try:
            # Solve for the two cable tensions
            tensions_AB = np.linalg.solve(M, RHS)
            T_A, T_B = np.maximum(0, tensions_AB) # Cables can't push (tension is non-negative)
            self.tensions = np.array([T_A, T_A, T_B, T_B])

            # --- Back-substitute to find all accelerations ---
            self.foci_accels = [
                (F1 - T_A * u1) / m1,
                (F2 - T_A * u2) / m2,
                (F3 - T_B * u3) / m3,
                (F4 - T_B * u4) / m4
            ]
            F_net_hitch = T_A * J1 + T_B * J2
            self.hitch_accel = F_net_hitch / m0

        except np.linalg.LinAlgError:
            print("Warning: Coupled dynamics solver failed. System may be singular.")
            self.tensions = np.zeros(4); self.foci_accels = [np.zeros(2) for _ in range(4)]; self.hitch_accel = np.zeros(2)

    def update_dynamics(self, p_news: List[np.ndarray], v_news: List[np.ndarray], external_forces: List[np.ndarray]):
        """ Updates the entire state of the system. """
        self.foci_vels = v_news
        super().update(*p_news)
        self.calculate_dynamics(external_forces)

## Example Usage
if __name__ == '__main__':
    # --- Simulation Parameters ---
    num_steps = 300
    dt = 0.1 # Time step
    hitch_mass = 0.01
    foci_masses = [1.0, 1.0, 1.0, 1.0]

    # --- Initial Dynamic State ---
    initial_pos = [np.array([-3, 2]), np.array([-3, -2]), np.array([-1, -3]), np.array([-1, 3])]
    initial_vels = [np.array([0.5, 0.8]), np.array([-0.5, 0.5]), np.array([0.3, -0.6]), np.array([-0.2, -0.4])]
    
    ellipse1 = Ellipse(p1=initial_pos[0], p2=initial_pos[1], d=10.0)
    ellipse2 = Ellipse(p1=initial_pos[2], p2=initial_pos[3], d=10.0)

    print("--- Initializing Simulation ---")
    dyn_hitch = DynamicHitch(e1=ellipse1, e2=ellipse2, foci_vels=initial_vels, mass=hitch_mass, foci_masses=foci_masses)
    
    # --- Set up the plot for animation ---
    fig, ax = plt.subplots(figsize=(10, 10))
    x_range = np.linspace(-8, 8, 200)
    y_range = np.linspace(-8, 8, 200)
    X, Y = np.meshgrid(x_range, y_range)

    def animate(frame):
        ax.clear()

        # Define external forces for the current frame (zero for now)
        external_forces = [np.zeros(2) for _ in range(4)]

        if frame > 0:
            # --- Update State using accelerations from the PREVIOUS step ---
            new_vels = [dyn_hitch.foci_vels[i] + dyn_hitch.foci_accels[i] * dt for i in range(4)]
            new_pos = [
                dyn_hitch.e1.p1 + new_vels[0] * dt,
                dyn_hitch.e1.p2 + new_vels[1] * dt,
                dyn_hitch.e2.p1 + new_vels[2] * dt,
                dyn_hitch.e2.p2 + new_vels[3] * dt,
            ]
            
            try:
                # This call updates positions and then re-calculates accelerations for the NEXT step
                dyn_hitch.update_dynamics(new_pos, new_vels, external_forces)
            except ValueError as e:
                print(f"Frame {frame}: Simulation stopped. {e}")
                ani.event_source.stop()
                return

        # --- Redraw everything for the current frame ---
        p1, p2, p3, p4 = dyn_hitch.e1.p1, dyn_hitch.e1.p2, dyn_hitch.e2.p1, dyn_hitch.e2.p2
        
        Z1 = np.sqrt((X - p1[0])**2 + (Y - p1[1])**2) + np.sqrt((X - p2[0])**2 + (Y - p2[1])**2)
        ax.contour(X, Y, Z1, levels=[dyn_hitch.e1.d], colors='#1f77b4', linewidths=2)
        Z2 = np.sqrt((X - p3[0])**2 + (Y - p3[1])**2) + np.sqrt((X - p4[0])**2 + (Y - p4[1])**2)
        ax.contour(X, Y, Z2, levels=[dyn_hitch.e2.d], colors='#ff7f0e', linewidths=2)

        ax.scatter(p1[0], p1[1], c='#1f77b4', s=100, zorder=5, label='Ellipse 1 Foci')
        ax.scatter(p2[0], p2[1], c='#1f77b4', s=100, zorder=5)
        ax.scatter(p3[0], p3[1], c='#ff7f0e', s=100, zorder=5, label='Ellipse 2 Foci')
        ax.scatter(p4[0], p4[1], c='#ff7f0e', s=100, zorder=5)

        point = dyn_hitch.intersections[0]
        ax.scatter(point[0], point[1], c='red', s=60, zorder=6, label='Hitch Point')
        
        cable_points = [p1, p2, p3, p4]
        cable_colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
        for i in range(4):
            start_pt = cable_points[i]
            ax.plot([start_pt[0], point[0]], [start_pt[1], point[1]], color=cable_colors[i], linestyle='--', alpha=0.8)
            mid_point = (start_pt + point) / 2
            ax.text(mid_point[0], mid_point[1], f'T={dyn_hitch.tensions[i]:.2f}', fontsize=9, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        vel = dyn_hitch.hitch_vel
        accel = dyn_hitch.hitch_accel
        ax.arrow(point[0], point[1], vel[0], vel[1], head_width=0.3, head_length=0.3, fc='cyan', ec='blue', label='Velocity')
        ax.arrow(point[0], point[1], accel[0], accel[1], head_width=0.3, head_length=0.3, fc='magenta', ec='purple', label='Acceleration')

        ax.set_title(f'Dynamic Hitch Simulation (Frame {frame+1})')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)

    print(f"\n--- Starting animation for {num_steps} steps... ---")
    ani = FuncAnimation(fig, animate, frames=num_steps, interval=50, repeat=False)
    plt.show()

    print("\n--- Animation Finished ---")
    if dyn_hitch.update_times:
        update_times_np = np.array(dyn_hitch.update_times)
        mean_time = np.mean(update_times_np)
        std_time = np.std(update_times_np)
        print("\n--- Intersection Update Time Profiling ---")
        print(f"Mean update time: {mean_time:.4f} ms")
        print(f"Standard deviation: {std_time:.4f} ms")
