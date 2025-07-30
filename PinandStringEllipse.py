import numpy as np
from scipy.optimize import fsolve
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

        The equation is in the form f(C) = 0, where C is a point (x, y).
        This function is designed to be called when the equation is needed,
        rather than during initialization.

        Returns:
            Callable[[np.ndarray], float]: A function that takes a 2D point 
                                           and returns the value of the implicit equation.
                                           The value is 0 for points on the ellipse.
        """
        # The lambda function captures the ellipse's properties (p1, p2, d)
        # and returns the result of the ellipse formula for any point C.
        return lambda C: float(np.linalg.norm(C - self.p1) + np.linalg.norm(C - self.p2) - self.d)

class Hitch:
    """
    Manages the interaction between two Ellipse objects, finding and updating
    their intersection points.
    """
    def __init__(self, e1: Ellipse, e2: Ellipse):
        """
        Initializes the Hitch with two ellipses and finds their intersection.

        Args:
            e1 (Ellipse): The first ellipse object.
            e2 (Ellipse): The second ellipse object.
        """
        self.e1 = e1
        self.e2 = e2
        
        # Time the initial calculation
        start_time = time.perf_counter()
        self.intersections = self._calculate_intersections()
        end_time = time.perf_counter()
        print(f"Initial calculation took: {(end_time - start_time) * 1000:.4f} ms")


    def _calculate_intersections(self, initial_guesses_list: 'Optional[List[np.ndarray]]' = None, num_default_guesses=12) -> np.ndarray:
        """
        Finds intersection points by solving the system of ellipse equations.
        It can be seeded with initial guesses for faster convergence.

        Args:
            initial_guesses_list (List[np.ndarray], optional): A list of points to use as
                                                              initial guesses for the solver.
            num_default_guesses (int): Number of guesses to generate if none are provided.

        Returns:
            np.ndarray: An array of unique intersection points.
        """
        eq1 = self.e1.get_implicit_equation()
        eq2 = self.e2.get_implicit_equation()
        def system(C: np.ndarray) -> tuple[float, float]:
            return (eq1(C), eq2(C))

        guesses = []
        if initial_guesses_list is not None and len(initial_guesses_list) > 0:
            guesses.extend(initial_guesses_list)
        else:
            center = (self.e1.p1 + self.e1.p2 + self.e2.p1 + self.e2.p2) / 4.0
            radius = max(float(np.linalg.norm(self.e1.p1 - center)), float(np.linalg.norm(self.e2.p1 - center))) * 1.5
            guesses.append(center)
            for i in range(num_default_guesses):
                angle = 2 * np.pi * i / num_default_guesses
                guess = center + radius * np.array([np.cos(angle), np.sin(angle)])
                guesses.append(guess)
        
        solutions = []
        for guess in guesses:
            solution, _, ier, _ = fsolve(system, guess, full_output=True)
            if ier == 1 and not any(np.allclose(solution, s, atol=1e-6) for s in solutions):
                solutions.append(solution)
        return np.array(solutions)

    def update(self, p1_new: Optional[np.ndarray] = None, p2_new: Optional[np.ndarray] = None, p3_new: Optional[np.ndarray] = None, p4_new: Optional[np.ndarray] = None):
        """
        Updates the position of any of the four foci, validates the change,
        and recalculates the intersections iteratively.

        Args:
            p1_new (np.ndarray, optional): New position for focus 1 of ellipse 1.
            p2_new (np.ndarray, optional): New position for focus 2 of ellipse 1.
            p3_new (np.ndarray, optional): New position for focus 1 of ellipse 2.
            p4_new (np.ndarray, optional): New position for focus 2 of ellipse 2.
        """
        # Update foci positions
        if p1_new is not None: self.e1.p1 = p1_new
        if p2_new is not None: self.e1.p2 = p2_new
        if p3_new is not None: self.e2.p1 = p3_new
        if p4_new is not None: self.e2.p2 = p4_new

        # Re-validate that the major axis lengths are still valid
        focal_dist1 = np.linalg.norm(self.e1.p1 - self.e1.p2)
        if self.e1.d <= focal_dist1:
            raise ValueError(f"Update failed for Ellipse 1: Major axis length ({self.e1.d}) is not greater than new focal distance ({focal_dist1:.4f}).")
        
        focal_dist2 = np.linalg.norm(self.e2.p1 - self.e2.p2)
        if self.e2.d <= focal_dist2:
            raise ValueError(f"Update failed for Ellipse 2: Major axis length ({self.e2.d}) is not greater than new focal distance ({focal_dist2:.4f}).")

        # Iteratively update intersections using the old intersections as initial guesses
        self.intersections = self._calculate_intersections(initial_guesses_list=[pt for pt in self.intersections])


## Example Usage
if __name__ == '__main__':
    # --- Simulation Parameters ---
    num_steps = 150
    max_velocity = 0.1 # Max movement per step

    # --- Define Initial Ellipses ---
    ellipse1 = Ellipse(p1=np.array([-2, 0]), p2=np.array([2, 0]), d=7.0)
    ellipse2 = Ellipse(p1=np.array([0, -2]), p2=np.array([0, 2]), d=7.0)

    # --- Create the Hitch and find initial intersections ---
    print("--- Initializing Simulation ---")
    hitch = Hitch(e1=ellipse1, e2=ellipse2)
    
    # --- Store history for plotting ---
    foci_history = [[hitch.e1.p1.copy(), hitch.e1.p2.copy(), hitch.e2.p1.copy(), hitch.e2.p2.copy()]]
    intersection_history = [hitch.intersections]
    
    # --- Set up the plot for animation ---
    fig, ax = plt.subplots(figsize=(10, 10))
    x_range = np.linspace(-8, 8, 200)
    y_range = np.linspace(-8, 8, 200)
    X, Y = np.meshgrid(x_range, y_range)

    # This function will be called for each frame of the animation
    def animate(frame):
        ax.clear() # Clear the previous frame

        # --- Update simulation state for the current frame ---
        if frame > 0: # Don't move on the first frame
            # Generate random velocities for each focus
            v1 = (np.random.rand(2) - 0.5) * 2 * max_velocity
            v2 = (np.random.rand(2) - 0.5) * 2 * max_velocity
            v3 = (np.random.rand(2) - 0.5) * 2 * max_velocity
            v4 = (np.random.rand(2) - 0.5) * 2 * max_velocity
            
            # Calculate new positions
            p1_new = hitch.e1.p1 + v1
            p2_new = hitch.e1.p2 + v2
            p3_new = hitch.e2.p1 + v3
            p4_new = hitch.e2.p2 + v4

            try:
                # Update the hitch
                hitch.update(p1_new, p2_new, p3_new, p4_new)

                # Store results for plotting the trail
                foci_history.append([p1_new.copy(), p2_new.copy(), p3_new.copy(), p4_new.copy()])
                intersection_history.append(hitch.intersections)
            except ValueError as e:
                print(f"Frame {frame}: Simulation stopped. {e}")
                # Stop the animation by doing nothing further
                return

        # --- Redraw everything for the current frame ---
        current_foci = foci_history[-1]
        
        # Plot current ellipses
        Z1 = np.sqrt((X - current_foci[0][0])**2 + (Y - current_foci[0][1])**2) + np.sqrt((X - current_foci[1][0])**2 + (Y - current_foci[1][1])**2)
        ax.contour(X, Y, Z1, levels=[hitch.e1.d], colors='#1f77b4', linewidths=2)
        Z2 = np.sqrt((X - current_foci[2][0])**2 + (Y - current_foci[2][1])**2) + np.sqrt((X - current_foci[3][0])**2 + (Y - current_foci[3][1])**2)
        ax.contour(X, Y, Z2, levels=[hitch.e2.d], colors='#ff7f0e', linewidths=2)

        # Plot trajectories of foci
        foci_history_np = np.array(foci_history)
        ax.plot(foci_history_np[:, 0, 0], foci_history_np[:, 0, 1], 'o-', color='#1f77b4', markersize=3, alpha=0.7, label='Foci 1 Path')
        ax.plot(foci_history_np[:, 1, 0], foci_history_np[:, 1, 1], 'o-', color='#1f77b4', markersize=3, alpha=0.7)
        ax.plot(foci_history_np[:, 2, 0], foci_history_np[:, 2, 1], 'o-', color='#ff7f0e', markersize=3, alpha=0.7, label='Foci 2 Path')
        ax.plot(foci_history_np[:, 3, 0], foci_history_np[:, 3, 1], 'o-', color='#ff7f0e', markersize=3, alpha=0.7)

        # Plot trajectory of intersections
        all_intersections = [p for points in intersection_history if points.size > 0 for p in points]
        if all_intersections:
            all_intersections_np = np.array(all_intersections)
            ax.scatter(all_intersections_np[:, 0], all_intersections_np[:, 1], c='red', s=15, alpha=0.6, zorder=5, label='Intersection Path')

        # Set plot properties for each frame
        ax.set_title(f'Dynamic Simulation of Ellipse Intersections (Frame {frame+1})')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)

    # --- Create and run the animation ---
    print(f"\n--- Starting animation for {num_steps} steps... ---")
    ani = FuncAnimation(fig, animate, frames=num_steps, interval=50, repeat=False)
    plt.show()

    print("\n--- Animation Finished ---")
