import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Callable, List

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
        return lambda C: np.linalg.norm(C - self.p1) + np.linalg.norm(C - self.p2) - self.d

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
        self.intersections = self._calculate_intersections()

    def _calculate_intersections(self, initial_guesses_list: List[np.ndarray] = None, num_default_guesses=12) -> np.ndarray:
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
            radius = max(np.linalg.norm(self.e1.p1 - center), np.linalg.norm(self.e2.p1 - center)) * 1.5
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

    def update(self, p1_new: np.ndarray = None, p2_new: np.ndarray = None, p3_new: np.ndarray = None, p4_new: np.ndarray = None):
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
        self.intersections = self._calculate_intersections(initial_guesses_list=self.intersections)


## Example Usage
if __name__ == '__main__':
    # --- Define Initial Ellipses ---
    ellipse1 = Ellipse(p1=np.array([-2, 0]), p2=np.array([2, 0]), d=5.0)
    ellipse2 = Ellipse(p1=np.array([0, -3]), p2=np.array([0, 3]), d=7.0)

    # --- Create the Hitch and find initial intersections ---
    hitch = Hitch(e1=ellipse1, e2=ellipse2)
    initial_intersections = hitch.intersections
    print("--- Initial State ---")
    print("Intersection Points Found:")
    print(initial_intersections)

    # --- Store original ellipses for plotting ---
    original_e1_p1, original_e1_p2 = hitch.e1.p1.copy(), hitch.e1.p2.copy()
    original_e2_p1, original_e2_p2 = hitch.e2.p1.copy(), hitch.e2.p2.copy()

    # --- Update the position of one focus ---
    print("\n--- Updating Focus p1 ---")
    hitch.update(p1_new=np.array([-3.5, 0.5]))
    updated_intersections = hitch.intersections
    print("New Intersection Points Found:")
    print(updated_intersections)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(9, 9))
    x_range = np.linspace(-6, 6, 400)
    y_range = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x_range, y_range)

    # Plot Original Ellipse 1 (dashed blue)
    Z1_orig = np.sqrt((X - original_e1_p1[0])**2 + (Y - original_e1_p1[1])**2) + \
              np.sqrt((X - original_e1_p2[0])**2 + (Y - original_e1_p2[1])**2)
    ax.contour(X, Y, Z1_orig, levels=[hitch.e1.d], colors=['#1f77b4'], linestyles='dashed', alpha=0.7)

    # Plot Original Ellipse 2 (dashed orange)
    Z2_orig = np.sqrt((X - original_e2_p1[0])**2 + (Y - original_e2_p1[1])**2) + \
              np.sqrt((X - original_e2_p2[0])**2 + (Y - original_e2_p2[1])**2)
    ax.contour(X, Y, Z2_orig, levels=[hitch.e2.d], colors=['#ff7f0e'], linestyles='dashed', alpha=0.7)
    
    # Plot Updated Ellipse 1 (solid blue)
    Z1_new = np.sqrt((X - hitch.e1.p1[0])**2 + (Y - hitch.e1.p1[1])**2) + \
             np.sqrt((X - hitch.e1.p2[0])**2 + (Y - hitch.e1.p2[1])**2)
    ax.contour(X, Y, Z1_new, levels=[hitch.e1.d], colors=['#1f77b4'], linewidths=2)

    # Plot Updated Ellipse 2 (solid orange - unchanged in this example)
    ax.contour(X, Y, Z2_orig, levels=[hitch.e2.d], colors=['#ff7f0e'], linewidths=2)

    # Plot foci and intersections
    if initial_intersections.size > 0:
        ax.scatter(initial_intersections[:, 0], initial_intersections[:, 1], c='lightcoral', s=80, zorder=4, label='Initial Intersections')
    if updated_intersections.size > 0:
        ax.scatter(updated_intersections[:, 0], updated_intersections[:, 1], c='red', s=80, zorder=5, label='Updated Intersections')

    ax.scatter([original_e1_p1[0], hitch.e1.p1[0]], [original_e1_p1[1], hitch.e1.p1[1]], c='#1f77b4', marker='x', s=100, label='Foci E1 (Orig/New)')
    ax.scatter(hitch.e1.p2[0], hitch.e1.p2[1], c='#1f77b4', marker='x', s=100)
    ax.scatter([hitch.e2.p1[0], hitch.e2.p2[0]], [hitch.e2.p1[1], hitch.e2.p2[1]], c='#ff7f0e', marker='x', s=100, label='Foci E2')

    ax.set_title('Updating Ellipse Intersections')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    plt.show()
