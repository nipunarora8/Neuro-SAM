from abc import ABC, abstractmethod
import numpy as np

class Heuristic(ABC):
    """Abstract class for heuristic estimates to goal
    """

    @abstractmethod
    def estimate_cost_to_goal(self, current_point: np.ndarray, goal_point: np.ndarray) -> float:
        """calculates the estimated cost from current point to the goal
        (implementation depends on the heuristic function)

        Parameters
        ----------
        current_point : numpy ndarray
            the coordinates of the current point
                - for 2D points, x and y coordinates
                - for 3D points, x, y and z coordinates
        goal_point : numpy ndarray
            the coordinates of the current point
                - for 2D points, x and y coordinates
                - for 3D points, x, y and z coordinates
        
        Returns
        -------
        float
            the estimated cost to goal
        """
        pass
