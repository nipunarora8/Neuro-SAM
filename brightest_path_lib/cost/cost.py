from abc import ABC, abstractmethod

class Cost(ABC):
    """Base class for cost function
    """

    @abstractmethod
    def cost_of_moving_to(self, intensity_at_new_point: float) -> float:
        """calculates the cost of moving to a point

        Parameters
        ----------
        intensity_at_new_point : float
            The intensity of the new point under consideration
        
        Returns
        -------
        float
            the cost of moving to the new point
        """
        pass

    @abstractmethod
    def minimum_step_cost(self) -> float:
        """calculates the minimum step cost
        (depends on the cost function implementation)
        
        Returns
        -------
        float
            the minimum step cost
        """
        pass
