import numpy as np

class KArmedBandit:
    """
    Implements the k-armed bandit problem.

    Attributes:
        n_arms: number of k different options (arms) available to the agent
        q_star: true value (expected reward) of each arm
        optimal: index of the optimal arm
    """
    def __init__(self, n_arms: int):
        """
        Initializes the k-armed bandit problem.

        Args:
            n_arms: number of k different options (arms) available to the agent
        """
        self.n_arms = n_arms
        self.q_star = np.random.normal(size=n_arms)
        self.optimal = np.argmax(self.q_star)
    
    def __call__(self, action: int) -> float:
        """
        Returns the reward for the selected action, sampled from a normal distribution centered at the true value of the action.

        Args:
            action: index of the selected arm
        """
        return np.random.normal(loc=self.q_star[action])