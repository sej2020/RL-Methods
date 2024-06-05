import numpy as np
from environments.k_armed_bandit import KArmedBandit


class BanditPolicy:
    """
    The apparatus that selects the action to take based on the current estimates of the action values of a k-armed bandit.

    Attributes:
        n_arms: number of k different options (arms) available to the agent
        q: current estimate of the action values
        n: number of times each action has been selected
        epsilon: probability of selecting a random action
        optimistic_q_init: initial value for the action values for the optimistic initialization
        alpha: fixed step size for updating the action values
    """
    def __init__(self, n_arms: int, epsilon = 0.01, optimistic_q_init: int = None, alpha: float = None):
        """
        Initializes the bandit policy.

        Args:
            n_arms: number of k different options (arms) available to the agent
            epsilon: probability of selecting a random action
            optimistic_q_init: initial value for the action values for the optimistic initialization. If None, the initial values are 0
            alpha: fixed step size for updating the action values. If None, the step size is 1/n resulting in the sample-average method
        """
        self.n_arms = n_arms
        self.optimistic_q_init = optimistic_q_init
        if optimistic_q_init is not None:
            self.q = np.ones(n_arms) * optimistic_q_init
        else:
            self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)
        self.epsilon = epsilon
        self.alpha = alpha

    def __call__(self, bandit: KArmedBandit, ep_len: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Selects actions from the bandit and updates the action values based on the rewards received on each time step, for the duration
        of a single episode.

        Args:
            bandit: instance of the KArmedBandit class
            ep_len: number of actions to take in an episode
        
        Returns:
            an array of rewards received during the episode (one for each action), and an array indicating whether the optimal action was selected at each step
        """
        ep_reward = np.zeros(ep_len)
        op_action = np.zeros(ep_len)
        for idx in range(ep_len):
            action = np.argmax(self.q) if np.random.rand() > self.epsilon else np.random.randint(self.n_arms)
            reward = bandit(action)
            self.n[action] += 1
            if self.alpha is None:
                step_size = 1/self.n[action]
            else:
                step_size = self.alpha
            self.q[action] += step_size * (reward - self.q[action])
            ep_reward[idx] = reward
            op_action[idx] = action == bandit.optimal
        return ep_reward, op_action
            
    def _reset(self):
        """
        Resets the action values and action counts to their initial values.
        """
        if self.optimistic_q_init is not None:
            self.q = np.ones(self.n_arms) * self.optimistic_q_init
        else:
            self.q = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)


class GradientBanditPolicy:
    """
    The apparatus that selects the action to take based on relative numerical preferences of the actions.

    Attributes:
        n_arms: number of k different options (arms) available to the agent
        h: current preferences for each action
        n: number of times each action has been selected
        alpha: fixed step size for updating the action preferences
    """
    def __init__(self, n_arms: int, alpha: float = None):
        """
        Initializes the gradient bandit policy.

        Args:
            n_arms: number of k different options (arms) available to the agent
            alpha: fixed step size for updating the action preferences. If None, the step size is 1/n resulting in the sample-average method
        """
        self.n_arms = n_arms
        self.h = np.zeros(n_arms)
        self.n = np.zeros(n_arms)
        self.alpha = alpha

    def __call__(self, bandit: KArmedBandit, ep_len: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Selects actions from the bandit and updates the action preferences based on the rewards received over a single episode.

        Args:
            bandit: instance of the KArmedBandit class
            ep_len: number of actions to take in an episode
        
        Returns:
            an array of rewards received during the episode (one for each action), and an array indicating whether the optimal action was 
            selected at each step
        """
        ep_reward = np.zeros(ep_len)
        op_action = np.zeros(ep_len)

        running_reward_avg = 0
        softmax = lambda x: np.exp(x)/np.sum(np.exp(x))

        for idx in range(ep_len):
            action = np.random.choice(self.n_arms, p=softmax(self.h))
            reward = bandit(action)
            self.n[action] += 1
            if self.alpha is None:
                step_size = 1/self.n[action]
            else:
                step_size = self.alpha
            self.h[action] += step_size * (reward - running_reward_avg) * (1 - softmax(self.h)[action])
            mask = np.ones(self.n_arms)
            mask[action] = 0
            self.h = self.h - step_size * (reward - running_reward_avg) * softmax(self.h) * mask

            running_reward_avg += (reward - running_reward_avg) / (idx + 1)

            ep_reward[idx] = reward
            op_action[idx] = action == bandit.optimal
        return ep_reward, op_action
            
    def _reset(self):
        """
        Resets the action preferences and action counts to their initial values.
        """
        self.h = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)
