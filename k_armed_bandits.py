import numpy as np

class KArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_star = np.random.normal(size=n_arms)
        self.optimal = np.argmax(self.q_star)
    
    def __call__(self, action: int):
        return np.random.normal(loc=self.q_star[action])


class BanditPolicy:
    def __init__(self, n_arms, epsilon=0.01):
        self.n_arms = n_arms
        self.q = np.zeros(n_arms)
        self.n = np.zeros(n_arms)
        self.epsilon = epsilon

    def __call__(self, bandit, ep_len, n_episodes):
        ep_reward = []
        for _ in range(ep_len):
            action = np.argmax(self.q) if np.random.rand() > self.epsilon else np.random.randint(self.n_arms)
            reward = bandit(action)
            self.n[action] += 1
            self.q[action] += 1/self.n[action] * (reward - self.q[action])
            ep_reward += reward
            

def run_expr(n_arms, ep_len, n_episodes):
    bandit = KArmedBandit(n_arms)
    policy = BanditPolicy(n_arms)
    for _ in range(n_episodes):
        policy(bandit, ep_len)
