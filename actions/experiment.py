import numpy as np
import matplotlib.pyplot as plt
from environments.k_armed_bandit import KArmedBandit
from methods.bandit_policies import BanditPolicy, GradientBanditPolicy

## TODO: Use argparse to enable running this script from the command line

def run_expr(n_arms: int, ep_len: int, n_episodes: int, policy: BanditPolicy) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the bandit policy for a given number of episodes and returns the average reward and the percentage of the time the optimal action
    was selected for each timestep.

    Args:
        n_arms: number of k different options (arms) available to the agent
        ep_len: number of actions to take in an episode
        n_episodes: number of episodes to run the policy
        policy: instance of the BanditPolicy class
    
    Returns:
        an array of average rewards received at each timestep, and an array indicating the percentage of the time the optimal action was 
        selected at each step
    """
    for idx in range(n_episodes):
        if idx % 100 == 0:
            print(f'Running episode {idx}')
        bandit = KArmedBandit(n_arms)
        ep_reward, op_action = policy(bandit, ep_len)
        if idx == 0:
            avg_reward = ep_reward
            prcntg_op_action = op_action
        else:
            avg_reward = np.sum([avg_reward, ep_reward], axis=0)
            prcntg_op_action = np.sum([prcntg_op_action, op_action], axis=0)
        policy._reset()

    avg_reward /= n_episodes
    prcntg_op_action /= n_episodes
    return avg_reward, prcntg_op_action


def main(n_arms: int, ep_len: int, n_episodes: int, policy_list: list[BanditPolicy]):
    """
    Runs a bandit policy for a given number of episodes for each value of epsilon, and plots the average reward received and percentage of
    optimal actions selected over each timestep for each agent.

    Args:
        n_arms: number of k different options (arms) available to the agent
        ep_len: number of actions to take in an episode
        n_episodes: number of episodes to run the policy
        policy_list: list of instances of the BanditPolicy class
    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    for policy in policy_list:
        avg_reward, prcntg_op_action = run_expr(n_arms, ep_len, n_episodes, policy)
        ax[0].plot(avg_reward, label=f'e: {policy.epsilon}' if type(policy) == BanditPolicy else f'a: {policy.alpha}')
        ax[1].plot(prcntg_op_action, label=f'e: {policy.epsilon}' if type(policy) == BanditPolicy else f'a: {policy.alpha}')
    ax[0].set_title('Average Reward')
    ax[0].set_ylabel('Average Reward')
    ax[0].legend()
    ax[1].set_title('Optimal Action')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('% Optimal Action')
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    policy1 = GradientBanditPolicy(10, alpha=0.1)
    policy2 = GradientBanditPolicy(10, alpha=0.4)
    main(10, 1000, 500, [policy1, policy2])