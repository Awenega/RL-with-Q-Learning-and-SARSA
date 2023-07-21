import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import logging

class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def __call__(self, obs) -> int:
        return random.randint(0, self.n_actions - 1)

class GreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
    def __call__(self, obs) -> int:
        return np.argmax(self.Q[obs])

class EpsGreedyPolicy:
    def __init__(self, Q):
        self.Q = Q
        self.n_actions = len(Q[0])
    def __call__(self, obs, epsilon: float) -> int:
        greedy = random.uniform(0,1) > epsilon
        if greedy:
            return np.argmax(self.Q[obs])
        else:
            return random.randint(0, self.n_actions - 1)

#Function for Q-Learning
def Qlearn(enviroment, alpha, gamma, number_of_episodes, max_steps_per_episode):

    #Define Q table
    Q = defaultdict(lambda: np.zeros(enviroment.action_space.n))

    #Parameters
    EPS_MAX = 1
    EPS_MIN = 0.01
    decay_epsilon = 0.001
    epsilon = EPS_MAX
    WRONG_ACTION = -10
    cumulative_reward = 0
    number_of_wrong_actions = 0
    
    #Epsilon Greedy police used
    policy = EpsGreedyPolicy(Q)
    
    # Compute value for Q-table iterating for number of episodes
    for episode in range(number_of_episodes):

        state, _ = enviroment.reset()
        done = False 

        #Iterate until episode is Done
        for step in range(max_steps_per_episode):

            #WRONG_ACTION of passenger
            if done:
                break
            
            # Select action using Epsilon Greedy police
            action = policy(state, epsilon)
            next_state, reward, done, _ , _ = enviroment.step(action)

            #Illegal WRONG_ACTION
            if reward == WRONG_ACTION:
                number_of_wrong_actions += 1
                
            #Updating of Q table and updating of state
            Q[state][action] = (1- alpha) * Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]))
            state = next_state
            
            cumulative_reward += reward

        print("Episode: " + str(episode) + " reward is: " + str(cumulative_reward))
        print("Episode: " + str(episode) + " number of wrong actions is: " + str(number_of_wrong_actions))
        logger.error(f"Episode: {episode}, Reward:{cumulative_reward}, Epsilon:{epsilon}, Wrong_actions:{number_of_wrong_actions}")

        cumulative_reward = 0
        number_of_wrong_actions = 0

        # Exploration rate esponential decay
        epsilon = EPS_MIN  + (EPS_MAX - EPS_MIN) * np.exp(-decay_epsilon*episode)

    return Q

def Rollouts(enviroment, policy, number_of_episodes, render):
    #perform Rollouts and compute the average discounted return.
    total_reward = 0
    done = False
    state, _ = enviroment.reset()

    episodes = 0
    if render:
        enviroment.render()

    # Iterate steps
    while True:
        if done:
            if render:
                print("Episode " + str(episodes) + " terminated.")
            state, _ = enviroment.reset()
            episodes += 1
            if episodes >= number_of_episodes:
                break
        
        # Select action
        action = policy(state)
        state, reward, done, _ , _ = enviroment.step(action)

        total_reward += reward

        if render:
            enviroment.render()
    
    return total_reward / number_of_episodes

# Setup Logger
logger = logging.getLogger('urbanGUI')
filename = 'QTable'
logging.basicConfig(filename=f'{filename}.log', filemode='a',format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.ERROR)

#Build the QTable
taxi_env = gym.make("Taxi-v3")
qtable = Qlearn(taxi_env, 0.5, 0.99, 5000, 50)
print("--- Q Table builded ---")
print("\n\n")

#Run Rollouts in order to evaluate the performance if we are doing random actions, without any learning
taxi_env = gym.make("Taxi-v3")
avg_return = Rollouts(taxi_env, RandomPolicy(taxi_env.action_space.n), 10, False) 
print("Average return using Random Policy: ",avg_return)
print("\n\n")

#Run Rollouts in order to check the correctness of the learning, using greedy policy for exploitiong the QTable
taxi_env = gym.make("Taxi-v3",render_mode="human")
avg_return = Rollouts(taxi_env, GreedyPolicy(qtable), 3, True) 
print("Average return using Q-Table: ",avg_return)

