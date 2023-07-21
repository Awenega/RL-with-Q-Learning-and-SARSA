import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import logging
import absl.logging
from os.path import exists
import os
from tensorflow import keras

def setupLogger(filename):
    logger = logging.getLogger('urbanGUI')
    logging.basicConfig(filename=f'{filename}.log', filemode='a',format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    return logger
    
def create_model(enviroment):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(1, )))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(enviroment.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_model(experience,model):
    batch = random.sample(experience, BATCH_SIZE)
    dataset = np.array(batch)
    X = np.asarray(dataset[:,0])

    datasetY = []
    for current_state,current_action,current_reward,next_state, done in batch:
        
        predicted_actions = model.predict_on_batch(np.array([[current_state]]))[0]
        Q = predicted_actions[current_action] 
        
        if done:
            predicted_actions[current_action] = current_reward
        else:
            predicted_next_actions = model.predict_on_batch(np.array([[next_state]]))[0]
            Qmax = predicted_next_actions[np.argmax(predicted_next_actions)] 
            Q_new = Q + ALPHA*(current_reward + GAMMA * Qmax - Q)
            predicted_actions[current_action] = Q_new
        
        datasetY.append(predicted_actions)
    Y = np.asarray(datasetY)

    model.fit(X, Y, validation_split=0.2)
            
def get_action(enviroment, epsilon,model,state):
    #Exploration
    random_value = random.randint(1,100) 
    selected_action = enviroment.action_space.sample()
    #Exploitation
    if random_value >= epsilon:
        predicted_actions = model.predict_on_batch(np.array([[state]]))[0]
        selected_action = np.argmax(predicted_actions)
    return selected_action

def Rollout(enviroment, model, number_of_episodes, render):
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
        selected_action = get_action(enviroment, 0, model, state)
        state_next, reward, done, _ , _ = enviroment.step(selected_action)

        total_reward += reward 

        state = state_next

        if render:
            enviroment.render()
    
    return total_reward / number_of_episodes
    
def Qlearn(enviroment, model,number_of_episodes):
    #Experience Buffer
    experience = deque([],EXP_MAX_SIZE)

    #Parameters
    epsilon = EPS_MAX
    decay_epsilon = 0.001
    WRONG_ACTION = -10 

    cumulative_reward = 0
    number_of_wrong_actions = 0
    episode = 1

    state, _ = enviroment.reset()
    
    while(episode <= number_of_episodes):
        
        selected_action = get_action(enviroment, epsilon, model, state)
        state_next, reward, terminated, truncated, _ = enviroment.step(selected_action)

        if reward == WRONG_ACTION:
            number_of_wrong_actions += 1
        cumulative_reward += reward 

        # Record experience
        if len(experience)>=EXP_MAX_SIZE:
            experience.popleft() 

        experience.append([state,selected_action,reward,state_next, terminated or truncated])

        state = state_next # update current state

        if terminated or truncated:
            if len(experience) >= BATCH_SIZE:
                train_model(experience,model)

            #debug information
            print("----------------------------------episode ", episode)
            print("return=",cumulative_reward)
            print("epsilon=", epsilon)
            print("wrong actions=",number_of_wrong_actions)
            logger.error(f"Episode: {episode}, Reward:{cumulative_reward}, Epsilon:{epsilon}, Wrong_actions:{number_of_wrong_actions}")

            epsilon = EPS_MIN  + (EPS_MAX - EPS_MIN) * np.exp(-decay_epsilon*episode)
            episode += 1
            cumulative_reward = 0
            number_of_wrong_actions = 0
            state, _ = enviroment.reset()
    model.save("model_NN_taxi")
    enviroment.close()

#Setup logger and parameters
logger = setupLogger("Alpha 0.001, gamma 0.99, decay_rate 0.001")
EXP_MAX_SIZE=5000
BATCH_SIZE=80
GAMMA = 0.95 
ALPHA = 0.001
EPS_MAX = 100
EPS_MIN = 10

#Setup environment
enviroment = gym.make("Taxi-v3")

#Load or create the model
if exists("model_NN_taxi"):
    model = keras.models.load_model("model_NN_taxi")
else:
    model = create_model(enviroment)

#Train the model
Qlearn(enviroment, model, 2500)
