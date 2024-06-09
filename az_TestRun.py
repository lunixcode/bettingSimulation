import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
#from tensorflow.keras.utils import Progbar
from rl.memory import SequentialMemory
#from tensorflow.keras.models import Sequential
from rl.policy import EpsGreedyQPolicy
import gym
from gym import spaces
import numpy as np
from rl.agents import DQNAgent
#from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy
import datetime
from az_Env import BettingEnv


training_leagues = []
env = BettingEnv()
env.reset()

file_path = 'bettingSim/Data/2021.csv'  # Update this to your file path
df = pd.read_csv(file_path)
training_leagues.append(df)
file_path = 'bettingSim/Data/2022.csv'
df = pd.read_csv(file_path)
training_leagues.append(df)


state_size = env.observation_space.shape[0]
number_of_actions = env.action_space.n
num_episodes = 1

print(state_size, "     ", number_of_actions)

model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_actions, activation='linear'))

memory = SequentialMemory(limit=50000, window_length=1)

policy = EpsGreedyQPolicy()

dqn = DQNAgent(model=model, 
               nb_actions=number_of_actions, 
               policy= policy,
               memory=memory,  # Greedy policy for evaluation
               nb_steps_warmup=0,  # No warmup necessary for evaluation
               target_model_update=1,  # Update target model frequently
               enable_double_dqn=False,  # Simplify for evaluation
               enable_dueling_network=False  # Simplify for evaluation
               )

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

try:
    #dqn.load_weights('dqn_betting_model_weights.h5f')
    #dqn.load_weights('test_weights.h5f')
    dqn.load_weights('new_weights.h5f')
    print("Weights loaded successfully!")
except Exception as e:
    print("An error occurred:", str(e))

episodes = 38
for test_league in training_leagues:  # test_leagues contains your two new seasons
    env.load_data(test_league)
    total_reward = 0
    done = False
    observation = env.reset()
    #or i in range(episodes):
    while not done:
        reshaped_observation = observation.reshape(1, 1, 4)
        action = np.argmax(dqn.model.predict(reshaped_observation))
        
        observation, reward, done, _ = env.step(action)
        #print(reward)
        total_reward += reward
    input()
    print(f"Total reward for : {total_reward}")
    