import gym
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
#from rl.agents import ActorCriticAgent
#from rl.policy import ActorCriticPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
#       IMPORTANT
#   Has to run on python 3.9.10 or 3.9.x
#   pip install pandas
#   pip install tensorflow
#   pip install gym
#   pip install keras-rl2  its RL2 not R12

import os
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
from my_envV2 import BettingEnv
from rl.callbacks import Callback
from keras.callbacks import TensorBoard
#from tensorboard.plugins.core import TensorBoard
#from stable_baselines3.ppo.ppo import PPO
#from stable_baselines3.common.policies import MlpPolicy

def abbreviate_param(param_name, param_value):
    """Abbreviates a hyperparameter name and combines it with its value.

    Args:
        param_name (str): The name of the hyperparameter.
        param_value (str or float or int): The value of the hyperparameter.

    Returns:
        str: The abbreviated string (e.g., "lr01hls128afreluoArps6ts10ut20rf1sp0").
    """

    abbrev = param_name[:2].lower()  # Take the first two characters of the name
    value_str = str(param_value).replace(".", "")  # Remove any decimal point
    return f"{abbrev}{value_str}"

hyper_parameters = {
    "learning_rate": 0.00025,
    "hidden_layer_size": 64,
    "activation_function": "relu",
    "optimizer": "adam",
    "runsPerSeason": "6",
    "totalSeasons": "10",
    "uniqueTeams": "20",
    "rewardFunc": "1",
    "stateSpace": "1",
    "league": "prem"
}

#https://arxiv.org/pdf/2307.13807
#https://arxiv.org/pdf/2307.13807
# Start timing
start_time = time.time()
# Load the E0.csv file
#uniqueTeams = 24
rowsPerSeason = (int(hyper_parameters["uniqueTeams"]) - 1) * int(hyper_parameters["uniqueTeams"])
base_path = 'Data/'
DQNTest = 0
#file_ppath = 'bettingSim/Data/E0.csv'  # Update this to your file path
#e0_data = pd.read_csv(file_ppath)
training_leagues = []
league_table = {}
dataframes = []
file_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ,19, 20, 21, 22, 23, 24, 25, 26, 27, 28 , 29, 30 , 31, 32, 33, 34, 35, 36, 37, 38 , 39, 40]

# Okay i want to load lower leagues next, then foriegn leagues and finally have it pick leagues at random.

for number in file_numbers:
    # Generate the file path by combining the base path and the number
    if (number <= int(hyper_parameters['totalSeasons'])):
        file_name = f'E0 ({number}).csv'
        file_path = f'{base_path}{file_name}'
        
        # Use pd.read_csv() to read the CSV file
        df = pd.read_csv(file_path)

        # Append the DataFrame to the list
        training_leagues.append(df)

env = BettingEnv()
env.reset()

state_size = env.observation_space.shape[0]

number_of_actions = env.action_space.n
num_episodes = 1

print(state_size, "     ", number_of_actions)
'''
model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_actions, activation='linear'))
'''
model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
#model.add(Dense(243, activation='relu'))
#model.add(Dense(81, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(number_of_actions, activation='softmax'))

memory = SequentialMemory(limit=50000, window_length=1)

policy = EpsGreedyQPolicy()
#policy = EpsGreedyQPolicy(eps=1.0, eps_decay_rate=0.995, min_eps=0.05) 
#policy = EpsGreedyQPolicy(initial_eps=1.0, min_eps=0.01, decay_rate=0.995)

"""ppo_agent = PPO(
    #MlpPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)"""


def create_actor_network(state_size, action_space_size):
  """
  Creates the actor network.

  Args:
      state_size: The size of the state space.
      action_space_size: The number of available actions.

  Returns:
      A Keras Sequential model representing the actor network.
  """
  actor = Sequential()
  actor.add(Dense(units=64, activation='relu', input_shape=(state_size,)))  # Example hidden layer
  actor.add(Dense(units=action_space_size, activation='softmax'))  # Output layer with action probabilities
  return actor

def create_critic_network(state_size):
  """
  Creates the critic network.

  Args:
      state_size: The size of the state space.

  Returns:
      A Keras Sequential model representing the critic network.
  """
  critic = Sequential()
  critic.add(Dense(units=64, activation='relu', input_shape=(state_size,)))  # Example hidden layer
  critic.add(Dense(units=1, activation='linear'))  # Output layer with estimated value
  return critic

def create_actor_critic_model(state_size, action_space_size):
  """
  Creates a combined Actor-Critic model.

  Args:
      state_size: The size of the state space.
      action_space_size: The number of available actions.

  Returns:
      A Keras Model instance representing the combined Actor-Critic model.
  """
  actor = create_actor_network(state_size, action_space_size)
  critic = create_critic_network(state_size)

  # Combine state input for both networks
  state_input = tf.keras.Input(shape=(state_size,))

  # Pass the state through the actor network
  action_output = actor(state_input)

  # Pass the state through the critic network
  value_output = critic(state_input)

  # Combine outputs (optional for separate training)
  model = tf.keras.Model(inputs=state_input, outputs=[action_output, value_output])
  return model

class ActorCriticModel(tf.keras.Model):
  def __init__(self, state_size, action_space_size):
    super().__init__()
    self.actor = create_actor_network(state_size, action_space_size)
    self.critic = create_critic_network(state_size)

  def call(self, state_input):
    action_output = self.actor(state_input)
    value_output = self.critic(state_input)
    return [action_output, value_output]

  def forward(self, state):
    # Call the `call` method to get the action probabilities and value
    outputs = self.call(state)
    action_probs, value = outputs
    return action_probs, value



dqn = DQNAgent(model=model, 
               nb_actions=number_of_actions, 
               memory=memory, 
               nb_steps_warmup=1000,
               target_model_update=hyper_parameters['learning_rate'], 
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Create an empty league table
#league_table = {}
index = 1
runs = 6
fullRun = 0
lSize = 0

# Optional: Create a custom callback to track additional metrics



if DQNTest == 1:
# Iterate through each row in the dataset and update the league table
    for leagues in training_leagues:
        env.load_data(leagues)
        env.set_state(int(hyper_parameters['stateSpace']))
        for i in range(int(hyper_parameters["runsPerSeason"])):

            # ************ IMPORTANT ***********
            #env is environment, steps is how many games    

            dqn.fit(env, nb_steps=rowsPerSeason, visualize=False, verbose=2)
            print("RUN: ", index)
            if(index == index):

                abbreviated_params = "".join(abbreviate_param(name, value) for name, value in hyper_parameters.items())
                weights_folder="weights"
                folder_path = os.path.join(weights_folder, abbreviated_params)
                os.makedirs(folder_path, exist_ok=True)  # Create the folder if needed

                # Save the weights with an incremental filename
                weights_filename = os.path.join(folder_path, f"{index}.h5f")
                dqn.save_weights(weights_filename, overwrite=True)

                '''abbreviated_params = "".join(abbreviate_param(name, value) for name, value in hyper_parameters.items())
                os.makedirs(abbreviated_params, exist_ok=True)  # Create the folder if needed
                weights_filename = os.path.join(abbreviated_params, f"{index}.h5f")
                dqn.save_weights(weights_filename, overwrite=True)'''
            index+=1

        if fullRun == 1:
            saveWeights = input("Save weights?: 'Y' or 'N' \n")
            if saveWeights.upper() == 'Y':
                weightName = input("File name for these weights?: \n")
                weightName = f"{weightName}.{index}.h5f"
                dqn.save_weights(weightName, overwrite=True)
            elif saveWeights.upper() == 'E':
                exit()
            else:
                pass
        

    weights_folder = "weights"
    os.makedirs(weights_folder, exist_ok=True)  # Create the folder if needed

    abbreviated_params = "weights".join(abbreviate_param(name, value) for name, value in hyper_parameters.items())

    # Construct the filenames using the abbreviated string and extensions
    snapshot_filename = os.path.join(weights_folder, f"{abbreviated_params}.txt")
    weights_filename = os.path.join(weights_folder, f"{abbreviated_params}.h5f")

    # Write the dictionary data to the file (same as before)
    with open(snapshot_filename, "w") as f:
        import json
        json.dump(hyper_parameters, f)


    dqn.save_weights('weights/new_weights_Test_24.h5f', overwrite=True)
    dqn.save_weights(weights_filename, overwrite=True)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    end_time = time.time()

    # Calculate and print the total execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    '''

    env = BettingEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Replace with a specific action for testing
        state, reward, done, _ = env.step(action, 'H')
        print(reward)
        if done:
            env.reset()

    '''


import random

def minibatch(data, batch_size):
  """
  Samples a minibatch from the given data.

  Args:
      data: A list of lists containing experience data (states, actions, rewards, next states, dones).
      batch_size: The size of the minibatch to sample.

  Returns:
      A list of lists representing a minibatch of experience data.
  """

  # Randomly shuffle the data
  random.shuffle(data)

  # Sample a minibatch
  minibatch_data = data[:batch_size]

  return minibatch_data


# ... (Define your BettingEnv environment)


# Combine them for the agent
#model = ActorCriticModel(policy=ActorCriticPolicy(actor, critic))

#model = ActorCriticModel(policy=actor, value_function=critic)
#actor = create_actor_critic_model(state_size, 3)
#critic = create_critic_network(state_size)
#agent = create_actor_critic_model(state_size, 3)

agent = ActorCriticModel(state_size, 3)

# Configure memory, optimizer, etc.
memory = SequentialMemory(limit=50000, window_length=1)
optimizer = tf.keras.optimizers.Adam(lr=0.001)  # Adjust learning rate as needed

# ... (Modify the training loop to handle PPO specifics)

# Train the agent
# ... (Use agent.fit(env, nb_steps=...) for training)

# Save weights using your existing `create_and_save_weights` function
def ppo_training_loop(env, agent, num_episodes, steps_per_episode):
  """
  Training loop for the PPO agent.

  Args:
      env: The environment to interact with.
      agent: The PPO agent to train.
      num_episodes: The number of episodes to train for.
      steps_per_episode: The number of steps per episode.

  Returns:
      None
  """

  for episode in range(num_episodes):
    # Collect experience through rollout
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for _ in range(steps_per_episode):
      state = env.reset()
      done = False
      while not done:
        action = agent.forward(state)  # Get action from the agent
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        state = next_state

    # Calculate advantages using GAE (Generalized Advantage Estimation)
    last_gae = 0.0
    advantages = []
    for i in reversed(range(len(rewards))):
      delta = rewards[i] + gamma * (not dones[i]) * agent.predict_v(next_states[i]) - agent.predict_v(states[i])
      gae = delta + gamma * lambda_ * (not dones[i]) * last_gae
      advantages.insert(0, gae)
      last_gae = gae

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + eps)

    # Train the PPO agent for multiple epochs (e.g., 4)
    for _ in range(ppo_epochs):
      # Sample mini-batches from collected experience
      for batch in minibatch(states, actions, rewards, advantages, batch_size):
        s_batch, a_batch, r_batch, adv_batch = batch
        # Update actor and critic networks using PPO loss function
        # (Replace "..." with the actual PPO loss calculation)
        agent.backward(s_batch, a_batch, r_batch, adv_batch)
        agent.learn(s_batch, a_batch)

    # Print episode progress (optional)
    print(f"Episode: {episode+1}/{num_episodes}")

# Hyperparameters
gamma = 0.99  # Discount factor
lambda_ = 0.95  # GAE parameter
eps = 1e-8  # Epsilon for numerical stability
ppo_epochs = 4  # Number of PPO epochs per episode
batch_size = 32  # Mini-batch size for training

# Training loop
ppo_training_loop(env, agent, num_episodes=10, steps_per_episode=rowsPerSeason)
