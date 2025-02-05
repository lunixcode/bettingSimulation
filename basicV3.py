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

dqn = DQNAgent(model=model, 
               nb_actions=number_of_actions, 
               memory=memory, 
               nb_steps_warmup=1000,
               target_model_update=hyper_parameters['learning_rate'], 
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])




#for _ in range(num_episodes):
    # Iterate through each row in the dataset and update the league table
    #env.reset()
    #for index, row in e0_data.iterrows():
        

#dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
#dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Optionally, you can save your trained model
#dqn.save_weights('dqn_betting_model_weights.h5f', overwrite=True)

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Convert the league table to a DataFrame for better visualization
#league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
#league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# Display the league table
#print(league_table_df)
    

# Create an empty league table
#league_table = {}
index = 1
runs = 6
fullRun = 0
lSize = 0



# Optional: Create a custom callback to track additional metrics


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
