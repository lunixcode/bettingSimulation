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


class BettingEnv(gym.Env):
    """A betting environment for a league simulator."""

    def __init__(self, game_data):
        super(BettingEnv, self).__init__()
        self.game_data = game_data
        self.current_game_index = 0
        # Define action and observation space
        # Assuming actions are discrete bets: 0 = bet on team A, 1 = draw , 2 = bet on team B
        self.action_space = spaces.Discrete(3)

        # Example for observation space: normalized league standings and recent performance
        # This is a placeholder. You'll need to adjust the size based on your actual state representation
        num_form_values = 2  
        num_point_values = 1
        num_odds_values = 3

        # Define the observation space
        self.observation_space = spaces.Box(
            low=np.array([0] * (num_form_values + num_point_values + num_odds_values)),
            high=np.array([1] * (num_form_values + num_point_values + num_odds_values)),
            dtype=np.float32
        )
        '''self.observation_space = spaces.Box(
            low=np.array([0] * num_form_values + [-np.inf] * num_point_values + [0] * num_odds_values),
            high=np.array([1] * num_form_values + [np.inf] * num_point_values + [1] * num_odds_values),
            dtype=np.float32
        )'''
        #self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize state and other necessary variables
        self.state = None  # This will be your state representation

    def step(self, action, new_state):
    #def step(self, action):
        #Init the new state
        self.state = new_state
        # Get the result of the current game
        current_game = self.game_data.iloc[self.current_game_index]
        game_result = current_game['FTR']  # Assuming 'result' is a column in your CSV

        # Execute one time step within the environment
        # You need to update the state based on the action and return state, reward, done, info

        # Implement the logic of the action (betting)
        # For example, this could involve updating the league based on the bet,
        # determining if the bet was successful, etc.

        # Calculate the reward based on the outcome of the bet
        if action == 0:  # Bet on Team A
            reward = self.calculate_reward(bet_on='H' , game_result=game_result)
        elif action == 1:  # Bet on Team B
            reward = self.calculate_reward(bet_on='D', game_result=game_result)
        elif action == 2:  # Bet on Team B
            reward = self.calculate_reward(bet_on='A', game_result=game_result)
        else:  # No bet
            reward = self.calculate_reward(bet_on=None, game_result=game_result )

        # Check if the episode is done (e.g., all matches in a round are played)
        done = self.check_if_done()

        # Optional: Additional info, might be empty
        info = {}
        self.current_game_index+= 1

        return self.state, reward, done, info

    def calculate_reward(self, bet_on, game_result):
        # This function calculates the reward based on the outcome of the bet
        # Implement the reward logic here based on your design
        # For example:
        #if bet_on is None:
            #return 0  # No bet was placed
        if self.bet_successful(bet_on, game_result):
            return 2  # Example: positive reward for successful bet
        else:
            return -1  # Example: negative reward for unsuccessful bet

    def bet_successful(self, bet_on, game_result):
        # Implement logic to determine if the bet was successful
        # This will depend on the match outcome and the bet placed
        if(bet_on == game_result):
            return True
        else:
            return False            
        pass

    def check_if_done(self):
        # Implement logic to check if the current episode is done
        # This might depend on whether all matches in a round have been played
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        # Implement your reset logic here, typically resetting the league and matches
        self.current_game_index = 0
        self.state = None  # Reset to initial state
        home_form = 0.5  # Neutral starting form
        away_form = 0.5  # Neutral starting form
        points_diff = 0  # Neutral starting points difference
        odds_probability = [1/3, 1/3, 1/3]  # Equal starting probabilities

        # Combine into a state vector
        self.state = np.array([home_form, away_form, points_diff] + odds_probability)

        return self.state

    def render(self, mode='console'):
        # Render the environment to the screen
        # Implement any rendering logic here (optional, mainly for debugging)
        if mode == 'console':
            print(self.state)

    def update_state(self, new_data):
        # new_data contains the necessary information to update the state
        home_form = new_data['home_form']  # normalized value between 0 and 1
        away_form = new_data['away_form']  # normalized value between 0 and 1
        points_diff = new_data['points_diff']  # normalized value
        odds = new_data['odds']  # list of three values summing to 1

        self.state = np.array([home_form, away_form, points_diff] + odds)


# Start timing
start_time = time.time()
def populateLeague(row):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    if home_team not in league_table:
        league_table[home_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}
    if away_team not in league_table:
        league_table[away_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}


# Function to update the league table
def update_league_table(row, league_table):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_goals = row['FTHG']
    away_goals = row['FTAG']
    result = row['FTR']

    # Initialize teams in the league table
    if home_team not in league_table:
        league_table[home_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}
    if away_team not in league_table:
        league_table[away_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}

    # Update matches played, goals, and results
    league_table[home_team]['Matches Played'] += 1
    league_table[away_team]['Matches Played'] += 1
    league_table[home_team]['Goals For'] += home_goals
    league_table[home_team]['Goals Against'] += away_goals
    league_table[away_team]['Goals For'] += away_goals
    league_table[away_team]['Goals Against'] += home_goals

    # Update wins, draws, losses, and points
    if result == 'H':
        league_table[home_team]['Wins'] += 1
        league_table[home_team]['Points'] += 3
        league_table[away_team]['Losses'] += 1
    elif result == 'A':
        league_table[away_team]['Wins'] += 1
        league_table[away_team]['Points'] += 3
        league_table[home_team]['Losses'] += 1
    else:
        league_table[home_team]['Draws'] += 1
        league_table[away_team]['Draws'] += 1
        league_table[home_team]['Points'] += 1
        league_table[away_team]['Points'] += 1

    # Update goal difference
    league_table[home_team]['Goal Difference'] = league_table[home_team]['Goals For'] - league_table[home_team]['Goals Against']
    league_table[away_team]['Goal Difference'] = league_table[away_team]['Goals For'] - league_table[away_team]['Goals Against']
    teams_sorted = [(team, data) for team, data in league_table.items()]
    teams_sorted.sort(key=lambda x: (x[1]['Points'], x[1]['Goal Difference'], x[1]['Goals For']), reverse=True)
    for position, (team, data) in enumerate(teams_sorted, start=1):
        league_table[team]['Position'] = position

def backHomeSearch(row, index):
    revSearchVal = 4
    currentHomeIndex = revSearchVal
    currentAwayIndex = revSearchVal
    currentHomePoints = 0
    currentAwayPoints = 0
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    if home_team in league_table:
        games = league_table['Crystal Palace']['Matches Played']
    else:
        # Handle the case where 'Crystal Palace' is not in the dictionary
        games = 0
    #games = league_table[home_team]['Matches Played']
    print("Home team: ", row['HomeTeam'], "    Away team: ", row['AwayTeam'])

    if(index < revSearchVal):
        return

    #for num in range(revSearchVal, -1, -1):
    for indexVal, crow in e0_data[index::-1].iterrows():
        tempHomeTeam = crow['HomeTeam']
        tempAwayTeam = crow['AwayTeam']
        
        if currentHomeIndex >= 0:
            #print(currentHomeIndex)
            if tempHomeTeam == home_team:
                if crow['FTR'] == 'H':
                    currentHomePoints+=3
                    currentHomeIndex-=1
                elif crow['FTR'] == 'D':
                    currentHomePoints+=1
                    currentHomeIndex-=1
                else:
                    currentHomeIndex-=1
        if currentAwayIndex >= 0:
            if tempAwayTeam == away_team:
                if crow['FTR'] == 'A':
                    currentAwayPoints+=3
                    currentAwayIndex-=1
                elif crow['FTR'] == 'D':
                    currentAwayPoints+=1
                    currentAwayIndex-=1
                else:
                    currentAwayIndex-=1

        #index-=1
        if currentHomeIndex <= 0 and currentAwayIndex <= 0:
            #print("HOME Points from back search: ", currentHomePoints)
            #print("AWAY Points from back search: ", currentAwayPoints)
            break 

    normHomePoints = currentHomePoints / (revSearchVal * 3)
    normAwayPoints = currentAwayPoints / (revSearchVal * 3)  
    print("Normalised Home Form: ", round(normHomePoints, 2), "   Normalised Away Form: ", round(normAwayPoints, 2))
    return round(normHomePoints, 2), round(normAwayPoints, 2)

def normalizedLeaguePos(row):
    if row['HomeTeam'] not in league_table:
        return
    elif row['AwayTeam'] not in league_table:
        return
    else:
        teamApos = league_table[row['HomeTeam']]['Position']
        teamBpos =  league_table[row['AwayTeam']]['Position']
        #print(teamApos, "   ", teamBpos)
        if teamApos > teamBpos:
            #print(1 - ((teamApos- teamBpos) / len(league_table)))
            return round( 1 - ((teamApos- teamBpos) / len(league_table)), 2)
        else:
            #print(1 - ((teamBpos- teamApos) / len(league_table)))
            return round(1 - ((teamBpos- teamApos) / len(league_table)), 2)
        
    return 

def normalizedOdds(row):
    best_home_odds = round(1 / row['AvgH'],2)  # Home odds in column 'Y'
    best_draw_odds = round(1 / row['AvgD'],2)  # Draw odds in column 'Z'
    best_away_odds = round(1 / row['AvgA'],2) # Away odds in column 'AA'
    odds_probability = [best_home_odds, best_draw_odds,best_away_odds]
    print("Home Normalised: ", best_home_odds, "    Draw normalised: ", best_draw_odds,"    Away normaized: ", best_away_odds)
    return odds_probability



# Load the E0.csv file
file_path = 'E0.csv'  # Update this to your file path
e0_data = pd.read_csv(file_path)

# Create an empty league table
league_table = {}
for index, row in e0_data.iterrows():
    populateLeague(row)

env = BettingEnv(e0_data)
env.reset()

state_size = env.observation_space.shape[0]
number_of_actions = env.action_space.n
num_episodes = 1

print(state_size, "     ", number_of_actions)

model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(number_of_actions, activation='linear'))

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()

dqn = DQNAgent(model=model, 
               nb_actions=number_of_actions, 
               memory=memory, 
               nb_steps_warmup=1000,
               target_model_update=0.01, 
               policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

for _ in range(num_episodes):
    # Iterate through each row in the dataset and update the league table
    env.reset()
    for index, row in e0_data.iterrows():
        if index < 4:
            update_league_table(row, league_table)
        else:
            print(index)
            homeForm, awayForm = backHomeSearch(row, index)
            odds_probability =normalizedOdds(row)
            normalLeaguePos = normalizedLeaguePos(row)
            
            print("Normalised League Pos: ", normalLeaguePos)
            update_league_table(row, league_table)
            result = row['FTR']
            newstate = np.array([homeForm, awayForm, normalLeaguePos] + odds_probability)
            #Enter RL Algorithm for betting
            action = env.action_space.sample()  # Replace with a specific action for testing
            state, reward, done, _ = env.step(action, newstate)
            #print(reward)
            if done:
                env.reset()

dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Optionally, you can save your trained model
dqn.save_weights('dqn_betting_model_weights.h5f', overwrite=True)

# Convert the league table to a DataFrame for better visualization
league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# Display the league table
print(league_table_df)

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
