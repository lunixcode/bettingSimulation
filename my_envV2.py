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

class BettingEnv(gym.Env):
    """A betting environment for a league simulator."""

    def __init__(self):
        super(BettingEnv, self).__init__()
        self.league_table = {}
        self.uniqueTeams = 20
        self.rowsPerSeason = (self.uniqueTeams - 1) * self.uniqueTeams
        self.file_path = 'E0.csv'  # Update this to your file path
        self.e0_data = pd.read_csv(self.file_path)
        self.current_row_index = 0
        self.num_rows_in_csv = len(self.e0_data)
        for index, row in self.e0_data.iterrows():
            self.populateLeague(row)
        #self.game_data = game_data
        self.current_game_index = 0
        self.running_total = 0
        self.home_Wins = 0
        self.away_Wins = 0
        self.draws = 0
        self.printdbg = 0
        self.betVal = 100
        # Define action and observation space
        # Assuming actions are discrete bets: 0 = bet on team A, 1 = draw , 2 = bet on team B
        self.action_space = spaces.Discrete(3)
        self.countForEpisode = 0

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

    def step(self, action):
    #def step(self, action):
        #Init the new state
        #self.state = new_state
        cIndex = self.current_game_index
        #print(cIndex)
        try:
        # Get the result of the current game
            row = self.e0_data.iloc[cIndex]
        except Exception as e:
            #You will at the end of each season be trying to find the state for the next game that doesnt exist
            print("An error occurred:", str(e))
            row = self.e0_data.iloc[self.current_game_index]



        game_result = row['FTR']  # Assuming 'result' is a column in your CSV

        if game_result == 'H':
            self.home_Wins+=1
        elif game_result == 'D':
            self.draws+=1
        else:
            self.away_Wins+=1

        '''if self.current_game_index == 0:
            info = {}
            reward = 0.5  
            #done = False
            done = self.check_if_done()
            self.current_game_index+=1
            print(self.current_game_index)
            return self.state, reward, done, info'''
        # Execute one time step within the environment
        # You need to update the state based on the action and return state, reward, done, info
        #print("here")
        # Implement the logic of the action (betting)
        # For example, this could involve updating the league based on the bet,
        # determining if the bet was successful, etc.
        if self.current_game_index < 4:
            #dont really want the Agent to start predicting actions until 5 games into the season
            self.update_league_table(row, self.league_table)
        else:
            #print(self.current_game_index)
            #homeForm, awayForm = self.backHomeSearch( row, self.current_game_index)
            #odds_probability = self.normalizedOdds(row)
            #normalLeaguePos = self.normalizedLeaguePos(row)

            self.update_state( row)

            #print("Normalised League Pos: ", normalLeaguePos)
            self.update_league_table(row, self.league_table)
            result = row['FTR']
            #newstate = np.array([homeForm, awayForm, normalLeaguePos] + odds_probability)
            #Enter RL Algorithm for betting
            #action = env.action_space.sample()  # Replace with a specific action for testing
            #state, reward, done, _ = env.step(action)
            #print(reward)
            #if done:
                #env.reset()
            
        #  *************** DEBUG *****************
        if self.printdbg:
            print("Home Team :", row['HomeTeam'], "  Away Team : ", row['AwayTeam'])


        odds = []
        odds.append(row['B365H'])
        odds.append(row['B365D'])
        odds.append(row['B365A'])
        bet_amount = 100
        # Calculate the reward based on the outcome of the bet
        if action == 0:  # Bet on Team A
            
            reward = self.calculate_reward(action, bet_on='H' , game_result=game_result, bet_amount = bet_amount,  odds = odds)
        elif action == 1:  # Bet on Draw
            
            reward = self.calculate_reward(action, bet_on='D', game_result=game_result,  bet_amount = bet_amount,  odds = odds)
        elif action == 2:  # Bet on Team B
            
            reward = self.calculate_reward(action, bet_on='A', game_result=game_result,  bet_amount = bet_amount,  odds = odds)
        else:  # No bet
            reward = self.calculate_reward(action, bet_on=None, game_result=game_result,  bet_amount = bet_amount,  odds = odds )

        # Check if the episode is done (e.g., all matches in a round are played)
        

        # Optional: Additional info, might be empty
        info = {}
        self.current_game_index+= 1
        self.countForEpisode+=1
        done = self.check_if_done()

        return self.state, reward, done, info

    '''
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
            
    '''

    def calculate_reward(self, action, bet_on, game_result, bet_amount, odds):

        self.print_state()

        if self.bet_successful(bet_on, game_result):
            holder = bet_amount * (odds[action])
            self.running_total +=  self.betVal + (self.betVal * (odds[action]))
            #self.running_total +=  bet_amount * (odds[action])
            
            #   ********** DEBUG ***************
            if self.printdbg:
                print("WIN  ", "Action : ", bet_on , "   Result : ", game_result, "  Return : ",  round(bet_amount * (odds[action]), 2), "  Total: ", self.running_total, "  Odds : ", odds)
            #return 1
            if bet_on == 'H':
                self.home_Wins+=1
                if self.printdbg:
                    print("Reward for H: ", 1 -  (1 / (odds[action])), "\n") # (1 / odds[action]) * (1 / 0.46))
                #return 1 - (1 /odds[action])
                #return (odds[action] * (1 / 0.5))
                #return 1 -  (1 / (odds[action] ))
                #return (1 / odds[action]) * (1 / 0.5)
                #return ((1 / odds[action])  / 100) * 54
                #return 1 - (((1 / odds[action])  / 100) * 54)
                return self.betVal + (self.betVal * odds[action])
            
            elif bet_on == 'D':
                self.draws+=1
                if self.printdbg:
                    print("Reward for D: ",  (1 - (1 /odds[action])) , "\n")#((1 / odds[action])) * (1 / 0.24))
                #return (1 - (1 /odds[action])) #* 1.85
                #return  1 -  (1 / ( odds[action]  * ((1 / 0.24))/ (1 / 0.5)))
                #return (1 / odds[action]) * (1 / 0.24)
                #return (1 / odds[action]) / 100 * 76 # * 1.85
                #return (1 - (1 / odds[action])) * 1.85 # 1.47
                return self.betVal + (self.betVal * odds[action]) 
                
            else:
                self.away_Wins+=1
                if self.printdbg:
                    print("Reward for A: ", (1 - (1 /odds[action])) , "\n") # ((1 / odds[action])) * (1 / 0.3))
                #return (1 - (1 /odds[action])) #* 1.54
                #return 1 -  (1 / ( odds[action]  * ((1 / 0.3)) / (1 / 0.5)))
                #return (1 / odds[action]) * (1 / 0.3)
                #return (1 / odds[action])  / 100 * 70 #* 1.54
                #return (1 - (1 / odds[action])) * 1.54 #1.36
                return self.betVal + (self.betVal * odds[action]) 
            
            #return bet_amount * (odds) #reward V1
            #return odds #reward V2
            #return 1 - ( 1/ odds) #rerward V3
            return 1 / odds[action]
        else:
            self.running_total -=  self.betVal
            if self.printdbg:
                print("LOSE  ", "Action : ", bet_on , "   Result : ", game_result, "  Return : ", 0 - bet_amount, "  Total: ", self.running_total,"  Odds : ", odds)

            holder = -bet_amount
            
            #return - 0.5
            #return -bet_amount #r1
            #return 0 #r2
            #return 0 - (1 -(1 /odds[action]))
            if bet_on == 'H':
                if self.printdbg:
                    print("Minus Reward for H: ", 0 - (1 / odds[action]), "\n")
                self.home_Wins+=1
                #return 0
                #return 1 - odds[action]
                #return 0 - (1 / odds[action])
                #return 0 - odds[action]
                return 0 - self.betVal
            
            elif bet_on == 'D':
                if self.printdbg:
                    print("Minus Reward for D: ", 0 - (1 / odds[action]), "\n")
                self.draws+=1
                #return 0
                #return 0 - (1 / odds[action])
                #return 1 - odds[action]
                #return 0 - ((1 / odds[action]) / 100 * 53)
                #return 0 - odds[action]
                return 0 - self.betVal
            
            else:
                if self.printdbg:
                    print("Minus Reward for A: ", 0 - (1 / odds[action]), "\n")
                self.away_Wins+=1
                #return 0
                #return 0 - (1 / odds[action])
                #return 1 - odds[action]
                #return 0 - ((1 / odds[action])  / 100 * 64)
                #return 0 - odds[action]
                return 0 - self.betVal
            
    def print_state(self):
        if self.printdbg:
            print(self.state)

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
        if(self.rowsPerSeason <= self.current_game_index ):
            #   ************    DEBUG   ****************
            #print("Home wins:", self.home_Wins, "    Draws: ", self.draws, "    Away Wins: ", self.away_Wins)
            
            return True
        #if(self.countForEpisode <= 10):
            #return False
           # return True
        else:
            #return True
            return False

        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        # Implement your reset logic here, typically resetting the league and matches
        self.current_game_index = 0
        self.running_total = 0
        self.state = None  # Reset to initial state
        home_form = 0.5  # Neutral starting form
        away_form = 0.5  # Neutral starting form
        points_diff = 0  # Neutral starting points difference
        odds_probability = [1/3, 1/3, 1/3]  # Equal starting probabilities

        # Combine into a state vector
        self.state = np.array([home_form, away_form, points_diff] + odds_probability)
        self.countForEpisode = 0

        return self.state

    def render(self, mode='console'):
        # Render the environment to the screen
        # Implement any rendering logic here (optional, mainly for debugging)
        if mode == 'console':
            print(self.state)

    def update_state(self, row):
        # new_data contains the necessary information to update the state
        #home_form = new_data['home_form']  # normalized value between 0 and 1
        #away_form = new_data['away_form']  # normalized value between 0 and 1
        #points_diff = new_data['points_diff']  # normalized value
        #odds = new_data['odds']  # list of three values summing to 1

        homeForm, awayForm = self.backHomeSearch( row, self.current_game_index)
        hForm, aForm = self.formBackSearch(row, self.current_game_index )
        odds_probability = self.normalizedOdds(row)
        normalLeaguePos = self.normalizedLeaguePos(row)

        self.state = np.array([homeForm, awayForm, normalLeaguePos] + odds_probability)

    def normalizedLeaguePos(self, row):
        if row['HomeTeam'] not in self.league_table:
            return
        elif row['AwayTeam'] not in self.league_table:
            return
        else:
            teamApos = self.league_table[row['HomeTeam']]['Position']
            teamBpos =  self.league_table[row['AwayTeam']]['Position']
                #print(teamApos, "   ", teamBpos)
            if teamApos > teamBpos:
                    #print(1 - ((teamApos- teamBpos) / len(league_table)))
                return round( 1 - ((teamApos- teamBpos) / len(self.league_table)), 2)
            else:
                    #print(1 - ((teamBpos- teamApos) / len(league_table)))
                return round(1 - ((teamBpos- teamApos) / len(self.league_table)), 2)
                
        return 
    
    def normalizedOdds(self, row):
        best_home_odds = round(1 / row['B365H'],2)  # Home odds in column 'Y'
        best_draw_odds = round(1 / row['B365D'],2)  # Draw odds in column 'Z'
        best_away_odds = round(1 / row['B365A'],2) # Away odds in column 'AA'
        odds_probability = [best_home_odds, best_draw_odds,best_away_odds]
        #*******************DEBUG
        #print("Home Normalised: ", best_home_odds, "    Draw normalised: ", best_draw_odds,"    Away normaized: ", best_away_odds)
        return odds_probability

    def formBackSearch(self, row, index):
        revSearchVal = 7
        currentHomeIndex = revSearchVal
        currentAwayIndex = revSearchVal
        currentHomePoints = 0
        currentAwayPoints = 0
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        if(index < revSearchVal):
            return 0, 0
        
        for indexVal, crow in self.e0_data[index::-1].iterrows():
            tempHomeTeam = crow['HomeTeam']
            tempAwayTeam = crow['AwayTeam']

            if currentHomeIndex >= 0:
                if home_team == tempHomeTeam:
                    if crow['FTR'] == 'H':
                        currentHomePoints+=3
                        currentHomeIndex-=1
                    elif crow['FTR'] == 'D':
                        currentHomePoints+=1
                        currentHomeIndex-=1
                    else:
                        currentHomeIndex-=1
                    pass
                elif home_team == tempAwayTeam:
                    if crow['FTR'] == 'A':
                        currentHomePoints+=3
                        currentHomeIndex-=1
                    elif crow['FTR'] == 'D':
                        currentHomePoints+=1
                        currentHomeIndex-=1
                    else:
                        currentHomeIndex-=1
                    pass                
                    
            if currentAwayIndex >= 0:
                if away_team == tempHomeTeam:
                    if crow['FTR'] == 'H':
                        currentAwayPoints+=3
                        currentAwayIndex-=1
                    elif crow['FTR'] == 'D':
                        currentAwayPoints+=1
                        currentAwayIndex-=1
                    else:
                        currentAwayIndex-=1
                    pass
                elif away_team == tempAwayTeam:
                    if crow['FTR'] == 'A':
                        currentAwayPoints+=3
                        currentAwayIndex-=1
                    elif crow['FTR'] == 'D':
                        currentAwayPoints+=1
                        currentAwayIndex-=1
                    else:
                        currentAwayIndex-=1
                    pass
            if currentHomeIndex <= 0 and currentAwayIndex <= 0:
                #print("HOME Points from back search: ", currentHomePoints)
                #print("AWAY Points from back search: ", currentAwayPoints)
                break

        normHomePoints = currentHomePoints / (revSearchVal * 3) 
        normAwayPoints = currentAwayPoints / (revSearchVal * 3)
        #*******************DEBUG  
        if self.printdbg:
            print("Normalised h Form: ", round(normHomePoints, 2), "   Normalised a Form: ", round(normAwayPoints, 2))
        return round(normHomePoints, 2), round(normAwayPoints, 2)    
        return 0
    

    def backHomeSearch(self, row, index):
        revSearchVal = 4 #Amount of games to do a reverse search on
        currentHomeIndex = revSearchVal
        currentAwayIndex = revSearchVal
        currentHomePoints = 0
        currentAwayPoints = 0
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        if home_team in self.league_table:
            games = self.league_table['Crystal Palace']['Matches Played']
        else:
            # Handle the case where 'Crystal Palace' is not in the dictionary
            games = 0
        #games = league_table[home_team]['Matches Played']
            
            #*******************DEBUG
        #print("Home team: ", row['HomeTeam'], "    Away team: ", row['AwayTeam'])

        if(index < revSearchVal):
            return

        #for num in range(revSearchVal, -1, -1):
        for indexVal, crow in self.e0_data[index::-1].iterrows():
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
                        #if one goes below 0 tho, is there a catch? 
            if currentHomeIndex <= 0 and currentAwayIndex <= 0:
                #print("HOME Points from back search: ", currentHomePoints)
                #print("AWAY Points from back search: ", currentAwayPoints)
                break 

        normHomePoints = currentHomePoints / (revSearchVal * 3) 
        normAwayPoints = currentAwayPoints / (revSearchVal * 3)
        #*******************DEBUG  
        if self.printdbg:
            print("Normalised Home Form: ", round(normHomePoints, 2), "   Normalised Away Form: ", round(normAwayPoints, 2))
        return round(normHomePoints, 2), round(normAwayPoints, 2)
    
    def populateLeague(self, row):
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        if home_team not in self.league_table:
            self.league_table[home_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}
        if away_team not in self.league_table:
            self.league_table[away_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}

    def update_league_table(self, row, league_table):
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        result = row['FTR']

        # Initialize teams in the league table
        if home_team not in league_table:
            self.league_table[home_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}
        if away_team not in league_table:
            self.league_table[away_team] = {'Matches Played': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0, 'Goals For': 0, 'Goals Against': 0, 'Goal Difference': 0, 'Points': 0}

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

    def load_data(self, league_data):
        self.e0_data = league_data
        #print("HERE")
        #print(self.league_table)

    def get_state(self):
        return self.state
    
    def get_action(self):
        row = self.e0_data.iloc[self.current_game_index]
        if row['FTR'] == 'H':
            return 0
        elif row['FTR'] == 'D':
            return 1
        else:
            return 2
