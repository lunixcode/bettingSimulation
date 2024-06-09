import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier  # For classification
# Or:
from sklearn.ensemble import RandomForestRegressor # For regression

# Start timing
start_time = time.time()

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

def normalizedLeaguePos( row):  
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
    best_home_odds = round((1 / row['B365H']),2)  # Home odds in column 'Y'
    best_draw_odds = round((1 / row['B365D']),2)  # Draw odds in column 'Z'
    best_away_odds = round( (1 / row['B365A']),2) # Away odds in column 'AA'
    odds_probability = [best_home_odds, best_draw_odds,best_away_odds]
    #*******************DEBUG
    #print("Home Normalised: ", best_home_odds, "    Draw normalised: ", best_draw_odds,"    Away normaized: ", best_away_odds)
    return odds_probability

def formBackSearch( row, index):
    revSearchVal = 7
    currentHomeIndex = revSearchVal
    currentAwayIndex = revSearchVal
    currentHomePoints = 0
    currentAwayPoints = 0
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    if(index < revSearchVal):
        return 0, 0
    
    for indexVal, crow in e0_data[index::-1].iterrows():
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
    #print("Normalised h Form: ", round(normHomePoints, 2), "   Normalised a Form: ", round(normAwayPoints, 2))
    return round(normHomePoints, 2), round(normAwayPoints, 2)    
    return 0


def backHomeSearch( row, index):
    revSearchVal = 4 #Amount of games to do a reverse search on
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
        
        #*******************DEBUG
    #print("Home team: ", row['HomeTeam'], "    Away team: ", row['AwayTeam'])

    if(index < revSearchVal):
        return 0 , 0

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
    #*******************DEBUG  
    #print("Normalised Home Form: ", round(normHomePoints, 2), "   Normalised Away Form: ", round(normAwayPoints, 2))
    return round(normHomePoints, 2), round(normAwayPoints, 2)

# Load the E0.csv file
file_path = 'E0.csv'  # Update this to your file path
e0_data = pd.read_csv(file_path)

# Create an empty league table
league_table = {}
features_list = []
target = []

trainingset_size = 0.2
trainingset_percent = trainingset_size * 100
# Iterate through each row in the dataset and update the league table
for index, row in e0_data.iterrows():
    update_league_table(row, league_table)
    home_form, away_form = backHomeSearch(row, index)
    hometotal_form , awaytotal_form = formBackSearch(row, index)
    odds_probability = normalizedOdds(row)
    home_odds = odds_probability[0]
    draw_odds= odds_probability[1]
    away_odds = odds_probability[2]
    normalLeaguePos = normalizedLeaguePos(row)
    features_list.append([home_form, hometotal_form, away_form, awaytotal_form, normalLeaguePos, odds_probability[0], odds_probability[1], odds_probability[2] ])
    target.append(row['FTR'])
    #print(row['HomeTeam'], "    " , row['AwayTeam'])
    #backHomeSearch(row, index)
    #formBackSearch(row, index)
    #print('\n')
    #target = e0_data['FTR']

# Convert the league table to a DataFrame for better visualization
league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# Display the league table
print(league_table_df)

for i in features_list:
    print(i)
uniqueTeams = 20
rowsPerSeason = (uniqueTeams - 1) * uniqueTeams

print(len(e0_data), "    ", rowsPerSeason)
print(target)
end_time = time.time()


column_names = ["HomeForm1", "HomeMainForm2", "AwayForm1", "AwayMainForm2", "posDiff", "HOdds", "DOdds", "AOdds"] 
cname = ['Result']
for i in range(len(target)): 
    if target[i] == 'H':
        target[i] = 0
    elif target[i] == 'D':
        target[i] = 1
    elif target[i] == 'A':
        target[i] = 2

print(target)
# Create the DataFrame
X = pd.DataFrame(features_list, columns=column_names) 
y = pd.DataFrame(target, columns=cname)

import numpy as np 

# Assuming y is a pandas Series (if it's a DataFrame column, use y['Result'])
y = y.to_numpy()  # Convert to a NumPy array first
y = np.ravel(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% for training, 20% for testing

rf_model = RandomForestClassifier()  # You can customize parameters later
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
outcome_mapping = {0: 'H', 1: 'D', 2: 'A'}
outcome = []
for prediction in y_pred:
    outcome.append(outcome_mapping[prediction]) 
    print(f"Predicted Outcome: {outcome}")

print("Prediction: ", y_pred)

probabilities = rf_model.predict_proba(X_test)
home_win_prob = []
draw_prob = []
away_win_prob = []
for probs in probabilities:
    home_win_prob.append(probs[0]) 
    draw_prob.append(probs[1])
    away_win_prob.append(probs[2])

    print(f"Home Win Probability: {probs[0]:.2f}")
    print(f"Draw Probability: {probs[1]:.2f}")
    print(f"Away Win Probability: {probs[2]:.2f}")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy: {accuracy:.2f}")

# Calculate and print the total execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

print(int(len(e0_data) / 100 * trainingset_percent))

tdsval = int(len(e0_data) /100 * trainingset_percent) #training data size Value (the size of the training data)
startpoint = len(e0_data) - tdsval
i=0
for index, row in e0_data.iloc[startpoint:].iterrows():
    # Access elements within the row using row['column_name']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']
    #print(index)
    print ( outcome[i], "   ", home_team,"  ", away_team, "     ", "    H:", home_win_prob[i], "    D: ", draw_prob[i], "   A: ", away_win_prob[i] )
    print(result, features_list[i] ,'\n' )
    i+=1



base_path = 'bettingSim/Data/'
#file_ppath = 'bettingSim/Data/E0.csv'  # Update this to your file path
#e0_data = pd.read_csv(file_ppath)
training_leagues = []
dataframes = []
file_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]

for number in file_numbers:
    # Generate the file path by combining the base path and the number
    file_name = f'E0 ({number}).csv'
    file_path = f'{base_path}{file_name}'
    
    # Use pd.read_csv() to read the CSV file
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    training_leagues.append(df)

# Create an empty league table
#league_table = {}
index = 1
runs = 5
# Iterate through each row in the dataset and update the league table
for leagues in training_leagues:
    #env.load_data(leagues)
    for i in range(runs):
        
        print("RUN: ", index)
        index+=1

    print(leagues) 
    

