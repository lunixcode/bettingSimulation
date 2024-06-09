import pandas as pd
import time

# Start timing
start_time = time.time()

# Function to update the league table
def update_league_table(row, league_table):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_goals = row['FTHG']
    away_goals = row['FTAG']
    result = row['FTR']

    if(home_team == 'NULL'):
        print("Found the lost entry")

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

# Load the E0.csv file
base_path = 'bettingSim/Data/'
file_ppath = 'bettingSim/Data/E0.csv'  # Update this to your file path
e0_data = pd.read_csv(file_ppath)
training_leagues = []
dataframes = []
file_numbers = [ 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11]

for number in file_numbers:
    # Generate the file path by combining the base path and the number
    file_name = f'E0 ({number}).csv'
    file_path = f'{base_path}{file_name}'
    
    # Use pd.read_csv() to read the CSV file
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    training_leagues.append(df)

# Create an empty league table
league_table = {}

# Iterate through each row in the dataset and update the league table
for leagues in training_leagues:
    for index, row in leagues.iterrows():    
        #dataframes.apply(update_league_table(row, league_table), axis=1, args=(league_table,))
        #print(row)
        update_league_table(row, league_table)

# Convert the league table to a DataFrame for better visualization
league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# Display the league table
print(league_table_df)

end_time = time.time()

 

#print(dataframes)

# Calculate and print the total execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")