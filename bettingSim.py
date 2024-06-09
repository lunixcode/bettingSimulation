import pandas as pd
import time

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

# Preprocessing function to find the best odds
def preprocess_odds(row):
    # Accessing specific columns for odds
    best_home_odds = row['Y']  # Home odds in column 'Y'
    best_draw_odds = row['Z']  # Draw odds in column 'Z'
    best_away_odds = row['AA'] # Away odds in column 'AA'

    return best_home_odds, best_draw_odds, best_away_odds

# Betting system
def betting_system(row, betting_results):
    betting_results.append(profit)


# Load the E0.csv file
file_path = 'E0.csv'  # Update this to your file path
e0_data = pd.read_csv(file_path)

# Start timing
start_time = time.time()

# Create an empty league table and betting results list
league_table = {}
betting_results = []

# Iterate through each row in the dataset, update the league table, and apply the betting system
for index, row in e0_data.iterrows():
    update_league_table(row, league_table)
    #betting_system(row, betting_results)

# Convert the league table to a DataFrame for better visualization
league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Calculate total profit/loss from betting
total_profit = sum(betting_results)

# Display the league table and betting results
print(league_table_df)
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Total Betting Profit/Loss: {total_profit:.2f} units")