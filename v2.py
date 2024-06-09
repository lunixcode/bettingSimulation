
import pandas as pd
import time

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b

    
    return a

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
        #print(league_table[team] , "    ", league_table[team]['Position'])

def print_odds(data):
    for index, row in data.iterrows():
        home_team = row['HomeTeam']  # Assuming 'HomeTeam' is the column name for the home team
        away_team = row['AwayTeam']  # Assuming 'AwayTeam' is the column name for the away team
        home_odds = row['AvgH']  # Home odds in column 'AT'
        draw_odds = row['AvgD']  # Draw odds in column 'AU'
        away_odds = row['AvgA']  # Away odds in column 'AV'

        print(f"{home_team} vs {away_team} | Home Odds: {home_odds}, Draw Odds: {draw_odds}, Away Odds: {away_odds}")


# Preprocessing function to find the best odds
def preprocess_odds(row):
    # Accessing specific columns for odds
    best_home_odds = row['AvgH']  # Home odds in column 'Y'
    best_draw_odds = row['AvgD']  # Draw odds in column 'Z'
    best_away_odds = row['AvgA'] # Away odds in column 'AA'

    return best_home_odds, best_draw_odds, best_away_odds

def update_non_win_streak(row, current_non_win_streaks, longest_non_win_streaks):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']  # 'FTR' stands for full-time result

    # Check for non-win outcomes
    if result != 'H':  # If the home team didn't win
        current_non_win_streaks[home_team] += 1
        if current_non_win_streaks[home_team] > longest_non_win_streaks[home_team]:
            longest_non_win_streaks[home_team] = current_non_win_streaks[home_team]
    else:  # Home team won, reset their streak
        current_non_win_streaks[home_team] = 0

    if result != 'A':  # If the away team didn't win
        current_non_win_streaks[away_team] += 1
        if current_non_win_streaks[away_team] > longest_non_win_streaks[away_team]:
            longest_non_win_streaks[away_team] = current_non_win_streaks[away_team]
    else:  # Away team won, reset their streak
        current_non_win_streaks[away_team] = 0


def update_win_streak(row, win_streaks):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']

    if result == 'H':  # Home team won
        win_streaks[home_team] += 1  # Reset the streak
        win_streaks[away_team] = 0 
    elif result == 'A':
        win_streaks[home_team] = 0  # Reset the streak
        win_streaks[away_team] += 1
    else:
        win_streaks[home_team] = 0  # Reset the streak
        win_streaks[away_team] = 0
        

def betting_system_home_wins(row, betting_pots, home_win_streaks, minimum_bet=100):
    home_team = row['HomeTeam']
    pot = betting_pots.get(home_team, 0)

    # Check if the home team hasn't won in 5 home games and has enough in the pot
    if home_win_streaks.get(home_team, 0) >= 5 and pot >= minimum_bet:
        home_odds = preprocess_odds(row)[0]  # Assuming this function returns the home odds
        # Place a bet on a home win
        if row['FTR'] == 'H':  # Home team wins
            # Update pot based on the home odds
            betting_pots[home_team] += (minimum_bet * home_odds) - minimum_bet
        else:
            # Deduct the bet amount from the home team's pot
            betting_pots[home_team] -= minimum_bet
        # Reset home win streak for the home team after placing a bet
        home_win_streaks[home_team] = 0


# Function to update draw streaks
def update_draw_streak(row, draw_streaks):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']

    if result == 'D':  # If the result is a draw
        draw_streaks[home_team] = 0  # Reset draw streak for both teams
        draw_streaks[away_team] = 0
    else:
        draw_streaks[home_team] += 1  # Increment draw streak for both teams
        draw_streaks[away_team] += 1

def kegs_System(row, teamVal, league_table, current_non_win_streaks, win_streaks, totalTrades):
    
    mult = 3
    start = 10

    if(teamVal == 0):
        tName = row['HomeTeam']
    else:
        tName = row['AwayTeam']

    if(league_table[tName]['Wins'] <= (len(league_table) - league_table[tName]['Position'])):
        if(league_table[tName]['Matches Played'] >= 5):
            if(league_table[tName]['Position'] < (len(league_table) / 2)):
                #init pot
                #VERY IMPORTANT, the 10 is base amount, 2 is the multiplier
                #bet_amount = 100 * fibonacci(current_non_win_streaks[tName])
                bet_amount = start * mult ** current_non_win_streaks[tName]

                if(win_pots[tName] > bet_amount):
                    if(teamVal == 0):
                        home_odds = preprocess_odds(row)[0]
                        if(row['FTR'] == 'H'):
                            #make bet
                            #REALLY NEED TO CHECK THIS EQUATION
                            win_pots[tName] += (bet_amount * home_odds) + bet_amount
                            print("MAKE BET WIN  betsize= ", bet_amount, "    Odds = ", home_odds, "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                        else:
                            #reduce pot
                            win_pots[tName] -= bet_amount
                            print("MAKE BET LOSS  betsize= ", bet_amount, "    Odds = ", home_odds, "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                    else:
                        away_odds = preprocess_odds(row)[2]
                        if(row['FTR'] == 'A'):
                            #make bet
                            win_pots[tName] += (away_odds * bet_amount) + bet_amount
                            print("MAKE BET WIN  betsize= ", bet_amount, "    Odds = ", away_odds,  "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                        else:
                            #reduce pot
                            win_pots[tName] -= bet_amount
                            print("MAKE BET LOSS  betsize= ", bet_amount, "    Odds = ", away_odds,  "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades

    #Now for teams half way down         
    elif(league_table[tName]['Losses'] < league_table[tName]['Position']):
        if(league_table[tName]['Matches Played'] >= 5):
            if(league_table[tName]['Position'] > (len(league_table) / 2)):
                #init pot
                #VERY IMPORTANT, the 10 is base amount, 2 is the multiplier, 
                #  This one is wrong, need to fix
                bet_amount = start * mult ** win_streaks[tName]
                #bet_amount = start * mult ** current_non_win_streaks[tName]
                #bet_amount = 100 * fibonacci(current_non_win_streaks[tName])
                #Bit of a blag but, youre betting on the team to lose, so you want the result to be the opposite of the team
                if(win_pots[tName] > bet_amount):
                    if(teamVal == 1):
                        home_odds = preprocess_odds(row)[0]
                        if(row['FTR'] == 'H'):
                            #make bet
                            #REALLY NEED TO CHECK THIS EQUATION
                            win_pots[tName] += (bet_amount * home_odds) + bet_amount
                            print("2 MAKE BET WIN  betsize= ", bet_amount, "    Odds = ", home_odds, "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                        else:
                            #reduce pot
                            win_pots[tName] -= bet_amount
                            print("2 MAKE BET LOSS  betsize= ", bet_amount, "    Odds = ", home_odds, "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                    else:
                        away_odds = preprocess_odds(row)[2]
                        if(row['FTR'] == 'A'):
                            #make bet
                            win_pots[tName] += (away_odds * bet_amount) + bet_amount
                            print("2 MAKE BET WIN  betsize= ", bet_amount, "    Odds = ", away_odds,  "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                        else:
                            #reduce pot
                            win_pots[tName] -= bet_amount
                            print("2 MAKE BET LOSS  betsize= ", bet_amount, "    Odds = ", away_odds,  "   ", tName, "  New Pot = ", round(win_pots[tName]))
                            totalTrades+=1
                            return totalTrades
                        

    # Now for draws
    return totalTrades

# Betting system
def betting_system(row, betting_pots, draw_streaks, minimum_bet=100):
    for team, pot in betting_pots.items():
        # Check if the team's draw streak is at least 5 and the pot is sufficient
        if draw_streaks[team] >= 5 and pot >= minimum_bet:
            if row['HomeTeam'] == team or row['AwayTeam'] == team:
                draw_odds = preprocess_odds(row)[1]  # Assuming this function gets the draw odds
                # Place a bet on a draw
                if row['FTR'] == 'D':  # If the match result is a draw
                    # Update pot based on the draw odds
                    betting_pots[team] += (minimum_bet * draw_odds) - minimum_bet
                else:
                    # Deduct the bet amount from the team's pot
                    betting_pots[team] -= minimum_bet
                # Reset draw streak for the team after placing a bet
                draw_streaks[team] = 0

def update_away_streak(row, away_streaks):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']

    if result == 'A':  # If the result is a draw
        away_streaks[home_team] = 0  # Reset draw streak for both teams
        away_streaks[away_team] = 0
    else:
        away_streaks[home_team] += 1  # Increment draw streak for both teams
        away_streaks[away_team] += 1

def update_non_draw_streak(row, current_non_draw_streaks, longest_non_draw_streaks):
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']  # 'FTR' is typically the field for full-time result

    if result != 'D':  # Not a draw
        current_non_draw_streaks[home_team] += 1
        current_non_draw_streaks[away_team] += 1

        # Update longest streak if current streak is longer
        if current_non_draw_streaks[home_team] > longest_non_draw_streaks[home_team]:
            longest_non_draw_streaks[home_team] = current_non_draw_streaks[home_team]
        if current_non_draw_streaks[away_team] > longest_non_draw_streaks[away_team]:
            longest_non_draw_streaks[away_team] = current_non_draw_streaks[away_team]
    else:  # Reset streak on a draw
        current_non_draw_streaks[home_team] = 0
        current_non_draw_streaks[away_team] = 0


# Betting system
def betting_system_away(row, betting_pots, away_streaks, minimum_bet=100):
    for team, pot in betting_pots.items():
        # Check if the team's draw streak is at least 5 and the pot is sufficient
        if draw_streaks[team] >= 5 and pot >= minimum_bet:
            if row['HomeTeam'] == team or row['AwayTeam'] == team:
                away_odds = preprocess_odds(row)[2]  # Assuming this function gets the draw odds
                # Place a bet on a draw
                if row['FTR'] == 'A':  # If the match result is a draw
                    # Update pot based on the draw odds
                    betting_pots[team] += (minimum_bet * away_odds) - minimum_bet
                else:
                    # Deduct the bet amount from the team's pot
                    betting_pots[team] -= minimum_bet
                # Reset draw streak for the team after placing a bet
                away_streaks[team] = 0

# Load the E0.csv file
file_path = 'E1.csv'  # Update this to your file path
e0_data = pd.read_csv(file_path)

# Start timing
start_time = time.time()

# Create an empty league table, betting results list, and draw streaks dictionary
league_table = {}
betting_results = []
win_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}
draw_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}
away_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}


# Initialize the current non-draw streak counters for each team
#current_non_draw_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

# Initialize the longest non-draw streak counters for each team
longest_non_draw_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

# Initialize current non-win streak counters for each team
current_non_win_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

# Initialize longest non-win streak counters for each team
longest_non_win_streaks = {team: 0 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

# Create betting pots for each team
betting_pots = {team: 2000 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

win_pots = {team: 2000 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}

home_win_pots = {team: 2000 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}
away_win_pots = {team: 2000 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}
draw_pots = {team: 2000 for team in pd.concat([e0_data['HomeTeam'], e0_data['AwayTeam']]).unique()}


totalTrades = 0

# Iterate through each row in the dataset, update the league table, draw streaks, and apply the betting system
for index, row in e0_data.iterrows():
    update_league_table(row, league_table)
    #print("home odds ", preprocess_odds(row)[0], "draw odds ", preprocess_odds(row)[1], "away odds ", preprocess_odds(row)[2])

    #betting_system(row, betting_pots, draw_streaks)
    #betting_system_home_wins(row, betting_pots, home_win_streaks)
    #betting_system_away(row, betting_pots, away_streaks)
    totalTrades = kegs_System(row, 0, league_table, current_non_win_streaks, win_streaks ,totalTrades)
    totalTrades = kegs_System(row, 1, league_table, current_non_win_streaks, win_streaks, totalTrades)
    #update_draw_streak(row, draw_streaks)
    update_non_draw_streak(row, draw_streaks, longest_non_draw_streaks)
    update_win_streak( row, win_streaks)
    update_away_streak(row, away_streaks)
    update_non_win_streak(row, current_non_win_streaks, longest_non_win_streaks)
   

    
# Convert the league table to a DataFrame for better visualization
league_table_df = pd.DataFrame.from_dict(league_table, orient='index')
league_table_df.sort_values(by=['Points', 'Goal Difference', 'Goals For'], ascending=False, inplace=True)

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Calculate total profit/loss from betting
total_profit = sum(betting_results)

# Display the league table, betting pots, and betting results
print(league_table_df)
print("Betting Pots:")
runningT=0
'''for team, pot in betting_pots.items():
    runningT+=pot
    print(f"{team}: {pot}")
    print(f"{'                          running total'} :{runningT}")
    '''


#print("Longest Non-Draw Streaks for Each Team:")
#for team, streak in longest_non_draw_streaks.items():
    #print(f"{team}: {streak} matches")

#print("Longest Non-Win Streaks for Each Team:")
#for team, streak in longest_non_win_streaks.items():
    #print(f"{team}: {streak} matches")

runningTotal = 0
for team, pot in win_pots.items():
    #print(f"Team: {team}, Pot: {round(pot)}")
    runningTotal+=pot 
    #print(round(runningTotal))

print(f"Execution time: {execution_time:.2f} seconds")
print(f"Total Betting Profit/Loss: { round( runningTotal - 40000) } units,  Total Trades = {totalTrades}")
#print(len(league_table))
