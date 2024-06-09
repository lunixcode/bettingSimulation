class TeamStreaks:
    def __init__(self):
        self.streaks = {'home': {}, 'away': {}, 'non_draw': {}}

    def update_streak(self, team, result, game_type):
        if team not in self.streaks[game_type]:
            self.streaks[game_type][team] = 0

        if game_type != 'non_draw':
            if (game_type == 'home' and result == 'H') or (game_type == 'away' and result == 'A'):
                self.streaks[game_type][team] = 0  # Team won, reset streak
            else:
                self.streaks[game_type][team] += 1  # Team didn't win, increment streak
        else:
            if result == 'D':
                self.streaks['non_draw'][team] = 0  # Game was a draw, reset streak
            else:
                self.streaks['non_draw'][team] += 1  # No draw, increment streak

    def get_streak(self, team, game_type):
        return self.streaks[game_type].get(team, 0)

for index, row in data.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']

    team_streaks.update_streak(home_team, result, 'home')
    team_streaks.update_streak(away_team, result, 'away')
    team_streaks.update_streak(home_team, result, 'non_draw')
    team_streaks.update_streak(away_team, result, 'non_draw')


def betting_logic(row, betting_pots, team_streaks, minimum_bet=100):
    for team in [row['HomeTeam'], row['AwayTeam']]:
        pot = betting_pots.get(team, 0)
        non_draw_streak = team_streaks.get_streak(team, 'non_draw')

        if non_draw_streak >= 5 and pot >= minimum_bet:
            # Implement betting logic here, e.g., betting on a draw
            # Update betting pots based on the outcome
            ...
