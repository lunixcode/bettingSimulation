import pandas as pd
from sklearn import preprocessing

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


label_encoder = preprocessing.LabelEncoder() 
df['HomeTeamForm'] = label_encoder.fit_transform(df['HomeTeamForm'])
df['AwayTeamForm'] = label_encoder.fit_transform(df['AwayTeamForm'])
df['Result'] = label_encoder.fit_transform(df['Result'])


from sklearn.model_selection import train_test_split

X = df[['HomeTeamForm', 'AwayTeamForm']] # Input features
y = df['Result']                        # Target outcome 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% for training, 20% for testing
