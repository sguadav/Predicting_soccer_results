import pandas as pd
"""
This file imports, separates and cleans the data ready to be used

Data source: https://www.kaggle.com/analystmasters/world-soccer-live-data-feed?select=analystm_mode_1_v1.csv
"""


def get_leagues_data():
    df = pd.read_csv('soccer_data/analystm_mode_1_v1.csv')

    # Separating into top 5 European leagues
    premier_league = df[df['League_type_country'] == 'england']
    calcio = df[df['League_type_country'] == 'italy']
    laliga = df[df['League_type_country'] == 'spain']
    bundesliga = df[df['League_type_country'] == 'germany']
    france_ligue = df[df['League_type_country'] == 'france']

    # Sent to data cleaning process
    premier_league = data_clean(premier_league)
    calcio = data_clean(calcio)
    laliga = data_clean(laliga)
    bundesliga = data_clean(bundesliga)
    france_ligue = data_clean(france_ligue)

    # To save the Dataframes into csv
    premier_league.to_csv('cleaned_data\Premier_league.csv')
    calcio.to_csv('cleaned_data\calcio.csv')
    laliga.to_csv('cleaned_data\laliga.csv')
    bundesliga.to_csv('cleaned_data/bundesliga.csv')
    france_ligue.to_csv('cleaned_data/france_ligue.csv')

    return premier_league, calcio, laliga, bundesliga, france_ligue


def data_clean(df):
    # Last game info split
    last_game_info = df['Detail_H2H'].apply(lambda x: x.split('_')[0])

    # Adding last game info
    df['H2H_Side'] = last_game_info.apply(lambda x: x.split('/')[1])
    df['H2H_Outcome'] = last_game_info.apply(lambda x: x.split('/')[2])
    df['H2H_Goals_Home'] = last_game_info.apply(lambda x: x.split('/')[3])
    df['H2H_Goals_Away'] = last_game_info.apply(lambda x: x.split('/')[4])

    # Adding Goal Difference to csv
    df['Goals_scored_diff'] = df['Goals_Scored_1'] - df['Goals_Scored_2']
    df['Goals_Rec_diff'] = df['Goals_Rec_1'] - df['Goal_Rec_2']
    df['Goals_Diff_diff'] = df['Goals_Diff_1'] - df['Goals_Diff_2']

    # Adding the difference between match result
    df['Diff_Goal_Match'] = df['Results_1'] - df['Results_2']


    # Columns deselection
    df = df.drop(['Votes_for_Home', 'Detail_H2H', 'Votes_for_Draw', 'Votes_for_Away', 'Weekday', 'Day', 'Month'],
                 axis=1)
    df = df.drop(['Hour', 'Minute', 'Total_Bettors', 'Bet_Perc_on_Home', 'Bet_Perc_on_Draw', 'Bet_Perc_on_Away',
                  'Team_1_Found', 'Team_2_Found'], axis=1)
    df = df.drop(['Win_Perc_1', 'Win_Perc_2', 'Draw_Perc_1', 'Draw_Perc_2', 'Jumps', 'Odds_Home', 'Odds_Draw',
                  'Odds_Away'], axis=1)
    df = df.drop(['Country_1', 'Country_2', 'Indices_home', 'Indices_draw', 'Indices_away', 'Year', 'Total_teams',
                  'Max_points'], axis=1)
    df = df.drop(['Min_points', 'Rank_1', 'Rank_2', 'Number_of_H2H_matches'], axis=1)

    return df


if __name__ == '__main__':
    get_leagues_data()
