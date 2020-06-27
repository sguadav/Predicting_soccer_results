import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('eda_data/Premier_eda_data.csv')

# Get dummy columns
df_dum = pd.get_dummies(df)

# Train and Testing data
X_home = df_dum.drop('Results_1', axis=1)
y_home = df_dum.Results_1.values
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)

X_away = df_dum.drop('Results_2', axis=1)
y_away = df_dum.Results_2.values
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)

# Multiple Linear Regression

# Lasso Regression

# Radom forest

# Tune models GridsearchCV

# Test ensembles