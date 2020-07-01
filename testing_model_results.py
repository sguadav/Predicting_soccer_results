import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

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

# Testing
model_home = pd.read_pickle(r'model_home_file.p')
predict_home = model_home['model'].predict(X_home_test)

model_away = pd.read_pickle(r'model_away_file.p')
predict_away = model_away['model'].predict(X_away_test)

count_home_correct = 0
count_home_incorrect = 0

count_away_correct = 0
count_away_incorrect = 0

print("Predict | Actual | Same or not")
for i in range(len(y_home_test)):
    if np.round(predict_home[i]) == y_home_test[i]:
        print(math.floor(np.round(predict_home[i])), "|", y_home_test[i], "|", 1)
        count_home_correct += 1
    else:
        print(math.floor(np.round(predict_home[i])), "|", y_home_test[i], "|", 0)
        count_home_incorrect += 1

print("\nPredict | Actual | Same or not")
for i in range(len(y_away_test)):
    if np.round(predict_away[i]) == y_away_test[i]:
        print(math.floor(np.round(predict_away[i])), "|", y_away_test[i], "|", 1)
        count_away_correct += 1
    else:
        print(math.floor(np.round(predict_away[i])), "|", y_away_test[i], "|", 0)
        count_away_incorrect += 1

# Success measurement
print("\nHome team Results")
print("The for this model accuracy was:", round((count_home_correct / (count_home_incorrect + count_home_correct)) * 100),
      "%")

print("\nAway team Results")
print("The for this model accuracy was:", round((count_away_correct / (count_away_incorrect + count_away_correct)) * 100),
      "%")

print("\nTotal Accuracy of the model: ", ((94+26)/2))
