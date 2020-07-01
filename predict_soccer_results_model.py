import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

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

# # Multiple Linear Regression
# Home
print("\nLinear Regression")
print("Home")
X_home_sm = X_home = sm.add_constant(X_home)
model_home = sm.OLS(y_home, X_home_sm)
# print(model_home.fit().summary())

lm_home = LinearRegression()
lm_home.fit(X_home_train, y_home_train)

coef = lm_home.coef_
"""for cf in zip(X_home.columns, coef):
    print(cf[0], cf[1])
predict_home = lm_home.predict(X_home_test)"""

print(np.mean(cross_val_score(lm_home, X_home_train, y_home_train, scoring='neg_mean_absolute_error', cv=3)))

# Away
print("\nAway")
X_away_sm = X_away = sm.add_constant(X_away)
model_away = sm.OLS(y_away, X_away_sm)
# print(model_away.fit().summary())

lm_away = LinearRegression()
lm_away.fit(X_away_train, y_away_train)

coef_away = lm_away.coef_
"""for cf in zip(X_home.columns, coef):
    print(cf[0], cf[1])"""

predict_away = lm_away.predict(X_away_test)
print(np.mean(cross_val_score(lm_away, X_away_train, y_away_train, scoring='neg_mean_absolute_error', cv=3)))

# Lasso Regression
print("\nLasso")
print("Home")
lm_lasso_home = Lasso(alpha=0.1)
lm_lasso_home.fit(X_home_train, y_home_train)
print(np.mean(cross_val_score(lm_lasso_home, X_home_train, y_home_train, scoring='neg_mean_absolute_error', cv=3)))

print("\nAway")
lm_lasso_away = Lasso(alpha=0.1)
lm_lasso_away.fit(X_away_train, y_away_train)
print(np.mean(cross_val_score(lm_lasso_away, X_away_train, y_away_train, scoring='neg_mean_absolute_error', cv=3)))

# Random forest
print("\nRandom Forest")
print("Home")
rf_home = RandomForestRegressor()
print(np.mean(cross_val_score(rf_home, X_home_train, y_home_train, scoring='neg_mean_absolute_error', cv=3)))

print("\nAway")
rf_away = RandomForestRegressor()
print(np.mean(cross_val_score(rf_away, X_away_train, y_away_train, scoring='neg_mean_absolute_error', cv=3)))
# Tune models GridsearchCV
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}

gs_home = GridSearchCV(rf_home, parameters, scoring='neg_mean_absolute_error', cv=3)
gs_home.fit(X_home_train, y_home_train)

print("Gridsearch Home")
print(gs_home.best_score_)
print(gs_home.best_estimator_)

gs_away = GridSearchCV(rf_home, parameters, scoring='neg_mean_absolute_error', cv=3)
gs_away.fit(X_home_train, y_home_train)
print("Gridsearch Away")
print(gs_away.best_score_)
print(gs_away.best_estimator_)

# Test ensembles
tpred_home_lm = lm_home.predict(X_home_test)
tpred_home_lml = lm_lasso_home.predict(X_home_test)
tpred_home_rf = gs_home.best_estimator_.predict(X_home_test)

print("\nMSE Home team")
print(mean_squared_error(y_home_test, tpred_home_lm),
      mean_squared_error(y_home_test, tpred_home_lml),
      mean_squared_error(y_home_test, tpred_home_rf),
      mean_squared_error(y_home_test, (tpred_home_lml + tpred_home_rf)/2))
print("R_squared Home")
print(r2_score(y_home_test, tpred_home_lm),
      r2_score(y_home_test, tpred_home_lml),
      r2_score(y_home_test, tpred_home_rf),
      r2_score(y_home_test, (tpred_home_lml + tpred_home_rf)/2))

pickl = {'model': gs_home.best_estimator_}
pickle.dump(pickl, open('model_home_file' + ".p", "wb"))

file_name = "model_home_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print(model.predict(np.array(list(X_home_test.iloc[1, :])).reshape(1, -1))[0])

list(X_home_test.iloc[1, :])

# Away la
tpred_away_lm = lm_away.predict(X_away_test)
tpred_away_lml = lm_lasso_away.predict(X_away_test)
tpred_away_rf = gs_away.best_estimator_.predict(X_away_test)

print("\nMSE Away team")
print(mean_squared_error(y_away_test, tpred_away_lm),
      mean_squared_error(y_away_test, tpred_away_lml),
      mean_squared_error(y_away_test, tpred_away_rf),
      mean_squared_error(y_away_test, (tpred_away_lml + tpred_away_rf)/2))
print("R_squared Away team")
print(r2_score(y_away_test, tpred_away_lm),
      r2_score(y_away_test, tpred_away_lml),
      r2_score(y_away_test, tpred_away_rf),
      r2_score(y_away_test, (tpred_away_lml + tpred_away_rf)/2))

pickl = {'model': gs_away.best_estimator_}
pickle.dump(pickl, open('model_away_file' + ".p", "wb"))

file_name = "model_away_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print(model.predict(np.array(list(X_away_test.iloc[1, :])).reshape(1, -1))[0])

list(X_away_test.iloc[1, :])
