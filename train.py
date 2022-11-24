import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")
X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']

stnd_scaler = StandardScaler()
X = stnd_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=True)
del X, y

params = {"objective":"reg:linear", 'colsample_bytree': 0.8130879553572901, 'gamma': 1.6079127959890107, 'learning_rate': 1.9636154628539149, 'max_depth': int(4), 
          'min_child_weight': 3.0, 'reg_alpha': 43.0, 'reg_lambda': 0.0977374813321206, 'subsample': 0.9992807942435554, 'eval_metric': ["rmse"]}

model = xgb.XGBRegressor(n_estimators = int(430.52110343125906), max_depth = int(params['max_depth']), gamma = params['gamma'], reg_alpha = params['reg_alpha'],
                          min_child_weight=int(params['min_child_weight']), colsample_bytree = params['colsample_bytree'], learning_rate = params['learning_rate'],
                          subsample = params['subsample'], reg_lambda = params['reg_lambda'])

evaluation = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train, eval_set = evaluation, eval_metric=["rmse"], early_stopping_rounds=10, verbose=2)

preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)
print ("RMSE: {}".format(rmse))

test = pd.read_csv("test.csv")
ids = test[['Id']]
X = test.drop(['Id', 'MSSubClass_SC150'], axis=1)

X = stnd_scaler.transform(X)

preds = model.predict(X)

test['SalePrice'] = np.expm1(preds)
test['Id'] = ids
test = test[['Id', 'SalePrice']]

test.to_csv("submission.csv", index=False)