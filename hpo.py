import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")
X = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']

stnd_scaler = StandardScaler()
X = stnd_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=True)
del X, y

space={ 'max_depth': hp.quniform("max_depth", 3, 50, 1),
        'learning_rate': hp.uniform ('learning_rate', 1,9),
        'subsample': hp.uniform('subsample', 0,1),
        'gamma': hp.uniform ('gamma', 1,30),
        'reg_alpha' : hp.quniform('reg_alpha', 40,250,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1),
        'n_estimators': hp.uniform('n_estimators', 100, 1000),
        'seed': 0
    }

def objective(space):
    rxgb=xgb.XGBRegressor(n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'], reg_alpha = space['reg_alpha'],
                          min_child_weight=int(space['min_child_weight']), colsample_bytree = space['colsample_bytree'], learning_rate = space['learning_rate'],
                          subsample = space['subsample'], reg_lambda = space['reg_lambda'])
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    rxgb.fit(X_train, y_train,
            eval_set=evaluation, eval_metric=["rmse"],
            early_stopping_rounds=10, verbose=2)
    

    pred = rxgb.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    print ("RMSE: {}".format(rmse))
    return {'loss': rmse, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 5000,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)