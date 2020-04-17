import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

# function to calculate performance
def calc_simres(data, alpha, expr, int_balance=1, delay=1):
    simres = pd.DataFrame()
    simres.index.name = 'Dates'
    simres['dailytvr'] = alpha.replace(np.nan, 0).diff().abs().sum(axis=1) / alpha.abs().shift(1).sum(axis=1).replace(np.nan, 0)
    numdays = data['resRet2'].shape[0]
    simres['dailytvr'].replace(np.inf, np.nan, inplace=True)
    simres['tvr'] = simres['dailytvr'].rolling(252, min_periods=1).mean()
    stockpnl = alpha * data['resRet2']
    simres['dailypnl'] = (alpha.shift(delay) * data['resRet2']).sum(axis=1)
    simres['numstocks'] = stockpnl.notnull().sum(axis=1)
    simres['cumpnl'] = simres['dailypnl'].cumsum()
    simres['dailyret'] = 2 * simres['dailypnl'] / (alpha.abs().sum(axis=1)).replace(0, np.nan)
    simres['sharpe'] = simres['dailypnl'].mean() / simres['dailypnl'].std() * 252 ** 0.5
    simres['vol'] = simres['dailyret'].std() * 252 ** 0.5
    simres['booksize'] = alpha.abs().sum(axis=1)
    simres['ret'] = simres['dailyret'].mean() * 252
    simres['equity_curve'] = int_balance + simres['cumpnl']
    simres['max_index'] = simres['equity_curve'].rolling(numdays, min_periods=1).max()
    simres['dd'] = ((simres['max_index']-simres['equity_curve']) / int_balance).rolling(numdays, min_periods=1).max()
    simres['shortsize'] = alpha[alpha < 0].sum(axis=1)
    simres['longsize'] = alpha[alpha > 0].sum(axis=1)
    simres['shortstk'] = (alpha < 0).sum(axis=1)
    simres['longstk'] = (alpha > 0).sum(axis=1)
    simres['name'] = expr
    simres['delay'] = delay
    return simres

# function to enforce market neutral
def op_balance(data, preA, booksize=1):
    preA = pd.DataFrame(preA)
    preA[data['filter'] == 0] = np.nan
    preA[np.isinf(preA)] = np.nan
    preA = preA.sub(preA.mean(axis=1), axis=0)
    preA = 2 * booksize * preA.divide(preA.abs().sum(axis=1), axis=0)
    return preA

# plot the graph of performance
def plot_pnl(simres):
    fig = plt.figure(figsize=(12, 7))
    label = 'sp: ' + str(round(simres['sharpe'].values[-1], 2)) + \
            ' ret: ' + str(round(simres['ret'].values[-1], 2)) + \
            ' vol: ' + str(round(simres['vol'].values[-1], 2)) + \
            ' tvr: ' + str(round(simres['tvr'].values[-1], 2)) + \
            ' dd: ' + str(round(simres['dd'].values[-1], 2)) + \
            ' bks ' + str(round(simres['longsize'].rolling(252).mean().values[-1], 1)) + \
            'x' + str(-round(simres['shortsize'].rolling(252).mean().values[-1], 1)) + \
            ' stk: ' + str(int(simres['longstk'].rolling(252).mean().values[-1])) + \
            '-' + str(int(simres['shortstk'].rolling(252).mean().values[-1])) + \
            ' lag: ' + str(simres['delay'].values[-1]) + '\n' + simres['name'].values[-1]
    plt.plot(simres['equity_curve'], label=label, color='red', linewidth=1)
    plt.legend(loc='upper left')
    plt.grid(linestyle='--')
    plt.show()
    plt.close(fig)

# helper function to calcuate rolling mean
def ts_mean(inputs, window):
    return inputs.rolling(window, min_periods=1).mean()

# read h5 data
def read_h5(fname):
    f = h5py.File(fname, 'r')
    data = {}
    for k, v in f.items():
        data[k] = pd.DataFrame(v.value.T)
    return data

# grid search for parameters
def parameter_search(X_train, y_train):
    params = {
        'n_estimators': [20, 50],
        'max_depth': [3, 6, 9],
        "learning_rate": [0.1, 0.5, 1],
        'subsample': [0.8, 1],
        "colsample_bytree": [0.8, 1],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0, 0.5, 1],
        'gamma': [0, 0.5, 1],
    }
    model = xgb.XGBRegressor(nthreads=-1)
    gs = GridSearchCV(model, params, cv=3, scoring='neg_mean_absolute_error', verbose=1)
    gs.fit(X_train, y_train)
    return gs.best_params_

# xgboost algorithm
def ts_xgboost(x0, y0):
    features = [i for i in x0.keys()]
    x1 = []
    y_shape = y0.shape
    for i in features:
        x0[i].replace([np.inf, -np.inf], np.nan, inplace=True)
        if len(x1) != 0:
            x1 = np.append(x1, x0[i].values.reshape(-1, 1), axis=1)
        else:
            x1 = x0[i].values.reshape(-1, 1)
    x = pd.DataFrame(x1, columns=f)
    x['y'] = y0.values.reshape(-1, )
    x.dropna(axis=0, how='any', inplace=True)
    y = x['y']
    x.drop(['y'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
    # param = parameter_search(X_train, y_train)
    # print(param)
    model = xgb.XGBRegressor(learning_rate=1, max_depth=3, n_estimators=50, gamma=0, reg_alpha=0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
    model.fit(X_train, y_train)
    x1[np.isnan(x1)] = 0
    x1 = pd.DataFrame(x1)
    x1.columns = [features]
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_entire_pred = model.predict(x1)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print("RMSE train: %f" % (rmse_train))
    print("RMSE test: %f" % (rmse_test))
    output = pd.DataFrame(data=y_entire_pred[~np.isnan(x1).any(axis=1)].reshape(y_shape), index=y0.index, columns=y0.columns)
    xgb.plot_tree(model, num_trees=0)
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.show()
    xgb.plot_importance(model)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    return output


if __name__ == '__main__':
    fname = 'D:\Project\data.mat'
    data = read_h5(fname)
    data['resRet2'][data['filter'] == 0] = np.nan

    f = {
        "vol": np.log(data['volume']/ts_mean(data['volume'], 20)),
        "Ret20": -ts_mean(data['resRet2'], 20),
        'adjSplClo': data['adjSplClo'],
        'bB2P': data['bB2P'],
        'bLev': data['bLev'],
        'bBetaNL': data['bBetaNL'],
        'bDivYld': data['bDivYld'],
        'bErnYld': data['bErnYld'],
        'bMomentum': data['bMomentum'],
        'bResVol': data['bResVol'],
        'bSizeNL': data['bSizeNL'],
        'modelBeta': data['modelBeta'],
    }

    expr = "ts_xgboost(f, data['resRet2'].shift(-1))"
    alpha = eval(expr)
    simres = calc_simres(data, op_balance(data, alpha), expr, delay=1)
    plot_pnl(simres)

