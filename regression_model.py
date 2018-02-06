import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

import pdb

def loaddata():
    # loads cdc data and joins old and new data sets based upon common fields
    t2015_clinic = pd.read_csv('WHO_NREVSS_Clinical_Labs.csv', header=1)
    t2015_phlabs = pd.read_csv('WHO_NREVSS_Public_Health_Labs.csv', header=1)
    t_pre2015 = pd.read_csv('WHO_NREVSS_Combined_prior_to_2015_16.csv', header=1)
    
    acols_pre  = [
                  'A (2009 H1N1)', 
                  'A (H1)', 'A (H3)', 
                  'A (Subtyping not Performed)', 
                  'A (Unable to Subtype)'
                 ]
    
    acols_post = [
                  'A (2009 H1N1)', 
                  'A (H3)', 
                  'A (Subtyping not Performed)'
                 ]
    
    bcols_post = [
                  'B', 
                  'BVic', 
                  'BYam'
                 ]
    
    # exploration
    # post_A_percent = t2015_phlabs[acols_post].sum(axis=1)/t2015_phlabs['TOTAL SPECIMENS']
    # post_B_percent = t2015_phlabs[bcols_post].sum(axis=1)/t2015_phlabs['TOTAL SPECIMENS']
    
    A_inter = list(set(acols_pre).intersection(set(acols_post)))
    A_comb = pd.concat([t_pre2015[A_inter], t2015_phlabs[A_inter]], axis=0)
    
    B_comb = pd.concat([t_pre2015['B'], t2015_phlabs[bcols_post].sum(axis=1)], axis=0)
    B_comb.name = 'B'
    
    date_cols =  [
                  'YEAR',
                  'WEEK'
                 ]
    dates = pd.concat([t_pre2015[date_cols], t2015_clinic[date_cols]], axis=0)
    
    total_specimens_post = t2015_clinic['TOTAL SPECIMENS'] + t2015_phlabs['TOTAL SPECIMENS']
    total_spec = pd.concat([t_pre2015['TOTAL SPECIMENS'], total_specimens_post], axis=0)
    
    per_pos = pd.concat([t_pre2015['PERCENT POSITIVE'], t2015_clinic['PERCENT POSITIVE']], axis=0)

    df = pd.concat([dates, total_spec, A_comb, B_comb, per_pos], axis=1)
    df.index = range(df.shape[0]) # reindex after concatenating
    return df

def lookback(dataf, lookback_cols, iters):
    # pivots data to add previous weeks information to current week. iter defines the number of
    # weeks to look back
    for i in range(1,iters+1):
        new_data = dataf[lookback_cols].iloc[:-i]
        new_data.index = dataf.index[i:]
        new_data.columns = [col + 'lb' + str(i) for col in lookback_cols]
        dataf = pd.concat([dataf,new_data], axis=1)

    return dataf

def difference(dataf, diff_cols, iters):
    # generates week to week delta for a given week. iters is the number of week deltas to add
    for i in range(1,iters+1):
        diff_data = dataf[diff_cols].diff(periods=i)
        diff_data.columns = [col + 'diff' + str(i) for col in diff_cols]
        dataf = pd.concat([dataf, diff_data], axis=1)

    return dataf

def target_gen(dataf, col_name, forward_window=1):
    target = dataf[col_name].iloc[forward_window:]
    target.index = dataf.index[:-forward_window]
    target.name = 'target'
    return pd.concat([dataf, target], axis=1)
    
def updown_eval(df, y_test_pred, test_idx, target_col_name):
    # evaluates if model accurately predicts up or down trend in flu activity

    # pdb.set_trace()
    actual = df['updown'].loc[test_idx]
    pred = (y_test_pred - \
            df[target_col_name].loc[test_idx]).map(lambda x: 1 if x>0 else 0)


    percent_correct = float((actual==pred).sum())/pred.shape[0]
    return percent_correct

if __name__=="__main__":
    df = loaddata()
    data_cols = [
                 'TOTAL SPECIMENS',
                 'A (H3)',
                 'A (2009 H1N1)',
                 'A (Subtyping not Performed)',
                 'B',
                 'PERCENT POSITIVE'
                ]

    df = lookback(df, data_cols, 8)
    df = difference(df, data_cols, 8)
    target_col_name = 'PERCENT POSITIVE'
    # target_col_name = 'TOTAL SPECIMENS'

    df = target_gen(df, target_col_name, 1) 

    df['updown'] = (df.target-df[target_col_name]).map(lambda x: 1 if x>0 else 0)

    df.dropna(axis=0, inplace=True)

    X = df[df.columns[2:-2]]

    # uses poly function to generate interaction terms for model building
    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)
    y = df['target']

    # generate testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # select best features for model building using f-regression
    selector = SelectKBest(f_regression, k=25).fit(X_train,y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    
    # generate the model! all models were roughly the same accuracy
    model = LinearRegression()
    # model = SVR(kernel='rbf', C=1, verbose=True)
    # model = GradientBoostingRegressor(loss='huber', n_estimators=450, max_depth=4)
    model.fit(X_train, y_train)
    score_test = model.score(X_test, y_test)

    print "model score: %s " % score_test

    y_pred = model.predict(X_test)


    # predict this week's flu activity using last week's data
    last_week = df.iloc[-1]
    last_week_input = last_week[df.columns[2:-2]]
    this_week_percent = model.predict(selector.transform(poly.transform([last_week_input])))
    print 'this weeks projected flu activity: %s' % (this_week_percent)

    all_pred = model.predict(selector.transform(X))
    ud_acc = updown_eval(df, y_pred, y_test.index, target_col_name)
    print "accuracy of increasing or decreasing: %s " % (ud_acc)
    
    # plot results
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test)
    plt.show()

    plt.hist((y_test-y_pred), bins=40)
    plt.show()

    plt.plot(range(X.shape[0]), all_pred)
    plt.plot(range(X.shape[0]), y)
    plt.show()
