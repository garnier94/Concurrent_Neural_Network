# This model are used to compare with classical model
from sklearn.ensemble import RandomForestRegressor
import pdb
import pandas as pd

def split_label_features(X, target= "proportion", horizon = 6):
    lab = X[target]
    feat = X[[target +'_shift_' + str(horizon), target +'_shift_' + str(horizon+1), 'price', 'margin']]
    return feat, lab

def compute_basic_model(result, X_test, horizon):
    result.write('Null MAPE %.2f \n' % (100 * sum(abs(X_test['proportion'])) / sum(X_test['proportion'])))
    result.write('Simple_shift MAPE %.2f\n' % (
                100 * sum(abs(X_test['proportion_shift_'+str(horizon)] - X_test['proportion'])) / sum(X_test['proportion'])))
    result.write('Simple 2-mean MAPE %.2f\n' % (100 * sum(
        abs(X_test['proportion_shift_'+str(horizon)] / 2 + X_test['proportion_shift_'+str(horizon+1)] / 2 - X_test['proportion'])) / sum(
        X_test['proportion'])))


def compute_random_forest(result, X_train, X_valid, X_test, compute_scaled_RF = False):
    train_features, train_labels = split_label_features(X_train)
    valid_features, valid_labels = split_label_features(X_valid)
    test_features, test_labels = split_label_features(X_test)

    models = [RandomForestRegressor(), RandomForestRegressor(max_depth = 4), RandomForestRegressor(min_samples_leaf = 10), RandomForestRegressor(min_samples_leaf = 30)]
    err_liste =[]
    m_list =[]
    for model in models:
        model = RandomForestRegressor()
        model.fit(train_features, train_labels)
        y_pred_valid = model.predict(valid_features)
        err_liste.append( sum(abs( y_pred_valid - valid_labels)))
        m_list.append(model)
    min_model =err_liste.index(min(err_liste))
    model = m_list[min_model]
    print("meilleur modele RF :" + str(min_model))
    y_pred = model.predict(test_features)
    result.write('Random Forest MAPE %.2f\n' % (
                100 * sum(abs( y_pred - test_labels)) / sum(test_labels)))
    if compute_scaled_RF:
        df_pred = pd.DataFrame(data = y_pred, index=test_features.index, columns =['prev'])
        gpdf =  df_pred.groupby(level= 1).sum()
        gpdf.columns = ['sum_pred']
        df_pred = df_pred.join(gpdf)
        result.write('Scaled Random Forest MAPE %.2f\n' % (
                100 * sum(abs(df_pred['prev']/df_pred['sum_pred'] - test_labels)) / sum(test_labels)))