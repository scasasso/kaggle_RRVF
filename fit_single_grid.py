import pickle
from play import *
from copy import deepcopy
import defines

if __name__ == "__main__":

    # Load data
    df_train, df_test = load_data()

    # Fit the model
    model, (X_val, y_val), cols = fit(df_train, feat_list=defines.SEL_FEATURES, model_name='rfr', do_grid=False)

    # Print the score
    _, score = get_preds_and_score(model, X_val, y_val)
    print(' score = {0:.3f}'.format(score))

    # Dump the models
    pickle.dump(model, open('model_{}.pkl'.format(defines.TAG), 'wb'))

    # Predictions for the test dataset
    labels = df_test['label'].as_matrix()
    df_test = transform(df_test.copy(), defines.SEL_FEATURES)

    # Predictions on the test set
    X_test = df_test.loc[:, df_test.columns != defines.TARGET_COL].as_matrix()
    y_test = df_test.loc[:, df_test.columns == defines.TARGET_COL].as_matrix().ravel()
    y_pred, _ = get_preds_and_score(model, X_test, y_test)
    df_test['visitors'] = y_pred
    df_test['id'] = labels

    # Create the submission file
    df_test[['id', 'visitors']].to_csv('sub_{}.csv'.format(defines.TAG), index=False)
