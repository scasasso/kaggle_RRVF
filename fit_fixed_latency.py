import pickle
from play import *
from copy import deepcopy
import defines

if __name__ == "__main__":

    # Load data
    df_train, df_test = load_data()

    model_dict = {}
    scores = []
    for dlt, df in df_train.groupby('delta'):

        print('Delta = {0}, {1}'.format(dlt, len(df)))

        # Fit the model
        model, (X_val, y_val), cols = fit(df, feat_list=defines.SEL_FEATURES, model_name='rfr')

        model_dict[dlt] = deepcopy(model)

        _, score = get_preds_and_score(model, X_val, y_val)
        scores.append((score, len(df)))
        print(' score = {0:.3f}'.format(score))

        # Predictions for the test dataset
        df_test_tmp = transform(df_test.loc[df_test['delta'] == dlt, :].copy(), defines.SEL_FEATURES)

        # Fill columns which don't appear in the test set
        for c in cols:
            if c not in df_test_tmp.columns:
                df_test_tmp[c] = 0

        # Predictions on the test set
        X_test = df_test_tmp.loc[:, df_test_tmp.columns != defines.TARGET_COL].as_matrix()
        y_test = df_test_tmp.loc[:, df_test_tmp.columns == defines.TARGET_COL].as_matrix().ravel()
        y_pred, _ = get_preds_and_score(model, X_test, y_test)
        idxs = df_test_tmp.index.values
        df_test.iloc[idxs, df_test.columns.get_loc('visitors')] = y_pred

    # Dump the models
    pickle.dump(model_dict, open('model_dict_{}.pkl'.format(defines.TAG), 'wb'))

    # Print the scores
    w_score = np.average([s for s, _ in scores], weights=[w for _, w in scores])
    score = np.average([s for s, _ in scores])
    print('Weighted score = {0:.3f}'.format(w_score))
    print('Unweighted score = {0:.3f}'.format(score))

    # model_dict = pickle.load(open('model_dict.pkl', 'rb'))

    # Create the submission file
    df_test['id'] = df_test['label']
    df_test[['id', 'visitors']].to_csv('sub_{}.csv'.format(defines.TAG), index=False)
