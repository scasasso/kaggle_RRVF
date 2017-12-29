from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from utils import *
import defines


def load_data():

    print('Loading the data...')
    df_train = pd.read_csv('data/train.tsv', sep='\t')
    df_test = pd.read_csv('data/test.tsv', sep='\t')

    return df_train, df_test


def fillnas(df):

    # NaN -> 0
    for col in df.columns:
        if (pd.isnull(df[col])).sum() > 0:
            print('Column {col} has NaN which will be replaced with zeros...'.format(col=col))
            df[col] = df[col].fillna(0.)

    return df


def transform(df, feat_set=None):

    if feat_set is not None:
        df = df[feat_set + [defines.TARGET_COL]].copy()

    df = fillnas(df)
    df = pd.get_dummies(df)

    return df


def train_test_val_split(df_train, f_val=0.1, df_test=None):

    X = df_train.loc[:, df_train.columns != defines.TARGET_COL].as_matrix()
    y = df_train.loc[:, df_train.columns == defines.TARGET_COL].as_matrix().ravel()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=f_val, random_state=42, shuffle=False)
    if df_test is None:
        return X_train, y_train, X_val, y_val

    X_test = df_test.loc[:, df_test.columns != defines.TARGET_COL].as_matrix()
    y_test = df_test.loc[:, df_test.columns == defines.TARGET_COL].as_matrix().ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_preds_and_score(model, X, y):

    # Validation
    y_pred = model.predict(X)
    y_pred = np.array([max(0., p) for p in y_pred])
    score = rmsle(y, y_pred)

    return y_pred, score


def fit(df_train, feat_list, model_name='rfr', do_grid=False, f_val=0.1):

    # Transform
    print('One-hot encoding...')
    df_train = transform(df_train, feat_set=feat_list)
    print('... {} features in total'.format(len(df_train.columns) - 1))

    print('Splitting into train and val datasets...')
    X_train, y_train, X_val, y_val = train_test_val_split(df_train, f_val=f_val)
    print('Size:\n train {train}\n val {val}'.format(train=len(y_train), val=len(y_val)))

    # Build scorer
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    print('Building the model...')

    model, params = None, dict()

    if model_name == 'xgbr':

        # Boosted Tree
        model = XGBRegressor(n_estimators=30,
                             max_depth=20,
                             colsample_bytree=0.8,
                             gamma=0.1,
                             nthread=-1,
                             silent=False,
                             seed=42)

        params = {'n_estimators': [30, 50],
                  'learning_rate': [0.05, 0.1],
                  'gamma': [0., 0.1],
                  'colsample_bytree': [0.8, 1.]}

    elif model_name == 'rfr':

        # Random Forest
        model = RandomForestRegressor(n_estimators=50,
                                      max_features=0.8,
                                      min_samples_split=100,
                                      random_state=42,
                                      verbose=10,
                                      n_jobs=-1)

        params = {'n_estimators': [30, 50],
                  'min_samples_split': [50, 100],
                  'max_features': [0.5, 0.8]}

    elif model_name == 'etr':

        # Random Forest
        model = ExtraTreesRegressor(n_estimators=200,
                                    max_features=None,
                                    min_samples_split=2,
                                    random_state=42,
                                    verbose=10,
                                    n_jobs=-1)

        params = {'n_estimators': [30, 50],
                  'min_samples_split': [50, 100],
                  'max_features': [0.5, 0.8]}

    else:
        raise NotImplementedError('Model {} not implemented in function fit'.format(model_name))

    if do_grid:

        # GridSearchCV
        print('Will do grid search with the following grid:')
        pprint(params)
        grid = GridSearchCV(model, param_grid=params, scoring=rmsle_scorer,
                            error_score=100., n_jobs=-1, verbose=10)
        grid.fit(X_train, y_train)

        # Save best model
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)

    return model, (X_val, y_val), df_train.columns
