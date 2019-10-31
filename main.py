import pandas as pd
import os.path as ospath
import numpy as np
from os import makedirs
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import train_test_split


def perform_data_scaling(x_train, x_test):
    scaler = RobustScaler(quantile_range=(20.0, 80.0))
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def find_outliers(x):
    outlier_indices = np.zeros(x.shape[0], dtype=np.bool)
    isolation_forest = IsolationForest(contamination="auto", behaviour="new")
    isolation_forest.fit(x)
    predictions = isolation_forest.predict(x)
    outlier_indices[predictions == 1] = 1
    return outlier_indices


def main():
    output_pathname = "output"
    output_filepath = ospath.join(output_pathname, "out.csv")
    training_data_dir = ospath.join("data", "training")
    testing_data_dir = ospath.join("data", "testing")

    # read the data
    train_data_x = pd.read_csv(ospath.join(training_data_dir, "X_train.csv"), delimiter=",").drop(["id"], axis=1)
    train_data_y = pd.read_csv(ospath.join(training_data_dir, "y_train.csv"), delimiter=",")["y"]

    test_data_x = pd.read_csv(ospath.join(testing_data_dir, "X_test.csv"), delimiter=",")
    test_data_ids = test_data_x["id"]  # Get the ids for processing later on
    test_data_x = test_data_x.drop(["id"], axis=1)

    # separate the data between age and features and convert them into values (required after using pandas)
    x_train_orig = train_data_x.fillna(train_data_x.median()).values
    y_train_orig = train_data_y.values
    x_test_orig = test_data_x.fillna(test_data_x.median()).values

    # Preprocessing step #1: Whitening of data
    x_train_whitened, x_test_whitened = perform_data_scaling(x_train_orig, x_test_orig)

    # PCA step #1
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_train_whitened)
    print(pca.explained_variance_ratio_)
    print("\n")

    # Preprocessing step #2: Outlier detection and removal

    outlier_indices = find_outliers(x_pca)
    x_train_whitened = x_train_whitened[outlier_indices]
    y_train_orig = y_train_orig[outlier_indices]

    # # PCA step #2
    # pca = PCA()
    # pca.fit(x_train_whitened)
    # print(pca.explained_variance_ratio_)
    # print("\n")

    # Preprocessing step #3/training: XGBoost
    np.random.seed(1)

    X_train, X_eval, y_train, y_eval = train_test_split(x_train_whitened, y_train_orig, test_size=0.2, random_state=100)
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    deval = xgb.DMatrix(data=X_eval, label=y_eval)
    dtest = xgb.DMatrix(x_test_whitened)
    param = dict({'max_depth': 5,
                  'eta': 0.02,
                  'objective': 'reg:squarederror',
                  'reg_alpha': 2, 'reg_lambda': 0.5,
                  'colsample_bytree': 0.3})
    param['eval_metric'] = 'rmse'
    evallist = [(deval, 'eval'), (dtrain, 'train')]
    bst = xgb.train(dtrain=dtrain, params=param, evals=evallist, num_boost_round=500)

    # Do the prediction
    # y_predict = lasso_optimal.predict(x_test_whitened)
    y_predict = bst.predict(dtest)

    # Prepare results dataframe
    results = np.zeros((x_test_whitened.shape[0], 2))
    results[:, 0] = test_data_ids
    results[:, 1] = y_predict

    # save the output weights
    if not ospath.exists(output_pathname):
        makedirs(output_pathname)
    np.savetxt(output_filepath, results, fmt=["%1.1f", "%1.14f"], newline="\n", delimiter=",", header="id,y",
               comments="")


if __name__ == "__main__":
    main()
