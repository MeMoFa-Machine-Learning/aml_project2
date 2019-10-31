import pandas as pd
import os.path as ospath
import numpy as np
from os import makedirs
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, KernelPCA
from xgboost import XGBClassifier
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
    x_train_orig = train_data_x.values
    y_train_orig = train_data_y.values
    x_test_orig = test_data_x.values

    # PCA step #1
    pca = KernelPCA(kernel="rbf", n_components=2)
    x_pca = pca.fit_transform(x_train_orig)

    # Preprocessing step #2: Outlier detection and removal

    outlier_indices = find_outliers(x_pca)
    x_train_whitened = x_train_orig[outlier_indices]
    y_train_orig = y_train_orig[outlier_indices]

    # Preprocessing step #3/training: XGBoost
    np.random.seed(1)

    model = XGBClassifier(learning_rate=0.05, n_estimators=2, max_depth=5)
    model.fit(x_train_whitened, y_train_orig)

    # Do the prediction
    y_predict = model.predict(x_test_orig)

    # Prepare results dataframe
    results = np.zeros((x_test_orig.shape[0], 2))
    results[:, 0] = test_data_ids
    results[:, 1] = y_predict

    # save the output weights
    if not ospath.exists(output_pathname):
        makedirs(output_pathname)
    np.savetxt(output_filepath, results, fmt=["%1.1f", "%1.1f"], newline="\n", delimiter=",", header="id,y",
               comments="")


if __name__ == "__main__":
    main()
