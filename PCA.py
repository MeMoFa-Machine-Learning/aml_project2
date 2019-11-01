import pandas as pd
import os.path as ospath
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt


def two_vs_zero_KernelPCA(x, y, kernel_type, perform_oversampling=False):
    pca = KernelPCA(kernel=kernel_type, n_components=2)
    res_list = [i for i, value in enumerate(y) if value == 1]
    x_reduced = np.delete(x, res_list, axis=0)
    y_reduced = np.delete(y, res_list)
    x_pca = pca.fit_transform(x_reduced)
    plt.figure()
    colors = ['navy', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 2], [0, 2]):
        plt.scatter(x_pca[y_reduced == i, 0], x_pca[y_reduced == i, 1], color=color, alpha=.3, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('KernelPCA')
    plt.xlabel('PC' + "1")
    plt.ylabel('PC' + "2")
    plt.savefig("test_{}.png".format(perform_oversampling))


def perform_data_scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def oversampling(X, y):
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    idx = ros.sample_indices_
    noise = np.random.normal(0, 1, (X_res.shape[0] - X.shape[0], X.shape[1]))
    X_res[X.shape[0]:X_res.shape[0]] += noise
    return X_res, y_res


def main():
    training_data_dir = ospath.join("data", "training")
    testing_data_dir = ospath.join("data", "testing")

    # read the data
    train_data_x = pd.read_csv(ospath.join(training_data_dir, "X_train.csv"), delimiter=",").drop(["id"], axis=1)
    train_data_y = pd.read_csv(ospath.join(training_data_dir, "y_train.csv"), delimiter=",")["y"]

    test_data_x = pd.read_csv(ospath.join(testing_data_dir, "X_test.csv"), delimiter=",")
    test_data_ids = test_data_x["id"]  # Get the ids for processing later on
    test_data_x = test_data_x.drop(["id"], axis=1)

    # separate the data between label and features and convert them into values (required after using pandas)
    x_train_orig = train_data_x.values
    y_train_orig = train_data_y.values
    x_test_orig = test_data_x.values

    # Preprocessing step #1: Perform data scaling
    x_train_whitened, x_test_whitened = perform_data_scaling(x_train_orig, x_test_orig)

    # PCA step #1: w/o oversampling
    kernel_type = 'rbf'
    two_vs_zero_KernelPCA(x_train_whitened, y_train_orig, kernel_type, perform_oversampling=False)

    # Preprocessing step #2: Oversampling
    x_res, y_res = oversampling(x_train_whitened, y_train_orig)

    # PCA step #2: w/ oversampling
    two_vs_zero_KernelPCA(x_res, y_res, kernel_type, perform_oversampling=True)


if __name__ == "__main__":
    main()
