import pandas as pd
import os.path as ospath
import numpy as np
from os import makedirs
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb


def perform_data_scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def oversampling(X, y):
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    noise = np.random.normal(0, 1, (X_res.shape[0] - X.shape[0], X.shape[1]))
    X_res[X.shape[0]:X_res.shape[0]] += noise
    return X_res, y_res


def plot3clusters(x, y, title, vtitle, out_filename="training_data.png"):
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
        plt.scatter(x[y == i, 0], x[y == i, 1], color=color, alpha=.3, lw=lw,
                  label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.xlabel(vtitle + "1")
    plt.ylabel(vtitle + "2")
    plt.savefig(out_filename)


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

    # separate the data between label and features and convert them into values (required after using pandas)
    x_train_orig = train_data_x.values
    y_train_orig = train_data_y.values
    x_test_orig = test_data_x.values

    # Preprocessing step #1: Perform data scaling
    x_train_whitened, x_test_whitened = perform_data_scaling(x_train_orig, x_test_orig)

    # Preprocessing step #2: Oversampling
    x_res, y_res = oversampling(x_train_whitened, y_train_orig)

    # PCA step #1
    kernel_type = 'rbf'
    pca = KernelPCA(kernel=kernel_type, n_components=7)
    x_pca = pca.fit_transform(x_res)
    plot3clusters(x_pca, y_res, 'PCA', 'PC', "training_samples_{}.png".format(kernel_type))

    # Preprocessing step #3/training: XGBoost
    np.random.seed(1)

    # model = XGBClassifier(learning_rate=0.05, n_estimators=2, max_depth=5)
    # model.fit(x_res, y_res)

    X_train, X_eval, y_train, y_eval = train_test_split(x_pca, y_res, test_size=0.1, random_state=100)
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    deval = xgb.DMatrix(data=X_eval, label=y_eval)
    dtest = xgb.DMatrix(pca.fit_transform(x_test_whitened))

    param = {'max_depth': 3, 'eta': 1, 'objective': 'multi:softmax', 'num_class': 3, 'reg_alpha': 1, 'reg_lambda': 1,
             'colsample_bytree': 0.3}
    param['eval_metric'] = 'mlogloss'
    evallist = [(deval, 'eval'), (dtrain, 'train')]
    bst = xgb.train(dtrain=dtrain, params=param, evals=evallist, early_stopping_rounds=10, num_boost_round=500)

    # Do the prediction
    y_predict = bst.predict(dtest)
    unique_elements, counts_elements = np.unique(y_predict, return_counts=True)
    print("test set labels and their corresponding counts")
    print(np.asarray((unique_elements, counts_elements)))

    # Confusion matrix
    # By definition a confusion matrix C is such that C_{i,j} is equal to the number of observations known to be in
    # group i but predicted to be in group j.
    y_pred = bst.predict(xgb.DMatrix(pca.fit_transform(x_train_whitened)))
    y_true = y_train_orig
    print("training set confusion matrix")
    print(confusion_matrix(y_true, y_pred))

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
