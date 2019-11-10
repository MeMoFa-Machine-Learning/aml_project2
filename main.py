import pandas as pd
import os.path as ospath
import numpy as np
from os import makedirs
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


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

    # Preprocessing step #3: Feature selection
    n_components = [10, 100, 1000, ]
    class_weight = [{0: 10, 1: 80, 2: 10}, {0: 12.5, 1: 75, 2: 12.5}]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    reg_param = list(np.logspace(start=-2, stop=2, num=5, endpoint=True, base=10))

    # n_components = [1000, ]
    # class_weight = [{0: 25, 1: 75, 2: 25}, ]
    # kernels = ['linear', ]
    # reg_param = list(np.logspace(start=-2, stop=2, num=1, endpoint=True, base=10))

    wclf = SVC(kernel='rbf', probability=True)
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('wclf', wclf)])
    param_grid = dict(
        pca__n_components=n_components,
        wclf__C=reg_param,
        wclf__class_weight=class_weight,
        wclf__kernel=kernels
    )
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    # C-support vector classification according to a one-vs-one scheme
    grid_search = GridSearchCV(pipe, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=0)
    grid_result = grid_search.fit(x_res, y_res)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Do the prediction
    y_predict = grid_search.predict(x_test_whitened)
    unique_elements, counts_elements = np.unique(y_predict, return_counts=True)
    print("test set labels and their corresponding counts")
    print(np.asarray((unique_elements, counts_elements)))

    # Confusion matrix
    # By definition a confusion matrix C is such that C_{i,j} is equal to the number of observations known to be in
    # group i but predicted to be in group j.
    y_pred = grid_search.predict(x_train_whitened)
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
