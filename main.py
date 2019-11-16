import pandas as pd
import os.path as ospath
import numpy as np
from os import makedirs
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s')


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
    output_filepath = ospath.join(output_pathname, "out_tree.csv")
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

    # Preprocessing step #2: Oversampling (Skipping for now)
    x_res, y_res = x_train_whitened, y_train_orig

    # Training Step #1: Grid Search
    x_train_gs, x_ho, y_train_gs, y_ho = train_test_split(x_res, y_res, test_size=0.1, random_state=0)

    k_best = [50, 100, 200, 400, 600, ]
    max_depth = [3, 4, 5, 7, 9, 11]
    min_samples_split = [2, 3, 4, 5]
    n_estimators = [50, 75, 100, 150, 200]

    # k_best = [200]
    # max_depth = [9]
    # min_samples_split = [0.4]
    # n_estimators = [100]

    parameters = [
        {
            'skb__k': k_best,
            'rfc__criterion': ['gini'],
            'rfc__max_depth': max_depth,
            'rfc__min_samples_split': min_samples_split,
            'rfc__n_estimators': n_estimators,
            'rfc__class_weight': ['balanced'],
        },
        {
            'skb__k': k_best,
            'rfc__criterion': ['entropy'],
            'rfc__max_depth': max_depth,
            'rfc__min_samples_split': min_samples_split,
            'rfc__n_estimators': n_estimators,
            'rfc__class_weight': ['balanced'],
        },
    ]

    # Perform the cross-validation
    best_models = []
    for kernel_params in parameters:

        pl = Pipeline([('skb', SelectKBest()), ('rfc', RandomForestClassifier())])
        kfold = StratifiedKFold(n_splits=15, shuffle=True, random_state=6)

        # C-support vector classification according to a one-vs-one scheme
        grid_search = GridSearchCV(pl, kernel_params, scoring="balanced_accuracy", n_jobs=-1, cv=kfold, verbose=1)
        grid_result = grid_search.fit(x_train_gs, y_train_gs)

        # Calculate statistics and calculate on hold-out
        logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        y_ho_pred = grid_search.predict(x_ho)
        hold_out_score = balanced_accuracy_score(y_ho_pred, y_ho)
        best_models.append((hold_out_score, grid_result.best_params_))
        logging.info("Best score on hold-out: {}".format(hold_out_score))

    # Pick best params and fit model
    final_model_params_i = int(np.argmax(np.array(best_models)[:, 0]))
    final_model_params = best_models[final_model_params_i][1]
    logging.info("Picked the following model: {}".format(final_model_params))

    logging.info("Fitting the final model...")
    final_model = Pipeline([('skb', SelectKBest()), ('rfc', RandomForestClassifier())])
    final_model.set_params(**final_model_params)
    final_model.fit(x_res, y_res)

    # Do the prediction
    y_predict = final_model.predict(x_test_whitened)
    unique_elements, counts_elements = np.unique(y_predict, return_counts=True)
    print("test set labels and their corresponding counts")
    print(np.asarray((unique_elements, counts_elements)))

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
