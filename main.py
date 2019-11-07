# system
import os.path as ospath
from os import makedirs

# general packages
import numpy as np
import pandas as pd

# scikit-learn helpers
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# keras machinery
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE


########################################################
################ Helper functions ######################
########################################################

def check_for_imbalance(Y):
    unique, counts = np.unique(Y, return_counts=True)
    # format output nicely
    for i, cl in enumerate(unique):
        print("Class label: {} \tCount: {}".format(int(cl), counts[i]))
    return class_weight.compute_class_weight('balanced', np.unique(Y), Y)


def encode_labels(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    return np_utils.to_categorical(encoder.transform(Y))


########################################################
################## define model ########################
########################################################
def create_model(npl=100, lrs=3, activation_fct='relu', dropout=0, learning_rate=1e-3):
    """
    npl      neurons per layer
    lrs      number of layers
    """
    model = Sequential()
    model.add(Dense(npl, activation=activation_fct, input_dim=1000))
    model.add(Dense(units=npl, activation=activation_fct, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    if dropout > 0:
        model.add(Dropout(rate=dropout))
    if lrs > 1:
        for i in range(lrs-1):
            model.add(Dense(units=npl, activation=activation_fct, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
            if dropout > 0:
                model.add(Dropout(rate=dropout))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    return model


########################################################
#################### main ##############################
########################################################


def plot3clusters(x, y, title, vtitle, out_filename="training_data.png"):
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
        plt.scatter(x[y == i, 0], x[y == i, 1], color=color, alpha=1., lw=lw,
                  label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.xlabel(vtitle + "1")
    plt.ylabel(vtitle + "2")
    plt.savefig(out_filename)


def perform_data_scaling(x_train, x_test):
    scaler = StandardScaler()
    x_train_whitened = scaler.fit_transform(x_train)
    x_test_whitened = scaler.transform(x_test)
    return x_train_whitened, x_test_whitened


def main():
    # set critical parameters
    seed = 1
    folds = RepeatedKFold(n_splits=10, n_repeats=1, random_state=np.random.seed(1))  # used in grid search
    # setting miscellaneous parameters including those for I/O
    kernel_type = "rbf"
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

    # Preprocessing step #1: Perform data scaling
    x_train_whitened, x_test_whitened = perform_data_scaling(x_train_orig, x_test_orig)

    # PCA step #1
    pca = KernelPCA(kernel=kernel_type, n_components=2)
    x_pca = pca.fit_transform(x_train_whitened)
    plot3clusters(x_pca, y_train_orig, 'PCA', 'PC', "training_samples_{}.png".format(kernel_type))

    # SMOTE Sampling
    sm = SMOTE(random_state=42)
    x_train_res, y_train_res = sm.fit_sample(x_train_whitened, y_train_orig)
    # Encode data and labels
    y_train_encoded = encode_labels(y_train_res)

    ######################################
    # grid search for optimal parameters #
    ######################################
    model = KerasClassifier(build_fn=create_model, verbose=False)
    neurons = [1000, 2000]
    layers = [2, 3, 4]
    dropout = [0.3, 0.5, 0.6]
    epochs = [50, 100, 200]
    learning_rate = [1e-3, ]
    batch_size = [800, ]
    early_stopping = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True, verbose=1)

    param_grid = dict(
        npl=neurons,
        lrs=layers,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=folds)
    grid_result = grid.fit(x_train_whitened, y_train_encoded, callbacks=[early_stopping])

    # log the results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print('*' * 50)
    print('mean (stdev) \t parameters')
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    opt_npl, opt_lrs, opt_dropout, opt_epochs, opt_learning_rate, opt_batch_size = grid_result.best_params_['npl'], \
        grid_result.best_params_['lrs'],\
        grid_result.best_params_['dropout'], \
        grid_result.best_params_['epochs'], \
        grid_result.best_params_['learning_rate'], \
        grid_result.best_params_['batch_size']

    # opt_npl, opt_lrs, opt_dropout, opt_epochs, opt_learning_rate, opt_batch_size = 2000, 10, 0.5, 800, 1e-3, 50

    # Do the prediction
    model = create_model(npl=opt_npl,
                         lrs=opt_lrs,
                         activation_fct='relu',
                         dropout=opt_dropout,
                         learning_rate=opt_learning_rate)
    model.fit(x_train_res, y_train_encoded, epochs=opt_epochs, batch_size=opt_batch_size, verbose=False, callbacks=[early_stopping])
    model.summary()
    y_test_predict = model.predict(x_test_whitened)

    # Prepare results dataframe
    results = np.zeros((x_test_orig.shape[0], 2))
    results[:, 0] = test_data_ids
    results[:, 1] = np.argmax(y_test_predict, axis=1)

    # save the output weights
    if not ospath.exists(output_pathname):
        makedirs(output_pathname)
    np.savetxt(output_filepath, results, fmt=["%1.1f", "%1.1f"], newline="\n", delimiter=",", header="id,y",
               comments="")


if __name__ == "__main__":
    main()
