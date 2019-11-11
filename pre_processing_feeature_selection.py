# pre_processing dimensionality reduction

import numpy as np
import os.path as ospath
import pandas as pd
from os import makedirs
from sklearn.decomposition import PCA

def add_ids_and_headings_back(ids, new_table):
    new_column_names = list(np.arange(0, new_table.shape[1]))
    new_column_names = new_column_names.insert(0, "id")
    complete_table = pd.DataFrame(new_table, columns= new_column_names, index = ids)
    return complete_table

# import data
kernel_type = "rbf"
output_pathname = "output_processed"
output_filepath_train = ospath.join(output_pathname, "out_train.csv")
output_filepath_test = ospath.join(output_pathname, "out_test.csv")
training_data_dir = ospath.join("data", "training")
testing_data_dir = ospath.join("data", "testing")

train_data_x = pd.read_csv(ospath.join(training_data_dir, "X_train.csv"), delimiter=",")
train_data_ids = train_data_x["id"]  # Get the ids for processing later on
train_data_x = train_data_x.drop(["id"], axis=1)

test_data_x = pd.read_csv(ospath.join(testing_data_dir, "X_test.csv"), delimiter=",")
test_data_ids = test_data_x["id"]  # Get the ids for processing later on
test_data_x = test_data_x.drop(["id"], axis=1)

# pca dimensionality reduction
pca = PCA(n_components=0.9, copy=True, whiten=True, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=1)
pca.fit(train_data_x)
new_train_data = pca.transform(train_data_x)
new_test_data = pca.transform(test_data_x)

# add ids back and heading
complete_table_train = add_ids_and_headings_back(train_data_ids, new_train_data)
complete_table_test = add_ids_and_headings_back(test_data_ids, new_test_data)

# output file
if not ospath.exists(output_pathname):
        makedirs(output_pathname)
complete_table_train.to_csv(output_filepath_train, sep=",", index_label = "id")
complete_table_train.to_csv(output_filepath_test, sep=",", index_label = "id")
