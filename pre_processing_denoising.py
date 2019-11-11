# Noise reduction and feature selection of images

import numpy as np
import cv2 as cv
import os.path as ospath
import pandas as pd


# import data
kernel_type = "rbf"
output_pathname = "output_denoised"
output_filepath = ospath.join(output_pathname, "out.csv")
training_data_dir = ospath.join("data", "training")
testing_data_dir = ospath.join("data", "testing")

train_data_x = pd.read_csv(ospath.join(training_data_dir, "X_train.csv"), delimiter=",").drop(["id"], axis=1)
train_data_ids = train_data_x["id"]  # Get the ids for processing later on
train_data_x = train_data_x.drop(["id"], axis=1)

test_data_x = pd.read_csv(ospath.join(testing_data_dir, "X_test.csv"), delimiter=",")
test_data_ids = test_data_x["id"]  # Get the ids for processing later on
test_data_x = test_data_x.drop(["id"], axis=1)

# add ids back and heading
complete_table_test = np.zeros((test_data_x.shape[0]+1, test_data_x.shape[1]+1))

complete_table_train = np.zeros((train_data_x.shape[0]+1, train_data_x.shape[1]+1))


# denoise image
new_img_matrix = np.zeros(train_data_x.values.shape[0], train_data_x.values.shape[1])
for index, image in enumerate(train_data_x):
    denoised_img = cv.fastNlMeansDenoising(image,None,10,10,7,21) 
    new_img_matrix[index] = denoised_img

