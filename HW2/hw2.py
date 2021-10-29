import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# importing the data set (clothing images and image labels)
total_images = pd.read_csv("hw02_images.csv", header=None)
total_labels = pd.read_csv("hw02_labels.csv", header=None)

print(total_images)

# dividing data set to test and train
train_dataSet_len = 30000
test_dataSet_len = 5000

train_images = total_images[:train_dataSet_len]
test_images = total_images[-test_dataSet_len:]

train_labels = total_labels[:train_dataSet_len]
test_labels = total_labels[-test_dataSet_len:]

# converting the data to numpy for ease of use
train_images = train_images.to_numpy()
test_images = test_images.to_numpy()
train_labels = train_labels.T.to_numpy().astype(int).flatten()
test_labels = test_labels.T.to_numpy().astype(int).flatten()

# estimating the mean parameters
num_of_classes = len(np.unique(train_labels))

sample_means = np.vstack(([np.mean(train_images[train_labels == (class_num + 1)], axis=0)
                           for class_num in range(num_of_classes)]))

print('\nsample_means: \n' + str(sample_means))

# estimating the standard deviation parameters
num_of_pixels = len(train_images[0])
sample_covariances = np.zeros((num_of_classes, num_of_pixels))

for class_num in range(num_of_classes):
    sample_covariances[class_num] = np.sqrt((np.sum((train_images[train_labels == class_num + 1]
                                                     - sample_means[class_num]) *
                                                    (train_images[train_labels == class_num + 1] -
                                                     sample_means[class_num]), axis=0)) /
                                            len(train_images[train_labels == class_num + 1]))

print('\nsample_deviations: \n' + str(sample_covariances))

# calculating the prior probabilities
prob = pd.value_counts(train_labels).to_numpy()
class_priors = prob / sum(prob)

print('\nclass_priors: \n' + str(class_priors))

# calculate the confusion matrix using the parametric classification rule for train data set
y_pred_train = np.ones((train_dataSet_len, num_of_classes))

pred_list_train = []

for class_num in range(num_of_classes):
    for image in range(train_dataSet_len):
        y_pred_train[image, class_num] = np.sum(-1 / 2 * (np.log(2 * np.pi *
                                                                 np.power(sample_covariances[class_num, :], 2))) -
                                                ((np.power((train_images[image, :] - sample_means[class_num, :]), 2)) /
                                                 (2 * np.power((sample_covariances[class_num, :]), 2))))

for image in range(train_dataSet_len):
    pred_list_train.append(np.argmax(y_pred_train[image, :]))

# calculate the confusion matrix using the parametric classification rule for test data set
y_pred_test = np.ones((test_dataSet_len, num_of_classes))

pred_list_test = []

for class_num in range(num_of_classes):
    for image in range(test_dataSet_len):
        y_pred_test[image, class_num] = np.sum(-1 / 2 * (np.log(2 * np.pi *
                                                                np.power(sample_covariances[class_num, :], 2))) -
                                               ((np.power((test_images[image, :] - sample_means[class_num, :]), 2)) /
                                                (2 * np.power((sample_covariances[class_num, :]), 2))))

for image in range(test_dataSet_len):
    pred_list_test.append(np.argmax(y_pred_test[image, :]))

""" creating the confusion matrix using pred_list_train 
    pred_list_train = sum for specific predicted classes = sum of each row for train confusion matrix table given
"""
pred_list_train = np.array(pred_list_train)

y_actual_train = train_labels
y_pred_train = pred_list_train

conf_mat_train = confusion_matrix(y_actual_train, y_pred_train).T[0:5, 1:]

number_of_classes = ['1', '2', '3', '4', '5']
format_row = '{:>10}' * (len(number_of_classes) + 1)
print(format_row.format('\ny_truth' + '', *number_of_classes))
print('y_pred')

for group, row in zip(number_of_classes, conf_mat_train):
    print(format_row.format(group, *row))

""" creating the confusion matrix using pred_list_test
    pred_list_test = sum for specific predicted classes = sum of each row for test confusion matrix table given
"""

pred_list_test = np.array(pred_list_test)

y_actual_test = test_labels
y_pred_test = pred_list_test

conf_mat_test = confusion_matrix(y_actual_test, y_pred_test).T[0:5, 1:]

number_of_classes = ['1', '2', '3', '4', '5']
format_row = '{:>10}' * (len(number_of_classes) + 1)
print(format_row.format('\ny_truth' + '', *number_of_classes))
print('y_pred')

for group, row in zip(number_of_classes, conf_mat_test):
    print(format_row.format(group, *row))
