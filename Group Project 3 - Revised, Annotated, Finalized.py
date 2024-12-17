#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Group 15: Jennifer Xue, Christina Wang, Grace Wu
"""

# Part 1

import numpy as np

def load_data(csv_filename): # reading files
    wine_data = [] # empty list for the data
    with open(csv_filename, 'r') as file: # opens file using with, no need to close
        next(file) # skips header row
        for line in file:
            line = line.strip()
            data = [float(value.strip()) for value in line.split(';')][:-1] #  take a float of the stripped values, split by ;, excluding last column (quality)
            wine_data.append(data) # append the float values into the empty list made for each csv file
    return np.array(wine_data) # 
red_wine = load_data('redwine.csv') # loading data
white_wine = load_data('whitewine''.csv')
#print(red_wine)
#print(white_wine)
def split_data(dataset, ratio): # splitting to training and testing portions based on given ratios
    num_rows  = dataset.shape[0] 
    if not isinstance(ratio * num_rows, int): # if the ratio times the number of rows is not an integer (eg: a float)
        splitting_point = int(ratio * num_rows) # convert type of the product into an integer via truncation; sets splitting point as this product
    else:
        splitting_point = ratio * num_rows # sets splitting point value as product of ratio & number of rows
    training_set = dataset[0:splitting_point] # makes the first [ratio*num_rows] rows serve as the training rows / set
    testing_set = dataset[splitting_point:num_rows] # makes the remaining rows serve as the testing rows / set
    set_tuple = (training_set, testing_set) 
    return set_tuple # return tuple containing the training & testing sets
red_wine_training, red_wine_testing = split_data(red_wine, .9) # unpacking training and testing tuple, as an example we will use 0.9 as the ratio (makes 90 percent of rows serve as training, 10 percent as testing)
white_wine_training, white_wine_testing = split_data(white_wine, .9) # makes 90 percent of rows serve as training, 10 percent as testing
# printing shapes
print("Red wine file shape:", red_wine.shape, "Training set shape:", red_wine_training.shape, "Testing set shape:", red_wine_testing.shape)
print("White wine file shape:", white_wine.shape, "Training set shape:", white_wine_training.shape, "Testing set shape:", white_wine_testing.shape)

# PART 2

# centroid function (from lecture):
import math

def compute_centroid(labeled_examples): 
    return sum(labeled_examples[:,:]) / labeled_examples.shape[0] # add up all rows and the divide by how many rows there are (this produces the average / centroid value)

# euclidean distance function (from lecture):
def euclidean_distance(a,b): 
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i])**2
    return math.sqrt(total) # scalar distance between vectors a and b 

#experiment / testing function to see if running the testing set yields incorrect or correct predictions
def experiment(ww_training, rw_training, ww_test, rw_test):
    correct_for_ww = 0 # initiate counts for number of correctly predicted values from testing set 
    correct_for_rw = 0
    total = 0 # initiate count for total number of values tested
    ww_centroid = compute_centroid(ww_training) # compute the centroid for the white wine's training values
    rw_centroid = compute_centroid(rw_training) # compute the centroid for the red wine's training values
    #print(red_centroid.shape, white_centroid.shape) # checking to see if this matches the shapes of the four data sets (should be 11)

    for data in ww_test: # for values in the white wine testing set
        distance_to_rw_centroid = euclidean_distance(data, rw_centroid) # calculate the distance of the white wine testing value from the red wine's centroid value 
        distance_to_ww_centroid = euclidean_distance(data, ww_centroid) # calculate the distance of the white wine testing value from the white wine's centroid value 
        total += 1 # increment the total prediction count
        if distance_to_ww_centroid < distance_to_rw_centroid: # only correct if the data in the white wine test set is closer to the white wine centroid than the red wine centroid
            correct_for_ww += 1 # if this is correct, then add 1 to the corect count
    
    for data in rw_test: # same thing but for red wine 
        distance_to_rw_centroid = euclidean_distance(data, rw_centroid)
        distance_to_ww_centroid = euclidean_distance(data, ww_centroid)
        total += 1 # increment the total prediction count
        if distance_to_rw_centroid < distance_to_ww_centroid:
            correct_for_rw += 1
    
    # print(correct_for_ww, correct_for_rw) #just wanted to see which type of wine it was more accurate for
    correct = correct_for_ww + correct_for_rw # total correct predictions is the sum of correct red wine predictions and correct white wine predictions
    accuracy = correct/total # accuracy is correct productions divided by total predictions
    print("Total number of predictions made:", total, "\nTotal number of correct predictions", correct, "\nAccuracy of the model:", accuracy)
    return accuracy # returns the final accuracy 

experiment(white_wine_training, red_wine_training, white_wine_testing, red_wine_testing)

# part 3
def cross_validation(ww_data, rw_data, k):
    # performs k-fold cross-validation on two datasets: white wine (ww_data) and red wine (rw_data)
    # k is the number of folds for cross-validation
    if len(ww_data) == len(rw_data): # check for that the data sets are of equal length
        fold_size = len(ww_data)//k  # determine size of each fold so later, we can make each fold the same size
        accuracy_sum = 0 # sum of all accuracies over folds
        accuracy_count = 0 # count the number of accuracies (used for averaging)
    else:
        raise ValueError("Datasets must have the same length for cross-validation.")
        
    for x in range(k):  # iterate over k-folds
        if x < (k-1):  # for all folds except the last one
            i = x * fold_size  # starting index of the fold
            j = (x + 1) * fold_size  # ending index of the fold

            # test sets for current fold
            ww_test_set = ww_data[i:j]
            rw_test_set = rw_data[i:j]

            # training sets: combine data before and after the test fold
            # for white wine data set
            ww_1 = ww_data[:i]  # data before the test fold
            ww_2 = ww_data[j+1:len(ww_data)]  # data after the test fold
            ww_training_set = np.vstack((ww_1, ww_2))  # combine the two parts to create the training set
            
            # for red wine data set
            rw_1 = rw_data[:i]  # data before the test fold
            rw_2 = rw_data[j+1:len(rw_data)]  # data after the test fold
            rw_training_set = np.vstack((rw_1, rw_2))  # combine the two parts to create the training set

            # run the experiment and accumulate accuracy
            accuracy = experiment(ww_training_set, rw_training_set, ww_test_set, rw_test_set)
            accuracy_sum += accuracy
            accuracy_count += 1

        elif x == (k-1):  # to account for when k doesn't divide evenly into the data, include the "extra" data in the last fold
            i = x * fold_size  # starting index of the last test fold

            # test sets: remainder of the data
            ww_test_set = ww_data[i:len(ww_data)+1]
            rw_test_set = rw_data[i:len(rw_data)+1]

            # training sets: all data before the last test fold
            ww_training_set = ww_data[:i]
            rw_training_set = rw_data[:i]

            # run the experiment and accumulate accuracy
            accuracy = experiment(ww_training_set, rw_training_set, ww_test_set, rw_test_set)
            accuracy_sum += accuracy
            accuracy_count += 1

    avg_accuracy = accuracy_sum/accuracy_count
    print("Average Accuracy from Cross Validation:", avg_accuracy)
    return avg_accuracy

cross_validation(white_wine, red_wine, 5) # call cross validation with five folds

