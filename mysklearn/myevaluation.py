##############################################
# Programmer: Adrian Rabadam & Michael Finch
# Class: CptS 322-01, Spring 2021
# Final Project
# 4/21/21
# 
# 
# Description: This file defines the evaluator 
# functions to aid in training and testing
##############################################

import myutils as myutils
import random
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
        random.seed(random_state)
        pass
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        for i in range(len(X)):
            # generate a random index to swap the element and i with 
            rand_index = random.randrange(0, len(X)) # [0, len(alist))
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]
        pass
    num_instances = len(X) 
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size
    
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:] # TODO: fix this

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []

    # split X into folds
    fold_indices = []
    fold_size_1 = (len(X) // n_splits) + 1
    group_1_num = len(X) % n_splits
    fold_size_2 = len(X) // n_splits
    k = 0
    for i in range(n_splits):
        fold = []
        if (i < group_1_num):
            for j in range(fold_size_1):
                fold.append(k)
                k += 1
        else:
            for j in range(fold_size_2):
                fold.append(k)
                k += 1
        fold_indices.append(fold)
    for j in range(n_splits):
        test_temp = fold_indices[j]
        X_test_folds.append(test_temp)
        k = 0
        test_train = []
        while k < len(fold_indices):
            if (k == j):
                k += 1
            if (k == len(fold_indices)): # j was last index
                break
            test_train.extend(fold_indices[k])
            k += 1
        X_train_folds.append(test_train)
    return X_train_folds, X_test_folds # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_train_folds = []
    X_test_folds = []

    # split X into folds
    header = ["X sample", "classifier"]
    table = []
    i = 0
    for sample in X:
        row = []
        row.append(sample)
        row.append(y[i])
        i += 1
        table.append(row)
    classifiers, classifier_tables = myutils.group_by(table, header, "classifier")
    fold_indices = []
    fold_size_1 = (len(X) // n_splits) + 1
    group_1_num = len(X) % n_splits
    fold_size_2 = len(X) // n_splits
    indices = []
    for i in range(len(classifier_tables[0])):
        for j in range(len(classifier_tables)):
            if i < len(classifier_tables[j]):
                indices.append(X.index(classifier_tables[j][i][0]))
    k = 0
    for i in range(group_1_num):
        temp = []
        for j in range(fold_size_1):
            if k < len(indices):
                temp.append(indices[k])
            k += 1
        fold_indices.append(temp)
    for i in range(group_1_num, n_splits):
        temp = []
        for j in range(fold_size_2):
            if k < len(indices):
                temp.append(indices[k])
            k += 1
        fold_indices.append(temp)
    for j in range(n_splits):
        test_temp = fold_indices[j]
        X_test_folds.append(test_temp)
        k = 0
        test_train = []
        while k < len(fold_indices):
            if (k == j):
                k += 1
            if (k == len(fold_indices)): # j was last index
                break
            test_train.extend(fold_indices[k])
            k += 1
        X_train_folds.append(test_train)
    return X_train_folds, X_test_folds # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []

    for i in range(len(labels)):
        temp = [0] * len(labels)
        matrix.append(temp)
    for i in range(len(labels)):
        for j in range(len(y_true)):
            if y_pred[j] == labels[i] and y_true[j] == labels[i]:
                matrix[i][i] += 1
            elif y_pred[j] == labels[i] and y_true[j] != labels[i]:
                matrix[labels.index(y_true[j])][i] += 1

    return matrix # TODO: fix this