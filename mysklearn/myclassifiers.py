##############################################
# Programmer: Adrian Rabadam & Michael Finch
# Class: CptS 322-01, Spring 2021
# Final Project
# 4/21/21
# 
# 
# Description: This file defines classifiers 
# for prediction
##############################################

import mysklearn.myutils as myutils
import operator
import random
import copy

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        mean_x = sum(X_train) / len(X_train)
        mean_y = sum(y_train) / len(y_train)
        self.slope = sum([(X_train[i] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) \
        / sum([(X_train[i] - mean_x) ** 2 for i in range(len(X_train))])
        self.intercept = mean_y - self.slope * mean_x
        pass # TODO: copy your solution from PA4 here

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test in X_test:
            y_predicted.append(test[0] * self.slope + self.intercept)
        return y_predicted # TODO: copy your solution from PA4 here


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        new_X_train = []
        for i, instance in enumerate(self.X_train):
            temp = instance.copy()
            temp.append(i)
            for test in X_test: 
                dist = myutils.compute_euclidean_distance(temp[:len(X_test[0])], test)
                temp.append(dist)
            new_X_train.append(temp)
        
        for i in range(len(X_test)):
            x_sorted_dist = sorted(new_X_train, key=operator.itemgetter(len(X_test[0]) + i + 1))  # sorted by distance
            top_k = x_sorted_dist[:self.n_neighbors]
            top_k_sorted = sorted(top_k, key=operator.itemgetter(-2))  # sorted by index
            distances_temp = []
            indices_temp = []
            for neighbor in top_k_sorted:
                indices_temp.append(neighbor[len(X_test[0])])
                distances_temp.append(neighbor[len(X_test[0]) + 1])
            distances.append(distances_temp)
            neighbor_indices.append(indices_temp)
        return distances, neighbor_indices # TODO: copy your solution from PA4 here

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, indices = self.kneighbors(X_test)
        for row in indices:
            knn_classifiers = []
            for index in row:
                knn_classifiers.append(self.y_train[index])
            values, freqs = myutils.get_frequencies(knn_classifiers)
            prediction_index = freqs.index(max(freqs))
            prediction = values[prediction_index]
            y_predicted.append(prediction)
        
        return y_predicted # TODO: copy your solution from PA4 here

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.priors = []
        self.posteriors =  []
        header  = []
        X_train_copy = X_train.copy()
        for i in range(len(X_train_copy)):
            X_train_copy[i].append(y_train[i])
        for i in range(len(X_train_copy[0])):
            header.append(str(i + 1))
        classifier_names, classifier_subtables = myutils.group_by(X_train_copy, header, str(len(X_train_copy[0])))
        self.priors.append(classifier_names)
        for subtable in classifier_subtables:
            self.priors.append(len(subtable) / len(X_train_copy))
        posteriors_header = []
        #print(self.priors)
        for i in range(len(X_train_copy[i]) - 1, 0, -1):
            temp_names, temp_subtables = myutils.group_by(X_train_copy, header, str(i))
            for name in temp_names:
                posteriors_header.append(name)
        posteriors_row = []
        for i in range(len(posteriors_header) + 1):
            for j in range(len(classifier_names) + 1):
                posteriors_row.append(0)
            self.posteriors.append(posteriors_row)
            posteriors_row = []
        self.posteriors[0][0] = "label"
        for i in range(len(self.posteriors[0]) - 1):
            self.posteriors[0][i + 1] = classifier_names[i]
        for i in range(1, len(self.posteriors)):
            self.posteriors[i][0] = str(posteriors_header[i - 1]) 
        for k in range(len(classifier_subtables)):
            header_col = myutils.get_column(self.posteriors, self.posteriors[0], self.posteriors[0][0])
            for i in range(len(header) - 1):
                col = myutils.get_column(classifier_subtables[k], header, header[i])
                values, counts = myutils.get_frequencies(col)
                for j in range(len(counts)):
                    row_index = header_col.index(str(values[j]))
                    header_col[row_index] = 0
                    col_index = self.posteriors[0].index(classifier_names[k])
                    self.posteriors[row_index][col_index] = counts[j]/len(classifier_subtables[k])
        pass # TODO: copy your solution from PA5 here

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_predicted = []
        for test in X_test:
            temp_classifier_table = self.priors.copy()
            temp_classifier_table.pop(0)  # remove header
            labels_col = myutils.get_column(self.posteriors, self.posteriors[0], "label")
            for label in test:
                label = str(label)
                i = 0  # for counting through priors
                for classifier in self.priors[0]:
                    col = myutils.get_column(self.posteriors, self.posteriors[0], classifier)
                    #print(col)
                    p_index = labels_col.index(label)
                    p_value = col[p_index]
                    #print(p_value)
                    temp_classifier_table[i] = temp_classifier_table[i] * p_value
                    i += 1
                labels_col.pop(p_index)
            max_index = temp_classifier_table.index(max(temp_classifier_table))
            y_predicted.append(self.priors[0][max_index])
        return y_predicted # TODO: copy your solution from PA5 here

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        heading = []
        for i in range(len(X_train[0])):
            heading_value = "att" + str(i)
            heading.append(heading_value)
        att_domain = {}
        for item in heading:
            values, counts = myutils.get_table_frequencies(X_train, heading, item)
            att_domain[item] = values
        # stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = heading.copy() #  pass by object reference
        # call tdidt()
        tree = myutils.tdidt(train, available_attributes, heading, att_domain)
        self.X_train = X_train
        self.y_train = y_train
        self.tree = tree
        pass # TODO: fix this
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        heading = []
        y_predicted = []
        for i in range(len(self.X_train[0])):
            heading_value = "att" + str(i)
            heading.append(heading_value)
        for test in X_test:
            y_predicted.append(myutils.tdidt_predict(heading, self.tree, test))
        return y_predicted # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        att = ""
        if attribute_names != None:
            temp = self.tree[1]
            index = int(temp[len(temp) - 1])
            att = attribute_names[index]
        else:
            att = self.tree[1]
        print_statement = "IF " + att + " "
        for i in range(2, len(self.tree)):
            myutils.decision_rules_rec(attribute_names, self.tree[i], print_statement, class_name)
        pass # TODO: fix this

class MyRandomForrestClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, M=3, N=5, F=2):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None 
        self.y_train = None
        self.trees = None
        self.test_set = None 
        self.remainder_set = None
        self.attributes = None
        self.attribute_indexes = None
        self.M = M
        self.N = N
        self.F = F

    def fit(self, X_train, y_train, seed=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        if seed != None:
            random.seed(seed)
        # split train set into test_set and remainder set
        heading = []
        self.X_train = X_train
        self.y_train = y_train
        for i in range(len(X_train[0])):
            heading_value = "att" + str(i)
            heading.append(heading_value)
        test_set, remainder_set = myutils.random_stratifed_test_set(self.X_train, self.y_train, seed)
        self.test_set = test_set
        self.remainder_set = remainder_set
        self.attributes = copy.deepcopy(heading)
        # N random decision trees

        #select F random attributes 
        # call random_startified_test on remainder_set
        # limit training set to only include instances for random attributes 
        # use training set to build tree 
        # predict using validation set 
        # get accuracy 
        # Repeat N times 
        accuracies = []
        trees = []
        att_indexes = []
        for i in range(self.N):
            attributes = []
            attribute_indexes = []
            while len(attributes) < self.F:
                index = random.randint(0, len(heading)-1)
                attribute = heading[index]
                if attribute not in attributes:
                    attributes.append(attribute)
                    attribute_indexes.append(index)
            att_indexes.append(attribute_indexes)
            X_set = []
            for instance in remainder_set:
                temp = []
                sub_set = []
                for index in attribute_indexes:
                    sub_set.append(instance[0][index])
                temp.append(sub_set)
                temp.append(instance[1])
                X_set.append(temp)
            #print(X_set, Y_set)
            validation_set, train_set = myutils.bootstrap_sets(X_set, seed)
            X_train = []
            y_train = []
            for instance in train_set: 
                train = []
                for item in instance[0]:
                    train.append(item)
                X_train.append(train)
                y_train.append(instance[1])
            X_test = []
            y_test = []
            for instance in validation_set: 
                train = []
                for item in instance[0]:
                    train.append(item)
                X_test.append(train)
                y_test.append(instance[1])

            d_tree = MyDecisionTreeClassifier()
            d_tree.fit(X_train, y_train)
            trees.append(d_tree.tree)
            
            #find accuracy 
            predicted = d_tree.predict(X_test)

            num_correct = 0
            total = len(y_test)
            for j in range(len(y_test)):
                if y_test[j] == predicted[j]:
                    num_correct += 1
            accuracy = num_correct/total
            accuracies.append(accuracy)
            #d_tree.print_decision_rules()
            #print()
        # Select M most accurate of N decision trees
        best_trees = []
        best_tree_att_indexes = []
        for i in range(self.M):
            max_accuracy = max(accuracies)
            index = accuracies.index(max_accuracy)
            best_trees.append(trees[index])
            best_tree_att_indexes.append(att_indexes[index])
            att_indexes.remove(att_indexes[index])
            trees.remove(trees[index])
            accuracies.remove(max_accuracy)
        self.trees = best_trees
        self.attribute_indexes = best_tree_att_indexes

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        all_predictions = []
        for test in X_test:
            temp = []
            for i in range(len(self.trees)):
                tree = self.trees[i]
                heading = []
                test_sub_set = []
                for j in range(len(self.attribute_indexes[i])):
                    heading_value = "att" + str(j)
                    heading.append(heading_value)
                    test_sub_set.append(test[self.attribute_indexes[i][j]])
                temp.append(myutils.tdidt_predict(heading, tree, test_sub_set))
            all_predictions.append(temp)
        for item in all_predictions:
            y_predicted.append(myutils.forest_majority_voting(item))
        return  y_predicted
