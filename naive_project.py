# Naive Bayes Project : Predicting breast cancer malignant or benign using naive bayes classifier

import pandas as pd
import numpy as np

GD_THRESHOLD = 2 * 10**(-2)

# Prior probabilities: P(Y=benign), P(Y=malignant)
def prior_probabilities(part):
    benign = part[part[1] == 'B']
    malignant = part[part[1] == 'M']
    benign_count = benign.shape[0]
    malignant_count = malignant.shape[0]
    total_count = benign_count + malignant_count
    benign_prior = benign_count / total_count
    malignant_prior = malignant_count / total_count
    return benign_prior, malignant_prior

# In this dataset, input samples have continuous values
# Use Gaussian Naive Bayes classifier
# Calculate mean and standard deviation for each attribute
def mean_std(part):
    benign = part[part[1] == 'B']
    malignant = part[part[1] == 'M']
    benign = benign.drop(columns=[0, 1])
    malignant = malignant.drop(columns=[0, 1])
    benign_mean = benign.mean()
    malignant_mean = malignant.mean()
    benign_std = benign.std()
    malignant_std = malignant.std()
    return benign_mean, malignant_mean, benign_std, malignant_std

# Calculate the probability of each attribute given the class
# P(Xi|Y=benign), P(Xi|Y=malignant) = 1 / (sqrt(2 * pi) * std) * exp(-((x - mean)^2 / (2 * std^2)))
# since the input samples have continuous values and the distribution is Gaussian
# x = input sample, mean = mean of the attribute, std = standard deviation of the attribute
def prob_attribute_given_class(x, mean, std):
    exponent = np.exp(-((x - mean)**2 / (2 * std**2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# read wdbc.data file
df = pd.read_csv('wdbc.data', header=None)

# Number of instances: 569
# Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

# Class distribution: 357 benign, 212 malignant

# Divide the dataset into training and testing sets
# Training set: 80%
# Testing set: 20%

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Apply 5-fold cross validation
# Divide the dataset into 5 equal parts
# 1 part for testing and 4 parts for training
# Repeat the process 5 times and average the accuracy

# Split the dataset into 5 equal parts
part1 = df.iloc[0:114, :]
part2 = df.iloc[114:228, :]
part3 = df.iloc[228:342, :]
part4 = df.iloc[342:456, :]
part5 = df.iloc[456:570, :]
parts = [part1, part2, part3, part4, part5]
test_accuracy = []
training_accuracy = []
# Apply 5-fold cross validation
for i in range(5):
    test = parts[i]
    train = pd.concat([part for j, part in enumerate(parts) if j != i])
    benign_prior, malignant_prior = prior_probabilities(train)
    benign_mean, malignant_mean, benign_std, malignant_std = mean_std(train)
    correct = 0
    for index, row in test.iterrows():
        benign_prob = benign_prior
        malignant_prob = malignant_prior
        for j in range(0, 30):
            benign_prob *= prob_attribute_given_class(row[j+2], benign_mean[j+2], benign_std[j+2])
            malignant_prob *= prob_attribute_given_class(row[j+2], malignant_mean[j+2], malignant_std[j+2])
        if benign_prob > malignant_prob:
            prediction = 'B'
        else:
            prediction = 'M'
        if prediction == row[1]:
            correct += 1
    test_accuracy.append(correct / test.shape[0])
    correct = 0
    for index, row in train.iterrows():
        benign_prob = benign_prior
        malignant_prob = malignant_prior
        for j in range(0, 30):
            benign_prob *= prob_attribute_given_class(row[j+2], benign_mean[j+2], benign_std[j+2])
            malignant_prob *= prob_attribute_given_class(row[j+2], malignant_mean[j+2], malignant_std[j+2])
        if benign_prob > malignant_prob:
            prediction = 'B'
        else:
            prediction = 'M'
        if prediction == row[1]:
            correct += 1
    training_accuracy.append(correct / train.shape[0])

avg_test_accuracy = sum(test_accuracy) / len(test_accuracy)
avg_training_accuracy = sum(training_accuracy) / len(training_accuracy)
# Average test accuracy
print('naive bayes test accuracy:', avg_test_accuracy * 100)
# Average training accuracy
print('naive bayes training accuracy:', avg_training_accuracy * 100)

# With the conditional independence assumption,
# we need number of parameters = 2 * 30 = 60 since we have 30 attributes
# and 2 classes (benign and malignant)

def z_score_normalization(X):
    mean = np.mean(X, axis=0) # axis=0 -> calculate the mean for each column.
    std = np.std(X, axis=0)
    # don't normalize the bias term (first column)
    mean[0] = 0
    std[0] = 1
    X = (X - mean) / std
    return X

# GD
def gradient_descent(X, y, w, learning_rate):
    N = X.shape[0] # shape[0] -> number of rows
    grad_norm = 1
    iterations = 0
    while(grad_norm > GD_THRESHOLD):
        iterations += 1
        nominator = -(y[:, np.newaxis] * X)  # both nominator and nominator.T is fine
        expression = y.T * np.dot(X, w)
        denominator = 1 + np.exp(expression).T
        fraction = np.divide(nominator, denominator[:, np.newaxis])
        grad_vector = fraction.mean(axis=0)
        w -= learning_rate * grad_vector
        grad_norm = np.linalg.norm(grad_vector)
    return w, iterations

def logistic_regression(X, y, learning_rate):
    X = z_score_normalization(X)
    w = np.zeros(X.shape[1])
    return gradient_descent(X, y, w, learning_rate)

def determine_learning_rate(parts):
    accuracy = []
    training_accuracy = []
    learning_rates = [10, 5, 1]
    for i in range(5):
        test = parts[i]
        train = pd.concat([part for j, part in enumerate(parts) if j != i])
        X_train = train.drop(columns=[0, 1]).to_numpy()
        X_train = np.insert(X_train, 0, 1, axis=1)
        y_train = np.where(train[1] == 'B', -1, 1)
        X_test = test.drop(columns=[0, 1]).to_numpy()
        X_test = np.insert(X_test, 0, 1, axis=1)
        y_test = np.where(test[1] == 'B', -1, 1)
        wArray = [0]*len(learning_rates)
        iterations = [0]*len(learning_rates)
        for i in range(len(learning_rates)):
            wArray[i], iterations[i] = logistic_regression(X_train, y_train, learning_rates[i])

        X_test = z_score_normalization(X_test)

        accuracyOfW = []

        for w in wArray:
            correct = 0
            for j in range(len(X_test)):
                prediction = np.sign(np.dot(w, X_test[j]))
                if prediction == y_test[j]:
                    correct += 1
            accuracyOfW.append(correct / len(X_test))
        print(accuracyOfW)
        accuracy.append(accuracyOfW)

    accuracies = np.array([0.0]*len(learning_rates))
    for j in range(len(learning_rates)):
        accuracies[j] = np.mean([accuracy[i][j] for i in range(5)])
        print("Average accuracy for learning rate", learning_rates[j], ":", accuracies[j])
    print("Best average accuracy for learning rate:", learning_rates[np.argmax(accuracies)])
    print("The learning rates perform similar. Pick one of them. Pick 1")

test_accuracy = []
training_accuracy = []
# run logistic regression on the dataset, apply 5-fold cross validation and print the average accuracy
for i in range(5):
    test = parts[i]
    train = pd.concat([part for j, part in enumerate(parts) if j != i])
    # drop columns ID and diagnosis and shift the columns to the left by 2
    # and insert a column of ones at the beginning for the bias term
    X_train = train.drop(columns=[0, 1]).to_numpy()
    X_train = np.insert(X_train, 0, 1, axis=1)
    y_train = np.where(train[1] == 'B', -1, 1)
    X_test = test.drop(columns=[0, 1]).to_numpy()
    X_test = np.insert(X_test, 0, 1, axis=1)
    y_test = np.where(test[1] == 'B', -1, 1)

    w, iterations = logistic_regression(X_train, y_train, 1)

    X_test = z_score_normalization(X_test)
    X_train = z_score_normalization(X_train)

    correct = 0
    for j in range(X_test.shape[0]):
        if np.dot(w, X_test[j]) > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction == y_test[j]:
            correct += 1
    test_accuracy.append(correct / X_test.shape[0])

    correct = 0
    for j in range(X_train.shape[0]):
        if np.dot(w, X_train[j]) > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction == y_train[j]:
            correct += 1
    training_accuracy.append(correct / X_train.shape[0])
determine_learning_rate(parts)
avg_training_accuracy = sum(training_accuracy) / len(training_accuracy)
avg_test_accuracy = sum(test_accuracy) / len(test_accuracy)
print('logistic regression test accuracy:', avg_test_accuracy * 100)
print('logistic regression training accuracy:', avg_training_accuracy * 100)