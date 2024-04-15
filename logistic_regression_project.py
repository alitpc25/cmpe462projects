import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import time
import sys

arff_file = arff.loadarff('Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(arff_file[0])
HISTORY_LENGTH = 10

# Logistic loss function = 1/N * sum(log(1 + exp(-y * w^T * x)))
# Gradient of the logistic loss function = 1/N * sum(-y * x * exp(-y * w^T * x) / (1 + exp(-y * w^T * x)))
# which is equal to 1/N * sum(-y * x / (1 + exp(y * w^T * x))) since sigmoid(-t) = 1 / (1 + exp(t))

GD_THRESHOLD = 10**(-4)
ITERATION_THRESHOLD = 500

# HW Part 1 : Z-score normalization to normalize the data.
def z_score_normalization(X):
    mean = np.mean(X, axis=0) # axis=0 -> calculate the mean for each column.
    std = np.std(X, axis=0)
    # don't normalize the bias term (first column)
    mean[0] = 0
    std[0] = 1
    X = (X - mean) / std
    return X

def loss_function(X, y, w):
    expression = y.T * np.dot( X, w) # may remove .T from both?
    ln_expr = np.log(1 + np.exp(-expression))
    mean = np.mean(ln_expr)
    return mean

loss_values_GD = []
loss_values_SGD = []

# GD
def gradient_descent(X, y, w, learning_rate):
    N = X.shape[0] # shape[0] -> number of rows
    loss_values_temp = []
    loss_values_temp.append(loss_function(X, y, w))
    grad_norm = 1
    iterations = 0
    norm_history = [0.0] * HISTORY_LENGTH
    history_index = 0
    while(True):
        iterations += 1
        nominator = -(y[:, np.newaxis] * X)  # both nominator and nominator.T is fine
        expression = y.T * np.dot(X, w)
        denominator = 1 + np.exp(expression).T
        fraction = np.divide(nominator, denominator[:, np.newaxis])
        grad_vector = fraction.mean(axis=0)
        w -= learning_rate * grad_vector
        loss = loss_function(X, y, w)
        loss_values_temp.append(loss)
        grad_norm = np.linalg.norm(grad_vector)
        norm_history[history_index] = grad_norm
        history_index = (history_index + 1) % HISTORY_LENGTH
        if np.mean(norm_history) < GD_THRESHOLD or iterations > ITERATION_THRESHOLD:
            break
    loss_values_GD.append(loss_values_temp)
    return w, iterations

def logistic_regression(X, y, learning_rate):
    X = z_score_normalization(X)
    w = np.zeros(X.shape[1])
    return gradient_descent(X, y, w, learning_rate)


# SGD
# stochastic gradient descent is an optimization method that updates the weights after picking a random sample from the dataset and calculating the gradient of the loss function for that sample
def stochastic_gradient_descent(X, y, w, learning_rate):
    N = X.shape[0]
    loss_values_temp = []
    loss_values_temp.append(loss_function(X, y, w))
    grad_norm = 1
    iterations = 0
    norm_history = [0.0] * HISTORY_LENGTH
    history_index = 0
    while(True):
        iterations += 1
        j = np.random.randint(0, N) # pick a random sample
        gradient = -y[j] * X[j] / (1 + np.exp(y[j] * np.dot(w, X[j])))
        w -= learning_rate * gradient
        loss = loss_function(X, y, w)
        loss_values_temp.append(loss)
        grad_norm = np.linalg.norm(gradient)
        norm_history[history_index] = grad_norm
        history_index = (history_index + 1) % HISTORY_LENGTH
        if np.mean(norm_history) < GD_THRESHOLD or iterations > ITERATION_THRESHOLD:
            break
    loss_values_SGD.append(loss_values_temp)
    return w, iterations

def logistic_regression_stochastic(X, y, learning_rate):
    X = z_score_normalization(X)
    w = np.zeros(X.shape[1])
    return stochastic_gradient_descent(X, y, w, learning_rate)


# Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function
# Regularized logistic loss function = 1/N * sum(log(1 + exp(-y * w^T * x))) + lambda/2 * w^T * w
# Gradient of the regularized logistic loss function = 1/N * sum(-y * x / (1 + exp(y * w^T * x))) + lambda * w

# Regularized GD
def gradient_descent_regularized(X, y, w, learning_rate, lambda_val):
    iterations = 0
    norm_history = [0.0] * HISTORY_LENGTH
    history_index = 0
    while True:
        iterations += 1
        nominator = -(y[:, np.newaxis] * X)
        expression = y.T * np.dot(X, w)
        denominator = 1 + np.exp(expression).T
        fraction = np.divide(nominator, denominator[:, np.newaxis])
        grad_vector = fraction.mean(axis=0)
        grad_vector += lambda_val * w
        w -= learning_rate * grad_vector
        grad_norm = np.linalg.norm(grad_vector)
        norm_history[history_index] = grad_norm
        history_index = (history_index + 1) % HISTORY_LENGTH
        if np.mean(norm_history) < GD_THRESHOLD or iterations > ITERATION_THRESHOLD:
            break
    return w, iterations

def logistic_regression_regularized(X, y, learning_rate, lambda_val):
    X = z_score_normalization(X)
    w = np.zeros(X.shape[1])
    return gradient_descent_regularized(X, y, w, learning_rate, lambda_val)


# Regularized SGD
def gradient_descent_stochastic_regularized(X, y, w, learning_rate, lambda_val):
    N = X.shape[0]
    loss_values_temp = []
    loss_values_temp.append(loss_function(X, y, w))
    grad_norm = 1
    iterations = 0
    norm_history = [0.0] * HISTORY_LENGTH
    history_index = 0
    while(True):
        iterations += 1
        j = np.random.randint(0, N)
        gradient = -y[j] * X[j] / (1 + np.exp(y[j] * np.dot(w, X[j])))
        gradient += lambda_val * w
        w -= learning_rate * gradient
        grad_norm = np.linalg.norm(gradient)
        loss = loss_function(X, y, w)
        loss_values_temp.append(loss)
        norm_history[history_index] = grad_norm
        history_index = (history_index + 1) % HISTORY_LENGTH
        if np.mean(norm_history) < GD_THRESHOLD or iterations > ITERATION_THRESHOLD:
            break
    return w, iterations

def logistic_regression_stochastic_regularized(X, y, learning_rate, lambda_val):
    X = z_score_normalization(X)
    w = np.zeros(X.shape[1])
    return gradient_descent_stochastic_regularized(X, y, w, learning_rate, lambda_val)



# HW Part 2
def hw_part2():
    lambdas = [0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0]

    #mse_results = []
    avg_accuracy_results = []
    avg_training_accuracy_results = []

    for lambda_val in lambdas:
        accuracy = []
        training_accuracy = []
        #mse_values = []

        for i in range(5):
            test = parts[i]
            train = pd.concat([part for j, part in enumerate(parts) if j != i])
            X_train = train.iloc[:, :-1].values
            y_train = train.iloc[:, -1].values
            y_train = np.where(y_train == b'Cammeo', 1, -1)
            X_train = np.insert(X_train, 0, 1, axis=1)
            w, iterations = logistic_regression_regularized(X_train, y_train, 0.05, lambda_val)

            X_train = z_score_normalization(X_train)
            correct = 0
            for j in range(len(X_train)):
                prediction = np.sign(np.dot(w, X_train[j]))
                if prediction == y_train[j]:
                    correct += 1
            training_accuracy.append(correct / len(X_train))

            X_test = test.iloc[:, :-1].values
            X_test = np.insert(X_test, 0, 1, axis=1)
            y_test = test.iloc[:, -1].values
            y_test = np.where(y_test == b'Cammeo', 1, -1)
            X_test = z_score_normalization(X_test)

            correct = 0
            for j in range(len(X_test)):
                prediction = np.sign(np.dot(w, X_test[j]))
                if prediction == y_test[j]:
                    correct += 1
            accuracy.append(correct / len(X_test))

            #mse = mean_squared_error(y_test, np.dot(X_test, w))
            #mse_values.append(mse)

        accuracy_avg = np.mean(accuracy)
        training_accuracy_avg = np.mean(training_accuracy)
        #mse_avg = np.mean(mse_values)

        #mse_results.append(mse_avg)
        avg_accuracy_results.append(accuracy_avg)
        avg_training_accuracy_results.append(training_accuracy_avg)
        print("Lambda: ", lambda_val)
        print("Test Accuracy: ", accuracy_avg * 100)
        print("Training accuracy: ", training_accuracy_avg * 100)
        #print("MSE: ", mse_avg)

    # plot lambda and mse. Found lambda = 0.01 to be the best
    print("Best regularization parameter (lambda)",lambdas[np.argmax(avg_accuracy_results)])

    plt.plot(lambdas, avg_accuracy_results)
    plt.xlabel('Lambda')
    plt.ylabel('Test Accuracy')
    plt.title('Lambda vs Test Accuracy')
    plt.show()

    return lambdas[np.argmax(avg_accuracy_results)]


# HW Part 3 - Found lambda = 0.01 to be the best
def hw_part3(parts):
    best_lambda = hw_part2()
    print("Because of Regularized SGD, this part will take some time")
    accuracy = []
    training_accuracy = []
    exec_times_gd = []
    exec_times_sgd = []
    for i in range(5):
        test = parts[i]
        train = pd.concat([part for j, part in enumerate(parts) if j != i])
        X_train = train.iloc[:, :-1].values # iloc -> integer-location based indexing for selection by position [row, column], : means all rows, :-1 means all columns except the last one
        y_train = train.iloc[:, -1].values
        y_train = np.where(y_train == b'Cammeo', 1, -1)
        X_train = np.insert(X_train, 0, 1, axis=1) # bias term. Insert a column of 1s at the beginning of the matrix
        start_time = time.time()
        w1, iterations1 = logistic_regression(X_train, y_train, 0.05)
        mid_time = time.time()
        w2, iterations2 = logistic_regression_stochastic(X_train, y_train, 0.05)
        end_time = time.time()
        exec_times_gd.append(mid_time-start_time)
        exec_times_sgd.append(end_time-mid_time)
        w3, iterations3 = logistic_regression_regularized(X_train, y_train, 0.05, best_lambda)
        w4, iterations4 = logistic_regression_stochastic_regularized(X_train, y_train, 0.05, best_lambda)
        wArray = [w1, w2, w3, w4]

        # calculate training accuracy
        X_train = z_score_normalization(X_train)

        training_accuracy_of_w = []

        for w in wArray:
            correct = 0
            for j in range(len(X_train)):
                prediction = np.sign(np.dot(w, X_train[j]))
                if prediction == y_train[j]:
                    correct += 1
            training_accuracy_of_w.append(correct / len(X_train))

        training_accuracy.append(training_accuracy_of_w)

        X_test = test.iloc[:, :-1].values
        X_test = np.insert(X_test, 0, 1, axis=1) # bias term. Insert a column of 1s at the beginning of the matrix
        y_test = test.iloc[:, -1].values
        y_test = np.where(y_test == b'Cammeo', 1, -1)
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

    print("average execution time for GD", np.mean(exec_times_gd))
    print("average execution time for SGD", np.mean(exec_times_sgd))
    w1_avg_accuracy = np.mean([accuracy[i][0] for i in range(5)])
    w2_avg_accuracy = np.mean([accuracy[i][1] for i in range(5)])
    w3_avg_accuracy = np.mean([accuracy[i][2] for i in range(5)])
    w4_avg_accuracy = np.mean([accuracy[i][3] for i in range(5)])

    w1_avg_training_accuracy = np.mean([training_accuracy[i][0] for i in range(5)])
    w2_avg_training_accuracy = np.mean([training_accuracy[i][1] for i in range(5)])
    w3_avg_training_accuracy = np.mean([training_accuracy[i][2] for i in range(5)])
    w4_avg_training_accuracy = np.mean([training_accuracy[i][3] for i in range(5)])

    print("Average accuracy for GD: ", w1_avg_accuracy)
    print("Average accuracy for SGD: ", w2_avg_accuracy)
    print("Average accuracy for Regularized GD: ", w3_avg_accuracy)
    print("Average accuracy for Regularized SGD: ", w4_avg_accuracy)

    print("Average training accuracy for GD: ", w1_avg_training_accuracy)
    print("Average training accuracy for SGD: ", w2_avg_training_accuracy)
    print("Average training accuracy for Regularized GD: ", w3_avg_training_accuracy)
    print("Average training accuracy for Regularized SGD: ", w4_avg_training_accuracy)

    # CREATE A PLOT OF BAR CHARTS
    labels = ['GD', 'SGD', 'GD Regularized', 'SGD Regularized']
    accuracy = [w1_avg_accuracy, w2_avg_accuracy, w3_avg_accuracy, w4_avg_accuracy]
    training_accuracy = [w1_avg_training_accuracy, w2_avg_training_accuracy, w3_avg_training_accuracy, w4_avg_training_accuracy]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, accuracy, width, label='Accuracy')
    ax.bar(x + width/2, training_accuracy, width, label='Training Accuracy')

    ax.set_ylabel('Accuracies')
    ax.set_title('Accuracies by method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

# HW Part 4
def hw_part4(parts):
    hw_part3(parts)

    # plot loss values
    loss_values_GD_arr = np.array(loss_values_GD, dtype=object)
    loss_values_SGD_arr = np.array(loss_values_SGD, dtype=object)

    for i in range(5):
        plt.plot(range(len(loss_values_GD_arr[i])), loss_values_GD_arr[i], label='GD')
        plt.plot(range(len(loss_values_SGD_arr[i])), loss_values_SGD_arr[i], label='SGD')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        title_string = "Loss vs Iterations for test part {}".format(i)
        plt.title(title_string)
        plt.legend()
        plt.show()

# HW Part 5
def hw_part5(parts):
    accuracy = []
    training_accuracy = []
    for i in range(5):
        test = parts[i]
        train = pd.concat([part for j, part in enumerate(parts) if j != i])
        X_train = train.iloc[:, :-1].values # iloc -> integer-location based indexing for selection by position [row, column], : means all rows, :-1 means all columns except the last one
        y_train = train.iloc[:, -1].values
        y_train = np.where(y_train == b'Cammeo', 1, -1)
        X_train = np.insert(X_train, 0, 1, axis=1) # bias term. Insert a column of 1s at the beginning of the matrix
        w1, iterations1 = logistic_regression_stochastic(X_train, y_train, 0.01)
        w2, iterations2 = logistic_regression_stochastic(X_train, y_train, 0.1)
        w3, iterations3 = logistic_regression_stochastic(X_train, y_train, 1)
        wArray = [w1, w2, w3]

        # calculate training accuracy
        X_train = z_score_normalization(X_train)

        training_accuracy_of_w = []

        for w in wArray:
            correct = 0
            for j in range(len(X_train)):
                prediction = np.sign(np.dot(w, X_train[j]))
                if prediction == y_train[j]:
                    correct += 1
            training_accuracy_of_w.append(correct / len(X_train))

        training_accuracy.append(training_accuracy_of_w)

        X_test = test.iloc[:, :-1].values
        X_test = np.insert(X_test, 0, 1, axis=1) # bias term. Insert a column of 1s at the beginning of the matrix
        y_test = test.iloc[:, -1].values
        y_test = np.where(y_test == b'Cammeo', 1, -1)
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

    w1_avg_accuracy = np.mean([accuracy[i][0] for i in range(5)])
    w2_avg_accuracy = np.mean([accuracy[i][1] for i in range(5)])
    w3_avg_accuracy = np.mean([accuracy[i][2] for i in range(5)])

    w1_avg_training_accuracy = np.mean([training_accuracy[i][0] for i in range(5)])
    w2_avg_training_accuracy = np.mean([training_accuracy[i][1] for i in range(5)])
    w3_avg_training_accuracy = np.mean([training_accuracy[i][2] for i in range(5)])

    print("Average accuracy for learning rate 0.01: ", w1_avg_accuracy)
    print("Average accuracy for learning rate 0.1: ", w2_avg_accuracy)
    print("Average accuracy for learning rate 1: ", w3_avg_accuracy)

    print("Average training accuracy for learning rate 0.01: ", w1_avg_training_accuracy)
    print("Average training accuracy for learning rate 0.1: ", w2_avg_training_accuracy)
    print("Average training accuracy for learning rate 1: ", w3_avg_training_accuracy)

    # CREATE A PLOT OF BAR CHARTS
    labels = ['SGD 0.01', 'SGD 0.1', 'SGD 1']
    accuracy = [w1_avg_accuracy, w2_avg_accuracy, w3_avg_accuracy]
    training_accuracy = [w1_avg_training_accuracy, w2_avg_training_accuracy, w3_avg_training_accuracy]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, accuracy, width, label='Accuracy')
    ax.bar(x + width/2, training_accuracy, width, label='Training Accuracy')

    ax.set_ylabel('Accuracies')
    ax.set_title('Accuracies by method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

    # Plot loss values
    loss_values_SGD_arr = np.array(loss_values_SGD, dtype=object)

    for i in range(5):
        plt.plot(range(len(loss_values_SGD_arr[3*i])), loss_values_SGD_arr[3*i], label='SGD with learning rate 0.01', color='red')
        plt.plot(range(len(loss_values_SGD_arr[3*i+1])), loss_values_SGD_arr[3*i+1], label='SGD with learning rate 0.1', color='blue')
        plt.plot(range(len(loss_values_SGD_arr[3*i+2])), loss_values_SGD_arr[3*i+2], label='SGD with learning rate 1', color='green')

        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        title_string = "Loss vs Iterations for test part {}".format(i)
        plt.title(title_string)
        plt.legend()
        plt.show()


part_to_run = sys.argv[1]

# Apply 5-fold cross validation
# Divide the dataset into 5 equal parts
# 1 part for testing and 4 parts for training
# Repeat the process 5 times and average the accuracy
# Number of samples = 3810

# First, shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True) # frac=1 -> return all rows in random order, reset_index -> reset the index of the DataFrame, drop=True -> do not save the old index as a column

part1 = df.iloc[0:762, :]
part2 = df.iloc[762:1524, :]
part3 = df.iloc[1524:2286, :]
part4 = df.iloc[2286:3048, :]
part5 = df.iloc[3048:3810, :]
parts = [part1, part2, part3, part4, part5]

if (part_to_run == "4"):
    hw_part4(parts)
elif (part_to_run == "5"):
    hw_part5(parts)