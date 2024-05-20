# Soft-margin SVM
from keras.datasets import mnist
import numpy as np
from cvxopt import matrix, solvers, spmatrix, sparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# shuffle the data and take only 50000 training samples and 10000 test samples
np.random.seed(0)
indices = np.random.permutation(x_train.shape[0])
x_train = x_train[indices]
y_train = y_train[indices]
indices = np.random.permutation(x_test.shape[0])
x_test = x_test[indices]
y_test = y_test[indices]
x_train = x_train[:50000]
y_train = y_train[:50000]
x_test = x_test[:10000]
y_test = y_test[:10000]

# take into account only 2, 3, 8, 9 digits
mask_list = [2, 3, 8, 9] # contains the indices of the images that are 2, 3, 8 or 9

train_mask = np.isin(y_train, mask_list) # is in, returns a boolean array of the same shape as y_train
test_mask = np.isin(y_test, mask_list)

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

# flatten the image 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Helper method for part1a
def create_Q_sparse_matrix(N, d):
    values = []
    rows = []
    cols = []

    # the first d+1 diagonal elements are equal to 1
    for i in range(d+1):
        values.append(1.0)
        rows.append(i)
        cols.append(i)

    matrix = spmatrix(values, rows, cols, (N+d+1, N+d+1))
    return matrix

# Helper method for part1a
def create_A_sparse_matrix(N, d, y):
    values = []
    rows = []
    cols = []

    for i in range(0, d):
        for j in range(N):
            values.append(-y[j]*x_train[j, i])
            rows.append(j)
            cols.append(i)

    for j in range(N):
        values.append(-y[j])
        rows.append(j)
        cols.append(d)

    for i in range(d + 1, d + N + 1):
        values.append(-1.0)
        rows.append(i - d - 1)
        cols.append(i)

    for i in range(d + 1, d + N + 1):
        values.append(-1.0)
        rows.append(i - d - 1 + N)
        cols.append(i)

    matrix = spmatrix(values, rows, cols, (2*N, N+d+1))

    return matrix

# ---------------------------- SVM ----------------------------
# 4-class linear SVM with one-vs-all strategy using cvxopt qp solver
# including the regularization parameter C and the tolerance epsilon
def one_vs_all_linear_svm(x_train, y_train, x_test, y_test, C):
    N = y_train.shape[0] # number of training samples
    d = x_train.shape[1] # number of features
    Q, p, A, c = None, None, None, None
    # Q is a matrix of size (N+d+1) x (N+d+1) with the first d+1 diagonal elements equal to 1
    Q = create_Q_sparse_matrix(N, d)
    p = np.zeros(N+d+1)
    p[d+1:] = C
    c = np.zeros(2*N)
    c[:N] = -1.0
    p = matrix(p)
    c = matrix(c)

    test_accuracy_arr = []
    training_accuracy_arr = []

    for i in mask_list:
        # convert the multi-class problem into a one-vs-all problem
        y = np.zeros(N)
        y[y_train == i] = 1
        y[y_train != i] = -1

        A = create_A_sparse_matrix(N, d, y)

        sol = solvers.qp(Q, p, A, c)
        # u is the solution of the optimization problem
        u = np.array(sol['x']).flatten()

        # calculate the bias term
        w = u[0:d]
        b = u[d]
        epsilon = u[d+1:]

        # calculate the training accuracy
        y_pred = np.sign(np.dot(x_train, w) + b)
        y_pred[y_pred == -1] = 0

        y_train_temp = np.zeros(y_train.shape)
        y_train_temp[y_train == i] = 1
        y_train_temp[y_train != i] = 0

        accuracy = np.mean(y_pred == y_train_temp)
        training_accuracy_arr.append(accuracy)
        print('Training accuracy for class', i, ':', accuracy)

        # test the model
        y_pred = np.sign(np.dot(x_test, w) + b)
        y_pred[y_pred == -1] = 0

        y_test_temp = np.zeros(y_test.shape)
        y_test_temp[y_test == i] = 1
        y_test_temp[y_test != i] = 0

        accuracy = np.mean(y_pred == y_test_temp)
        test_accuracy_arr.append(accuracy)
        print('Test accuracy for class', i, ':', accuracy)

    print('Overall test accuracy:', test_accuracy_arr)
    print('Overall training accuracy:', training_accuracy_arr)

    print('Average test accuracy:', np.mean(test_accuracy_arr))
    print('Average training accuracy:', np.mean(training_accuracy_arr))

def part1a(x_train, y_train, x_test, y_test, C_arr):
    for C in C_arr:
        time_start = time.time()
        one_vs_all_linear_svm(x_train, y_train, x_test, y_test, C)
        time_end = time.time()
        print('Time:', time_end - time_start, 'seconds')
        print('--------------------------------')





# PART B - SVM with scikit-learn ------------------------------------------------
# SVM with scikit-learn
def part1b(x_train, y_train, x_test, y_test, C_arr):
    for C in C_arr:
        time_start = time.time()
        clf = SVC(C=C, kernel='linear')
        clf.fit(x_train, y_train)

        # test the model
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        time_end = time.time()
        print('Test accuracy with C =', C, ':', accuracy)
        print('Time:', time_end - time_start, 'seconds')

        # training accuracy
        y_pred = clf.predict(x_train)
        accuracy = accuracy_score(y_train, y_pred)
        print('Training accuracy with C =', C, ':', accuracy)
        print('--------------------------------')







# ---------------------------- SVM DUAL ----------------------------
# PART C - non-linear SVM with one-vs-all strategy from scratch using qp solver dual form
def rbf_kernel(x, y, gamma):
    return np.exp(-gamma*np.linalg.norm(x - y)**2) # np.linalg.norm(x - y) is the Euclidean distance

def rbf_kernel_matrix_form(XN, X, gamma):
    # XN is a matrix of size N x d, where N is the number of samples and d is the number of features
    # X is a matrix of size M x d, where M is the number of samples and d is the number of features
    # The function returns a matrix of size N x M, where K[i, j] = K(XN[i], X[j])
    N = XN.shape[0]
    M = X.shape[0]
    K = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            K[i, j] = rbf_kernel(XN[i], X[j], gamma)

    return K
    
def g_func(xn, X, alpha, b, gamma, y):
    # first arg of np.dot is a vector of size N, second arg is b, which is a scalar
    # to add b to all elements of the vector, we need to add b to the dot product
    dot_product = np.dot(alpha*y, rbf_kernel_matrix_form(xn, X, gamma))
    dot_product = dot_product + b
    return np.sign(dot_product)

# Helper method for part1c
def create_Q_matrix(N, y, X, gamma):
    Q = np.zeros((N, N))
    # since Q is symmetric, we only need to compute the upper triangular part
    for i in range(N):
        for j in range(i, N):
            Q[i, j] = y[i]*y[j]*rbf_kernel(X[i], X[j], gamma)
            Q[j, i] = Q[i, j]

    return Q

# PART C - non-linear SVM with one-vs-all strategy from scratch using qp solver dual form
def one_vs_all_nonlinear_svm(x_train, y_train, x_test, y_test, C, gamma):
    N = y_train.shape[0] # number of training samples
    d = x_train.shape[1] # number of features
    Q, p, A, c, G, b_vec = None, None, None, None, None, None
    # Q: The matrix representing the quadratic coefficients in the objective function.
    # p: The vector representing the linear coefficients in the objective function.
    # A: The matrix representing the coefficients of the inequality constraints
    # c: The vector representing the right-hand side of the inequality constraints
    # G: The matrix representing the coefficients of the equality constraints
    # b_vec: The vector representing the right-hand side of the equality constraints

    # Q is a matrix of size N x N, where Q[i, j] = y[i]*y[j]*K(x[i], x[j]) K is the kernel function, here we use the RBF kernel
    # RBF kernel: K(x, y) = exp(-gamma*||x-y||^2)
    Q = create_Q_matrix(N, y_train, x_train, gamma)
    Q = matrix(Q)

    p = -np.ones(N)
    p = matrix(p)
    c = np.zeros(N)
    c = matrix(c)

    A = np.zeros((N, N))
    # A matrix is for the inequality constraints: A*u <= c, for alpha >= 0
    # A is a sparse matrix of size N x N, where A[i, i] = -1
    values = []
    rows = []
    cols = []

    for i in range(N):
        values.append(-1.0)
        rows.append(i)
        cols.append(i)

    A = spmatrix(values, rows, cols, (N, N))

    b_vec = np.zeros(1)
    b_vec = matrix(b_vec)

    test_accuracy_arr = []
    training_accuracy_arr = []

    for i in mask_list:
        # convert the multi-class problem into a one-vs-all problem
        y = np.zeros(N)
        y[y_train == i] = 1
        y[y_train != i] = -1
        
        # Also, there is another constraint: sum(alpha[i]*y[i]) = 0
        # G is a sparse matrix of size 1 x N, where G[0, i] = y[i]
        G = np.zeros((1, N))
        G[0, :] = y
        G = matrix(G)

        # solve the quadratic programming problem
        sol = solvers.qp(Q, p, A, c, G, b_vec)
        u = np.array(sol['x']).flatten()

        # find the support vectors
        support_vector_indices = np.where(u > 0)[0]

        b = y[support_vector_indices] - np.dot(u*y, rbf_kernel_matrix_form(x_train, x_train[support_vector_indices], gamma))
        b = np.mean(b)
        
        # calculate the training accuracy
        # size of the b vector here is number of training samples, but it should be number of test samples
        y_pred = g_func(x_train[support_vector_indices], x_train, u[support_vector_indices], b, gamma, y)
        y_pred[y_pred == -1] = 0

        y_train_temp = np.zeros(y_train.shape)
        y_train_temp[y_train == i] = 1
        y_train_temp[y_train != i] = 0

        accuracy = np.mean(y_pred == y_train_temp)
        training_accuracy_arr.append(accuracy)
        print('Training accuracy for class', i, ':', accuracy)

        # test the model
        y_pred = g_func(x_train[support_vector_indices], x_test, u[support_vector_indices], b, gamma, y)
        y_pred[y_pred == -1] = 0

        y_test_temp = np.zeros(y_test.shape)
        y_test_temp[y_test == i] = 1
        y_test_temp[y_test != i] = 0

        accuracy = np.mean(y_pred == y_test_temp)
        test_accuracy_arr.append(accuracy)
        print('Test accuracy for class', i, ':', accuracy)

    print('Overall test accuracy:', test_accuracy_arr)
    print('Overall training accuracy:', training_accuracy_arr)

def part1c(x_train, y_train, x_test, y_test, C_arr):
    for C in C_arr:
        time_start = time.time()
        one_vs_all_nonlinear_svm(x_train, y_train, x_test, y_test, C, gamma=10)
        time_end = time.time()
        print('Time:', time_end - time_start)
        print('--------------------------------')





# PART D - non-linear SVM with scikit-learn
def part1d(x_train, y_train, x_test, y_test):
    gamma_arr = [1e-2, 1e-2, 1e-1]
    for gamma in gamma_arr:
        time_start = time.time()
        clf = SVC(kernel='poly', gamma=gamma)
        clf.fit(x_train, y_train)

        # test the model
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        time_end = time.time()
        print('Time:', time_end - time_start, 'seconds')
        print('Test accuracy with gamma =', gamma, ':', accuracy)

        # training accuracy
        y_pred = clf.predict(x_train)
        accuracy = accuracy_score(y_train, y_pred)
        print('Training accuracy with gamma =', gamma, ':', accuracy)
        print('--------------------------------')




# ---------------------------- PCA ----------------------------


# part 2
# Feature extraction using PCA

def pca_feature_extraction(x_train, x_test, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca

def part2():
    n_components_arr = [10, 20, 50]
    for n_components in n_components_arr:
        x_train_pca, x_test_pca = pca_feature_extraction(x_train, x_test, n_components)
        print('Number of components:', n_components)
        print(x_train_pca.shape)
        #part1a(x_train_pca, y_train, x_test_pca, y_test, [1e-2])
        print('--------------------------------')
        print('Number of components:', n_components)
        part1b(x_train_pca, y_train, x_test_pca, y_test, [1e-2])
        print('--------------------------------')
        print('Number of components:', n_components)
        #part1c(x_train_pca, y_train, x_test_pca, y_test)
        print('--------------------------------')
        print('Number of components:', n_components)
        #part1d(x_train_pca, y_train, x_test_pca, y_test)
        print('--------------------------------')




# ---------------------------- Support Vectors ----------------------------


# part 3
def part3(x_train, y_train, x_test, y_test):
    clf = SVC(C=1e-4, kernel='poly', gamma=10)
    clf.fit(x_train, y_train)

    # test the model
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Test accuracy:', accuracy)

    # training accuracy
    y_pred = clf.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print('Training accuracy:', accuracy)

    print('Number of support vectors:', len(clf.support_))
    five_support_vectors_indices = clf.support_[:5]
    print('5 Support vector indices:', five_support_vectors_indices)
    five_not_support_vectors_indices = np.setdiff1d(np.arange(x_train.shape[0]), clf.support_)[:5]
    print('5 Not support vector indices:', five_not_support_vectors_indices)

    # inspect the images of the support vectors of 5 images
    for j in range(5):
        plt.imshow(x_train[clf.support_[five_support_vectors_indices[j]]].reshape(28, 28), cmap='gray')
        plt.show()

    # inspect the images of the not support vectors of 5 images
    for j in range(5):
        plt.imshow(x_train[clf.support_[five_not_support_vectors_indices[j]]].reshape(28, 28), cmap='gray')
        plt.show()





# --------------- MAIN ---------------
if __name__ == '__main__':
    # take program arguments to run the desired part
    part_to_run = sys.argv[1]
    if part_to_run == '1a':
        part1a(x_train, y_train, x_test, y_test, [1e-4, 1e-2, 1e-1, 1e1])
    elif part_to_run == '1b':
        part1b(x_train, y_train, x_test, y_test, [1e-4, 1e-2, 1e-1, 1e1])
    elif part_to_run == '1c':
        print('Running part 1c')
        part1c(x_train, y_train, x_test, y_test, [1e-4, 1e-2, 1e-1, 1e1])
    elif part_to_run == '1d':
        part1d(x_train, y_train, x_test, y_test)
    elif part_to_run == '2':
        part2()
    elif part_to_run == '3':
        part3(x_train, y_train, x_test, y_test)
    else:
        print('Invalid part number')
        sys.exit(1)