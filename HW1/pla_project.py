import numpy as np
import matplotlib.pyplot as plt
import sys

def select_point(w, dataset, labels):
    """

    :param w: the list of current weights with three elements
    :param dataset: 2d n*3 numpy array
    :param labels: 1d numpy array of labels
    :return: index of a misclassified sample
    """
    for i in range(len(dataset)):
        dot_product = np.dot(w.T, dataset[i])
        if np.all(w == 0):
            # since it is a zero vector, every point is misclassified. return the first one
            return 0
        if labels[i] * dot_product < 0:
            # return a misclassified sample
            return i
    # every sample is correctly classified
    return -1


def update_weight(w, x, y):
    """

    :param w: the list of current weights with three elements
    :param x: the list of coordinates of a data point with three elements
    :param y: label of a data point. either 1 or -1
    :return: the updated weights w
    """
    for i in range(len(w)):
        w[i] += y * x[i]
    return w


def plot_line(slope, intercept):
    """

    :param slope: slope of the decision boundary
    :param intercept: intercept of the decision boundary
    """
    # Generate x values for the line
    x_values = np.linspace(0, 1, 100)  # Generate 100 x values from -10 to 10
    # Calculate y values using the line equation y = slope * x + intercept
    y_values = slope * x_values + intercept
    # Plot the line
    plt.plot(x_values, y_values, label='y = {slope}x + {intercept}', color='blue')

part_to_run = sys.argv[1]

# to use the large dataset, uncomment these lines and comment the next two lines
if part_to_run == 'large':
    data = np.load('data_large.npy')
    labels = np.load('label_large.npy')
elif part_to_run == 'small':
    data = np.load('data_small.npy')
    labels = np.load('label_small.npy')

data_size = len(data)
# initialize the weights
w = np.array([0.0,0.0,0.0])
# select the index of a misclassified sample
point = select_point(w, data, labels)
# initialize the number of iterations
num_iterations = 0
# update weights until all samples are correctly classified
while point != -1:
    # update weights with the current misclassified sample
    w = update_weight(w, data[point], labels[point])
    # select the index of a misclassified sample
    point = select_point(w, data, labels)
    num_iterations += 1

# print the decision boundary and the total number of iterations
print(w, num_iterations)
# plot the dataset with different markers according to their labels
plt.plot([data[i][1] for i in range(data_size) if labels[i] == 1.0], [data[i][2] for i in range(data_size) if labels[i] == 1.0], marker='o', linestyle='', label="'o' points")
plt.plot([data[i][1] for i in range(data_size) if labels[i] == -1.0], [data[i][2] for i in range(data_size) if labels[i] == -1.0], marker='s', linestyle='', label="'s' points")
# calculate the slope and the intercept and then plot the decision boundary
slope = -1*w[1]/w[2]
intercept = -1*w[0]/w[2]
plot_line(slope, intercept)

plt.xlabel('First attributes')  # Set X axis label
plt.ylabel('Second attributes')  # Set Y axis label
plt.title(f'y={slope}x+{intercept} or {w}')
plt.grid(True)  # Show grid
plt.show()  # Show the plot