import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def kmeans(X, k, max_iter=1000, measure="euclidean"):
    # Initialize Centroids Randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        distances = np.zeros((X.shape[0], k))

        # Assign All Points to the Closest Centroid
        for i, data_point in enumerate(X):
            if measure == "euclidean":
                distances[i, :] = np.sqrt(np.sum((data_point - centroids) ** 2, axis=1))
            elif measure == "cosine":
                dot_product = np.dot(centroids, data_point)
                centroids_norm = np.linalg.norm(centroids, axis=1)
                data_point_norm = np.linalg.norm(data_point)
                distances[i, :] = 1 - (dot_product / (centroids_norm * data_point_norm))
        y = np.argmin(distances, axis=1)

        # Update Centroids
        new_centroids = []
        for i in range(k):
            cluster_points_indices = np.where(y == i)[0]
            if len(cluster_points_indices) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroid = np.mean(X[cluster_points_indices], axis=0)
                new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        # Check Convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return y, centroids


# Clustering Accuracy
def calculate_cluster_accuracy(true_labels, cluster_labels):
    # Relabel true_labels to 0, 1, 2, 3
    relabel_map = {2: 0, 3: 1, 8: 2, 9: 3}
    relabeled_true_labels = np.array([relabel_map[label] for label in true_labels])

    cm = confusion_matrix(relabeled_true_labels, cluster_labels)
    # plot_confusion_matrix(cm, title="Original Confusion Matrix")

    indexes = np.transpose(np.asarray(linear_sum_assignment(-cm + np.max(cm))))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    # plot_confusion_matrix(cm2, title="Reordered Confusion Matrix")

    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy


# SSE (Sum of Squared Errors)
def sse(X, centroids, cluster_labels):
    sse_score = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[cluster_labels == i]
        sse_score += np.sum((cluster_points - centroid) ** 2)
    return sse_score


# Feature extraction using PCA
def pca_feature_extraction(x_train, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    return x_train_pca


def plot_clusters(X, y_train, labels, centroids):
    plt.figure(figsize=(10, 8))

    # Scatter plot of data points
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], c=f'C{labels[i]}')
        plt.text(X[i, 0], X[i, 1], str(int(y_train[i])), fontsize=9)

    # Plot centroids
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], c=f'C{i}', marker='x', s=200, linewidths=3)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('K-Means Clustering with Centroids')
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()


def main(part):
    # Load data
    (x_train, y_train), (_, _) = mnist.load_data()

    # Filter 2,3,8,9
    train_mask = np.isin(y_train, [2, 3, 8, 9])
    X_train = x_train[train_mask]
    y_train = y_train[train_mask]

    # Flatten
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

    # Normalization (Min-Max)
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

    # Define measures
    pca_components = 10
    iter_for_average = 50

    if part == 2:
        measure = "euclidean"
    elif part == 3:
        measure = "cosine"
    else:
        print("Argument should be 2 (euclidean) or 3 (cosine).")

    results = []
    results_pca = []

    for _ in range(iter_for_average):
        # K-means without PCA
        labels, centers = kmeans(X_train, 4, measure=measure)
        accuracy = calculate_cluster_accuracy(y_train, labels)
        sse_score = sse(X_train, centers, labels)
        results.append((accuracy, sse_score))
        #print(f"Accuracy ({measure}): {accuracy}")
        #print(f"SSE ({measure}): {sse_score}")

        # K-means with PCA
        X_train_pca = pca_feature_extraction(X_train, pca_components)
        labels_ext, centers_ext = kmeans(X_train_pca, 4, measure=measure)
        accuracy_ext = calculate_cluster_accuracy(y_train, labels_ext)
        sse_score_ext = sse(X_train_pca, centers_ext, labels_ext)
        results_pca.append((accuracy_ext, sse_score_ext))
        #print(f"Accuracy with PCA ({measure}): {accuracy_ext}")
        #print(f"SSE with PCA ({measure}): {sse_score_ext}")

    # Calculate average results
    avg_accuracy = np.mean([result[0] for result in results])
    avg_sse = np.mean([result[1] for result in results])

    avg_accuracy_pca = np.mean([result[0] for result in results_pca])
    avg_sse_pca = np.mean([result[1] for result in results_pca])

    print(f"Average Clustering Accuracy ({measure.capitalize()}): {avg_accuracy}")
    print(f"Average Clustering Accuracy ({measure.capitalize()}_PCA): {avg_accuracy_pca}")
    print(f"Average SSE Score ({measure.capitalize()}): {avg_sse}")
    print(f"Average SSE Score ({measure.capitalize()}_PCA): {avg_sse_pca}")

    # # For visualization, reduce to 2 PCA components
    # X_train_pca_2d = pca_feature_extraction(X_train, 2)
    # labels_2d, centers_2d = kmeans(X_train_pca_2d, 4, measure=measure)
    #
    # plot_clusters(X_train_pca_2d, y_train, labels_2d, centers_2d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", type=int, choices=[2, 3], help="Specify which part of the script to run")
    args = parser.parse_args()
    main(args.part)
