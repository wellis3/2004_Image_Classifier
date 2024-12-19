import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def image_to_reduced_feature(images, split='train'):

    principle_components = 70

    covx = np.cov(images, rowvar=0)
    N = covx.shape[0]
    w, v = np.linalg.eigh(covx)
    w = w[-principle_components:]
    v = v[:, -principle_components:]
    v = np.fliplr(v)
    v.shape

    N = 70
    mean_train = np.mean(images, axis=0)

    # Create a figure for displaying the images
    plt.figure(figsize=(10, 10))

    reconstructed = (
        np.dot(
            np.dot(images - mean_train, v[:, 0: N - 1]),
            v[:, 0: N - 1].transpose(),
        )
        + mean_train
    )

    return reconstructed


def training_model(train_features, train_labels):
    model = KNeighboursClassifier()
    model.fit(train_features, train_labels)
    return model

class KNeighboursClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbours=6):
        self.n_neighbours = n_neighbours
        self.train_data = None
        self.train_labels = None

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        return self

    def predict(self, test_data):
        # Compute cosine similarity as before
        x = np.dot(test_data, self.train_data.T)
        modtest = np.sqrt(np.sum(test_data * test_data, axis=1))
        modtrain = np.sqrt(np.sum(self.train_data * self.train_data, axis=1))
        dist = x / np.outer(modtest, modtrain)  # Cosine similarity matrix

        # Since we're using cosine similarity, convert it to a distance measure (1 - similarity)
        distances = 1 - dist

        # Find indices of the nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbours]

        # Retrieve the nearest labels
        nearest_labels = self.train_labels[nearest_indices]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

        # Compute weights as inverse of distances (adding a small epsilon to avoid division by zero)
        weights = 1 / (nearest_distances + 1e-8)

        # Perform weighted voting
        weighted_votes = np.zeros((test_data.shape[0], np.max(self.train_labels) + 1))  # Assuming labels start from 0
        for i in range(self.n_neighbours):
            for j in range(test_data.shape[0]):
                label = nearest_labels[j, i]
                weighted_votes[j, label] += weights[j, i]

        # Predict the label with the highest weighted vote
        predictions = np.argmax(weighted_votes, axis=1)

        return predictions

