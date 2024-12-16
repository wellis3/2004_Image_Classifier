import scipy

from utils import *
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
from scipy.linalg import eigh


def image_to_reduced_feature(images, split='train'):

    mean_images = np.mean(images, axis=0)
    std_images = np.std(images, axis=0) + 1e-8
    standardized_images = (images - mean_images) / std_images

    # Compute covariance matrix
    cov_matrix = np.cov(standardized_images, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    num_components = 100

    # Project the data onto the top components
    principal_components = np.dot(standardized_images, eigenvectors[:, :num_components])

    return principal_components


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
        # Super compact implementation of nearest neighbour
        x = np.dot(test_data, self.train_data.T)
        modtest = np.sqrt(np.sum(test_data * test_data, axis=1))
        modtrain = np.sqrt(np.sum(self.train_data * self.train_data, axis=1))
        dist = x / np.outer(modtest, modtrain)  # Cosine similarity matrix

        nearest_indices = np.argsort(dist, axis=1)[:, -self.n_neighbours:]

        nearest_labels = self.train_labels[nearest_indices]

        predictions = mode(nearest_labels, axis=1).mode.flatten()

        return predictions
