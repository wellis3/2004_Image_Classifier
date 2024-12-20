import numpy as np

def image_to_reduced_feature(images, split='train'):

    # Number of priciple components wanted
    principle_components = 80

    #Covarience Matrix is calculated
    covx = np.cov(images, rowvar=0)
    N = covx.shape[0]
    # Eigenvectors and values are calculated
    w, v = np.linalg.eigh(covx)
    # Eigenvectors and values and sorted in descending order of varience and truncated at the number of priciple components
    w = w[-principle_components:]
    v = v[:, -principle_components:]
    v = np.fliplr(v)

    # Mean of the data is found which is used to normalise the data set, mean of 0 and range -1 - 1
    mean_train = np.mean(images, axis=0)

    # Images are reconstruced from the eigenvectors only keeping the most important information
    reconstructed = (
        np.dot(
            np.dot(images - mean_train, v[:, 0: N - 1]),
            v[:, 0: N - 1].transpose(),
        )
        + mean_train
    )
    # The smaller reconstructed images are retruned, reducing the time taken to run the program
    return reconstructed


def training_model(train_features, train_labels):
    model = KNeighboursClassifier()
    model.fit(train_features, train_labels)
    return model

class KNeighboursClassifier():
    def __init__(self, n_neighbours=4):
        self.n_neighbours = n_neighbours
        self.train_data = None
        self.train_labels = None

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        return self

    def predict(self, test_data):
        # Calculate cosine similarity
        x = np.dot(test_data, self.train_data.T)
        modtest = np.sqrt(np.sum(test_data * test_data, axis=1))
        modtrain = np.sqrt(np.sum(self.train_data * self.train_data, axis=1))
        dist = x / np.outer(modtest, modtrain)  # Cosine similarity matrix

        # Convert to a distance measure (1 - similarity)
        distances = 1 - dist

        # Find indices of the nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbours]

        # Retrieve the nearest labels
        nearest_labels = self.train_labels[nearest_indices]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

        # Compute weights as inverse of distances (adding tiny number (does not effect reuslts) in order to avoid devision by zero errors)
        weights = 1 / (nearest_distances + 1e-8)

        # Perform weighted voting
        weighted_votes = np.zeros((test_data.shape[0], np.max(self.train_labels) + 1))  # Assuming labels start from 0
        for i in range(self.n_neighbours):
            for j in range(test_data.shape[0]):
                label = nearest_labels[j, i]
                weighted_votes[j, label] += weights[j, i]

        # Predict the label with the highest weighted vote
        predictions = np.argmax(weighted_votes, axis=1)
        
        # Return the predictions
        return predictions

