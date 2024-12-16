from utils import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle


def image_to_reduced_feature(images, split='train'):
    return images[:, :49]
    

def training_model(train_features, train_labels):
    return NullModel()
