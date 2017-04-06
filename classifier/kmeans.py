import logging

import numpy
from sklearn.cluster import KMeans
from sklearn import metrics


class KMeansClassifier:

    def __init__(self):
        self.label = ""
        self.kmeans = KMeans(n_clusters=1)

    def get_parameters_string(self):
        return "kmeans"

    def print_details(self):
        print self.get_details()

    def get_details(self):
        result = "--------------------------\n"
        result += "Classifier type: Kmeans\n"
        result += "Parameters\n"
        result += "--------------------------\n"
        return result

    def train(self,dataset):
        first_label = dataset.feature_vectors[0].label
        for feature_vector in dataset.feature_vectors:
            if feature_vector.label != first_label:
                print "Training set vectors should be of the same label!!!"
                return None
        self.label = dataset.feature_vectors[0].label
        self.kmeans.fit([feature_vector.values for feature_vector in dataset.feature_vectors])
        return None

    def test(self,dataset):
        result = []
        # Returns the area under the roc curve. The higher the better.
        distances = self.kmeans.transform([feature_vector.values for feature_vector in dataset.feature_vectors])
        # We have to assign the opposite labels as in our case, our values do not correspond to confidence
        # values. They correspond to distances, which work in the opposite way. The lower the distance the
        # better.
        labels = [-1 if feature_vector.label == self.label else 1 for feature_vector in dataset.feature_vectors]
        roc_score = metrics.roc_auc_score(y_true=labels,y_score=distances)
        return roc_score
