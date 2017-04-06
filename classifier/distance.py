import itertools
import logging

import numpy
from scipy.spatial.distance import pdist
from sklearn import metrics
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial import distance

available_distance_functions = ['fastdtw','dtw','euclidean', 'cityblock','seuclidean','sqeuclidean','cosine','correlation','chebyshev','canberra','braycurtis','mahalanobis']
available_distance_threshold_types = ["mean","min"]

class AverageDistanceClassifier:

    def __init__(self, distance_function='euclidean', kind="mean"):
        if distance_function not in available_distance_functions:
            raise Exception
        self.distance_function = distance_function
        self.kind = kind
        self.user_id = None
        self.mean = 0
        self.std = 0
        self.train_samples = 0

    def get_parameters_string(self):
        return self.distance_function+"_"+self.kind

    def print_details(self):
        print self.get_details()

    def get_details(self):
        result = "--------------------------\n"
        result += "Classifier type: Distance\n"
        result += "Parameters\n"
        result += "Distance Function :"+self.distance_function+" \n"
        result += "Threshold type :"+self.kind+" \n"
        result += "--------------------------\n"
        return result

    def distance(self, X, metric='euclidean'):
        if len(X)==2:
            if metric == 'dtw':
                dist, cost, acc, path = dtw(X[0], X[1], dist=distance.euclidean)
                return dist
            elif metric == 'fastdtw':
                dist, path = fastdtw(X[0], X[1], dist=distance.euclidean)
                return dist
            else:
                return pdist(X,metric=metric)
        elif len(X)>2:
            pairs = itertools.combinations(X,2)
            if metric == 'dtw' or metric == 'fastdtw':
                return [self.distance(pair,metric) for pair in pairs]
            else:
                return pdist(X,metric)
        else:
            return [0]

    def train(self, dataset):
        first_label = dataset.feature_vectors[0].label
        for feature_vector in dataset.feature_vectors:
            if feature_vector.label != first_label:
                print "Training set vectors should be of the same label!!!"
                return None
        self.user_id = dataset.feature_vectors[0].label
        to_measure = [feature_vector.values for feature_vector in dataset.feature_vectors]
        distances = self.distance(numpy.array(to_measure),
                                  metric=self.distance_function)
        self.mean = numpy.mean(distances)
        self.std = numpy.std(distances)
        self.train_samples = dataset.feature_vectors
        return self.mean,self.std

    def get_mean_distance(self, feature_vector):
        samples_train = [train_sample.values for train_sample in self.train_samples]
        return numpy.mean([self.distance([feature_vector, float_train_sample], metric=self.distance_function) for float_train_sample in samples_train])

    def get_sum_distance(self, feature_vector):
        samples_train = [train_sample.values for train_sample in self.train_samples]
        return numpy.sum([self.distance([feature_vector, float_train_sample], metric=self.distance_function) for float_train_sample in samples_train])

    def get_min_distance(self, feature_vector):
        samples_train = [train_sample.values for train_sample in self.train_samples]
        return numpy.min([self.distance([feature_vector, float_train_sample], metric=self.distance_function) for float_train_sample in samples_train])

    def test(self,dataset):
        result = []
        # Returns the area under the roc curve. The higher the better.
        if self.kind == 'mean':
            distances = [self.get_mean_distance(feature_vector.values) for feature_vector in dataset.feature_vectors]
        elif self.kind == 'min':
            distances = [self.get_min_distance(feature_vector.values) for feature_vector in dataset.feature_vectors]
        else:
            distances = [self.get_sum_distance(feature_vector.values) for feature_vector in dataset.feature_vectors]
        # We have to assign the opposite labels as in our case, our values do not correspond to confidence
        # values. They correspond to distances, which work in the opposite way. The lower the distance the
        # better.
        labels = [-1 if feature_vector.label == self.user_id else 1 for feature_vector in dataset.feature_vectors]
        roc_score = metrics.roc_auc_score(y_true=labels,y_score=distances)
        return roc_score


    def get_roc(self,labeled_samples,type='mean',):
        if type == 'mean':
            distances = [self.get_mean_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        elif type == 'min':
            distances = [self.get_min_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        else:
            distances = [self.get_sum_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        labels = [1 if labeled_sample.user_id == self.user_id else -1 for labeled_sample in labeled_samples]
        fpr, tpr, thresholds = metrics.roc_curve(labels, distances,pos_label=-1)
        return fpr, tpr, thresholds
