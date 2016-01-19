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

    def train(self,labeled_samples):
        self.user_id = labeled_samples[0].user_id
        distances = self.distance([labeled_sample.data for labeled_sample in labeled_samples],
                                  metric=self.distance_function)
        self.mean = numpy.mean(distances)
        self.std = numpy.std(distances)
        self.train_samples = labeled_samples
        logging.info("Training finished")
        return self.mean,self.std

    def get_mean_distance(self, sample):
        samples_train = [train_sample.data for train_sample in self.train_samples]
        return numpy.mean([self.distance([sample,float_train_sample],metric=self.distance_function) for float_train_sample in samples_train])

    def get_sum_distance(self,sample):
        samples_train = [train_sample.data for train_sample in self.train_samples]
        return numpy.sum([self.distance([sample,float_train_sample],metric=self.distance_function) for float_train_sample in samples_train])

    def get_min_distance(self,sample):
        samples_train = [train_sample.data for train_sample in self.train_samples]
        return numpy.min([self.distance([sample, float_train_sample], metric=self.distance_function) for float_train_sample in samples_train])

    def test(self,labeled_samples,activity_info=None):
        result = []
        # Returns the area under the roc curve. The higher the better.
        if self.kind == 'mean':
            distances = [self.get_mean_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        elif self.kind == 'min':
            distances = [self.get_min_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        else:
            distances = [self.get_sum_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        # We have to assign the opposite labels as in our case, our values do not correspond to confidence
        # values. They correspond to distances, which work in the opposite way. The lower the distance the
        # better.
        labels = [-1 if labeled_sample.user_id == self.user_id else 1 for labeled_sample in labeled_samples]
        roc_score = metrics.roc_auc_score(y_true=labels,y_score=distances)
        result.append(roc_score)
        if activity_info:
            for i in range(4):
                labels_activity = [-1 if labeled_sample.user_id == self.user_id else 1 for labeled_sample in labeled_samples if labeled_sample.activity==i+1]
                distances_activity = [distances[j] for j in range(len(distances)) if labeled_samples[j].activity==i+1]
                if len(labels_activity) > 0 and len(set(labels_activity))>1:
                    activity_roc_score =  metrics.roc_auc_score(y_true=labels_activity,y_score=distances_activity)
                    result.append(activity_roc_score)
                elif len(set(labels_activity))>1:
                    result.append(0.5)
        logging.info("Test finished")
        return result


    def get_roc(self,labeled_samples,activity_info=None,type='mean',):
        if type == 'mean':
            distances = [self.get_mean_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        elif type == 'min':
            distances = [self.get_min_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        else:
            distances = [self.get_sum_distance(labeled_sample.data) for labeled_sample in labeled_samples]
        labels = [1 if labeled_sample.user_id == self.user_id else -1 for labeled_sample in labeled_samples]
        fpr, tpr, thresholds = metrics.roc_curve(labels, distances,pos_label=-1)
        return fpr, tpr, thresholds
