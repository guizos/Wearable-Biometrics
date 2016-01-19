import logging

import numpy
from sklearn.cluster import KMeans
from sklearn import metrics


class KMeansClassifier:

    def __init__(self):
        self.user_id = ""
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

    def train(self,labeled_samples):
        self.user_id = labeled_samples[0].user_id
        self.kmeans.fit([labeled_sample.data for labeled_sample in labeled_samples])
        logging.info("Training finished")
        return None

    def test(self,labeled_samples,activity_info=None):
        result = []
        # Returns the area under the roc curve. The higher the better.
        distances = self.kmeans.transform([labeled_sample.data for labeled_sample in labeled_samples])
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
                    activity_roc_score = metrics.roc_auc_score(y_true=labels_activity,y_score=distances_activity)
                    result.append(activity_roc_score)
                elif len(set(labels_activity))>1:
                    result.append(0.5)
        logging.info("Test finished")
        return result
