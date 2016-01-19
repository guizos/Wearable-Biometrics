import logging

from sklearn import metrics
from sklearn.svm import OneClassSVM


class SVMClassifier:


    def __init__(self,nu=0.1,kernel="rbf", gamma=0.1):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def get_parameters_string(self):
        return self.kernel+"_"+str(self.nu)+"_"+str(self.gamma)

    def print_details(self):
        print self.get_details()

    def get_details(self):
        result = "--------------------------\n"
        result += "Classifier type: SVM\n"
        result += "Parameters\n"
        result += "Kernel :"+self.kernel+"\n"
        result += "nu :"+str(self.nu)+"\n"
        result += "gamma :"+str(self.gamma)+"\n"
        result += "--------------------------\n"
        return result

    def train(self,labeled_samples):
        self.user_id = labeled_samples[0].user_id
        self.svm.fit([labeled_sample.data for labeled_sample in labeled_samples])
        logging.info("Training finished")

    def test(self,labeled_samples,activity_info=None):
        # each row is a labeled_sample
        samples = [labeled_sample.data for labeled_sample in labeled_samples]
        result = []
        labels = [1 if labeled_sample.user_id == self.user_id else -1 for labeled_sample in labeled_samples]
        predictions = self.svm.predict(samples)
        tp = len([1 for index in range(len(predictions)) if labels[index]==1==predictions[index]])
        tn = len([1 for index in range(len(predictions)) if labels[index]==-1==predictions[index]])
        fp = len([1 for index in range(len(predictions)) if labels[index]==-1 and predictions[index]==1])
        fn = len([1 for index in range(len(predictions)) if labels[index]==1 and predictions[index]==-1])
        result.append([tp, tn, fp, fn])
        if activity_info:
            for i in range(4):
                tp_a = len([1 for index in range(len(predictions)) if labels[index]==1==predictions[index] and labeled_samples[index].activity==i+1])
                tn_a = len([1 for index in range(len(predictions)) if labels[index]==-1==predictions[index] and labeled_samples[index].activity==i+1])
                fp_a = len([1 for index in range(len(predictions)) if labels[index]==-1 and predictions[index]==1 and labeled_samples[index].activity==i+1])
                fn_a = len([1 for index in range(len(predictions)) if labels[index]==1 and predictions[index]==-1 and labeled_samples[index].activity==i+1])
                result.append([tp_a, tn_a, fp_a, fn_a])
        logging.info("Test finished")
        return result

class SVMROCClassifier(SVMClassifier):

    def test(self,labeled_samples,activity_info=None):
        # each row is a labeled_sample
        samples = [labeled_sample.data for labeled_sample in labeled_samples]
        result = []
        labels = [1 if labeled_sample.user_id == self.user_id else -1 for labeled_sample in labeled_samples]
        distances = self.svm.decision_function(samples)
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
