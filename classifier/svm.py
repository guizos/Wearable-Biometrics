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

    def train(self,dataset):
        first_label = dataset.feature_vectors[0].label
        for feature_vector in dataset.feature_vectors:
            if feature_vector.label != first_label:
                print "Training set vectors should be of the same label!!!"
                return None
        self.user_id = dataset.feature_vectors[0].label
        self.svm.fit([feature_vector.values for feature_vector in dataset.feature_vectors])

    def test(self,dataset):
        # each row is a labeled_sample
        samples = [feature_vector.values for feature_vector in dataset.feature_vectors]
        result = []
        labels = [1 if feature_vector.label == self.user_id else -1 for feature_vector in dataset.feature_vectors]
        predictions = self.svm.predict(samples)
        tp = len([1 for index in range(len(predictions)) if labels[index]==1==predictions[index]])
        tn = len([1 for index in range(len(predictions)) if labels[index]==-1==predictions[index]])
        fp = len([1 for index in range(len(predictions)) if labels[index]==-1 and predictions[index]==1])
        fn = len([1 for index in range(len(predictions)) if labels[index]==1 and predictions[index]==-1])
        result.append([tp, tn, fp, fn])
        return result

class SVMROCClassifier(SVMClassifier):

    def test(self,dataset):
        # each row is a labeled_sample
        samples = [feature_vector.values for feature_vector in dataset.feature_vectors]
        result = []
        labels = [1 if feature_vector.label == self.user_id else -1 for feature_vector in dataset.feature_vectors]
        distances = self.svm.decision_function(samples)
        roc_score = metrics.roc_auc_score(y_true=labels,y_score=distances)
        return roc_score
