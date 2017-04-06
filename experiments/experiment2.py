import os

import numpy


class OneClassExperimenter:

    def __init__(self, dataset, classifier, number_of_tries_per_subject):
        self.dataset = dataset
        self.classifier = classifier
        self.tries = number_of_tries_per_subject

    def execute_all_subjects(self,num_training_samples):
        user_ids = self.dataset.get_all_labels()
        results = []
        for user_id in user_ids:
            this_user_result = self.execute_subject(user_id,num_training_samples)
            results.append(this_user_result)
        final_average = numpy.mean(results,axis=0)
        print "Final average of execution is: {0}".format(final_average)
        return final_average

    def execute_subject(self,user_id,num_training_samples):
        results = []
        print "{0} : ".format(user_id),
        for i in range(self.tries):
            (train,test) = self.dataset.split_dataset(label=user_id,num_feature_vectors=num_training_samples)
            self.classifier.train(train)
            this_try_result = self.classifier.test(test)
            results.append(this_try_result)
            print "{0}, ".format(this_try_result),
        average_result_tables = numpy.mean(results,axis=0)
        print average_result_tables
        return average_result_tables

