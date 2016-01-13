import logging
import experiment
from feature import beats
from datasource import original
from classifier import distance
import cPickle as pickle
import time

logging.basicConfig(level=logging.DEBUG)
start_time = time.time()
with open('ecg_database_smoothed_before_splitting.pkl','rb') as input:
    ecg_database = pickle.load(input)
# your code
elapsed_time = time.time() - start_time
print "Database Created in " + str(elapsed_time)
classifier = distance.AverageDistanceClassifier(distance_function='cosine',kind="min")
classifier.print_details()
exp = experiment.OneClassROCAreaExperimenter(labeled_database=ecg_database,classifier=classifier, number_of_tries_per_subject=5,verbose=True)
exp.execute_all_subjects(activity_training=1,num_training_samples=5)
print "Finished"