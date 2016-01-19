import time

from classifier import distance
from data import source
from data.segmentation import ecg

start_time = time.time()
data_folder = source.OriginalDataFolder("original_data")
#extract interpolated beat samples
ecg_database = ecg.ECGBeatLabeledSamplesDatabase(data_folder, 80, True)
# your code
elapsed_time = time.time() - start_time
print "Database Created in " + str(elapsed_time)
start_time = time.time()
(train,test) = ecg_database.get_labeled_training_and_test_samples(1,'ad45363b-7632-432e-a368-215d3fb0ca10',10)
# your code
elapsed_time = time.time() - start_time
print "Train and Test sets Created in " + str(elapsed_time)
start_time = time.time()
classifier = distance.AverageDistanceClassifier(distance_function='cosine')
elapsed_time = time.time() - start_time
print "Classifier Created in " + str(elapsed_time)
start_time = time.time()
classifier.train(train)
elapsed_time = time.time() - start_time
print "Classifier Trained in " + str(elapsed_time)
start_time = time.time()
roc_min_score = classifier.test(test,type='min',activity_info=True)
elapsed_time = time.time() - start_time
print "ROC Min Obtained in " + str(elapsed_time)
print "ROCMIN"
print roc_min_score
start_time = time.time()
roc_mean_score = classifier.test(test,type='mean',activity_info=True)
elapsed_time = time.time() - start_time
print "ROC MEAN Obtained in " + str(elapsed_time)
print "ROCMEAN "
print roc_mean_score
print "Finished"