from matplotlib.pyplot import plot
from matplotlib import pyplot
from feature import beats
from datasource import original
from classifier import distance
import pickle
import time
start_time = time.time()
with open('ecg_database_smothed_before_splitting.pkl','rb') as input:
    ecg_database = pickle.load(input)
# your code
elapsed_time = time.time() - start_time
print "Database Created " + str(elapsed_time)
start_time = time.time()
(train,test) = ecg_database.get_labeled_training_and_test_samples(1,'ad45363b-7632-432e-a368-215d3fb0ca10',10)
# your code
elapsed_time = time.time() - start_time
print "Train and Test sets Created" + str(elapsed_time)
start_time = time.time()
classifier = distance.AverageDistanceClassifier(1, 'fastdtw')
elapsed_time = time.time() - start_time
print "Classifier Created" + str(elapsed_time)
start_time = time.time()
classifier.train(train)
elapsed_time = time.time() - start_time
print "Classifier Trained" + str(elapsed_time)
start_time = time.time()
roc_min = classifier.get_roc(test,type='min',activity_info=1)
elapsed_time = time.time() - start_time
print "ROC MIN Obtained" + str(elapsed_time)
start_time = time.time()
roc_mean = classifier.get_roc(test,type='mean',activity_info=1)
elapsed_time = time.time() - start_time
print "ROC MEAN Obtained" + str(elapsed_time)
#roc_sum = classifier.get_roc(test,type='sum',activity_info=1)
plot(roc_min[0],roc_min[1],'r-')
plot(roc_mean[0],roc_mean[1],'y-')
plot([0,1],[1,0],'b-')
pyplot.savefig("result.png")
#plot(roc_sum[0],roc_sum[1],'b-')
print "Finished"