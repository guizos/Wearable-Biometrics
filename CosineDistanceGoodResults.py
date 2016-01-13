from matplotlib.pyplot import plot

from feature import beats
from datasource import original
from classifier import distance
data_folder = original.OriginalDataFolder("../data/original")
ecg_database = beats.ECGBeatLabeledSamplesDatabase(data_folder, 20, True)
(train,test) = ecg_database.get_labeled_training_and_test_samples(1,'ad45363b-7632-432e-a368-215d3fb0ca10',10)
classifier = distance.AverageDistanceClassifier(1,'cosine')
classifier.train(train)
roc_min = classifier.get_roc(test,type='min',activity_info=1)
roc_mean = classifier.get_roc(test,type='mean',activity_info=1)
#roc_sum = classifier.get_roc(test,type='sum',activity_info=1)
plot(roc_min[0],roc_min[1],'r-')
plot(roc_mean[0],roc_mean[1],'y-')
plot([0,1],[1,0],'b-')
#plot(roc_sum[0],roc_sum[1],'b-')
print "Finished"