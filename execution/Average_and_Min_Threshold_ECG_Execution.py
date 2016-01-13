import cPickle as pickle
from feature import beats
import experiment
from classifier import distance
#get data folder
#data_folder = original.OriginalDataFolder("../data/original")
#extract interpolated beat samples
#ecg_database = beats.ECGBeatLabeledSamplesDatabase(data_folder, 80, True)
with open('ecg_database_smoothed_before_splitting.pkl','rb') as input:
    ecg_database = pickle.load(input)
print "Database read. Beginning experiments"
distances = distance.available_distance_functions
number_training_samples = [5, 10, 15, 20, 30]
activity_training = [0, 1, 2, 4]
num_repetitions = 10
kinds = distance.available_distance_threshold_types
for activity in activity_training:
    for distance in distances:
        for kind in kinds:
            for n_train in number_training_samples:
                classifier = distance.AverageDistanceClassifier(distance_function=distance,kind=kind)
                classifier.print_details()
                exp = experiment.OneClassROCAreaExperimenter(labeled_database=ecg_database, classifier=classifier,
                                                             number_of_tries_per_subject=num_repetitions, verbose=True,
                                                             output_foler="results_ROCArea")
                exp.exectue_all_subjects(activity_training=activity,num_training_samples=n_train)


