import cPickle as pickle
import experiment
from feature import beats
from classifier import svm
#get data folder
#data_folder = original.OriginalDataFolder("../data/original")
#extract interpolated beat samples
#ecg_database = beats.ECGBeatLabeledSamplesDatabase(data_folder, 80, True)
with open('ecg_database_smothed_before_splitting.pkl','rb') as input:
    ecg_database = pickle.load(input)
print "Database read. Beginning experiments"
nus = [0.01,0.05,0.2,0.3,0.4,0.5,0.6,0.7]
kernels = ["rbf","linear","poly","sigmoid"]
gammas = [0.01, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
number_training_samples = [5, 10, 15, 20, 30]
activity_training = [0, 1, 2, 4]
for activity in activity_training:
    for nu in nus:
        for kernel in kernels:
            for gamma in gammas:
                for n_train in number_training_samples:
                    classifier = svm.SVMClassifier(nu=nu,kernel=kernel,gamma=gamma)
                    classifier.print_details()
                    exp = experiment.OneClassExperimenter(ecg_database,classifier,10,verbose=True,output_foler="results_svm")
                    exp.execute_all_subjects(activity,n_train)


