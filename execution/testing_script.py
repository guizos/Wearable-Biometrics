from classifier import svm
from data import source
from data.features.generation.ecg_only_functions import ecg_fiducial
from experiments.experiment2 import OneClassExperimenter
from data.features.generator import FeatureExtractor
#Read the data folder
data_folder = source.PumpPrimingDataFolder("dataset_activity1")
print "Done reading data folder"

from data.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor(ppg=[["subset",{"start":500,"end":-200}],
                                 ["savitzky_golay",{'window_size':5, 'order':2, 'deriv':0, 'rate':1}],
                                 ["normf", {}]],
                            ecg=[["subset",{"start":500,"end":-200}],
                                 ["savitzky_golay",{'window_size':5, 'order':2, 'deriv':0, 'rate':1}],
                                 ["normf", {}]])

preprocessed_datafolder = preprocessor.preprocess_datafolder(data_folder)
print "Preprocessing done"

from data.segmentation.segmenter import Segmenter

segmenter = Segmenter(datafile=["peak_midpoint_beat_segmentation",{'frequency':100,'length':100}])

segmented_datafolder = segmenter.segment_datafolder(preprocessed_datafolder)
print "Segmentation done"


generator = FeatureExtractor(segmented_datafolder=segmented_datafolder)
dataset = generator.generate_dataset(ecg_fiducial)

classifier = svm.SVMROCClassifier()
classifier.print_details()
exp = OneClassExperimenter(dataset=dataset,classifier=classifier,number_of_tries_per_subject=5)
exp.execute_all_subjects(num_training_samples=60)
print "Done"