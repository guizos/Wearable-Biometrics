import inspect

from classifier import svm, distance, kmeans
from data import source
from data.features.generation import ecg_only_functions
from data.features.generation import multimodal_functions
from data.features.generation import single_signal_functions
from data.preprocessing.preprocessor import Preprocessor
from data.segmentation.segmenter import Segmenter
from experiments.experiment2 import OneClassExperimenter
from data.features.generator import FeatureExtractor


#Read the data folder
data_folder = source.PumpPrimingDataFolder("dataset_activity1")
print "** Done reading data folder **"

# Preprocessing
preprocessor = Preprocessor(ecg=[["subset",{"start":500,"end":-200}],
                                 ["savitzky_golay",{'window_size':5, 'order':2, 'deriv':0, 'rate':1}],
                                 ["normf", {}]],
                            ppg=[["subset", {"start": 500, "end": -200}],
                                 ["savitzky_golay", {'window_size': 5, 'order': 2, 'deriv': 0, 'rate': 1}],
                                 ["normf", {}]],
                            gsr=[["subset",{"start":500,"end":-200}],
                                 ["savitzky_golay",{'window_size':5, 'order':2, 'deriv':0, 'rate':1}],
                                 ["normf", {}]]
                            )

preprocessed_datafolder = preprocessor.preprocess_datafolder(data_folder)
print "** Preprocessing done **"
results = {}
segmenter_configurations = [["peak_midpoint_beat_segmentation",{'frequency':100,'length':100}],
                            ["t40_beat_segmentation", {'frequency': 100}]]
signals = [["ecg"],["ppg"],["gsr"],["ecg","ppg"],["ecg","ppg","gsr"]]
single_signal_feature_generation_functions = inspect.getmembers(single_signal_functions, inspect.isfunction)
ecg_only_feature_generation_functions = inspect.getmembers(ecg_only_functions, inspect.isfunction)
multi_signal_feature_generation_functions = inspect.getmembers(multimodal_functions, inspect.isfunction)
classifiers = {"distance": distance.AverageDistanceClassifier(),"svm": svm.SVMROCClassifier(), "kmeans":kmeans.KMeansClassifier()}#,"fastdtw": distance.AverageDistanceClassifier("fastdtw")}
training_samples = [5, 10, 20, 30, 40]
summary_output = open("activity1_summary_results.txt",'w')

for num_training_samples in training_samples:
    print "--Training with {0} samples --".format(num_training_samples)
    for seg_config in segmenter_configurations:
        segmenter = Segmenter(datafile=seg_config)
        segmented_datafolder = segmenter.segment_datafolder(preprocessed_datafolder)
        print "-- Segmentation done for : {0} --".format(seg_config[0])
        #ECG Feature extraction
        generator = FeatureExtractor(segmented_datafolder=segmented_datafolder)
        for signal in signals:
            if len(signal)==1:
                print "-- Using signal: {0} --".format(signal)
                feature_generation_functions = single_signal_feature_generation_functions
                if signal[0] == "ecg":
                    feature_generation_functions.extend(ecg_only_feature_generation_functions)
            else:
                print "-- Using signals: {0} --".format(signal)
                feature_generation_functions = multi_signal_feature_generation_functions
            for function_name,function in feature_generation_functions:
                print "-- Extracting features with : {0} --".format(function_name)
                dataset = generator.generate_dataset(function,signals=signal)
                for classifier_name in classifiers:
                    classifier = classifiers[classifier_name]
                    classifier.print_details()
                    exp = OneClassExperimenter(dataset=dataset, classifier=classifier, number_of_tries_per_subject=5)
                    # {signal}_{seg_config}_{function_name}_{classifier_name}_{num_training_sampes}: result
                    res = exp.execute_all_subjects(num_training_samples=num_training_samples)
                    mapping_key = "{0}_{1}_{2}_{3}_{4}".format(signal, seg_config[0], function_name, classifier_name,
                                                               num_training_samples)
                    results[mapping_key] = res
                    summary_output.write("{0}: {1}\n".format(mapping_key,res))
                    print "Experiment finished"
print "**** ALL Experiments finished ****"
summary_output.close()
print "**** Average Results ****"
for key,value in results:
    print "{0} : {1}".format(key,value)





