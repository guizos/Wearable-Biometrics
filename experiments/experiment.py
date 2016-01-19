from __future__ import division

import logging
import os

import numpy

from data.preprocessing import interpolation
from utils.sampling import reservoir_sampling


class OneClassExperimenter:

    def __init__(self,labeled_database,classifier, number_of_tries_per_subject,verbose=None,output_foler=None):
        self.labeled_database = labeled_database
        self.classifier = classifier
        self.tries = number_of_tries_per_subject
        self.verbose = verbose
        self.output_folder = output_foler
        if self.output_folder:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

    def execute_all_subjects(self,activity_training,num_training_samples):
        user_ids = self.labeled_database.get_all_user_ids()
        results = []
        for user_id in user_ids:
            this_user_result = self.execute_subject(user_id,activity_training,num_training_samples)
            results.append(this_user_result)
            if self.verbose == "Full":
                self.print_experiment_summary(activity_training,num_training_samples,this_user_result)
        final_average = numpy.mean(results,axis=0)
        if self.verbose:
            print "---- FINAL ----"
            self.print_experiment_summary(activity_training,num_training_samples,final_average)
        if self.output_folder :
            output_file = self.output_folder+"/"+"final_"+self.classifier.get_parameters_string()+"_"\
                          + str(activity_training) + "_" + str(num_training_samples)
            self.write_results_to_file(result_table=final_average, output_file=output_file, user_id="SUMMARY",
                                       activity_training=activity_training, num_training_samples=num_training_samples)
        return final_average

    def execute_subject(self,user_id,activity_training,num_training_samples):
        results = []
        for i in range(self.tries):
            (train,test) = self.labeled_database.get_labeled_training_and_test_samples(activity_training,user_id,num_training_samples)
            self.classifier.train(train)
            this_try_result = self.classifier.test(test,activity_info=None)
            results.append(this_try_result)
            logging.debug("Subject Finished with id "+user_id+" finished try "+str(i))
        average_result_tables = numpy.mean(results,axis=0)
        # ONLY IF OUTPUT FOLDER IS DEFINED
        if self.output_folder:
            output_file = self.output_folder+"/"+self.classifier.get_parameters_string()+"_"+user_id+"_"\
                          + str(activity_training) + "_" + str(num_training_samples)
            self.write_results_to_file(result_table=average_result_tables, output_file=output_file, user_id=user_id,
                                       activity_training=activity_training, num_training_samples=num_training_samples)
        return average_result_tables

    def write_results_to_file(self, result_table, output_file, user_id, activity_training, num_training_samples):
        with open(output_file,'w') as f:
            f.write("**** SUMMARY ****\n")
            f.write("User ID: "+str(user_id)+"\n")
            f.write("Tr. Activity: "+str(activity_training)+"\n")
            f.write("Tr. Samples: "+str(num_training_samples)+"\n")
            f.write(self.classifier.get_details())
            tp_line = "TP :"
            fn_line = "FN :"
            tn_line = "TN :"
            fp_line = "FP :"
            h_line = "H :"
            tpr_line = "TPR :"
            fpr_line = "FPR :"
            i = 0
            for result in result_table:
                if sum(result) != 0 and result[0]+result[2]!=0 and result[1]+result[3]!=0:
                    tp_line += " ("+str(i)+") "+str(result[0]) + "\t"
                    tn_line += " ("+str(i)+") "+str(result[1]) + "\t"
                    fp_line += " ("+str(i)+") "+str(result[2]) + "\t"
                    fn_line += " ("+str(i)+") "+str(result[3]) + "\t"
                    tpr_line += " ("+str(i)+") "+str(result[0] / (result[0] + result[2])) + "\t"
                    fpr_line += " ("+str(i)+") "+str(result[3] / (result[3] + result[1])) + "\t"
                    h_line += " ("+str(i)+") "+str((result[1] + result[0]) / (sum(result))) + "\t"
                else:
                    tp_line += " ("+str(i)+") "+str(result[0]) + "\t"
                    tn_line += " ("+str(i)+") "+str(result[1]) + "\t"
                    fp_line += " ("+str(i)+") "+str(result[2]) + "\t"
                    fn_line += " ("+str(i)+") "+str(result[3]) + "\t"
                i +=1
            f.write(tp_line+"\n")
            f.write(fn_line+"\n")
            f.write(tn_line+"\n")
            f.write(fp_line+"\n")
            f.write(h_line+"\n")
            f.write(tpr_line+"\n")
            f.write(fpr_line+"\n")

    def print_experiment_summary(self,activity_training, number_samples_training, result_tables):
        print "**** SUMMARY ****"
        print "Tr. Activity: "+str(activity_training)
        print "Tr. Samples: "+str(number_samples_training)
        tp_line = "TP :"
        fn_line = "FN :"
        tn_line = "TN :"
        fp_line = "FP :"
        h_line = "H :"
        tpr_line = "TPR :"
        fpr_line = "FPR :"
        i = 0
        for result_table in result_tables:
            if sum(result_table)!=0:
                tp_line += " ("+str(i)+") "+str(result_table[0])+"\t"
                tn_line += " ("+str(i)+") "+str(result_table[1])+"\t"
                fp_line += " ("+str(i)+") "+str(result_table[2])+"\t"
                fn_line += " ("+str(i)+") "+str(result_table[3])+"\t"
                tpr_line += " ("+str(i)+") "+str(result_table[0]/(result_table[0]+result_table[2]))+"\t"
                fpr_line += " ("+str(i)+") "+str(result_table[3]/(result_table[3]+result_table[1]))+"\t"
                h_line += " ("+str(i)+") "+str((result_table[1]+result_table[0])/(sum(result_table)))+"\t"
            i +=1
        print tp_line
        print fn_line
        print tn_line
        print fp_line
        print h_line
        print tpr_line
        print fpr_line


class OneClassROCAreaExperimenter(OneClassExperimenter):

    def write_results_to_file(self,result_table,output_file,user_id,activity_training,num_training_samples):
        with open(output_file,'w') as f:
            f.write("**** SUMMARY ****\n")
            f.write("User ID: "+str(user_id)+"\n")
            f.write("Tr. Activity: "+str(activity_training)+"\n")
            f.write("Tr. Samples: "+str(num_training_samples)+"\n")
            f.write(self.classifier.get_details())
            roc_line = "ROC Area :"
            i = 0
            for result in result_table:
                roc_line += " ("+str(i)+") "+str(result)+"\t"
                i +=1
            f.write(roc_line+"\n")

    def print_experiment_summary(self,activity_training, number_samples_training, result_table):
        print "**** SUMMARY ****"
        print "Tr. Activity: "+str(activity_training)
        print "Tr. Samples: "+str(number_samples_training)
        roc_line = "ROC Area :"
        i = 0
        for result in result_table:
            roc_line += " ("+str(i)+") "+str(result)+"\t"
            i +=1
        print roc_line

class OneClassDistanceExperimentEnrollmentOneActivity():

    # Uses the samples from the first activity as
    def __init__(self,person, others, num_enrollment, distance, threshold,sample_type,*arguments):
        """

        :param person:
        :param others:
        :param samples_enrollment:
        :param distance:
        :param threshold:
        :param sample_type:
        :param signal:
        :return:
        """
        self.person = person
        self.others = others
        #Select the enrollment samples randomly between the initial activity.
        print "Selecting enrollment samples"
        self.samples_enrollment = reservoir_sampling(person.get_activity_samples("1",sample_type,arguments), num_enrollment)
        #Select user samples to be used for testing. All minus the ones used during enrollment
        print "Selecting the rest of the user as samples for testing"
        all_user_samples = person.get_all_samples(sample_type,arguments)
        self.samples_user_test = [sample for sample in all_user_samples if sample not in self.samples_enrollment]
        print "Selecting rest of the samples from other users for testing"
        self.samples_others_test = [sample for datafile in others.get_all_data_files() for sample in datafile.get_samples(sample_type,arguments)]
        self.distance = distance
        self.threshold = threshold
        print "Measuring the mean sample size for interpolation"
        self.size_samples = numpy.mean([len(sample.data) for sample in self.samples_enrollment])

    def train(self):
        """
            Measures the distance between all training samples to create a confidence measure
        :return: The average distance and std deviation in the training samples
        """
        distances = []
        print "training"
        for sample in self.samples_enrollment:
            for second_sample in self.samples_enrollment:
                distances.append(self.distance(interpolation.linear(sample.data, self.size_samples),
                                               interpolation.linear(second_sample.data, self.size_samples)))
        return (numpy.mean(distances),numpy.std(distances))

    def test(self,threshold):
        # We initialize the values result values
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        # First we measure the distances of the ones that belong to the user.
        same_user_distances = []
        other_user_distances = []
        print "testing"
        print "measuring same user distances"
        for same_user_sample in self.samples_user_test:
            same_user_distances.append([self.distance(interpolation.linear(same_user_sample.data, self.size_samples),
                                                      interpolation.linear(enrollment_sample.data, self.size_samples)) for enrollment_sample in self.samples_enrollment])
        print "measuring other user distances"
        for other_user_sample in self.samples_others_test:
            other_user_distances.append([self.distance(interpolation.linear(other_user_sample.data, self.size_samples),
                                                       interpolation.linear(enrollment_sample.data, self.size_samples)) for enrollment_sample in self.samples_enrollment])
        print "Measuring classification values"
        tp = len([1 for distance_array in same_user_distances if numpy.mean(distance_array)<threshold])
        fn = len(same_user_distances)-tp
        tn = len([1 for distance_array in other_user_distances if numpy.mean(distance_array)<threshold])
        fp = len(other_user_distances)-tn
        return (tp,fn,tn,fp)

    def roc(self):
        pass


class OneClassDistanceExperimentRandomEnrollment(OneClassDistanceExperimentEnrollmentOneActivity):

    def __init__(self,person, others, samples_enrollment, distance, treshold):
        pass
    #Persona, Otros Readings como entrada




