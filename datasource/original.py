import ntpath
import os
from os import listdir
from os.path import isfile


from datasource.generaldata import LabeledTimeSignal, OriginalTimeSignalCollection, ModifiedTimeSignalCollection


class OriginalDataFolder:

    def __init__(self,folder):
        if folder[-1] != '/':
            folder +='/'
        files = [folder + f for f in listdir(folder) if isfile(os.path.join(folder, f)) and f != ".DS_Store"]
        self.data_files =[]
        for file in files:
            self.data_files.append(OriginalDataFile(file))

    def get_all_user_ids(self):
        result = [data_file.user_id for data_file in self.data_files]
        return list(set(result))


class OriginalDataFile:

    def __init__(self,filename):
        self.name = filename
        self.user_id = ntpath.basename(filename).split("_")[0]
        self.activity = int(ntpath.basename(filename).split("_")[1])
        self.gender = ntpath.basename(filename).split("_")[2]
        self.years = ntpath.basename(filename).split("_")[3]
        self.ecg = []
        self.ppg = []
        self.gsr = []
        self.acc = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ppg.append(int(line.strip().split(",")[0].strip()))
                self.gsr.append(int(line.strip().split(",")[1].strip()))
                if self.activity != 3:
                    self.ecg.append(int(line.strip().split(",")[2].strip()))
                self.acc.append(int(line.strip().split(",")[3].strip()))

    def save(self):
        save_name = self.name+"_modified"
        with open(save_name, 'w') as f:
            for i in range(len(self.ecg)):
                line = str(self.ppg[i])+","+str(self.gsr[i])+","+str(self.ecg[i])+","+str(self.acc[i])+"\n"
                f.write(line)

    def print_ecg(self):
        print "*** PRINTING ECG OF "+self.name+" ***"
        for measure in self.ecg:
            print measure

    def print_ppg(self):
        print "*** PRINTING PPG OF "+self.name+" ***"
        for measure in self.ppg:
            print measure

    def print_gsr(self):
        print "*** PRINTING GSR OF "+self.name+" ***"
        for measure in self.gsr:
            print measure

    def print_acc(self):
        print "*** PRINTING ACC OF "+self.name+" ***"
        for measure in self.acc:
            print measure

    def get_samples(self,sample_type, *arguments):
        """
        :param activity:
        :param sample_type: a list. First element, signal, second kind
        :param arguments*: List with arguments for the samples.
        :return:
        """
        call = sample_type
        if call == "ecg_beat" and self.activity != 3:
            return self.get_ecg_beat_set().time_signals
        elif call == "ppg_beat":
            return self.get_ppg_beat_set().time_signals
        elif call == "ecg_time" and self.activity != 3:
            return self.get_ecg_time_signals(seconds=arguments[0]).time_signals
        elif call == "ppg_time":
            return self.get_ppg_time_signals(seconds=arguments[0]).time_signals
        else:
            return []

    def get_ecg_time_signal(self):
        return LabeledTimeSignal(self.ecg,self.user_id,self.activity)

    def get_ppg_time_signal(self):
        return LabeledTimeSignal(self.ppg,self.user_id,self.activity)

    def get_gsr_time_signal(self):
        return LabeledTimeSignal(self.gsr,self.user_id,self.activity)

    def get_acc_time_signal(self):
        return LabeledTimeSignal(self.acc,self.user_id,self.activity)

    def get_ecg_time_signals(self,seconds=5):
        return OriginalTimeSignalCollection(self.user_id,self.activity,self.get_ecg_time_signal(),5)

    def get_ppg_time_signals(self,seconds=5):
        return OriginalTimeSignalCollection(self.user_id,self.activity,self.get_ppg_time_signal(),5)

    def get_gsr_time_signals(self,seconds=5):
        return OriginalTimeSignalCollection(self.user_id,self.activity,self.get_gsr_time_signal(),5)

    def get_acc_time_signals(self,seconds=5):
        return OriginalTimeSignalCollection(self.user_id,self.activity,self.get_acc_time_signal(),5)


class ModifiedTimeDataFile:

    def __init__(self,filename):
        self.user_id = ntpath.basename(filename).split("_")[0]
        self.activity = ntpath.basename(filename).split("_")[1]
        self.gender = ntpath.basename(filename).split("_")[2]
        self.years = ntpath.basename(filename).split("_")[3]
        name_ecg = filename+"_ecg"
        name_ppg = filename+"_ppg"
        self.ecg_signals = []
        self.ppg_signals = []
        with open(name_ecg, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ecg = [int(value) for value in line.split(", ")]
                self.ecg_signals.append(LabeledTimeSignal(ecg,self.user_id,self.activity))
        with open(name_ppg, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ppg = [int(value) for value in line.split(", ")]
                self.ppg_signals.append(LabeledTimeSignal(ppg,self.user_id,self.activity))

    def get_ecg_time_signals(self):
        return ModifiedTimeSignalCollection(self.user_id,self.activity,self.ecg_signals)

    def get_ppgtime_signals(self):
        return ModifiedTimeSignalCollection(self.user_id,self.activity,self.ppg_signals)