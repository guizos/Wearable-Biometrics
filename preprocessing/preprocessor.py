import os
from os import listdir
from os.path import isfile, join

from datasource import generaldata
from preprocessing import normalization
from preprocessing import smoothing



def get_all_datafiles(folder):
    if folder[-1] != '/':
        folder +='/'
        files = [folder+f for f in listdir(folder) if isfile(join(folder, f)) and f != ".DS_Store"]
        data_files = []
        for file in files:
            d = generaldata.DataFile(file)
            data_files.append(d)
    return data_files

def write_array_to_file(filename,array):
    with open(filename,'w+') as f:
        for line in array:
            f.write(", ".join([str(x) for x in line])+"\n")


def preprocess_time_signals(folder, normalize, interpolation_method, smoothe,output_folder, kind, time=0):
    data_files = get_all_datafiles(folder)
    data_files = get_all_datafiles(folder)
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process data
    for data_file in data_files:
        if kind == 'beats':
            ecg_original = data_file.get_ecg_beat_set().time_signals
            ppg_original = data_file.get_ppg_beat_set().time_signals
        elif kind == 'time':
            ecg_original = data_file.get_ecg_time_signals(time).time_signals
            ppg_original = data_file.get_ppg_time_signals(time).time_signals
        if normalize:
            ecg_normalized = [normalization.normalize(ecg_time_signal.data) for ecg_time_signal in ecg_original]
            ppg_normalized = [normalization.normalize(ppg_time_signal.data) for ppg_time_signal in ppg_original]
        else:
            ecg_normalized = ecg_original
            ppg_normalized = ppg_original
        if interpolation_method:
            ecg_interpoled = [interpolation_method(ecg_beat,80) for ecg_beat in ecg_normalized]
            ppg_interpoled = [interpolation_method(ppg_beat,80) for ppg_beat in ppg_normalized]
        else:
            ecg_interpoled = ecg_normalized
            ppg_interpoled = ppg_normalized
        if smoothe:
            ecg_smoothed = [smoothing.savitzky_golay(ecg_beat,9,3) for ecg_beat in ecg_interpoled]
            ppg_smoothed = [smoothing.savitzky_golay(ppg_beat,9,3) for ppg_beat in ppg_interpoled]
        else:
            ecg_smoothed = ecg_interpoled
            ppg_smoothed = ppg_interpoled
        write_array_to_file(output_folder+"/"+data_file.user_id+"_"+data_file.activity+"_"+data_file.gender+"_"+data_file.years+"_ecg",ecg_smoothed)
        write_array_to_file(output_folder+"/"+data_file.user_id+"_"+data_file.activity+"_"+data_file.gender+"_"+data_file.years+"_ppg",ppg_smoothed)


#Deprecated, the previous method should be used instead.
def preprocess_beats(folder, normalize, interpolation_method, smoothe,output_folder):
    data_files = get_all_datafiles(folder)
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process data
    for data_file in data_files:
        ecg_original_beats = data_file.get_ecg_beat_set().beats
        ppg_original_beats = data_file.get_ppg_beat_set().beats
        if normalize:
            ecg_normalized_beats = [normalization.normalize(ecg_time_signal.data) for ecg_time_signal in ecg_original_beats]
            ppg_normalized_beats = [normalization.normalize(ppg_time_signal.data) for ppg_time_signal in ppg_original_beats]
        else:
            ecg_normalized_beats = ecg_original_beats
            ppg_normalized_beats = ppg_original_beats
        if interpolation_method:
            ecg_interpoled_beats = [interpolation_method(ecg_beat,80) for ecg_beat in ecg_normalized_beats]
            ppg_interpoled_beats = [interpolation_method(ppg_beat,80) for ppg_beat in ppg_normalized_beats]
        else:
            ecg_interpoled_beats = ecg_normalized_beats
            ppg_interpoled_beats = ppg_normalized_beats
        if smoothe:
            ecg_smoothed_beats = [smoothing.savitzky_golay(ecg_beat,9,3) for ecg_beat in ecg_interpoled_beats]
            ppg_smoothed_beats = [smoothing.savitzky_golay(ppg_beat,9,3) for ppg_beat in ppg_interpoled_beats]
        else:
            ecg_smoothed_beats = ecg_interpoled_beats
            ppg_smoothed_beats = ppg_interpoled_beats
        write_array_to_file(output_folder+"/"+data_file.user_id+"_"+data_file.activity+"_"+data_file.gender+"_"+data_file.years+"_ecg",ecg_smoothed_beats)
        write_array_to_file(output_folder+"/"+data_file.user_id+"_"+data_file.activity+"_"+data_file.gender+"_"+data_file.years+"_ppg",ppg_smoothed_beats)



