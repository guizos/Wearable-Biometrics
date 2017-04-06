from biosppy.signals import ecg
from numpy import mean

from data import source

#Read the data folder
data_folder = source.PumpPrimingDataFolder("dataset")
print "Done reading data folder"

from data.preprocessing.preprocessor import Preprocessor
hr = {}
for data_file in data_folder.data_files:
    if data_file.user_id not in hr.keys():
        hr[data_file.user_id] = {}
    if data_file.activity != 3:
        d = data_file.ecg
        try:
            out = ecg.ecg(signal=d, sampling_rate=100, show=None)
            hr[data_file.user_id][data_file.activity]= mean(out[6])
        except Exception:
            hr[data_file.user_id][data_file.activity] = 0
for user_id in data_folder.get_all_user_ids():
    print "{0}: {1}, {2}, {3}, {4}".format(user_id,hr[user_id][1],hr[user_id][2],hr[user_id][4],mean(hr[user_id].values()))

