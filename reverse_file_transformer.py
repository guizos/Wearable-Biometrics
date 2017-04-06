from data import source

data_folder = source.PumpPrimingDataFolder("real_original_data/")
data_files = []
data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="f225a13c-c149-4e6f-a530-5f1ca0722d77"])
data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="47b3fe43-f579-4375-9b84-c162fa007"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ad45363b-7632-432e-a368-215d3fb0ca9a"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ad45363b-7632-432e-a368-215d3fb0ca9d"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ea124ef1-64c5-4dbf-8f19-d96d7804cf21"])
for data_file in data_files:
    ecg = [1024-x for x in data_file.ecg]
    data_file.ecg = ecg
    data_file.save()
