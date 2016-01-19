from data import source

data_folder = source.OriginalDataFolder("../data/original")
data_files = []
data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="de46f888-a646-40e9-b817-c0f9aff7a362"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ad45363b-7632-432e-a368-215d3fb0ca10"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ad45363b-7632-432e-a368-215d3fb0ca9a"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ad45363b-7632-432e-a368-215d3fb0ca9d"])
#data_files.extend([data_file for data_file in data_folder.data_files if data_file.user_id=="ea124ef1-64c5-4dbf-8f19-d96d7804cf21"])
for data_file in data_files:
    ecg = [1024-x for x in data_file.ecg]
    data_file.ecg = ecg
    data_file.save()
