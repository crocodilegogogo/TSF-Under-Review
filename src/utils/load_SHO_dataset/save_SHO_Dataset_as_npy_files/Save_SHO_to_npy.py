"""SHO dataset can be Downloaded at https://www.utwente.nl/en/eemcs/ps/research/dataset/"""
"""Put the downloaded dataset into the current folder"""
"""Run this py file"""

import pandas as pd
import numpy as np
import os

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

# there are typos in the data!!!
def change_the_typo(label):
    if label == "upsatirs":
        return "upstairs"
    else:
        return label

root_path = os.path.join('sensors-activity-recognition-dataset-shoaib', 'DataSet')
CUR_DIR   = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','SHO', "Clean_SHO.npy")

file_list = os.listdir(root_path)
file_list = [file for file in file_list if "Participant" in file] # in total , it should be 10
assert len(file_list) == 10

used_cols = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, # Left_pocket
             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, # Right_pocket
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, # Wrist
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # Upper_arm
             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Belt
             69]

df_dict = {}

col_list = ["acc_x","acc_y","acc_z","lacc_x","lacc_y","lacc_z","Gyro_x","Gyro_y","Gyro_z","mag_x","mag_y","mag_z"]
pos_list = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
col_names = [item for sublist in [[col+"_"+pos for col in col_list] for pos in pos_list] for item in sublist]+["activity_id"]

file_encoding = {"Participant_1.csv":1, "Participant_2.csv":2, "Participant_3.csv":3,
                 "Participant_4.csv":4, "Participant_5.csv":5, "Participant_6.csv":6,
                 "Participant_7.csv":7, "Participant_8.csv":8, "Participant_9.csv":9,
                 "Participant_10.csv":10} 

sub_ids_of_each_sub = {}

label_map = [(0, 'walking'), 
             (1, 'standing'),
             (2, 'jogging'),
             (3, 'sitting'), 
             (4, 'biking'),
             (5, 'upstairs'),
             (6, "downstairs")]

labelToId = {int(x[0]): i for i, x in enumerate(label_map)}

for file in file_list:
    sub_data = pd.read_csv(os.path.join(root_path, file), skiprows=[0,1], header=None)
    sub_data = sub_data.iloc[:, used_cols]

    sub_data.columns = col_names

    # if missing values, imputation TODO
    sub_data = sub_data.interpolate(method='linear', limit_direction='both')

    sub = int(file_encoding[file])
    sub_data['sub_id'] = sub
    sub_data["sub"] = sub

    df_dict[sub] = sub_data

df_all = pd.concat(df_dict)
df_all = df_all.set_index('sub_id')

# this is for first encode the str label to nummeric number
label_mapping = {item[1]:item[0] for item in label_map}

df_all["activity_id"] = df_all["activity_id"].apply(change_the_typo)
df_all["activity_id"] = df_all["activity_id"].map(label_mapping)
df_all["activity_id"] = df_all["activity_id"].map(labelToId)

# reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
df_all = df_all[col_names[:-1]+["sub"]+["activity_id"]]

all_mat = df_all.values
now_col_index = []
for pos_id in range(5):
    now_col_index.extend(list(np.array([0,1,2,6,7,8,3,4,5])+12*pos_id))

all_mat_data = all_mat[:,:60]
all_mat_data = all_mat_data[:, now_col_index]
all_mat_data[np.isnan(all_mat_data)] = 0

sub_label_mat = all_mat[:,60:]
count = 0
for sub_id in list(np.unique(all_mat[:,60])):
    for act_id in list(np.unique(all_mat[:,61])):

        cur_sub_label_mat = sub_label_mat[sub_label_mat[:,0] == sub_id, :]
        cur_sub_label_mat = cur_sub_label_mat[cur_sub_label_mat[:,1] == act_id, :]
        cur_count_mat     = np.array([count] * cur_sub_label_mat.shape[0])
        cur_count_mat     = cur_count_mat.reshape((cur_count_mat.shape[0], 1))
        cur_sub_label_count_mat = np.concatenate((cur_count_mat, cur_sub_label_mat), axis=1)
        
        if count == 0:
            count_sub_label_mat = cur_sub_label_count_mat
        else:
            count_sub_label_mat = np.concatenate((count_sub_label_mat, cur_sub_label_count_mat), axis=0)
        
        count += 1

save_npy_dir_flag = create_directory(os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                                     'datasets','SHO'))
all_mat = np.concatenate((count_sub_label_mat, all_mat_data), axis=1)
np.save(save_npy_dir, all_mat)