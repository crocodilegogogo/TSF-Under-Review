"""SHL2018 dataset can be Downloaded at http://www.shl-dataset.org/activity-recognition-challenge/"""
"""Put the downloaded dataset into the 'dataset/SHL2018' folder"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple
from utils.load_SHL2018_dataset.save_SHL2018_dataset_as_npy import save_SHL2018_to_npy
from utils.load_SHL2018_dataset.preprocess_raw_data import *
from pandas import Series

def load_SHL2018_raw_data(data_dir, cal_attitude_angle, std_all, std,
                          ACT_LABELS, ActID, SPLIT_NUM,
                          DATASET_SIZE=-1, save_npy_flag = False,
                          to_NED_flag = True):
    
    if save_npy_flag == True:
        save_SHL2018_to_npy(data_dir, std_all, DATASET_SIZE)
    
    for data_type in ['train','test']:
        
        label_file_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                         'label', 'label.npy')
        
        # concatenate all sensor data
        for (std_id,std_name) in enumerate(std):
            
            if std_name != 'ori':
                data_file_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                             std_name, std_name + '_xyz.npy')
                if std_id == 0:
                    cur_sensor_data  = np.load(data_file_dir)
                    
                    # Perform linear interpolation, fill up missing data
                    cur_sensor_data_interp = cur_sensor_data.reshape(-1, cur_sensor_data.shape[-1])
                    cur_sensor_data_interp = np.array([Series(i).interpolate() for i in cur_sensor_data_interp])
                    cur_sensor_data  = cur_sensor_data_interp.reshape(cur_sensor_data.shape[0],
                                                                      cur_sensor_data.shape[1],
                                                                      cur_sensor_data.shape[-1])
                    
                    # if nan, set zeros
                    cur_sensor_data[np.isnan(cur_sensor_data)] = 0
                    all_sensors_data = cur_sensor_data
                else:
                    cur_sensor_data  = np.load(data_file_dir)
                    
                    # Perform linear interpolation, fill up missing data
                    cur_sensor_data_interp = cur_sensor_data.reshape(-1, cur_sensor_data.shape[-1])
                    cur_sensor_data_interp = np.array([Series(i).interpolate() for i in cur_sensor_data_interp])
                    cur_sensor_data  = cur_sensor_data_interp.reshape(cur_sensor_data.shape[0],
                                                                      cur_sensor_data.shape[1],
                                                                      cur_sensor_data.shape[-1])
                    
                    # if nan, set zeros
                    cur_sensor_data[np.isnan(cur_sensor_data)] = 0
                    all_sensors_data = np.concatenate((all_sensors_data, cur_sensor_data),
                                                      axis=1)
            
            elif to_NED_flag == True:
                data_file_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                             std_name, std_name + '_wxyz.npy')
                ori_data = np.load(data_file_dir)
                ori_data_interp = ori_data.reshape(-1, ori_data.shape[-1])
                ori_data_interp = np.array([Series(i).interpolate() for i in ori_data_interp])  # 线性插值
                ori_data = ori_data_interp.reshape(ori_data.shape[0],
                                                   ori_data.shape[1],
                                                   ori_data.shape[-1])
                ori_data[np.isnan(ori_data)] = 0
                R_ori_data = NED_R(ori_data)
        
        # correct the data using quarternion
        if to_NED_flag == True:
            split_sensor_data = np.split(all_sensors_data, 3, axis=1)
            for (data_id, cur_data) in enumerate(split_sensor_data):
                if data_id == 0:
                    all_sensors_data = pre_threeaxis_data(to_NED_flag, cur_data, R_ori_data)
                else:
                    cur_sensors_data = pre_threeaxis_data(to_NED_flag, cur_data, R_ori_data)
                    all_sensors_data = np.concatenate((all_sensors_data, cur_sensor_data),
                                                      axis=1)
        
        # seg the dataset to split_num pieces
        dataset     = seg_dataset(all_sensors_data, SPLIT_NUM)
        
        # get the labels of all samples
        data_labels = get_sample_labels(label_file_dir, SPLIT_NUM)
        
        if data_type == 'train':
            X_train = dataset
            Y_train = data_labels
        elif data_type == 'test':
            X_test  = dataset
            Y_test  = data_labels
    
    X_train, Y_train = log_remaining_index(ActID, X_train, Y_train)
    X_test , Y_test  = log_remaining_index(ActID, X_test , Y_test)
    
    # normalize train and test datasets together
    trainset_size = Y_train.shape[0]
    concat_data   = np.concatenate((X_train, X_test), axis=0)
    if 'ori' in std:
        std.remove('ori')
        len_std = len(std)
    else:
        len_std = len(std)
    concat_data   = np.split(concat_data, len_std, axis = 1)
    for i in range(len_std):
        if i == 0:
            X_data   = scale_raw_data(concat_data[i], std[i], cal_attitude_angle)
        else:
            cur_data = scale_raw_data(concat_data[i], std[i], cal_attitude_angle)
            X_data   = np.concatenate((X_data, cur_data), axis=1)
    X_train = X_data[:trainset_size,:,:]
    X_test  = X_data[trainset_size:,:,:]
    
    # get act and label distionary data
    label2act, act2label = act_label_transform(ACT_LABELS)
    
    User_ids_train = [1] * X_train.shape[0]
    User_ids_test  = [1] * X_test.shape[0]
    
    return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1),\
           (Y_train-1), (Y_test-1), label2act, act2label, User_ids_train, User_ids_test