# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:35:26 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple


def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def scale_raw_data(data_all_axis_array, std_id, cal_attitude_angle, scaler="normalize"):
    
    data_all_axis_array = np.transpose(data_all_axis_array, (0,2,1))
    data_shape = data_all_axis_array.shape
    data_all_axis_array = data_all_axis_array.reshape(-1, data_shape[-1])
    
    ####cal_attitude_angle####
    if std_id == 'gra' or std_id == 'mag':
        if cal_attitude_angle == True:
            attitude_angle = np.zeros_like(data_all_axis_array)
            attitude_angle[:,0] = np.arctan2(data_all_axis_array[:,1], data_all_axis_array[:,2])
            attitude_angle[:,1] = np.arctan2(data_all_axis_array[:,0], data_all_axis_array[:,2])
            attitude_angle[:,2] = np.arctan2(data_all_axis_array[:,0], data_all_axis_array[:,1])
            attitude_angle = pd.DataFrame(
                attitude_angle,
                columns=['x','y','z']
            )
            data_all_axis_array = scale(attitude_angle, scaler)
        else:
            data_all_axis_array = pd.DataFrame(
                data_all_axis_array,
                columns=['x','y','z']
            )
            data_all_axis_array = scale(data_all_axis_array, scaler)
    
    ####cal_attitude_angle####
    else:
        data_all_axis_array = pd.DataFrame(
            data_all_axis_array,
            columns=['x','y','z']
        )
        data_all_axis_array = scale(data_all_axis_array, scaler)
    
    data_all_axis_array = data_all_axis_array.values
    data_all_axis_array = data_all_axis_array.reshape(data_shape[0],
                                                      data_shape[1],
                                                      data_shape[2])
    data_all_axis_array = np.transpose(data_all_axis_array, (0,2,1))
    
    return data_all_axis_array

def seg_dataset(all_sensors_data, split_num):
    
    # split 6000 to 12*500, then recurrently concat them and reshape to (sample_num, 12, 500)
    all_sensors_data = np.split(all_sensors_data, split_num, axis=2)
    for i in range(len(all_sensors_data)):
        if i == 0:
            dataset = np.expand_dims(all_sensors_data[i], axis=2)
        else:
            cur_dataset = np.expand_dims(all_sensors_data[i], axis=2)
            dataset     = np.concatenate((dataset, cur_dataset),axis=2)
    dataset = np.transpose(dataset, (0,2,1,3))
    dataset = dataset.reshape(-1, dataset.shape[2], dataset.shape[3])
    
    return dataset

def get_sample_labels(label_file_dir, split_num):
    
    labels = np.load(label_file_dir)
    labels = np.split(labels, split_num, axis=1)
    
    for i in range(len(labels)):
        if i == 0:
            data_labels = np.expand_dims(labels[i], axis=1)
        else:
            cur_labels  = np.expand_dims(labels[i], axis=1)
            data_labels = np.concatenate((data_labels, cur_labels),axis=1)
    data_labels = data_labels.reshape(-1, data_labels.shape[-1])
    
    # cal the most repetitive category of each axis
    u, indices  = np.unique(data_labels, return_inverse=True)
    
    # Find categories 1 to 8, cal the repeated times of each category for each sample
    counted     = np.apply_along_axis(np.bincount, 1,
                                  indices.reshape(data_labels.shape),
                                  None, np.max(indices) + 1)
    
    # Cal the index of the largest value as the category number
    data_labels = u[np.argmax(counted, axis=1)]
    
    return data_labels

def act_label_transform(ACT_LABELS):
    label2act, act2label = {}, {}
    for label, act in enumerate(ACT_LABELS):
        label2act[int(label)] = act
        act2label[act] = int(label)
    return label2act, act2label

def log_remaining_index(ActID, X_dataset, Y_dataset):
    
    # log the index of remaining ActID
    for (count_id, act_id) in enumerate(ActID):
        if count_id == 0:
            act_index = np.where(Y_dataset==act_id)[0]
        else:
            act_flag  = np.where(Y_dataset==act_id)[0]
            act_index = np.concatenate((act_index, act_flag), axis=0)
    X_dataset = X_dataset[act_index]
    Y_dataset = Y_dataset[act_index]
    
    return X_dataset, Y_dataset

def pre_threeaxis_data(to_NED, concat_data, R, scaler = "normalize"):
    
    if to_NED == True:
        data_all_axis_array = to_ned(concat_data,R)  #(40000, 3, 500) R:(40000,500,3,3)
    
    return data_all_axis_array

def NED_R(ori_data):

    ori_data_concat = np.transpose(ori_data,[0,2,1])  # (n,500,3)
    ori_data_concat = np.expand_dims(ori_data_concat,axis=3)


    qw = ori_data_concat[:, :, 0]
    qx = ori_data_concat[:, :, 1]
    qy = ori_data_concat[:, :, 2]
    qz = ori_data_concat[:, :, 3]

    one = np.ones((ori_data_concat.shape[0], ori_data_concat.shape[1], 1))  # (40000,500,1)


    R = one - (qy * qy + qz * qz) - (qy * qy + qz * qz)  # (16308,6000,1)
    R = np.append(R, (qx * qy - qw * qz) + (qx * qy - qw * qz), axis=2)  # (40000,500,2)
    R = np.append(R, (qx * qz + qw * qy) + (qx * qz + qw * qy), axis=2)
    R = np.append(R, (qx * qy + qw * qz) + (qx * qy + qw * qz), axis=2)
    R = np.append(R, one - (qx * qx + qz * qz) - (qx * qx + qz * qz), axis=2)
    R = np.append(R, (qy * qz - qw * qx) + (qy * qz - qw * qx), axis=2)
    R = np.append(R, (qx * qz - qw * qy) + (qx * qz - qw * qy), axis=2)
    R = np.append(R, (qy * qz + qw * qx) + (qy * qz + qw * qx), axis=2)
    R = np.append(R, one - (qy * qy + qx * qx) - (qy * qy + qx * qx), axis=2)
    # R (40000,500,9)
    R  = R.reshape([ori_data_concat.shape[0],-1,3,3])
    # R (40000,500,3,3)

    return R


def to_ned(concat_data, R):
    
    concat_data = np.transpose(concat_data,[0,2,1])
    concat_data = np.expand_dims(concat_data, axis=3)
    concat_ned_data = np.matmul(R, concat_data).squeeze()

    return np.transpose(concat_ned_data,[0,2,1])