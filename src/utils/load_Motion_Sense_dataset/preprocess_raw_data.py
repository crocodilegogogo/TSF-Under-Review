"""Preprocess raw data for creating input for DL tasks."""
import glob
import os
import sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.load_Motion_Sense_dataset.preprocessing import Preprocess
from utils.load_Motion_Sense_dataset.preprocessing import *

def get_ds_infos(data_dir):
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_table(os.path.join(data_dir,"data_subjects_info.txt"), sep=',')
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    # _signal = of.apply_filter(_signal, filter="median")
    # _signal = of.apply_filter(_signal, filter="butterworth")
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal

def creat_time_series(data_dir, dt_list, ds_list, act_labels, trial_codes,
                      cal_attitude_angle,
                      mode="mag", labeled=True, scaler="normalize"):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.
        scaler: Normalization mode

    Returns:
        It returns a time-series of sensor data and the column labels.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos(data_dir)
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list.iloc[:,0]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = os.path.join(data_dir, 'A_DeviceMotion_data', 'A_DeviceMotion_data', \
                                     act+'_'+str(trial), 'sub_'+str(int(sub_id))+'.csv')
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                raw_data = scale_raw_data(raw_data, dt_list, cal_attitude_angle, scaler)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    
    # add the column labels
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
    
    if labeled:
        cols += ["act_id", "sub_id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset, cols

def scale_raw_data(raw_data, dt_list, cal_attitude_angle, scaler):
    
    ####cal_attitude_angle####
    if cal_attitude_angle == True:
        raw_gravity = raw_data[dt_list[1]].values
        scale_gravity = np.zeros_like(raw_gravity)
        scale_gravity[:,0] = np.arctan2(raw_gravity[:,1], raw_gravity[:,2])
        scale_gravity[:,1] = np.arctan2(raw_gravity[:,0], raw_gravity[:,2])
        scale_gravity[:,2] = np.arctan2(raw_gravity[:,0], raw_gravity[:,1])
        scale_gravity = pd.DataFrame(
            scale_gravity,
            index=raw_data[dt_list[1]].index, columns=raw_data[dt_list[1]].columns
        )
        scale_gravity = scale(scale_gravity, scaler)
    
    ####cal_attitude_angle####
    else:
        scale_gravity = scale(raw_data[dt_list[1]], scaler)
    scale_rotationRate = scale(raw_data[dt_list[2]], scaler)
    scale_userAcceleration = scale(raw_data[dt_list[3]], scaler)
    
    raw_data[dt_list[1]] = scale_gravity
    raw_data[dt_list[2]] = scale_rotationRate
    raw_data[dt_list[3]] = scale_userAcceleration
    
    return raw_data

def seperate_train_test(dataset, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID):
    
    train_dataset = []
    test_dataset = []
    selected_col = ['gravity.x', 'gravity.y', 'gravity.z',
                    'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
                    'userAcceleration.x', 'userAcceleration.y','userAcceleration.z',
                    'act_id','sub_id','trial']
    
    for train_sub_id in (np.array(TRAIN_SUBJECTS_ID)-1).tolist():
        if len(train_dataset) == 0:
            cur_sub_dataset = dataset[dataset['sub_id']==train_sub_id]
            train_dataset   = cur_sub_dataset[selected_col]
        else:
            cur_sub_dataset = dataset[dataset['sub_id']==train_sub_id]
            train_dataset   = pd.concat([train_dataset,cur_sub_dataset[selected_col]])
            
    for test_sub_id in (np.array(TEST_SUBJECTS_ID)-1).tolist():
        if len(test_dataset) == 0:
            cur_sub_dataset = dataset[dataset['sub_id']==test_sub_id]
            test_dataset = cur_sub_dataset[selected_col]
        else:
            cur_sub_dataset = dataset[dataset['sub_id']==test_sub_id]
            test_dataset = pd.concat([test_dataset,cur_sub_dataset[selected_col]])
            
    return train_dataset, test_dataset

def seg_dataset(dataset, trial_codes, seg_window_size, overlap):
    
    X_data  = np.array([])
    Y_data  = np.array([])
    Sub_ids = np.array([])
    for act_id in set(dataset['act_id']):
        for trial_id in trial_codes[int(act_id)]:
            for sub_id in set(dataset['sub_id']):
            
                select_dataset = dataset[dataset["act_id"]==act_id]
                select_dataset = select_dataset[select_dataset["trial"]==trial_id]
                select_dataset = select_dataset[select_dataset["sub_id"]==sub_id]
                cur_seg_x = preprocess_signal(select_dataset.loc[:,'gravity.x':'userAcceleration.z'], seg_window_size, overlap)
                features  = np.zeros((len(cur_seg_x), cur_seg_x[0].shape[0], cur_seg_x[0].shape[1]))
                
                labels = np.array([int(act_id)]*len(cur_seg_x))
                labels = np.expand_dims(labels, axis=1)
                
                sub_ids = np.array([int(sub_id)]*len(cur_seg_x))
                sub_ids = np.expand_dims(sub_ids, axis=1)
                
                for i in range(len(cur_seg_x)):
                    features[i] = cur_seg_x[i]
                
                if len(X_data) == 0:
                    X_data = features
                else:
                    X_data = np.vstack((X_data, features))
                
                if len(Y_data) == 0:
                    Y_data = labels
                else:
                    Y_data = np.vstack((Y_data, labels))
                
                if len(Sub_ids) == 0:
                    Sub_ids = sub_ids
                else:
                    Sub_ids  = np.vstack((Sub_ids, sub_ids))
                
    X_data = np.swapaxes(X_data.squeeze(),1,2)
    return X_data, Y_data, Sub_ids

def active_matrix_from_angle(basis, angle):
    """Compute active rotation matrix from rotation about basis vector.
    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)
    angle : float
        Rotation angle
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    rep_time = angle.shape[0]

    if basis == 0:
        
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0., 0.],
                      [0.0, 0., 0.]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,1,1] = c
        R[:,1,2] = -s
        R[:,2,1] = s
        R[:,2,2] = c
    elif basis == 1:
        
        R = np.array([[0., 0.0, 0.],
                      [0.0, 1.0, 0.0],
                      [0., 0.0, 0.]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,0,0] = c
        R[:,0,2] = s
        R[:,2,0] = -s
        R[:,2,2] = c
    elif basis == 2:
        
        R = np.array([[0., 0., 0.0],
                      [0., 0., 0.0],
                      [0.0, 0.0, 1.0]])
        R = np.expand_dims(R,0).repeat(rep_time,axis=0)
        R[:,0,0] = c
        R[:,0,1] = -s
        R[:,1,0] = s
        R[:,1,1] = c
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R

def active_matrix_from_extrinsic_euler_xyz(e):
    """Compute active rotation matrix from extrinsic xyz Cardan angles.
    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = np.matmul(active_matrix_from_angle(0, alpha), active_matrix_from_angle(1, beta))
    R = np.matmul(active_matrix_from_angle(2, gamma), R)

    return R

def correct_orientation(dataset):
    
    # cal the orientation correction mat numpy
    orientation_x      = dataset['attitude.pitch'].values
    orientation_y      = dataset['attitude.roll'].values
    orientation_z      = dataset['attitude.yaw'].values
    orientation_xyz    = np.array([orientation_x, orientation_y, orientation_z])
    correct_R          = active_matrix_from_extrinsic_euler_xyz(orientation_xyz)
    
    # correct the orientation of grav, acc and gyro
    grav_xyz           = dataset[['gravity.x','gravity.y','gravity.z']].values
    lacc_xyz           = dataset[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].values
    gyro_xyz           = dataset[['rotationRate.x','rotationRate.y','rotationRate.z']].values
    grav_xyz           = np.matmul(correct_R, np.expand_dims(grav_xyz,2)).squeeze()
    lacc_xyz           = np.matmul(correct_R, np.expand_dims(lacc_xyz,2)).squeeze()
    gyro_xyz           = np.matmul(correct_R, np.expand_dims(gyro_xyz,2)).squeeze()
    dataset[['gravity.x','gravity.y','gravity.z']] = grav_xyz
    dataset[['userAcceleration.x','userAcceleration.y','userAcceleration.z']] = lacc_xyz
    dataset[['rotationRate.x','rotationRate.y','rotationRate.z']] = gyro_xyz
    
    return dataset