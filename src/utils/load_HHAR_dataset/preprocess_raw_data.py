import os
import sys
import numpy as np 
from copy import deepcopy
from time import gmtime, strftime
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from utils.load_HHAR_dataset.preprocessing import Preprocess
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal

def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal) # normalize the xyz axes
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def preprocess_raw_data(read_data_dir, SUBJECTS, TRAIN_SUBJECTS_ID, window_size, overlap, cal_attitude_angle,
                        scaler, separate_gravity_flag=True):

    X_train = np.array([])
    Y_train = np.array([])
    X_test = np.array([])
    Y_test = np.array([])
    
    SubID = range(len(SUBJECTS))
    subIdxDict  = dict(zip(SUBJECTS, SubID))
    
    dataList = os.listdir(read_data_dir)
    
    for sub_file in dataList:
        
        if os.path.exists(os.path.join(read_data_dir, sub_file)):
            sub_data = np.load(os.path.join(read_data_dir, sub_file), allow_pickle=True)
        for cur_frag_id in range(sub_data.shape[1]):
            print('Read num ' + str(cur_frag_id) + ' fragment of ' + sub_file)
            cur_frag = sub_data[0][cur_frag_id]
            if cur_frag.shape[0] < window_size:
                continue
            acc_raw  = cur_frag[:,0:3]
            acc_raw  = pd.DataFrame(acc_raw, columns = ["x", "y", "z"])
            gyro_raw = cur_frag[:,3:6]
            gyro_raw  = pd.DataFrame(gyro_raw, columns = ["x", "y", "z"])
            if separate_gravity_flag:
                
                # separate gravity and linear acc
                pre_obj = Preprocess()
                acc_body, acc_grav = pre_obj.separate_gravity(acc_raw, cal_attitude_angle) # seperate gravity from acc
                
                # scale the current fragment
                acc_body = scale(acc_body, scaler=scaler)
                acc_grav = scale(acc_grav, scaler=scaler)
                gyro_raw = scale(gyro_raw, scaler=scaler)
                
                # seg the current fragment
                tGravityAccXYZ = preprocess_signal(acc_grav, window_size, overlap)
                tBodyGyroXYZ   = preprocess_signal(gyro_raw, window_size, overlap)
                tBodyAccXYZ    = preprocess_signal(acc_body, window_size, overlap)
                channel_num = 9
                column_ind = ["GravAccX", "GravAccY", "GravAccZ",
                              "GyroX",    "GyroY",    "GyroZ",
                              "BodyAccX", "BodyAccY", "BodyAccZ"]
                
            else:
                # scale the current fragment
                acc_raw  = scale(acc_raw, scaler=scaler)
                gyro_raw = scale(gyro_raw, scaler=scaler)
                
                # seg the current fragment
                tAccXYZ = preprocess_signal(acc_raw, window_size, overlap)
                tBodyGyroXYZ = preprocess_signal(gyro_raw, window_size, overlap)
                channel_num = 6
                column_ind = ["AccX",  "AccY",  "AccZ",
                              "GyroX", "GyroY", "GyroZ"]
            
            # the number of windows in current experiment, user and action, put them into features
            features = np.zeros((len(tBodyGyroXYZ), window_size, channel_num)) # len(tAccXYZ):number of windows, 128:window size, 6:the axises of channels
            
            # concatenate acc and gyro data into 'feature', then the seg windows of data are put into features
            for i in range(len(tBodyGyroXYZ)):
                if separate_gravity_flag:
                    np_concat = np.concatenate((tGravityAccXYZ[i], tBodyGyroXYZ[i], tBodyAccXYZ[i]), 1)
                else:
                    np_concat = np.concatenate((tAccXYZ[i], tBodyGyroXYZ[i]), 1)
                feature = pd.DataFrame(
                    np_concat,
                    columns=column_ind,
                )
                features[i] = feature
            
            # Record the y_labels
            if np.unique(cur_frag[:,-1]).shape[0] == 1:
                act_id   = int(np.unique(cur_frag[:,-1])[0])
            else:
                print('class_label is more than one, check the code!')
            y_labels = [act_id] * len(tBodyGyroXYZ)
            
            # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
            if subIdxDict[sub_file[0]] in TRAIN_SUBJECTS_ID:
                if len(X_train) == 0:
                    X_train = features
                    Y_train = y_labels
                    User_ids_train = [subIdxDict[sub_file[0]]] * len(y_labels)
                else:
                    X_train = np.vstack((X_train, features))
                    Y_train = Y_train + y_labels
                    User_ids_train = User_ids_train + [subIdxDict[sub_file[0]]] * len(y_labels)
            else:
                if len(X_test) == 0:
                    X_test = features
                    Y_test = y_labels
                    User_ids_test = [subIdxDict[sub_file[0]]] * len(y_labels)
                else:
                    X_test = np.vstack((X_test, features))
                    Y_test = Y_test + y_labels
                    User_ids_test = User_ids_test + [subIdxDict[sub_file[0]]] * len(y_labels)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], channel_num, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], channel_num, 1)

    return X_train, X_test, Y_train, Y_test, User_ids_train, User_ids_test
