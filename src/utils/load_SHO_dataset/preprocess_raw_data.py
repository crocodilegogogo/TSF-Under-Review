import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from time import gmtime, strftime
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from utils.load_SHO_dataset.preprocessing import *
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas import Series

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal

def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal) # 这个是对xyz三轴同时做归一化
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def preprocess_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                        window_size, overlap, cal_attitude_angle,
                        scaler):
    
    X_train = np.array([])
    Y_train = np.array([])
    X_test  = np.array([])
    Y_test  = np.array([])
    
    all_data = np.load(os.path.join(DATA_DIR,'Clean_SHO.npy'), allow_pickle=True)
    SUBJECTS = list(np.unique(all_data[:,1]).astype(np.int))
    LABEL_DATA = list(np.unique(all_data[:,2]).astype(np.int))
    
    # get discontinuous fragments, save them to list
    curFragList = []
    for Frag_id in list(np.unique(all_data[:,0]).astype(np.int)):
        curFragList.append(all_data[all_data[:,0]==Frag_id])
    
    # scale and separate_gravity from each frag
    for (id_F,Frag) in enumerate(curFragList):
        if Frag.shape[0] < window_size:
            continue
        for pos_id in range(5):
            cur_pos_frag       = Frag[:, (3+pos_id*9) : (3+9+pos_id*9)]
            
            cur_pos_grav       = pd.DataFrame(cur_pos_frag[:,0:3], columns = ["x", "y", "z"])
            cur_pos_gyro       = pd.DataFrame(cur_pos_frag[:,3:6], columns = ["x", "y", "z"])
            cur_pos_acc_body   = pd.DataFrame(cur_pos_frag[:,6:9], columns = ["x", "y", "z"])
            
            # norm per sensor
            cur_pos_grav     = scale(cur_pos_grav)
            cur_pos_acc_body = scale(cur_pos_acc_body)
            cur_pos_gyro     = scale(cur_pos_gyro)
            
            cur_pos_fea      = np.concatenate((cur_pos_grav, cur_pos_gyro, cur_pos_acc_body), axis=1)
            
            # concat the sensor data of different positions in cur frag
            if pos_id == 0:
                cur_sub_label_frag   = cur_pos_fea
            else:
                cur_sub_label_frag   = np.concatenate((cur_sub_label_frag, cur_pos_fea), axis=1)
        
        # get the IMU data only
        concat_cur_sub_label_fea = pd.DataFrame(cur_sub_label_frag)
        
        cur_label_segments_mid   = preprocess_signal(concat_cur_sub_label_fea, window_size, overlap)
        cur_label_segments_mid   = np.array(cur_label_segments_mid)
        
        cur_label_y_labels       = [Frag[0,2]] * len(cur_label_segments_mid)
        cur_label_sub_labels     = [Frag[0,1]] * len(cur_label_segments_mid)
        
        # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
        sub = Frag[0,1]
        if (sub in TRAIN_SUBJECTS_ID):
            if len(X_train) == 0:
                X_train        = cur_label_segments_mid
                Y_train        = cur_label_y_labels
                User_ids_train = cur_label_sub_labels
            else:
                X_train        = np.vstack((X_train, cur_label_segments_mid))
                Y_train        = Y_train + cur_label_y_labels
                User_ids_train = User_ids_train + cur_label_sub_labels
        else:
            if len(X_test) == 0:
                X_test         = cur_label_segments_mid
                Y_test         = cur_label_y_labels
                User_ids_test  = cur_label_sub_labels
            else:
                X_test         = np.vstack((X_test, cur_label_segments_mid))
                Y_test         = Y_test + cur_label_y_labels
                User_ids_test  = User_ids_test + cur_label_sub_labels
                    
    return X_train, X_test, Y_train, Y_test, User_ids_train, User_ids_test