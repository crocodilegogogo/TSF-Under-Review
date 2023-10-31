import os
import sys
import numpy as np 
from copy import deepcopy
from time import gmtime, strftime

from scipy.interpolate import interp1d
from scipy.fftpack import fft
from utils.load_Pamap2_dataset.preprocessing import *
# from preprocessing import *
import pandas as pd
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
        signal = StandardScaler().fit_transform(signal) # normalize the xyz axes
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def preprocess_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                        window_size, overlap, cal_attitude_angle,
                        scaler, separate_gravity_flag=True):

    X_train = np.array([])
    Y_train = np.array([])
    X_test  = np.array([])
    Y_test  = np.array([])
    
    df             = pd.read_csv(os.path.join(DATA_DIR, 'clean_pamap.csv'))
    FEATURES       = [str(i) for i in range(20)]
    LABEL_COL      = 20
    SUBJECT_COL    = 21
    LABEL_DATA     = df.values[:, LABEL_COL]
    LABEL_DATA[np.where(LABEL_DATA == 11)] = 8
    LABEL_DATA[np.where(LABEL_DATA == 12)] = 9
    LABEL_DATA[np.where(LABEL_DATA == 13)] = 10
    LABEL_DATA[np.where(LABEL_DATA == 14)] = 11
    LABEL_DATA[np.where(LABEL_DATA == 18)] = 12
    df.values[:, LABEL_COL]                = LABEL_DATA-1
    
    mis_length     = 0.25
    
    LABELS         = np.unique(df.values[:, LABEL_COL]).tolist()
    
    for (sub_id,sub) in enumerate(SUBJECTS):
        for (lbl_id,label) in enumerate(LABELS):
            curLabelFragList   = []
            
            # get the array data of current label
            cur_sub_label_data = df[df[str(SUBJECT_COL)] == sub]
            cur_sub_label_data = cur_sub_label_data[cur_sub_label_data[str(LABEL_COL)] == label].values
            
            if cur_sub_label_data.shape[0] < window_size:
                print('subject:', sub, 'label:', label, ', not included')
                continue
            
            # If cur_sub_label_data is discontinous in time, get the diff time of current file.
            cur_data_timestamps = np.array(cur_sub_label_data[:, 0]).squeeze()
            cur_data_timestamps = np.diff(cur_data_timestamps)
            cur_data_timestamps = np.insert(cur_data_timestamps, obj=0, values=0)
            
            # find discontinuous position
            discontin_positions     = np.where(cur_data_timestamps>=mis_length)[0]
            # insert the last timestamp to discontin_positions
            if discontin_positions.shape[0] == 0: # if discontin_positions do not exist
                discontin_positions = np.array([len(cur_data_timestamps)])
            elif cur_data_timestamps.shape[0] not in discontin_positions:
                discontin_positions = np.insert(discontin_positions, obj=-1,
                                                values=len(cur_data_timestamps)) # insert the last point
                discontin_positions = np.sort(discontin_positions)
            
            # get discontinuous fragments, save them to list
            cur_frag_start          = 0
            for (dis_pos_id, dis_pos) in enumerate(discontin_positions):
                curFrag             = cur_sub_label_data[cur_frag_start:dis_pos, :]
                
                # update start time
                cur_frag_start      = dis_pos
                
                # only interpolate the feature part of current fragment
                if curFrag.shape[0] == 1:
                    continue
                else:
                    curFragFea = Interpolation(curFrag)
                    curLabelFragList.append(curFragFea)
            
            # scale and separate_gravity from each frag
            for (id_F,Frag) in enumerate(curLabelFragList):
                if Frag.shape[0] < window_size:
                    continue
                for pos_id in range(3):
                    cur_pos_frag       = Frag[:, pos_id*6 : (pos_id+1)*6]
                    
                    cur_pos_acc        = pd.DataFrame(cur_pos_frag[:,0:3], columns = ["x", "y", "z"])
                    cur_pos_gyro       = pd.DataFrame(cur_pos_frag[:,3:6], columns = ["x", "y", "z"])
                    
                    if separate_gravity_flag == True:
                        # separate gravity and linear acc
                        pre_obj = Preprocess(800) # 500
                        cur_pos_acc_body, cur_pos_grav = pre_obj.separate_gravity(cur_pos_acc, cal_attitude_angle) # seperate gravity from acc
                        # norm per sensor
                        cur_pos_grav     = scale(cur_pos_grav)
                        cur_pos_acc_body = scale(cur_pos_acc_body)
                        cur_pos_gyro     = scale(cur_pos_gyro)
                        
                        cur_pos_fea      = np.concatenate((cur_pos_grav, cur_pos_gyro, cur_pos_acc_body), axis=1)
                    
                    elif separate_gravity_flag == False:
                        
                        # norm per sensor
                        cur_pos_acc     = scale(cur_pos_acc)
                        cur_pos_gyro    = scale(cur_pos_gyro)
                        
                        cur_pos_fea     = np.concatenate((cur_pos_acc, cur_pos_gyro), axis=1)
                    
                    # concat the sensor data of different positions in cur frag
                    if pos_id == 0:
                        cur_sub_label_frag   = cur_pos_fea
                    else:
                        cur_sub_label_frag   = np.concatenate((cur_sub_label_frag, cur_pos_fea), axis=1)
            
                # get the IMU data only
                concat_cur_sub_label_fea = pd.DataFrame(cur_sub_label_frag)
                
                cur_label_segments_mid   = preprocess_signal(concat_cur_sub_label_fea, window_size, overlap)
                cur_label_segments_mid   = np.array(cur_label_segments_mid)
                
                cur_label_y_labels       = [label] * len(cur_label_segments_mid)
                cur_label_sub_labels     = [sub] * len(cur_label_segments_mid)
                
                # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
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