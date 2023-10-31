from copy import deepcopy
from time import gmtime, strftime
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from utils.load_DSADS_dataset.preprocessing import *
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas import Series
import os
import sys
import numpy as np
import pandas as pd

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal

def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal) # normalize xyz axes
        return pd.DataFrame(signal, columns=["x", "y", "z"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["x", "y", "z"])

def preprocess_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID, ACT_ID,
                        cal_attitude_angle,
                        scaler, separate_gravity_flag=True,
                        ):

    X_train = np.array([])
    Y_train = np.array([])
    X_test  = np.array([])
    Y_test  = np.array([])
    
    df        = np.load(os.path.join(DATA_DIR, 'clean_DSADS.npy'))
    
    all_frags = df.reshape([19*8,60*125,45])
    
    for frag_id in range(all_frags.shape[0]):
        for pos_id in range(5):
            
            cur_pos_frag       = all_frags[frag_id, :, pos_id*9:(pos_id+1)*9]
            
            cur_pos_acc        = pd.DataFrame(cur_pos_frag[:,0:3], columns = ["x", "y", "z"])
            cur_pos_gyro       = pd.DataFrame(cur_pos_frag[:,3:6], columns = ["x", "y", "z"])
            
            if separate_gravity_flag == True:
                
                # separate gravity and linear acc
                pre_obj = Preprocess(200)
                cur_pos_acc_body, cur_pos_grav = pre_obj.separate_gravity(cur_pos_acc, cal_attitude_angle) # seperate gravity from acc
                
                # norm per sensor
                cur_pos_grav     = scale(cur_pos_grav)
                cur_pos_acc_body = scale(cur_pos_acc_body)
                cur_pos_gyro     = scale(cur_pos_gyro)
                
                cur_pos_fea = np.concatenate((cur_pos_grav, cur_pos_gyro, cur_pos_acc_body), axis=1)
            
            elif separate_gravity_flag == False:
                
                # norm per sensor
                cur_pos_acc      = scale(cur_pos_acc)
                cur_pos_gyro     = scale(cur_pos_gyro)
                cur_pos_fea = np.concatenate((cur_pos_acc, cur_pos_gyro), axis=1)
            
            # concat the sensor data of cur frag
            if pos_id == 0:
                novel_per_frag = cur_pos_fea
            else:
                novel_per_frag = np.concatenate((novel_per_frag, cur_pos_fea), axis=1)
        
        # concat all frags
        if frag_id == 0:
            novel_all_frags = np.expand_dims(novel_per_frag, 0)
        else:
            novel_all_frags = np.concatenate((novel_all_frags, np.expand_dims(novel_per_frag, 0)), axis=0)
    
    novel_all_frags = novel_all_frags.reshape(len(ACT_ID), len(SUBJECTS), 60, 125, -1)
    
    for (sub_id,sub) in enumerate(SUBJECTS):
        for (lbl_id,label) in enumerate(ACT_ID):
            
            # get the array data of current label
            cur_sub_label_frags      = novel_all_frags[(label-1),(sub-1),:,:,:].squeeze()
            
            cur_label_y_labels       = [label-1] * len(cur_sub_label_frags)
            cur_label_sub_labels     = [sub] * len(cur_sub_label_frags)
            
            # distinguish whether the current 'features' should be put into train set or test set, according to the predivided user info
            if (sub in TRAIN_SUBJECTS_ID):
                if len(X_train) == 0:
                    X_train        = cur_sub_label_frags
                    Y_train        = cur_label_y_labels
                    User_ids_train = cur_label_sub_labels
                else:
                    X_train        = np.vstack((X_train, cur_sub_label_frags))
                    Y_train        = Y_train + cur_label_y_labels
                    User_ids_train = User_ids_train + cur_label_sub_labels
            else:
                if len(X_test) == 0:
                    X_test         = cur_sub_label_frags
                    Y_test         = cur_label_y_labels
                    User_ids_test  = cur_label_sub_labels
                else:
                    X_test         = np.vstack((X_test, cur_sub_label_frags))
                    Y_test         = Y_test + cur_label_y_labels
                    User_ids_test  = User_ids_test + cur_label_sub_labels

    return X_train, X_test, Y_train, Y_test, User_ids_train, User_ids_test