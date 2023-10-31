"""MotionSense dataset can be Downloaded at https://github.com/mmalekzadeh/motion-sense"""
"""Put the downloaded dataset into the 'dataset/Motion-Sense' folder"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple
from utils.load_Motion_Sense_dataset.preprocessing import Preprocess
from utils.load_Motion_Sense_dataset.preprocess_raw_data import *

def load_Motion_Sense_raw_data(data_dir, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, 
                               ACT_LABELS, TRIAL_CODES, WINDOW_SIZE, OVERLAP,
                               to_NED_flag, cal_attitude_angle):

    ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
    ## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    sdt = ["attitude","gravity","rotationRate","userAcceleration"]
    print("[INFO] -- Selected sensor data types: "+str(sdt))
    act_labels = ACT_LABELS
    print("[INFO] -- Selected activites: "+str(act_labels))
    
    # the trials of every action
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    
    # the list of every kind of recorded data
    dt_list = set_data_types(sdt)
    
    # the information of the subject
    ds_list = get_ds_infos(data_dir)
    
    # create a labeled dataset of all recorded data; cols: the cloumns index of the dataset
    dataset, cols = creat_time_series(data_dir, dt_list, ds_list, act_labels, trial_codes,
                                      cal_attitude_angle,
                                      mode="raw", labeled=True)

    if to_NED_flag == True:
        
        # correct the orientation of grav, acc and gyro
        dataset = correct_orientation(dataset)

    print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))

    # seperate the dataset to train and test
    train_dataset, test_dataset = seperate_train_test(dataset, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID)
    
    # do the segmentation
    X_train, Y_train, User_ids_train = seg_dataset(train_dataset, trial_codes, WINDOW_SIZE, OVERLAP)
    X_test, Y_test, User_ids_test    = seg_dataset(test_dataset, trial_codes, WINDOW_SIZE, OVERLAP)
    
    # get act and label distionary data
    label2act, act2label = act_label_transform(ACT_LABELS)
    
    return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1), \
           Y_train.squeeze(), Y_test.squeeze(), np.array(User_ids_train+1), np.array(User_ids_test+1), \
           label2act, act2label