"""Load dataset"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.load_Pamap2_dataset.preprocess_raw_data import preprocess_raw_data
# from preprocess_raw_data import preprocess_raw_data

# CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
# # DATA_DIR = os.path.join(CUR_DIR, "../../data")
# DATA_DIR = CUR_DIR


def load_Pamap2_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID, ACT_LABELS, ACT_ID,
                     window_size, overlap, separate_gravity_flag,
                     cal_attitude_angle,
                     scaler: str = "normalize"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    """Load raw dataset.
    The following six classes are included in this experiment.
        - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    The following transition classes are excluded.
        - STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, and LIE_TO_STAND
    Args:
        scaler (str): scaler for raw signals, chosen from normalize or minmax
    Returns:
        X_train (pd.DataFrame):
        X_test (pd.DataFrame):
        y_train (pd.DataFrame):
        y_test (pd.DataFrame):
        label2act (Dict[int, str]): Dict of label_id to title_of_class
        act2label (Dict[str, int]): Dict of title_of_class to label_id
    """
    X_train, X_test, Y_train, Y_test, \
         User_ids_train, User_ids_test = preprocess_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                                                             window_size, overlap, cal_attitude_angle,
                                                             scaler=scaler, separate_gravity_flag=separate_gravity_flag)

    y_train = np.expand_dims(Y_train, 1)
    y_test  = np.expand_dims(Y_test, 1)
    
    ActID     = (np.array(ACT_ID) - 1).tolist()
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))
    
    X_train = np.swapaxes(X_train.squeeze(),1,2)
    X_test = np.swapaxes(X_test.squeeze(),1,2)
    
    return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1), y_train.squeeze(), y_test.squeeze(), \
           np.array(User_ids_train), np.array(User_ids_test), label2act, act2label