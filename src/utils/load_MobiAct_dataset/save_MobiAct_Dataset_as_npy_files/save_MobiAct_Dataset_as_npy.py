"""MobiAct dataset can be Downloaded at https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/"""
"""Put the downloaded dataset into the current folder"""
"""Run this py file"""

import pandas as pd
import glob
import os
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
pd.options.mode.chained_assignment = None

def Interpolation_normalization(curFrag, sepcturalSamples):
    
    # get time
    curTimeList = curFrag[:,0] - curFrag[0,0]
    
    # get data
    curDataList = curFrag[:, 1:10]
    
    # perform Interpolation
    Interp = interp1d(curTimeList, curDataList.T)
    
    InterpTime = np.arange(0.0,curTimeList[-1],sepcturalSamples)
    DataInterpVal = Interp(InterpTime).T
    DataInterpVal[np.isnan(DataInterpVal)] = 0
    
    # insert cls_label in the last column
    if np.unique(curFrag[:,-1]).shape[0] == 1:
        cls_label = np.unique(curFrag[:,-1])[0]
    else:
        print('Class label is not unique, please check the code!')
    DataInterpVal = np.insert(DataInterpVal, obj=DataInterpVal.shape[1], values = cls_label, axis=1)
    return DataInterpVal

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

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
root    = os.path.join(CUR_DIR, 'MobiAct_Dataset_v2.0', 'Annotated Data')
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','MobiAct','Per_subject_no_NED_npy')

paths = [os.path.join(root,'STD'),
         os.path.join(root,'WAL'),
         os.path.join(root,'JOG'),
         os.path.join(root,'JUM'),
         os.path.join(root,'STU'),
         os.path.join(root,'STN'),
         os.path.join(root,'SCH'),
         os.path.join(root,'SIT'),
         os.path.join(root,'CHU'),
         os.path.join(root,'CSI'),
         os.path.join(root,'CSO')]

Labels      = ['STD','WAL','JOG','JUM','STU','STN','SCH','SIT','CHU','CSI','CSO']
newColNames = {'azimuth':'yaw'}

mis_length = 0.5
sepcturalSamples = 0.005
numObs     = 256
overlap    = int(numObs/2)
subjectDataList = [[] for i in range(67)]

dfList = []
for folder in paths:
    files = glob.glob(folder + r'\*.csv')
    
    for f in files:
        curFileList = []
        data  = pd.read_csv(f)
        
        # get the user_id and act label of current file
        fname = os.path.basename(f)
        parts = fname.split('_')
        user_id  = int(parts[1])
        lbl   = parts[0]
        data.drop(columns = ['timestamp'], inplace = True)
        print('Read ' + fname + ' over!')
        
        # grab the wanted activity based on the folder name
        df                     = data[data['label']==lbl]
        df.reset_index(inplace = True, drop = True)
        df.rename(columns      = newColNames, inplace= True)
        
        # normalize the acc and orientation data
        df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']]/9.81
        df[['yaw', 'pitch', 'roll']]    = df[['yaw', 'pitch', 'roll']]*(np.pi/180) %(2*np.pi)
        
        # get df_mat composed of numpy data of acc, gyro, orientation and labels
        df_mat              = df.values
        df_mat[:,-1]        = Labels.index(lbl)
        df_mat              = df_mat.astype(np.float64)
        df_mat              = df_mat[df_mat[:,0].argsort()]
        
        # get the diff time of current file
        cur_file_timestamps = np.array(df[['rel_time']]).squeeze()
        cur_file_timestamps = np.diff(cur_file_timestamps)
        cur_file_timestamps = np.insert(cur_file_timestamps, obj=0, values=0)
        
        # find discontinuous position
        discontin_positions     = np.where(cur_file_timestamps>=mis_length)[0]
        
        # insert the last timestamp to discontin_positions
        if discontin_positions.shape[0] == 0: # if discontin_positions do not exist
            discontin_positions = np.array([len(cur_file_timestamps)])
        elif (cur_file_timestamps.shape[0]-1) not in discontin_positions:
            discontin_positions = np.insert(discontin_positions, obj=-1,
                                            values=len(cur_file_timestamps))
            discontin_positions = np.sort(discontin_positions)
        
        # get current fragment
        cur_frag_start       = 0
        for (pos_id, pos) in enumerate(discontin_positions):
            curFrag          = df_mat[cur_frag_start:pos,:]
            
            # update start time
            cur_frag_start   = pos
            
            # interpolate current fragment
            if curFrag.shape[0] == 1:
                continue
            else:
                curFrag      = Interpolation_normalization(curFrag, sepcturalSamples)
                
                # add the fragment to curDevList
                curFileList.append(curFrag)
        
        subjectDataList[user_id-1] = subjectDataList[user_id-1] + curFileList

save_npy_dir_flag = create_directory(os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                                    'datasets','MobiAct'))
for (sub_id,subject) in enumerate(subjectDataList):
    if subject != []:
        per_sub_data_save_dir = os.path.join(save_npy_dir, str(sub_id+1)+'_MobiActdata.npy')
        np.save(per_sub_data_save_dir,[subjectDataList[sub_id]])