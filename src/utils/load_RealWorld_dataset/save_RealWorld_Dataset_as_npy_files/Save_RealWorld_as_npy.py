"""RealWorld dataset can be Downloaded at https://www.uni-mannheim.de/dws/research/projects/activity-recognition/"""
"""Put the downloaded dataset into the current folder"""
"""Run this py file"""

from zipfile import ZipFile
from scipy.interpolate import interp1d
import os
import pandas as pd
import csv
import numpy as np
import glob
import os
import pandas as pd

# create a list with all zip file paths
def get_zip_paths_list(path, dirs):
    subpath = ''
    paths_list = []
    count = 0
    for proband in dirs:
        subpath = os.path.join(path, proband)
        subdir = os.listdir(subpath) # ['data', 'images', 'videos']
        for foldername in subdir:
            if foldername == 'data':
                subsubpath = os.path.join(subpath, "data")
                subsubdir = os.listdir(subsubpath) # all zips in the subject folder
                for file in subsubdir:
                    if file != 'extracted':
                        newpath = os.path.join(subsubpath, file)  # each zip
                        with ZipFile(newpath, 'r') as zip:
                            csvlist = zip.namelist()
                            if 'readMe' not in csvlist:
                                extracted_path = os.path.join(subsubpath, 'extracted')
                                if not os.path.isdir(extracted_path):
                                    os.makedirs(extracted_path)
                                zip.extractall(extracted_path)
                                zip.close()
                                for per_zip in csvlist:
                                    newpath = os.path.join(extracted_path, per_zip)
                                    paths_list.append(newpath)
                                    count += 1
                            else:
                                paths_list.append(newpath)
                                count += 1
            else:
                continue
    return paths_list

#input zipfilename, output list of csv files
def openzip(file_name, subject_id):
    labellist = []
    
    with ZipFile(file_name, 'r') as zip:
        csvlist = zip.namelist()
    
        if 'readMe' in csvlist:
            csvlist.remove('readMe')
        
        for name in csvlist:
            sublabellist = []
            zip.extract(name)
            
            #get labels
            templist = name.split("_")
            sensory_type = templist[0]
            activity = templist[1]
            body_position = templist[-1].split(".")
            body_position = body_position[0]
            
            #create sublist in csvlist per csv file for labels
            sublabellist.append(sensory_type)
            sublabellist.append(activity)
            sublabellist.append(body_position)
            if len(templist) > 3:
                record_id = templist[2]
                sublabellist.append(record_id)
            
            labellist.append(sublabellist)
    return csvlist, labellist

def Interpolation(selected_mat, sepcturalSamples):
    # get time
    curTimeList = selected_mat[:,0] - selected_mat[0,0]
    # get data
    curDataList = selected_mat[:, 1:]
    # perform Interpolation
    Interp      = interp1d(curTimeList, curDataList.T)
    InterpTime  = np.arange(0.0,curTimeList[-1],sepcturalSamples)
    DataInterpVal = Interp(InterpTime).T
    DataInterpVal[np.isnan(DataInterpVal)] = 0
    Ids         = np.arange(InterpTime.shape[0])  # get ids
    Attr_time   = InterpTime + selected_mat[0,0]  # get attr_time
    DataInterpVal = np.insert(DataInterpVal, obj=0, values = Attr_time, axis=1) # insert attr_time
    DataInterpVal = np.insert(DataInterpVal, obj=0, values = Ids, axis=1)       # insert ids
    return DataInterpVal

def create_dataframe(csvfile, labellist, count, subject_ID):
    df = pd.read_csv(csvfile)
    df.sort_values(by=['attr_time'], axis=0, inplace=True)
    if df.shape[0] > 3:
        selected_cols = list(df)
        selected_mat  = df.values[:,1:]
        interp_mat    = Interpolation(selected_mat, 20)
        df            = pd.DataFrame(interp_mat, index=list(np.arange(interp_mat.shape[0])), columns=selected_cols)
    
    if count == 0:
        
        # add columns to df using labellist
        df['Subject ID'] = subject_ID
        df['Sensor-type'] = labellist[count][0]
        df['Activity'] = labellist[count][1]
        
        columns = list(df)
        adjusted_cols = columns[5:] + columns[:5]
        df = df.loc[:, adjusted_cols]
        
    body_position = labellist[count][2]
    
    df = df.rename(columns=
                  {'attr_x': 'attr_x_' + body_position,
                  'attr_y': 'attr_y_' + body_position,
                  'attr_z': 'attr_z_' + body_position})
    return df

# The main code path down below, retrieves for each zip file from which subject it is and generates a csv file \
#     containing the merged dataframe of each sensor-type, activity, subject combination.
def merge_csv_of_positions(paths_list):
    body_position_names = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
    for path in paths_list:
        #retrieve labels
        templist = path.split("/")
        if len(templist)==1:
            templist = path.split("\\")
        # templist2 = templist[3].split("_")
        # os.path.split(path)[0]
    
        #get subject label
        # subject_ID = templist[14].replace("proband", '')
        subject_ID = templist[1].replace("proband", '')
        
        persubsubpath = ''
        for perpath in templist[:-1]:
            persubsubpath = os.path.join(persubsubpath, perpath)
        
        print("PATH: " + path + " SUBJECT: " + subject_ID)
        
        count = 0
        csvlist, labellist = openzip(path, subject_ID)
        
        mid_csvlist = []
        if len(csvlist) == 7:
            for pos_name in body_position_names:
                for mid_csv in csvlist:
                    if pos_name == mid_csv.split('_')[-1].split('.')[0]:
                        mid_csvlist.append(mid_csv)\
            
            # arrange csvlist in order
            csvlist = mid_csvlist
        
            for (csv_i,csv) in enumerate(csvlist):
                if count == 0:
                    # first_df
                    df1 = create_dataframe(csv, labellist, count, subject_ID)
                else:
                    # merge
                    df2 = create_dataframe(csv, labellist, count, subject_ID)
                    merged_df = df1.merge(df2, on='id', suffixes = ('_df'+str(csv_i+1), '_df1'))
                    df1 = merged_df
                
                count += 1
                
                # remove csv after use
                os.remove(csv)  
            
            if len(labellist[0]) > 3:
                #temporary csv
                filename = subject_ID + '_' + labellist[0][0] + '_' + labellist[0][1] + '_' + labellist[0][-1] + '_' + 'Merged.csv'
            else:
                # temporary csv
                filename = subject_ID + '_' + labellist[0][0] + '_' + labellist[0][1] + '_' + 'Merged.csv'
            print("MERGE FILE COMPLETE: " + filename)
            
            merged_df.to_csv(filename, index=True)
        else:
            for csv in csvlist:
                os.remove(csv)

def get_all_sensor_csvs_list(dirs, sensors):

    newdirs = []

    for (sensor_id, sensor) in enumerate(sensors):
        for item in dirs:
            cur_sensor_newdirs = glob.glob('*' + sensor + '*.csv')
        if sensor_id == 0:
            newdirs = cur_sensor_newdirs
        else:
            newdirs = newdirs + cur_sensor_newdirs
    return newdirs

def sep_newdirs_using_(newdirs):
    Merged_csv_dirs = []
    for each_dir in newdirs:
        dir_mid = each_dir.split('_')
        Merged_csv_dirs.append(dir_mid)
    return Merged_csv_dirs

def merge_csvs_of_all_sensors(activities, times, sensors, Merged_csv_dirs):
    all_sub_act_time_merge_sensors = []
    cur_sub_act_time_sensors = []
    for sub_id in range(15):
        sub_id += 1
        print(sub_id)
        for cur_act in activities:
            print(cur_act)
            for time_id in times:
                print(time_id)
                
                # During the temporal loop, the empty set is reset so that the final result is appended only once
                cur_sub_act_sensors = []
                for cur_sensor in sensors:
                    print(cur_sensor)
                    for (per_m_scv_id,per_m_scv) in enumerate(Merged_csv_dirs):
                        
                        # Put the data from different sensors of the same subject/activity together
                        if len(per_m_scv) == 4:
                            if int(per_m_scv[0]) == sub_id and \
                                 per_m_scv[1] == cur_sensor and \
                                   per_m_scv[2] == cur_act:
                                cur_sub_act_sensors.append(newdirs[per_m_scv_id])
                        
                        # Put the data from different sensors of the same subject/activity/timestamp together
                        elif len(per_m_scv) == 5:
                            if int(per_m_scv[0]) == sub_id and \
                                 per_m_scv[1] == cur_sensor and \
                                     per_m_scv[2] == cur_act and \
                                       int(per_m_scv[3]) == time_id:
                                cur_sub_act_time_sensors.append(newdirs[per_m_scv_id])
                
                # Put the data from different sensors of the same sub/act/timestamp (that have been put together) into the final list
                if cur_sub_act_time_sensors:
                    all_sub_act_time_merge_sensors.append(cur_sub_act_time_sensors)
                    cur_sub_act_time_sensors = []
            
            # Put the data from different sensors of the same sub/act (that have been put together) into the final list
            if cur_sub_act_sensors:
                all_sub_act_time_merge_sensors.append(cur_sub_act_sensors)
                cur_sub_act_sensors = []
    return all_sub_act_time_merge_sensors

def get_final_mat(all_sub_act_time_merge_sensors, activities):
    
    cur_frag_count = 0
    for (curfrag_id, curfrag) in enumerate(all_sub_act_time_merge_sensors):
        
        curfrag_acc  = pd.read_csv(curfrag[0])
        curfrag_gyro = pd.read_csv(curfrag[1])
        curfrag_mag  = pd.read_csv(curfrag[2])
        
        first_timestamps = [curfrag_acc.values[0,5], curfrag_gyro.values[0,5],\
                            curfrag_mag.values[0,5]]
        cur_frag_start   = max(first_timestamps)
        
        find_start_id_acc  = np.argmin(abs(curfrag_acc.values[:,5] - cur_frag_start))
        find_start_id_gyro = np.argmin(abs(curfrag_gyro.values[:,5] - cur_frag_start))
        find_start_id_mag  = np.argmin(abs(curfrag_mag.values[:,5] - cur_frag_start))
        
        [cur_acc_during, cur_gyro_during, cur_mag_during] = [curfrag_acc.shape[0]-find_start_id_acc,\
                                                    curfrag_gyro.shape[0]-find_start_id_gyro,\
                                                    curfrag_mag.shape[0]-find_start_id_mag]
        
        cur_frag_during  = min([cur_acc_during, cur_gyro_during, cur_mag_during])
        
        for i in range(7):
            curpos_acc  = curfrag_acc.values[find_start_id_acc:(find_start_id_acc+cur_frag_during), (6+4*i):(9+4*i)]
            curpos_gyro = curfrag_gyro.values[find_start_id_gyro:(find_start_id_gyro+cur_frag_during), (6+4*i):(9+4*i)]
            curpos_mag  = curfrag_mag.values[find_start_id_mag:(find_start_id_mag+cur_frag_during), (6+4*i):(9+4*i)]
            if i == 0:
                cur_frag_mat = np.concatenate((curpos_acc, curpos_gyro, curpos_mag), axis = 1)
            else:
                cur_frag_mat = np.concatenate((cur_frag_mat, curpos_acc, curpos_gyro, curpos_mag), axis = 1)
            
        cur_frag_count_mat = cur_frag_count + np.zeros((cur_frag_mat.shape[0],1))
        cur_label          = activities.index(curfrag_acc.iloc[0][3])
        cur_label_mat      = cur_label + np.zeros((cur_frag_mat.shape[0],1))
        sub_id             = int(curfrag_acc.iloc[0][1])
        sub_id_mat         = sub_id + np.zeros((cur_frag_mat.shape[0],1))
        
        cur_frag_mat       = np.concatenate((cur_frag_count_mat, sub_id_mat, cur_label_mat, cur_frag_mat), axis = 1)
        
        if curfrag_id == 0:
            all_frag_mat   = cur_frag_mat
        else:
            all_frag_mat   = np.concatenate((all_frag_mat, cur_frag_mat), axis=0)
        
        print(cur_frag_count)
        cur_frag_count += 1
    return all_frag_mat

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

# Opening data files
path = os.path.join('realworld2016_dataset')
CUR_DIR  = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','RealWorld', "Clean_Real_World.npy")
dirs = os.listdir(path)

#find folders
for file in dirs:
    print(file)

paths_list = get_zip_paths_list(path, dirs)
#all zip file paths
print(paths_list)

# Remove sqlite.zip files and thus only use csv's
paths_list = [x for x in paths_list if "sqlite" not in x]
print(paths_list)

# Generate csv after merging all locations
merge_csv_of_positions(paths_list)

dir = os.getcwd()
dirs = os.listdir(dir)

sensors = ['acc', 'Gyroscope', 'MagneticField']

newdirs = get_all_sensor_csvs_list(dirs, sensors)

Merged_csv_dirs = sep_newdirs_using_(newdirs)

activities = ['climbingdown','climbingup','jumping','lying',
              'running','sitting','standing','walking']
times = [2,3]

# Put the csv files of each act or different trials of each act of each subject (all sensor data) in a list component, for loading convenent
all_sub_act_time_merge_sensors = merge_csvs_of_all_sensors(activities, times, sensors, Merged_csv_dirs)

# put the csv files into the unified mat file
all_frag_mat = get_final_mat(all_sub_act_time_merge_sensors, activities)

save_npy_dir_flag = create_directory(os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                                     'datasets','RealWorld'))
np.save(save_npy_dir, all_frag_mat)

