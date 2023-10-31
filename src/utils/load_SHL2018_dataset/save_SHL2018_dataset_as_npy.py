import numpy as np
import pandas as pd
import os
from pandas import Series

def save_SHL2018_to_npy(data_dir, std, dataset_size=-1):
    
    for data_type in ['train', 'test']:
        
        for std_id in std:
            
            if std_id == 'label':
                
                # read the txt file of Label
                data_file_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                std_id, std_id.capitalize() + '.txt')
                data_all_axis_array = pd.read_table(data_file_dir, sep = ' ')
                data_all_axis_array = np.array(data_all_axis_array)[:dataset_size,:]
                print('Read '+std_id.capitalize()+' of '+data_type+' over.')
                
                # save the Label data to npy
                data_npy_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                            std_id, std_id + '.npy')
                np.save(data_npy_dir, data_all_axis_array)
                print('Save '+std_id+'.npy'+' of '+data_type+' over!!')
            
            elif std_id == 'ori':

                for axis_id in ['w', 'x', 'y', 'z']:
                    
                    # read the txt file of w,x,y,z
                    data_file_dir = os.path.join(data_dir, data_type, data_type + '_' +\
                                                 std_id, std_id.capitalize() + '_' + axis_id + '.txt')
                    data_per_axis = pd.read_table(data_file_dir, sep=' ')
                    print('Read '+std_id.capitalize()+'_'+axis_id+' of '+data_type+' over.')
                    
                    # concatenate the w,x,y,z data:(sample_num, 4, datalength)
                    if axis_id == 'w':
                        data_all_axis_array = np.expand_dims(np.array(data_per_axis), 1)[:dataset_size, :,
                                              :]  # (16309, 6000)
                    else:
                        data_per_axis = np.expand_dims(np.array(data_per_axis), 1)[:dataset_size, :, :]
                        data_all_axis_array = np.concatenate((data_all_axis_array,
                                                              data_per_axis),
                                                             axis=1)  # (16308, 3, 6000)

                data_npy_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                            std_id, std_id + '_wxyz.npy')
                np.save(data_npy_dir, data_all_axis_array)
                print('Save '+std_id+'_wxyz.npy'+' of '+data_type+' over!!')
            
            else:
                for axis_id in ['x', 'y', 'z']:
                    
                    # read the txt file of x,y,z
                    data_file_dir = os.path.join(data_dir, data_type, data_type + '_' + \
                                                 std_id, std_id.capitalize() + '_' + axis_id + '.txt')
                    data_per_axis  = pd.read_table(data_file_dir, sep = ' ')
                    print('Read '+std_id.capitalize()+'_'+axis_id+' of '+data_type+' over.')
                    
                    # concatenate the x,y,z data:(sample_num, 3, datalength)
                    if axis_id == 'x':
                        data_all_axis_array = np.expand_dims(np.array(data_per_axis), 1)[:dataset_size,:,:]
                    else:
                        data_per_axis       = np.expand_dims(np.array(data_per_axis), 1)[:dataset_size,:,:]
                        data_all_axis_array = np.concatenate((data_all_axis_array,
                                                             data_per_axis),
                                                             axis = 1)
                
                # save the concatenated data to npy
                data_npy_dir = os.path.join(data_dir, data_type, data_type + '_' +\
                                            std_id, std_id + '_xyz.npy')
                np.save(data_npy_dir, data_all_axis_array)
                print('Save '+std_id+'_xyz.npy'+' of '+data_type+' over!!')