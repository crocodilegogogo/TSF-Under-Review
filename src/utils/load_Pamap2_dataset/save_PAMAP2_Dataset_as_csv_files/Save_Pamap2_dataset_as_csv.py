"""PAMAP2 dataset can be Downloaded at https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"""
"""Put the downloaded dataset into the current folder"""
"""Run this py file"""

import os
import csv
import numpy as np
import pandas as pd
import yaml

class data_reader:
    def __init__(self, train_test_files, use_columns, save_npy_dir, labelToId):
        self.data = self.readPamap2(train_test_files, use_columns, labelToId)
        self.save_data(self.data, save_npy_dir)

    def save_data(self, data, save_npy_dir):
        all_data   = data['train']['inputs']
        labels     = np.expand_dims(data['train']['targets'], 1)
        sub_ids    = np.expand_dims(data['train']['subjects'], 1)
        saved_data = pd.DataFrame(np.concatenate((all_data, labels, sub_ids), axis=1))
        saved_data.to_csv(os.path.join(save_npy_dir, 'clean_pamap.csv'), index=False)
        print('Done.')

    def readPamap2(self,train_test_files,use_columns, labelToId):
        files = train_test_files
        cols = use_columns
        data = {dataset: self.readPamap2Files(files[dataset], cols, labelToId)
                for dataset in ('train', 'test', 'validation')}
        return data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        subjects = []
        base_path = os.path.join('PAMAP2_Dataset', 'Protocol')
        print(base_path)
        assert os.path.exists(base_path), "Please download the dataset first using the script"
        for i, filename in enumerate(filelist):
            # print('Reading file %d of %d' % (i+1, len(filelist)))
            with open(os.path.join(base_path, filename), 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    if line[1] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) < 54:
                        # data.append([float(x) / 100 for x in elem[:-1]])
                        data.append([float(x) for x in elem])
                        labels.append(labelToId[elem[1]])
                        subjects.append(i+1)
                        if elem[1] == 1:
                            print(labelToId[elem[1]])
        
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1, 'subjects': np.asarray(subjects, dtype=int)}

def read_dataset_pamap2(train_test_files, use_columns, save_npy_dir, label_to_id):
    dr = data_reader(train_test_files, use_columns, save_npy_dir, label_to_id)

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

# load metadata files
metadata_file = open('metadata.yaml', mode='r')
metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)[
                     'pamap2_preprocess']

# generate pathes for saving files
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','Pamap2')

train_test_files = metadata['file_list']
use_columns = metadata['columns_list']
label_to_id = metadata['label_to_id']

save_npy_dir_flag = create_directory(save_npy_dir)
read_dataset_pamap2(train_test_files, use_columns,
                    save_npy_dir, label_to_id)