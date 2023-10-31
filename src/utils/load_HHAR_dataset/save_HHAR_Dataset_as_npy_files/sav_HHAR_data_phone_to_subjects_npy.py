import os
import sys
import numpy as np 
from copy import deepcopy
from time import gmtime, strftime

from scipy.interpolate import interp1d
from scipy.fftpack import fft

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

def Interpolation_normalization(curFrag, sepcturalSamples):
    
    # get time
    curTimeList = curFrag[:,0] - curFrag[0,0]
    
    # get data
    curDataList = curFrag[:, 1:7]
    
    # perform Interpolation
    Interp = interp1d(curTimeList, curDataList.T)
    InterpTime = np.arange(0.0,curTimeList[-1],sepcturalSamples)
    DataInterpVal = Interp(InterpTime).T
    DataInterpVal[np.isnan(DataInterpVal)] = 0
    
    # insert cls_labell in the last column
    if np.unique(curFrag[:,-1]).shape[0] == 1:
        cls_label = np.unique(curFrag[:,-1])[0]
    else:
        print('Class label is not unique, please check the code!')
    DataInterpVal = np.insert(DataInterpVal, obj=DataInterpVal.shape[1], values = cls_label, axis=1)
    return DataInterpVal

timeLabel = 'Creation_Time'
pairDir = 'Per_subject_device_gt_npy'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','HHAR','Per_subject_npy')

Miss_length = 2
sepcturalSamples = 0.01
curTime = gmtime()
curRunDir = strftime("%a-%d-%b-%Y-%H_%M_%S+0000", curTime)

dataList = os.listdir(pairDir)
nameDevList = []
for dataFile in dataList:
	if dataFile[0] == '.':
		continue
	if dataFile[0] == '#':
		continue
	elems = dataFile.split('-')
	curLable = '-'.join(elems[:-1])
	if curLable not in nameDevList:
		nameDevList.append(curLable)

gtType = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
idxList = range(len(gtType))
gtIdxDict = dict(zip(gtType, idxList))
idxGtDict = dict(zip(idxList, gtType))

subjects    = ["a","b","c","d","e","f","g","h","i"]
id_sub_List = range(len(subjects))
subIdxDict  = dict(zip(subjects, id_sub_List))

subjectDataList = [[],[],[],[],[],[],[],[],[]]
for nameDev in nameDevList:
    curDevList = []
    for curGt in gtType:
        # curDevGtList = []
        curType = gtIdxDict[curGt]
        if os.path.exists(os.path.join(pairDir, nameDev+'-'+curGt+'.npy')):
            curMat = np.load(os.path.join(pairDir, nameDev+'-'+curGt+'.npy'))
            curMat[:,0] = curMat[:,0] / 10
            # current fragment start time
            cur_frag_start = 0
            for row_id in range(curMat.shape[0]):
                # get current fragment, interpolate and normalize the fragment, and add it to curDevList
                if row_id == (curMat.shape[0]-1) or (curMat[row_id+1, 0] - curMat[row_id, 0]) >= Miss_length:
                    # get current fragment
                    curFrag = curMat[cur_frag_start:(row_id+1),:]
                    # interpolate and normalize
                    if curFrag.shape[0] == 1:
                        cur_frag_start = row_id+1
                    else:
                        curFrag = Interpolation_normalization(curFrag, sepcturalSamples)
                        # add the fragment to curDevList
                        curDevList.append(curFrag)
                        # update start time
                        cur_frag_start = row_id+1
            print(nameDev+'_'+curGt+' over!')
    sub_id = subIdxDict[nameDev[0]]
    subjectDataList[sub_id] = subjectDataList[sub_id] + curDevList

save_npy_dir_flag = create_directory(save_npy_dir)
for (sub_id,subject) in enumerate(subjects):
    per_sub_data_save_dir = os.path.join(save_npy_dir, subject+'_HHARdata.npy')
    np.save(per_sub_data_save_dir,[subjectDataList[sub_id]])