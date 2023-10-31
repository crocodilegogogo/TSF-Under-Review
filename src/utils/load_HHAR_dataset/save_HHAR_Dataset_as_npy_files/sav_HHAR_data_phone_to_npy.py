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

timeLabel = 'Creation_Time'
pairDir = 'Dataset_AccGry_SourceDevice'+'-'+timeLabel+'-avgTime'+'_phone'
save_mat_dir = 'Per_subject_device_gt_npy'

save_mat_dir_flag = create_directory(save_mat_dir)

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
# print nameDevList


dataDict = {}
gtType = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
idxList = range(len(gtType))
gtIdxDict = dict(zip(gtType, idxList))
idxGtDict = dict(zip(idxList, gtType))

for nameDev in nameDevList:
    for curGt in gtType:
        curType = gtIdxDict[curGt]
        curData_Sep = []
        if os.path.exists(os.path.join(pairDir, nameDev+'-'+curGt)):
            fileIn = open(os.path.join(pairDir, nameDev+'-'+curGt))
            line = fileIn.readline()
            count = 0
            print(nameDev+'-'+curGt)
            while len(line) > 0:
                curElem = eval(line)
                curTime = curElem['Time']
                curAcc = curElem['Accelerometer']
                curGyro = curElem['Gyroscope']
                if count == 0:
                    cur_line = np.array([curTime]+curAcc+curGyro+[gtIdxDict[curGt]])
                    cur_line = cur_line.reshape(1,len(cur_line))
                    cur_Gt_save_mat = cur_line
                else:
                    cur_line = np.array([curTime]+curAcc+curGyro+[gtIdxDict[curGt]])
                    cur_line = cur_line.reshape(1,len(cur_line))
                    cur_Gt_save_mat = np.concatenate((cur_Gt_save_mat, cur_line), axis = 0)
                count = count + 1
                print(nameDev+'-'+curGt+':'+str(count))
                line = fileIn.readline()
            # close fileIn
            fileIn = fileIn.close()
            # sort the mat according to the timestamp
            cur_Gt_save_mat = cur_Gt_save_mat[cur_Gt_save_mat[:,0].argsort()] # sorting
            cur_Gt_save_mat[:,1:7] = (cur_Gt_save_mat[:,1:7]-np.mean(cur_Gt_save_mat[:,1:7],axis=0))/np.std(cur_Gt_save_mat[:,1:7],axis=0)
            np.save(save_mat_dir+'\\'+nameDev+'-'+curGt+'.npy', cur_Gt_save_mat)
            print(nameDev+'-'+curGt+'.npy'+' has been saved!')