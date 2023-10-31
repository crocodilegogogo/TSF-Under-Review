"""HHAR dataset can be Downloaded at https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition+Data+Set"""
"""Put the downloaded dataset into the current folder"""

import sys
import os
import numpy as np 

dataDir = 'Activity recognition exp'

fileInList = os.listdir(dataDir)

saveDir = 'Dataset_Alg_SourceDevice'
if not os.path.exists(saveDir):
	os.mkdir(saveDir)

for fileName in fileInList:
	if not '.csv' in fileName:
		continue

	fileIn = open(os.path.join(dataDir, fileName))
	line = fileIn.readline()
	print(fileName)
	headers = line[:-1].split(',')
	line = fileIn.readline()
	count = 0

	fileLable = fileName.split('.')[0]
	while len(line) > 0:
		count += 1
		elems = line[:-1].split(',')
		curDict = dict(zip(headers, elems))
		cur_User = curDict['User']
		cur_Device = curDict['Device']
		cur_gt = curDict['gt']

		subFileLable = '-'.join([cur_User, cur_Device, cur_gt, fileLable])
		fileOut = open(os.path.join(saveDir, subFileLable), 'a')
		fileOut.write(str(curDict)+'\n')
		fileOut.close()

		line = fileIn.readline()
		count += 1
		print('\r', count,)
		sys.stdout.flush()
	print('')


