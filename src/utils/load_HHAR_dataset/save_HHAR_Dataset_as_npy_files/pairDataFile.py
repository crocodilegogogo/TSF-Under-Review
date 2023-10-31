import os
import sys
from copy import deepcopy

dataDir = 'Dataset_Alg_SourceDevice'
fileList = os.listdir(dataDir)
timeLabel = 'Creation_Time'
pairSaveDir = 'Dataset_AccGry_SourceDevice'+'-'+timeLabel+'-avgTime'+'_phone'

dotList = []
for fileName in fileList:
	if fileName[0] == '.':
		dotList.append(fileName)

for elem in dotList:
	fileList.remove(elem)

# Combine acc and gyro, put them in the pairList
pairList = []
singleList = []
while len(fileList) > 1:
	curA = fileList[0]
	curPair = [curA]
	nameElem = curA.split('-')
	curLabel = '-'.join(nameElem[:-1])
	for idx in range(1,len(fileList)):
		if curLabel in fileList[idx]:
			curPair.append(fileList[idx])
	if len(curPair) == 2:
		pairList.append(deepcopy(curPair))
		for pairElem in curPair:
			fileList.remove(pairElem)
	elif len(curPair) == 1:
		# print curLabel
		fileList = fileList[1:]
		singleList.append(curPair[0])
	else:
		print(curLabel)
		print('Wrong', len(curPair), curPair)
		fileList = fileList[1:]
		fileList.append(curPair[0])

timeScale = 100000000.
if not os.path.exists(pairSaveDir):
	os.mkdir(pairSaveDir)

for pairElem in pairList:
	for fileElem in pairElem:
		if 'accelerometer' in fileElem:
			pair1 = fileElem
		else:
			pair2 = fileElem
	nameElem = pair1.split('-')
	curLabel = '-'.join(nameElem[:-1])
	# Read a generated novel file (acc file), put each line of the file into pairFile1
	pairFile1 = []
	print(pair1)
	fileIn1 = open(os.path.join(dataDir, pair1))
	line = fileIn1.readline()
	count = 0
	while len(line) > 0:
		curElem = eval(line)
		curElem['Index'] = eval(curElem['Index'])
		curElem['Creation_Time'] = eval(curElem['Creation_Time'])
		curElem['Arrival_Time'] = eval(curElem['Arrival_Time'])
		curElem['x'] = eval(curElem['x'])
		curElem['y'] = eval(curElem['y'])
		curElem['z'] = eval(curElem['z'])
		pairFile1.append(deepcopy(curElem))
		line = fileIn1.readline()
		count += 1
		print('\r', count,)
		sys.stdout.flush()
	print('')
	fileIn1.close()
	# Read a generated novel file (gyro file), put each line of the file into pairFile2
	pairFile2 = []
	print(pair2)
	fileIn2 = open(os.path.join(dataDir, pair2))
	line = fileIn2.readline()
	count = 0
	while len(line) > 0:
		curElem = eval(line)
		curElem['Index'] = eval(curElem['Index'])
		curElem['Creation_Time'] = eval(curElem['Creation_Time'])
		curElem['Arrival_Time'] = eval(curElem['Arrival_Time'])
		curElem['x'] = eval(curElem['x'])
		curElem['y'] = eval(curElem['y'])
		curElem['z'] = eval(curElem['z'])
		pairFile2.append(deepcopy(curElem))
		line = fileIn2.readline()
		count += 1
		print('\r', count,)
		sys.stdout.flush()
	print('')
	fileIn2.close()

# write the data into pairSaveDir line by line
	idx1 = 0
	idx2 = 0
	fileOut = open(os.path.join(pairSaveDir, curLabel),'w')
	print('Write', curLabel)
	count = 0
	while idx1 < len(pairFile1) and idx2 < len(pairFile2): # process the data in pairFile1 and pairFile2 line by line
		curItem1 = pairFile1[idx1]
		curItem2 = pairFile2[idx2]
		curTime1 = curItem1[timeLabel]/timeScale
		curTime2 = curItem2[timeLabel]/timeScale
		if abs(curTime1 - curTime2) < 0.1:
			curSaveElem = {}
			curSaveElem['Time'] = 0.5*(curTime1 + curTime2)
			curSaveElem['Accelerometer'] = deepcopy([curItem1['x'], curItem1['y'], curItem1['z']])
			curSaveElem['Gyroscope'] = deepcopy([curItem2['x'], curItem2['y'], curItem2['z']])
			fileOut.write(str(curSaveElem)+'\n')
			idx1 += 1
			idx2 += 1
			count += 1
			print('\r', count,)
			sys.stdout.flush()
		elif curTime1 - curTime2 >= 0.1:
			idx2 += 1
		else:
			idx1 += 1
	print('')
	fileOut.close()