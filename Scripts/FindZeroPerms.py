import os
import slab
import numpy as np
import scipy as sp
import nipype as npy
import mne
from mne.stats import fdr_correction
from nipype.interfaces.fsl.model import Cluster
zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]

Exp = '../Data/'
OutputFolder1 = '../Results/'
OutputFolder2 = 'perms_py_'
PermClusters = []
for iThresh in xrange(0,len(zthreshes)):
    for iTask in xrange(0,len(Tasks)):
        for iContrast in xrange(0,len(Contrasts[iTask])):
			Task = Tasks[iTask]
			Contrast = Contrasts[iTask][iContrast]
			zthresh = zthreshes[iThresh]
			OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(Contrast),OutputFolder2+str(zthresh))
			print('Working on '+OutputPath)
			PermClusters.append(slab.LoadPermResults(OutputPath,'perms','msgpack',0)[1])

sum = 0
for i in xrange(0,30):
    for j in xrange(0,5000):
        if len(PermClusters[i][j])==0:
            sum += 1
print(sum)
print(sum/float((30*5000)))
