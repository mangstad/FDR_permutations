#!/usr/bin/python

import os
import pickle
import numpy as np
import scipy as sp
import nipype as npy
from nipype.interfaces.fsl.model import Cluster

zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]

zthreshes = [3.1]
Tasks = ['RhymeJudgment']
Contrasts = [[1]]

Exp = '/net/pepper/Eklund/'
OutputFolder1 = '/net/pepper/Eklund/FDR_perms/'
OutputFolder2 = 'perms_py_'
StatsFolder1 = 'GroupAnalyses/'
StatsFolder2 = 'stats'

for iTask in xrange(0,len(Tasks)):
    for iContrast in xrange(0,len(Contrasts[iTask])):
        for iThresh in xrange(0,len(zthreshes)):
            Task = Tasks[iTask]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(iContrast+1),OutputFolder2+str(zthresh))
            print(OutputPath)

            StatsPath = os.path.join(Exp,Task,StatsFolder1,'OLS_'+str(zthresh)+'.gfeat','cope'+str(iContrast+1)+'.feat',StatsFolder2)
            print(StatsPath)

            #read smoothness file parse for DLH value
            smoothdata = np.genfromtxt(os.path.join(StatsPath,'smoothness'))
            dlh = smoothdata[0,1]
            vol = int(smoothdata[1,1])
            
            cl = Cluster()
            cl.inputs.threshold = zthresh
            cl.inputs.in_file = os.path.join(StatsPath,'zstat1.nii.gz')
            cl.inputs.dlh = dlh
            cl.inputs.volume = vol
            cl.inputs.pthreshold = 1
            cl.inputs.terminal_output = 'file'
            cl.run()
            clusterdata = np.genfromtext(os.path.join(StatsPath,'stdout.nipype'),skip_header=1)
            clustersizes = clusterdata[:,1]
            clusterfwep = clusterdata[:,2]
