#!/usr/bin/python

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

zthreshes = [3.1]
Tasks = ['RhymeJudgment']
Contrasts = [[1]]

Exp = '/net/pepper/Eklund/temp'
OutputFolder1 = '/net/pepper/Eklund/temp/FDR_perms/'
OutputFolder2 = 'perms_py_'
StatsFolder1 = 'Group/'
StatsFolder2 = 'stats'

for iTask in xrange(0,len(Tasks)):
    for iContrast in xrange(0,len(Contrasts[iTask])):
        for iThresh in xrange(0,len(zthreshes)):
            Task = Tasks[iTask]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(iContrast+1),OutputFolder2+str(zthresh))
            print(OutputPath)

            StatsPath = os.path.join(Exp,Task,StatsFolder1,'cope'+str(iContrast+1)+'.feat',StatsFolder2)
            print(StatsPath)

            #read smoothness file parse for DLH value
            smoothdata = np.genfromtxt(os.path.join(StatsPath,'smoothness'))
            dlh = smoothdata[0,1]
            vol = int(smoothdata[1,1])
            
            cwd = os.getcwd()
            os.chdir(StatsPath)
            cl = Cluster()
            cl.inputs.threshold = zthresh
            cl.inputs.in_file = os.path.join(StatsPath,'zstat1.nii.gz')
            cl.inputs.dlh = dlh
            cl.inputs.volume = vol
            cl.inputs.pthreshold = 1
            cl.inputs.terminal_output = 'file'
            cl.run()
            clusterdata = np.genfromtxt(os.path.join(StatsPath,'stdout.nipype'),skip_header=1)
            emp_c = clusterdata[:,1]
            fwe_p = clusterdata[:,2]

            Clusters = slab.LoadPermResults(OutputPath,1)[1]
            os.chdir(cwd)

            emp_p = np.zeros(emp_c.shape)
            
            for i in xrange(0,len(emp_c)):
                emp_p[i] = 1 - float(sum(emp_c[i] > Clusters))/ len(Clusters)
            
            #print(emp_c)
            #print(emp_p)
            #print(fwe_p)
            
            h, fdr_p = fdr_correction(emp_p,method='indep')
            print(fdr_p)

            #[h, crit, adj] = fdr_bh(emp_p,.05,'pdep','no',1);
            #sum(h);
            #save(fullfile(OutputPath,'fdr.mat'),'h','crit','adj','emp_c','emp_p','Clusters','-v7.3');
