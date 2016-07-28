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

#zthreshes = [3.1]
#Tasks = ['RhymeJudgment']
#Contrasts = [[1]]

Exp = '/net/pepper/Eklund/temp'
OutputFolder1 = '/net/pepper/Eklund/temp/FDR_perms/'
OutputFolder2 = 'perms_py_'
StatsFolder1 = 'Group/'
StatsFolder2 = 'stats'

output = []
eklsumscounter = 0
eklsumsr= [1,3,2,0,2,1,5,3,2,6,9,2,4,7,2,10,8,0,0,2,0,10,1,0,9,12,8,11,6,12];

for iThresh in xrange(0,len(zthreshes)):
    for iTask in xrange(0,len(Tasks)):
        for iContrast in xrange(0,len(Contrasts[iTask])):
            Task = Tasks[iTask]
            Contrast = Contrasts[iTask][iContrast]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(Contrast),OutputFolder2+str(zthresh))
            print(OutputPath)

            StatsPath = os.path.join(Exp,Task,StatsFolder1,'cope'+str(Contrast)+'.feat',StatsFolder2)
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
            cl.inputs.pthreshold = 1000
            cl.inputs.terminal_output = 'file'
            c = cl.run()
            clusterdata = np.genfromtxt(os.path.join(StatsPath,'stdout.nipype'),skip_header=1)
            emp_c = clusterdata[:,1]
            fwe_p = clusterdata[:,2]

            Clusters = slab.LoadPermResults(OutputPath,'perms','msgpack',1)[1]
            os.chdir(cwd)

            emp_p = np.zeros(emp_c.shape)
            
            #sum was super slow here, replaced with len(np.nonzero...
            for i in xrange(0,len(emp_c)):
                #emp_p[i] = 1 - float(sum(emp_c[i] > Clusters))/ len(Clusters)
                emp_p[i] = 1 - float(len(np.nonzero(emp_c[i]>Clusters)[0]))/len(Clusters)
            
            #print(emp_c)
            #print(emp_p)
            #print(fwe_p)
            
            h, fdr_p = fdr_correction(emp_p,method='indep')
            #print(fdr_p)

            slab.SavePermResults(OutputPath,'fdr','msgpack',h.tolist(),fdr_p.tolist(),emp_c.tolist(),emp_p.tolist(),Clusters)

            p00001 = [sum(fwe_p<0.00001),sum(fwe_p<0.00001)-sum(h[fwe_p<0.00001])]
            p00005 = [sum(fwe_p<0.00005),sum(fwe_p<0.00005)-sum(h[fwe_p<0.00005])]
            p0001 = [sum(fwe_p<0.0001),sum(fwe_p<0.0001)-sum(h[fwe_p<0.0001])]
            p0005 = [sum(fwe_p<0.0005),sum(fwe_p<0.0005)-sum(h[fwe_p<0.0005])]
            p001 = [sum(fwe_p<0.001),sum(fwe_p<0.001)-sum(h[fwe_p<0.001])]
            p005 = [sum(fwe_p<0.005),sum(fwe_p<0.005)-sum(h[fwe_p<0.005])]
            p01 = [sum(fwe_p<0.01),sum(fwe_p<0.01)-sum(h[fwe_p<0.01])]
            p05 = [sum(fwe_p<0.05),sum(fwe_p<0.05)-sum(h[fwe_p<0.05])]

            pout = np.concatenate([p00001,p00005,p0001,p0005,p001,p005,p01,p05])
            if len(output)==0:
                output = np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])
            else:
                output = np.vstack([output,np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])])
            eklsumscounter += 1
            #print(output)

cdt01 = [sum(output[0:14,6])/sum(output[0:14,5]),
         sum(output[0:14,8])/sum(output[0:14,7]),
         sum(output[0:14,10])/sum(output[0:14,9]),
         sum(output[0:14,12])/sum(output[0:14,11]),
         sum(output[0:14,14])/sum(output[0:14,13]),
         sum(output[0:14,16])/sum(output[0:14,15]),
         sum(output[0:14,18])/sum(output[0:14,17]),
         sum(output[0:14,20])/sum(output[0:14,19])]

cdt001 = [sum(output[15:29,6])/sum(output[15:29,5]),
          sum(output[15:29,8])/sum(output[15:29,7]),
          sum(output[15:29,10])/sum(output[15:29,9]),
          sum(output[15:29,12])/sum(output[15:29,11]),
          sum(output[15:29,14])/sum(output[15:29,13]),
          sum(output[15:29,16])/sum(output[15:29,15]),
          sum(output[15:29,18])/sum(output[15:29,17]),
          sum(output[15:29,20])/sum(output[15:29,19])]

#%% plot percent of pFWE values that are rejected at pFDR 0.05 for both CDT thresholds
x = -1*np.log10([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05]);

import pandas as p
import ggplot
from ggplot import *

d = p.DataFrame({'x':x,'y1':cdt01,'y2':cdt001})
print ggplot(d,aes(x='x')) + geom_line(aes(y='y1'),colour='blue') + geom_line(aes(y='y2'),colour='red')

