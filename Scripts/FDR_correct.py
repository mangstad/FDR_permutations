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

            os.chdir(cwd)

            emp_p = np.zeros(emp_c.shape)
            
            pmf = slab.LoadPermResults(OutputPath,'PMF','msgpack',0)[1]

            #now using PMF calculated across perms for a given contrast
            for i in xrange(0,len(emp_c)):
                if (emp_c[i]>len(pmf)):
                    emp_p[i] = pmf[-1]/np.round(np.sum(pmf))
                else:
                    emp_p[i] = np.sum(pmf[int(emp_c[i]):])/np.round(np.sum(pmf))
            
            h, fdr_p = fdr_correction(emp_p,method='indep')

            slab.SavePermResults(OutputPath,'fdr','msgpack',h.tolist(),fdr_p.tolist(),emp_c.tolist(),emp_p.tolist())

            ps = []
            pbins = [0,0.00001,0.0001,0.001,0.01,0.05]
            for ip in xrange(0,len(pbins)-1):
                fwe = sum(np.logical_and(fwe_p>pbins[ip],fwe_p<=pbins[ip+1]))
                ps.append([fwe,sum(h[np.logical_and(fwe_p>pbins[ip],fwe_p<=pbins[ip+1])])])
                
            pout = np.concatenate(ps)
            if len(output)==0:
                output = np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])
            else:
                output = np.vstack([output,np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])])
            eklsumscounter += 1

cdt01 = np.array([sum(output[0:14,6])/sum(output[0:14,5]),
         sum(output[0:14,8])/sum(output[0:14,7]),
         sum(output[0:14,10])/sum(output[0:14,9]),
         sum(output[0:14,12])/sum(output[0:14,11]),
         sum(output[0:14,14])/sum(output[0:14,13])])

cdt001 = np.array([sum(output[15:29,6])/sum(output[15:29,5]),
          sum(output[15:29,8])/sum(output[15:29,7]),
          sum(output[15:29,10])/sum(output[15:29,9]),
          sum(output[15:29,12])/sum(output[15:29,11]),
          sum(output[15:29,14])/sum(output[15:29,13])])


import prettyplotlib as ppl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

x = np.arange(11)+1
width = 0.70

xticks = ['(0,           \n0.00001]','(0.00001,\n0.0001]','(0.0001,\n0.001]','(0.001,\n0.01]','(0.01,\n0.05]']

cdt01sums = np.sum(output[0:15,5:15],axis=0)
cdt001sums = np.sum(output[15:30,5:15],axis=0)
cdt01sumsstr = ['%d' % n for n in cdt01sums]
cdt001sumsstr = ['%d' % n for n in cdt001sums]
cdt01annot = [i+'\n'+j+' clusters' for i,j in zip(xticks,cdt01sumsstr[::2])]
cdt001annot = [i+'\n'+j+' clusters' for i,j in zip(xticks,cdt001sumsstr[::2])]

xlabels = np.concatenate([cdt001annot,np.array(['']),cdt01annot])

#plt.xkcd()

fontAxis = FontProperties()
fontAxis.set_family('sans-serif')
fontAxis.set_weight('bold')
fontLabel = fontAxis.copy()
fontTitle = fontAxis.copy()

fontTitle.set_size(32)
fontAxis.set_size(24)
fontLabel.set_size(14)
  
dpi = 96
plt.figure(figsize=(1920/dpi,1080/dpi),dpi=dpi,facecolor='w')
ax = plt.gca()
ax.spines["top"].set_visible(False)     
ax.spines["right"].set_visible(False)  
plt.bar(x,100*np.concatenate([cdt001,np.array([0]),cdt01]),width,color=np.concatenate([np.repeat(['#76bf72'],5),np.repeat(['b'],1),np.repeat(['#597dbe'],5)]))
plt.ylabel('% of Clusters\nSurviving at FDR 0.05',fontproperties=fontAxis)
plt.xlabel('FWE Corrected P-Value Bins',fontproperties=fontAxis)
plt.xticks(x + width/2,xlabels,fontproperties=fontLabel)
plt.yticks(np.arange(0,105,10),fontproperties=fontLabel)
plt.tick_params(axis="both",which="both",bottom="off",top="off",labelbottom="on",left="on",right="off",labelleft="on")    
plt.ylim([0,105])
plt.xlim([0.55,12])
plt.figtext(0.27,0.91,'CDT 0.001',color='#76bf72',fontproperties=fontTitle)
plt.figtext(0.68,0.91,'CDT 0.01',color='#597dbe',fontproperties=fontTitle)
plt.savefig('foo.png', bbox_inches='tight',dpi=dpi,orientation='portrait',papertype='letter')
plt.show()
