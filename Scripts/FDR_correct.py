#!/usr/bin/python

import os
import slab
import numpy as np
import nipype as npy
import mne
from mne.stats import fdr_correction
from nipype.interfaces.fsl.model import Cluster

zthreshes = [2.3,3.1]
Tasks = ['RhymeJudgment','MixedGamblesTask','LivingNonliving','WordObject']
Contrasts = [[1,2,3,4],[1,4],[1,2,3],[1,2,3,4,5,6]]

Exp = '../Data/'
OutputFolder1 = '../Results/'
OutputFolder2 = 'perms_py_1980_'
StatsFolder1 = 'Group/'
StatsFolder2 = 'stats'

output = []
eklsumscounter = 0
eklsumsr= [1,3,2,0,2,1,5,3,2,6,9,2,4,7,2,10,8,0,0,2,0,10,1,0,9,12,8,11,6,12];

all_fdr=[]
all_fwe=[]

for iThresh in xrange(0,len(zthreshes)):
    for iTask in xrange(0,len(Tasks)):
        for iContrast in xrange(0,len(Contrasts[iTask])):
            Task = Tasks[iTask]
            Contrast = Contrasts[iTask][iContrast]
            zthresh = zthreshes[iThresh]
            OutputPath = os.path.join(OutputFolder1,Task,'contrast'+str(Contrast),OutputFolder2+str(zthresh))
            #print(OutputPath)

            StatsPath = os.path.join(Exp,Task,StatsFolder1,'cope'+str(Contrast)+'.feat',StatsFolder2)
            print('Now working on '+StatsPath)

            #read smoothness file parse for DLH value and volume
            smoothdata = np.genfromtxt(os.path.join(StatsPath,'smoothness'))
            dlh = smoothdata[0,1]
            vol = int(smoothdata[1,1])
            
            #cd to directory and run FSL cluster on image to get cluster
            #exents and FWE p-values
            cwd = os.getcwd()
            os.chdir(OutputPath)
            cl = Cluster()
            cl.inputs.threshold = zthresh
            cl.inputs.in_file = os.path.join(cwd,StatsPath,'zstat1.nii.gz')
            cl.inputs.dlh = dlh
            cl.inputs.volume = vol
            cl.inputs.pthreshold = 1000
            cl.inputs.terminal_output = 'file'
            c = cl.run()
            clusterdata = np.genfromtxt(os.path.join(cwd,OutputPath,'stdout.nipype'),skip_header=1)
            #observed cluster sizes
            emp_c = clusterdata[:,1]
            #RFT FWE corrected p-values
            fwe_p = clusterdata[:,2]

            os.chdir(cwd)

            emp_p = np.zeros(emp_c.shape)
            
            #using PMF calculated across perms for a given contrast
            #calculated in CombinePMF.py
            pmf = slab.LoadPermResults(OutputPath,'PMF','msgpack',0)[1]
            for i in xrange(0,len(emp_c)):
                if (emp_c[i]>len(pmf)):
                    emp_p[i] = pmf[-1]/np.round(np.sum(pmf))
                else:
                    emp_p[i] = np.sum(pmf[int(emp_c[i]):])/np.round(np.sum(pmf))

            #FDR correct
            h, fdr_p = fdr_correction(emp_p,method='indep')

            slab.SavePermResults(OutputPath,'fdr','msgpack',h.tolist(),fdr_p.tolist(),emp_c.tolist(),emp_p.tolist())
            
            #messy output stuff
            #select p-value bins, and concatenate rows of FWE
            #clusters and which are FDR clusters
            ps = []
            pbins = [-0.1,0.00001,0.0001,0.001,0.01,0.05]
            for ip in xrange(0,len(pbins)-1):
                fwe = sum(np.logical_and(fwe_p>pbins[ip],fwe_p<=pbins[ip+1]))
                temp = sum(fwe_p<=0.05)
                ps.append([fwe,sum(h[np.logical_and(fwe_p>pbins[ip],fwe_p<=pbins[ip+1])])])

            all_fdr.append(fdr_p)
            all_fwe.append(fwe_p)
                
            pout = np.concatenate(ps)
            if len(output)==0:
                output = np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])
            else:
                output = np.vstack([output,np.concatenate([np.array([iTask+1,Contrast,zthresh,sum(h),eklsumsr[eklsumscounter]]),pout])])
            eklsumscounter += 1

#cluster survival ratios for CDT 0.01
cdt01 = np.array([sum(output[0:14,6])/sum(output[0:14,5]),
         sum(output[0:14,8])/sum(output[0:14,7]),
         sum(output[0:14,10])/sum(output[0:14,9]),
         sum(output[0:14,12])/sum(output[0:14,11]),
         sum(output[0:14,14])/sum(output[0:14,13])])

#cluster survival ratios for CDT 0.001
cdt001 = np.array([sum(output[15:29,6])/sum(output[15:29,5]),
          sum(output[15:29,8])/sum(output[15:29,7]),
          sum(output[15:29,10])/sum(output[15:29,9]),
          sum(output[15:29,12])/sum(output[15:29,11]),
          sum(output[15:29,14])/sum(output[15:29,13])])

#make a nice plot
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches


fontAxis = FontProperties()
fontAxis.set_family('sans-serif')
fontAxis.set_weight('bold')
fontLabel = fontAxis.copy()
fontTitle = fontAxis.copy()
fontInset = fontAxis.copy()

fontTitle.set_size(48)
fontAxis.set_size(38)
fontLabel.set_size(28)
fontInset.set_size(20)


cdt01fdr = np.array(list(slab.flatten(all_fdr[0:15])))
cdt01fwe = np.array(list(slab.flatten(all_fwe[0:15])))
cdt001fdr = np.array(list(slab.flatten(all_fdr[15:30])))
cdt001fwe = np.array(list(slab.flatten(all_fwe[15:30])))

cdt01fwe[cdt01fwe==0] = min(cdt01fwe[cdt01fwe>0])
cdt001fwe[cdt001fwe==0] = min(cdt001fwe[cdt001fwe>0])

cdt01fdr = cdt01fdr[cdt01fwe<0.05]
cdt01fwe = cdt01fwe[cdt01fwe<0.05]
cdt001fdr = cdt001fdr[cdt001fwe<0.05]
cdt001fwe = cdt001fwe[cdt001fwe<0.05]

def get_colors(inp,colormap,vmin=None,vmax=None):
	norm = plt.Normalize(vmin,vmax)
	return colormap(norm(inp))

passfdr01 = cdt01fdr<0.05
passfdr001 = cdt001fdr<0.05
color01 = get_colors(passfdr01,plt.cm.winter)
color001 = get_colors(passfdr001,plt.cm.winter)


fontAxis = FontProperties()
fontAxis.set_family('sans-serif')
fontAxis.set_weight('bold')
fontLabel = fontAxis.copy()
fontTitle = fontAxis.copy()
fontInset = fontAxis.copy()

fontTitle.set_size(48)
fontAxis.set_size(38)
fontLabel.set_size(28)
fontInset.set_size(20)

xticklabels = ['0','','10','','20','','30','','40','','50']

dpi = 96
plt.figure(figsize=(1920/dpi,2160/dpi),dpi=dpi,facecolor='w')

plt.subplot(211)
ax = plt.gca()
ax.spines["top"].set_visible(False)	 
ax.spines["right"].set_visible(False)
plt.setp(ax.spines.values(),linewidth=5)
ax.yaxis.set_tick_params(width=5,length=10)
ax.xaxis.set_tick_params(width=5,length=10)
pfdrtext1 = '$p_{_{FDR}}\leq0.05$'
pfdrtext2 = '$p_{_{FDR}}>0.05$'
ax.text(15,-1*np.log10(0.05)+0.3,r''+pfdrtext1,fontproperties=fontTitle,color='#597dbe')
ax.text(15,-1*np.log10(0.05)-0.5,r''+pfdrtext2,fontproperties=fontTitle,color='#fe7d59')
plt.scatter(-1*np.log10(cdt001fwe[cdt001fdr<0.05]),-1*np.log10(cdt001fdr[cdt001fdr<0.05]),marker='o',s=100,edgecolors='black',zorder=1,facecolors='#597dbe')
plt.scatter(-1*np.log10(cdt001fwe[cdt001fdr>0.05]),-1*np.log10(cdt001fdr[cdt001fdr>0.05]),marker='o',s=100,edgecolors='black',zorder=1,facecolors='#fe7d59')
ax.add_patch(patches.Rectangle((0,-1*np.log10(0.05)),50,5-(-1*np.log10(0.05)),edgecolor=None,facecolor='#597dbe',alpha=0.1))
ax.add_patch(patches.Rectangle((0,0),50,-1*np.log10(0.05),edgecolor=None,facecolor='#fe7d59',alpha=0.1))
plt.ylabel('Clusterwise FDR Corrected P-Value\n'+r'$-log_{10}(p_{_{FDR}})$'+'\n',fontproperties=fontAxis,horizontalalignment='center')
#plt.xlabel('Clusterwise RFT FWE\nCorrected P-Value\n'+r'$-log_{10}(p_{_{RFT-FWE}})$',fontproperties=fontAxis)
plt.xticks(np.arange(0,51,5),xticklabels,fontproperties=fontLabel)
plt.yticks(np.arange(0,6,1),fontproperties=fontLabel)
plt.tick_params(axis="both",which="both",bottom="on",top="off",labelbottom="on",left="on",right="off",labelleft="on")	
plt.ylim([0,5])
plt.xlim([0,50])
plt.title('CDT .001',fontproperties=fontTitle)
ax.annotate('smaller p-values',xy=(45,.25),xytext=(25,.25),arrowprops=dict(facecolor='black',shrink=0.1),fontproperties=fontLabel,verticalalignment='center')
ax.get_yaxis().set_label_coords(0,0)

plt.subplot(212)
ax = plt.gca()
ax.spines["top"].set_visible(False)	 
ax.spines["right"].set_visible(False)
plt.setp(ax.spines.values(),linewidth=5)
ax.yaxis.set_tick_params(width=5,length=10)
ax.xaxis.set_tick_params(width=5,length=10)
ax.annotate('p=.05',xy=(-1*np.log10(0.05),0),xytext=(-1*np.log10(0.05),-0.5),arrowprops=dict(facecolor='black',shrink=0.1),fontproperties=fontLabel,horizontalalignment='right')
ax.annotate('p=.001',xy=(-1*np.log10(0.001),0),xytext=(-1*np.log10(0.001),-0.7),arrowprops=dict(facecolor='black',shrink=0.1),fontproperties=fontLabel,horizontalalignment='right')
ax.annotate('p=.00001',xy=(-1*np.log10(0.00001),0),xytext=(-1*np.log10(0.00001),-0.9),arrowprops=dict(facecolor='black',shrink=0.1),fontproperties=fontLabel,horizontalalignment='right')
pfdrtext1 = '$p_{_{FDR}}\leq.05$'
pfdrtext2 = '$p_{_{FDR}}>.05$'
ax.text(15,-1*np.log10(0.05)+0.3,r''+pfdrtext1,fontproperties=fontTitle,color='#597dbe')
ax.text(15,-1*np.log10(0.05)-0.5,r''+pfdrtext2,fontproperties=fontTitle,color='#fe7d59')
plt.scatter(-1*np.log10(cdt01fwe[cdt01fdr<0.05]),-1*np.log10(cdt01fdr[cdt01fdr<0.05]),marker='o',s=100,edgecolors='black',zorder=1,facecolors='#597dbe')
plt.scatter(-1*np.log10(cdt01fwe[cdt01fdr>0.05]),-1*np.log10(cdt01fdr[cdt01fdr>0.05]),marker='o',s=100,edgecolors='black',zorder=1,facecolors='#fe7d59')
ax.add_patch(patches.Rectangle((0,-1*np.log10(0.05)),50,5-(-1*np.log10(0.05)),edgecolor=None,facecolor='#597dbe',alpha=0.1))
ax.add_patch(patches.Rectangle((0,0),50,-1*np.log10(0.05),edgecolor=None,facecolor='#fe7d59',alpha=0.1))
#plt.ylabel('Clusterwise FDR\nCorrected P-Value\n'+r'$-log_{10}(p_{_{FDR}})$',fontproperties=fontAxis)
plt.xlabel('Clusterwise RFT-FWE\nCorrected P-Value\n'+r'$-log_{10}(p_{_{RFT-FWE}})$',fontproperties=fontAxis)
plt.xticks(np.arange(0,51,5),xticklabels,fontproperties=fontLabel)
plt.yticks(np.arange(0,6,1),fontproperties=fontLabel)
plt.tick_params(axis="both",which="both",bottom="on",top="off",labelbottom="on",left="on",right="off",labelleft="on")	
plt.ylim([0,5])
plt.xlim([0,50])
plt.title('CDT .01',fontproperties=fontTitle)
ax.annotate('smaller p-values',xy=(45,.25),xytext=(25,.25),arrowprops=dict(facecolor='black',shrink=0.1),fontproperties=fontLabel,verticalalignment='center')


plt.savefig(os.path.join(OutputFolder1,'FDR_surviving_clusters.png'), bbox_inches='tight',dpi=dpi,orientation='portrait',papertype='letter')
