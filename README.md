Permutation based clusterwise FDR correction
===================

This repository houses data, code, results, and manuscripts for preliminary investigations into investigating a permutation framework for clusterwise false discovery rate correction for fMRI data. 

###Data
This folder houses single subject and group level results from 4 OpenfMRI.org datasets<sup>[1](#1)</sup>. These are the results as reported in Eklund et. al.<sup>[2](#2)</sup> and were provided to us by Anders Eklund.

###Scripts
The scripts used to generate our permutations and test statistic. For details about the method, see the Letter and Methods folders.

 1. Perm_clusterP.py - This script generates permutations of each first level contrast by sign randomly sign flipping the original design matrix and calculating the number of significant clusters present in each permutation for multiple cluster defining thresholds.
 2. CombinePMF.py - This script estimates our final probability mass function by combining across the normalized histograms of each permutation. Permutations in which no clusters were observed assign all of their value to 0.
 3. FDR_correct.py - This script gathers the actual cluster extents and corrected pRFT-FWE values for all observed clusters from the group level Z images. It then compares each cluster extent to the PMF calculated above to determine an uncorrected p-value based on how often we observed a cluster of that size in the permutations. Finally, these uncorrected p-values are submitted to false discovery rate correction to obtain a corrected pFDR. We then plot the pRFT-FWE values against the pFDR values for two cluster defining thresholds.
 4. slab.py - This script houses a number of helper functions used elsewhere in the code.
 5. IPython notebook/Binder - We have set up a binder that accesses the IPython notebook in Scripts so that you can interact with the code that calculates the uncorrected p-values, FDR corrects them, and plots them. You can access it here [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/mangstad/fdr_permutations)

###Results
This houses saved results for each contrast. The msgpack python module was used for saving results. See the slab.SavePermResults calls in the above scripts for the order of variables saved. Additionally, output from FSL's cluster function is housed in the stdout.nipype files. If the LoadResults variable is set in FDR_correct.py it will load these files directly rather than trying to use FSL.

###Letter
LaTeX and PDF versions of our response to Eklund, et al.<sup>[2](#2)</sup> Submitted to arXiv on 8/3/2016.

###Methods
LaTeX and PDF versions of our extended methods from the above letter, going into more detail about our analysis.

###References
<a name="1">1</a>: https://openfmri.org/dataset/ds000003/<br/>
https://openfmri.org/dataset/ds000005/<br/>
https://openfmri.org/dataset/ds000006/<br/>
https://openfmri.org/dataset/ds000107/<br/>

<a name="2">2</a>: Eklund, A., Nichols, T. E., & Knutsson, H. (2016). Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. Proceedings of the National Academy of Sciences, 113(28), 7900–7905. http://doi.org/10.1073/pnas.1602413113

