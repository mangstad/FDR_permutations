%SPM needs to be on the path
%these results were generated using SPM8 (R4667)

%
% Eklund, A., Nichols, T. E., & Knutsson, H. (2016). Cluster failure: Why 
% fMRI inferences for spatial extent have inflated false-positive rates. 
% Proceedings of the National Academy of Sciences, 113(28), 7900-7905.
% http://doi.org/10.1073/pnas.1602413113
%

%if you do not have access to the parallel computing toolbox, make the
%following changes:
%
%
%
%

%% setting variables
p = 5000; %number of permutations per contrast
cores = 6; %number of cores to run if using parallel computing toolbox
seed = 1234; %random number seed

zthreshes = [2.3 3.1]; %Z thresholds to evaluate

Tasks = {
    'RhymeJudgment';
    'MixedGamblesTask';
    'LivingNonliving';
    'WordObject';
    };

Contrasts = {
    [1 2 3 4];
    [1 4];
    [1 2 3];
    [1 2 3 4 5 6];
    };

Exp = '/net/pepper/Eklund/FDR_perms/';

ResultsFolder = 'perms_3_'; %output folder for permutation results

LoadResults = 1; %load existing perms.mat file to use existing designs if possible



%% start calculation
rng(seed); %initialize RNG with seed for reproducible results

for iTask = 1:numel(Tasks)
    for iContrast = Contrasts{iTask}
        for iThresh = 1:numel(zthreshes)
            %% 
            clear permdata permstats data dataflat mask maskflat maskeddataflat Design;
            clear PermClusters Clusters;
            
            %% set up current contrast values
            Task = Tasks{iTask};
            sNum = sprintf('%d',iContrast);
            zthresh = zthreshes(iThresh);
            sThresh = sprintf('%2.1f',zthresh);
            
            fprintf(1,'Task: %s, Contrast: %d, Threshold: %2.1f, Permutations: %d\n',Task,iContrast,zthresh,p);

            OutputPath = [Exp Task '/contrast' sNum '/' ResultsFolder sThresh];
            mkdir(OutputPath);
            
            tic;
            
            %% load images
            InputPath = [Exp Task '/contrast' sNum];
            datafile = spm_select('FPList',InputPath,'.*_0.*\.nii');
            Vf = spm_vol(datafile);
            data = spm_read_vols(Vf);
            

            % reshape and mask data
            [dx dy dz n] = size(data);
            dataflat = reshape(data,dx*dy*dz,n)';
            
            maskfile = spm_select('FPList',InputPath,'.*mask.*\.nii');
            Vm = spm_vol(maskfile);
            mask = spm_read_vols(Vm);
            maskflat = reshape(mask,dx*dy*dz,1)';
            
            maskeddataflat = dataflat(:,logical(maskflat));
            
            %% calculate T threshold based on chosen Z threshold
            tthresh = icdf('t',cdf('norm',zthresh),n-1);
            Design = [ones(n,1)];
            
            %% load or generate permuted design matrices (via sign flipping)

            if (LoadResults && exist(fullfile(OutputPath,'perms.mat'),'file'))
                load(fullfile(OutputPath,'perms.mat'),'PermDesign');
            else
                PermDesign = sign(rand(n,p)-0.5);
            end
            
            
            %% run permutations
            PermClusters = cell(p,1);
            Clusters = [];
            
            %comment these three lines out if not using parallel computing
            %toolbox
            if (matlabpool('size') == 0)
                matlabpool('open',cores);
            end
            %change this to for i=1:p if not using parallel computing
            %toolbox
            parfor i = 1:p
                permstats(i) = mc_glm(maskeddataflat,PermDesign(:,i));
                permstats(i).b = [];
                permstats(i).res = [];
                permstats(i).pred = [];
                
                permflat = zeros(1,dx*dy*dz);
                permflat(logical(maskflat)) = permstats(i).t;
                permdata = reshape(permflat,dx,dy,dz);
                %uses SPM code to calculate connected clusters, but uses
                %FSL's default cluster connectivity of 26 rather than SPM's
                %default of 18 (face/edge/corner rather than just
                %face/edge)
                [cci num] = spm_bwlabel(double(permdata>tthresh),26);
                clust = sort(crosstab(cci(cci>0)));
                PermClusters{i} = clust;
            end
            
            %comment the next line out if not using parallel computing
            %toolbox
            matlabpool close
            
            %% gather/sort all discovered clusters across permutations and save
            Clusters = sort(vertcat(PermClusters{:}));
            
            save(fullfile(OutputPath,'perms.mat'),'PermClusters','Clusters','PermDesign','zthresh','tthresh','n','p','dx','dy','dz','maskflat','-v7.3');
            
            toc
            
        end
    end
end
