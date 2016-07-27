%
% Eklund, A., Nichols, T. E., & Knutsson, H. (2016). Cluster failure: Why 
% fMRI inferences for spatial extent have inflated false-positive rates. 
% Proceedings of the National Academy of Sciences, 113(28), 7900-7905.
% http://doi.org/10.1073/pnas.1602413113
%

%% setting variables
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

idx.con = [1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 5 6  1 2 3 4 1 2 3 4  1 2 3 4 1  2 3  4 5  6];
idx.tas = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 4 4  1 1 1 1 2 2 2 2  3 3 3 3 4  4 4  4 4  4];
idx.thr = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  2 2 2 2 2 2 2 2  2 2 2 2 2  2 2  2 2  2];
eklsums = [1 3 2 0 2 0 0 1 5 3 2 0 6 9 2 4 7 2 10 8 0 0 2 0 0 0 10 1 0 0 9 12 8 11 6 12];
eklsumsr= [1 3 2 0 2     1 5 3 2   6 9 2 4 7 2 10 8 0 0 2     0 10 1 0   9 12 8 11 6 12];

Exp = '/net/pepper/Eklund/FDR_perms/';

ResultsFolder = 'perms_'; %folder to load permutation results from

output = [];
eklsumscounter = 1;

for iThresh = 1:numel(zthreshes)
    for iTask = 1:numel(Tasks)
        for iContrast = Contrasts{iTask}
            %% set up current contrast values
            Task = Tasks{iTask};
            sNum = sprintf('%d',iContrast);
            zthresh = zthreshes(iThresh);
            sThresh = sprintf('%2.1f',zthresh);
            
            fprintf(1,'Task: %s, Contrast: %d, Threshold: %2.1f\n',Task,iContrast,zthresh);

            OutputPath = [Exp Task '/contrast' sNum '/' ResultsFolder sThresh];
            ContrastPath = [Exp Task '/contrast' sNum];

            %% grab smoothness values, volumes, and cluster sizes/p-values
            cmd = sprintf('cat %s',fullfile(ContrastPath,'smoothness'));
            [status, result] = system(cmd);
            dlh = str2num(result);
            cmd = sprintf('cat %s',fullfile(ContrastPath,'volume'));
            [status, result] = system(cmd);
            vol = str2num(result);
            
            cmd = sprintf('cluster -i %s -t %2.1f | awk ''{print $2}'' | tail -n +2',fullfile(ContrastPath,'zstat1.nii'),zthresh);
            [status, result] = system(cmd);
            emp_c = str2num(result);

            cmd = sprintf('cluster -i %s -t %2.1f -p 1000 --dlh=%.10f --volume=%d | awk ''{print $3}'' | tail -n +2',fullfile(ContrastPath,'zstat1.nii'),zthresh,dlh,vol);
            [status, result] = system(cmd);
            rft_fwe = str2num(result);
            
            if (size(emp_c,1)~=size(rft_fwe,1))
                fprintf(1,'error\n');
            end
            
            %% calculate empirical p-values based on observed null distribution of clusters and FDR correct
            emp_p = zeros(size(emp_c));
            
            load(fullfile(OutputPath,'perms.mat'),'Clusters');

            for (i=1:size(emp_c,1))
                emp_p(i) = 1 - sum(emp_c(i) > Clusters) ./ size(Clusters,1);
            end
            
            [h, crit, adj] = fdr_bh(emp_p,.05,'pdep','no',1);
            sum(h);
            save(fullfile(OutputPath,'fdr.mat'),'h','crit','adj','emp_c','emp_p','Clusters','-v7.3');

            %% calculate number of pFWE values meeting FDR 05 correction at various FWE thresholds
            p00001 = [sum(rft_fwe<0.00001) sum(rft_fwe<0.00001)-sum(h(rft_fwe<0.00001))];
            p00005 = [sum(rft_fwe<0.00005) sum(rft_fwe<0.00005)-sum(h(rft_fwe<0.00005))];
            p0001 = [sum(rft_fwe<0.0001) sum(rft_fwe<0.0001)-sum(h(rft_fwe<0.0001))];
            p0005 = [sum(rft_fwe<0.0005) sum(rft_fwe<0.0005)-sum(h(rft_fwe<0.0005))];
            p001 = [sum(rft_fwe<0.001) sum(rft_fwe<0.001)-sum(h(rft_fwe<0.001))];
            p005 = [sum(rft_fwe<0.005) sum(rft_fwe<0.005)-sum(h(rft_fwe<0.005))];
            p01 = [sum(rft_fwe<0.01) sum(rft_fwe<0.01)-sum(h(rft_fwe<0.01))];
            p05 = [sum(rft_fwe<0.05) sum(rft_fwe<0.05)-sum(h(rft_fwe<0.05))];
            
            output = [output;iTask iContrast zthresh sum(h) eklsumsr(eklsumscounter) p00001 p00005 p0001 p0005 p001 p005 p01 p05];
            eklsumscounter = eklsumscounter + 1;            
        end
    end
end

%% collect percent of FWE p-values that survived FDR 05 at each CDT
cdt01 = [
sum(output(1:15,7))/sum(output(1:15,6))
sum(output(1:15,9))/sum(output(1:15,8))
sum(output(1:15,11))/sum(output(1:15,10))
sum(output(1:15,13))/sum(output(1:15,12))
sum(output(1:15,15))/sum(output(1:15,14))
sum(output(1:15,17))/sum(output(1:15,16))
sum(output(1:15,19))/sum(output(1:15,18))
sum(output(1:15,21))/sum(output(1:15,20))]';

cdt001 = [
sum(output(16:30,7))/sum(output(16:30,6))
sum(output(16:30,9))/sum(output(16:30,8))
sum(output(16:30,11))/sum(output(16:30,10))
sum(output(16:30,13))/sum(output(16:30,12))
sum(output(16:30,15))/sum(output(16:30,14))
sum(output(16:30,17))/sum(output(16:30,16))
sum(output(16:30,19))/sum(output(16:30,18))
sum(output(16:30,21))/sum(output(16:30,20))]';

%% plot percent of pFWE values that are rejected at pFDR 0.05 for both CDT thresholds
x = -log10([0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05]);
figure;
plot(x,cdt01,'r-');
hold on;
plot(x,cdt001,'b-');
title('Plot of FDR survival by FWE p-value for CDT 0.01 and 0.001');
xlabel('-Log10(FWE p-value)');
ylabel('Percentage of FWE results that survive FDR 0.05');
hold off;
