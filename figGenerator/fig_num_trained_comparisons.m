% This script is to generate  AUC and CI for training on different number
% of comparison labels.
% 
% Date: April 2018
% Author: Peng Tian

clear all;
close all;
clc;

%% Parameter Setting
CIParameter = 0.10;
inputFileName = '../../data/ropResult/num_Comparisons/MS_LogLog_L1global_manual_Cmp_';
isSaveFigure = 0; % 1 for save the pdf figure.
alphaValue = {'20','40','60','80','100','120','140','160','180','200','400','600','800','1000','2000','3000'};%0.0:0.1:1.0;
lambdaStr = {'10000.0' '1000.0' '100.0' '10.0' '9.0' '7.0' '5.0' '3.0' '1.0' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.01' '0.001' '0.0001' '1e-05' '1e-06' };

%% Prepare Label Data
lambdaValue = cellfun(@str2num,lambdaStr(1:end));
expLabelFile = load('../../../Data/ExpertsRank/SetOf100/ExpertsLabel13.mat');
labeldata = expLabelFile.ExpertsLabel13;
labelExp13 = reshape(labeldata,[],1); % 2-Plus, 1-Preplus, 0-Normal
indPlusExp13 = find(labelExp13==2);
indPrePExp13 = find(labelExp13==1);
indOutExp13 = find(labelExp13==1.5);
labelExp13Plus = zeros(size(labelExp13,1),size(labelExp13,2));
labelExp13Plus(indPlusExp13) = 1;
labelExp13PreP = labelExp13Plus;
labelExp13PreP(indPrePExp13) = 1;
labelExp13PreP(indOutExp13) = 1;

RSDlabelFile = load('../../../Data/ProbalisticModel/iROPData_6DD.mat');
labelRSD = RSDlabelFile.classLabels1st;
indPlusRSD = find(labelRSD==1);
indPrePRSD = find(labelRSD==2);
labelRSDPlus = zeros(size(labelRSD,1),size(labelRSD,2));
labelRSDPlus(indPlusRSD) = 1;
labelRSDPreP = labelRSDPlus;
labelRSDPreP(indPrePRSD) = 1;

cmpLabelFile = load('../../../Data/ExpertsRank/SetOf100/cmpDataLabel.mat');
cmpLabels =  cmpLabelFile.cmpDatalabels';
cmpLabels(cmpLabels==-1)=0;

parameters = cell(16,3);

numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2); 
abs2RSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2CmpPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2RSDPlusAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPlusAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);
abs2CmpPlusAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);

abs2RSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue); 
abs2CmpPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2RSDPrePAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPrePAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);
abs2CmpPrePAUCStd = zeros(numOfAlphaValue,numOfLambdaValue);
for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
            try 
            file = load([inputFileName, alphaValue{alphaInd}, '_0.0_',lambdaStr{lambdaInd},'.mat']);
            % For the plot in Plus vs Not Plus Image
            abs2RSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucRSDPlusAbs,2);
            abs2AbsPlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusAbs,2);
            abs2CmpPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusAbs,2);

            abs2RSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucRSDPrePAbs,2);
            abs2AbsPrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePAbs,2);
            abs2CmpPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePAbs,2);
            
            abs2RSDPlusAUCStd(alphaInd, lambdaInd) = std(file.aucRSDPlusAbs);
            abs2AbsPlusAUCStd(alphaInd, lambdaInd) = std(file.aucLblPlusAbs);
            abs2CmpPlusAUCStd(alphaInd, lambdaInd) = std(file.aucCmpPlusAbs);

            abs2RSDPrePAUCStd(alphaInd, lambdaInd) = std(file.aucRSDPrePAbs);
            abs2AbsPrePAUCStd(alphaInd, lambdaInd) = std(file.aucLblPrePAbs);
            abs2CmpPrePAUCStd(alphaInd, lambdaInd) = std(file.aucCmpPrePAbs);
            catch 
            abs2RSDPlusAUC(alphaInd, lambdaInd) = 0;
            abs2AbsPlusAUC(alphaInd, lambdaInd) = 0;
            abs2CmpPlusAUC(alphaInd, lambdaInd) = 0;

            abs2RSDPrePAUC(alphaInd, lambdaInd) = 0;
            abs2AbsPrePAUC(alphaInd, lambdaInd) = 0;
            abs2CmpPrePAUC(alphaInd, lambdaInd) = 0;
            end
    end
end

bestAbs2RSDPlusAUC = zeros(numOfAlphaValue,2);
bestAbs2AbsPlusAUC = zeros(numOfAlphaValue,2);
bestAbs2CmpPlusAUC = zeros(numOfAlphaValue,2);

% bestAbs2RSDPrePAUC = zeros(numOfAlphaValue,2);
% bestAbs2AbsPrePAUC = zeros(numOfAlphaValue,2);
% bestAbs2CmpPrePAUC = zeros(numOfAlphaValue,2);

bestAbs2RSDPlusAUC(:,1) = max(abs2RSDPlusAUC,[],2);
bestAbs2AbsPlusAUC(:,1) = max(abs2AbsPlusAUC,[],2);
bestAbs2CmpPlusAUC(:,1) = max(abs2CmpPlusAUC,[],2);

% Compute confidence interval
for countAlphaInd = 1: numOfAlphaValue
[bestAbs2RSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2RSDPlusAUC(countAlphaInd,1),labelRSDPlus,CIParameter);
[bestAbs2AbsPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2AbsPlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
[bestAbs2CmpPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2CmpPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
end
% bestAbs2RSDPrePAUC(:,1) = max(abs2RSDPrePAUC,[],2);
% bestAbs2AbsPrePAUC(:,1) = max(abs2AbsPrePAUC,[],2);
% bestAbs2CmpPrePAUC(:,1) = max(abs2CmpPrePAUC,[],2);
            
colorSpace = {[0.12572087695201239, 0.47323337360924367, 0.707327968232772],...
              [0.21171857311445125, 0.63326415104024547, 0.1812226118410335],...
              [0.89059593116535862, 0.10449827132271793, 0.11108035462744099],...
              [0.99990772780250103, 0.50099192647372981, 0.0051211073118098693]};   
alphaNum = cellfun(@str2num,alphaValue);
aucPlus = [bestAbs2RSDPlusAUC(:,1),bestAbs2AbsPlusAUC(:,1),bestAbs2CmpPlusAUC(:,1)];
aucCIPlus = [bestAbs2RSDPlusAUC(:,2),bestAbs2AbsPlusAUC(:,2),bestAbs2CmpPlusAUC(:,2)];
aucCIPlus = 1.96*aucCIPlus;
fig1 = plotAUCCI_numComparison(aucPlus,1.96*aucCIPlus,'xLabel','Number of Trained Comparisons',...
    'yLabel','AUC','legendStr',{'Test on RSD Labels','Test on Class Labels', 'Test on Comparison Labels'},...
    'isSaveFig',1,'figName','../../pic/num_cmp_labels','xAxis',alphaNum);
% fig = figure(); 
% plot(alphaNum, bestAbs2RSDPlusAUC(:,1),'color',colorSpace{1},...
% 'LineStyle','-','LineWidth',2,...
% 'Marker','o','MarkerFaceColor',colorSpace{1},...
% 'MarkerEdgeColor',colorSpace{1},'MarkerSize',8);
% set(gca,'XTick',[20,100,1000,3000]);
% xlabel('Number of Trained Comparisons','FontSize',20,'FontWeight','Bold');
% ylabel('AUC','FontSize',20,'FontWeight','Bold');
% set(gca, 'FontSize', 15);
% axis([0,3000,0.5,1]);
% set(fig,'Units','Inches');
% set(gca,'XScale','log');
% pos = get(fig,'Position');
% set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
% print(fig, ['num_comparisons' '.pdf'],'-dpdf','-r0');