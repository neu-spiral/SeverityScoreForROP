% This script is to generate  AUC and CI for 
% The method inside the script contains a global unique beta(Training the
% RSD labels, beta with bias, beta train with bias do no use bias.)
% To goal is to generate the result on AUC for 13 expert absolute labels.
% Input:
% Output:
% 
% Date: Jan 2017
% Author: Peng Tian

clear all;
close all;
clc;

addpath('../../../ClinicalLabErrors/');
% fileName = '../../Data/GridSearch/Full/RSD&Bias2RSD&Exp13_L1_CV1_';
% fileName = '../../Data/Result/L1/RSD&Bias2RSD&Exp13_L1_CV1_manual_';
% fileName = '../../Data/Result/L1Weights/RSD&Bias2RSD&Exp13_L1_CV1_NWeights_manual_';
% fileName = '../../../Data/Result/L1_Auto/RSD&Bias2RSD&Exp13_L1_CV1_auto_';
% fileName = '../../../Data/Result/L1_Auto_SVM_RSD/RSD&Bias2RSD&Exp13_SVM_L1_CV1_auto_';
fileName = '../../../Projects/C_Tian_ProbalisticComparison_ICML2017/data/SVMAllRSD/RSD&Bias2RSD&Exp13_SVMAll_L1_CV1_auto_';
fileName = '../../../Data/Result/L1_Auto_SVMAll_RSD/'


%% Parameter Setting
CIParameter = 0.10;
savefileName = 'SVMAll2Abs.mat';

isSaving = 1;
alphaValue = 0.0:0.1:1.0;
lambdaValue = {'10000.0' '1000.0' '100.0' '10.0' '9.0' '7.0' '5.0' '3.0' '1.0' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.01' '0.001' '0.0001' '1e-05' '1e-06' };

%% Prepare Label Data
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



numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2);
FigAxisRange = [alphaValue(1),alphaValue(end),labelFigYLower,labelFigYUpper];
RSD2RSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue); % 
RSD2Exp13PlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2Exp13PlusAUC= zeros(numOfAlphaValue,numOfLambdaValue); 
bias2Exp13NoBPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpBiasPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpRSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);

RSD2RSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);% 
RSD2Exp13PrePAUC= zeros(numOfAlphaValue,numOfLambdaValue);
bias2Exp13PrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2Exp13NoBPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpBiasPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpRSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
            try file = load([fileName, num2str(alphaValue(alphaInd),'%.1f'), '_',lambdaValue{lambdaInd},'.mat']);
            % For the plot in Plus vs Not Plus Image
            RSD2RSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucLRSD2RSDPlus{1},2);
            RSD2Exp13PlusAUC(alphaInd, lambdaInd) = mean(file.aucLRSD2Exp13Plus{1},2);
            bias2Exp13PlusAUC(alphaInd, lambdaInd) = mean(file.aucLExp132Exp13Plus{1},2);
            bias2Exp13NoBPlusAUC(alphaInd, lambdaInd) = mean(file.aucLExp132Exp13NoBPlus{1},2);
            bias2RSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucLExp132RSDNoBPlus{1},2);
            cmpBiasPlusAUC(alphaInd, lambdaInd) = mean(file.aucCExp13Plus{1},2);
            cmpRSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucCRSDPlus{1},2);

            RSD2RSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucLRSD2RSDPreP{1},2);
            RSD2Exp13PrePAUC(alphaInd, lambdaInd) = mean(file.aucLRSD2Exp13PreP{1},2);
            bias2Exp13PrePAUC(alphaInd, lambdaInd) = mean(file.aucLExp132Exp13PreP{1},2);
            bias2Exp13NoBPrePAUC(alphaInd, lambdaInd) = mean(file.aucLExp132Exp13NoBPreP{1},2);
            bias2RSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucLExp132RSDNoBPreP{1},2);
            cmpBiasPrePAUC(alphaInd, lambdaInd) = mean(file.aucCExp13PreP{1},2);
            cmpRSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucCRSDPreP{1},2);
            catch 
            RSD2RSDPlusAUC(alphaInd, lambdaInd) = 0;
            RSD2Exp13PlusAUC(alphaInd, lambdaInd) = 0;
            bias2Exp13PlusAUC(alphaInd, lambdaInd) = 0;
            bias2Exp13NoBPlusAUC(alphaInd, lambdaInd) = 0;
            bias2RSDPlusAUC(alphaInd, lambdaInd) = 0;
            cmpBiasPlusAUC(alphaInd, lambdaInd) = 0;
            cmpRSDPlusAUC(alphaInd, lambdaInd) = 0;

            RSD2RSDPrePAUC(alphaInd, lambdaInd) = 0;
            RSD2Exp13PrePAUC(alphaInd, lambdaInd) = 0;
            bias2Exp13PrePAUC(alphaInd, lambdaInd) = 0;
            bias2Exp13NoBPrePAUC(alphaInd, lambdaInd) = 0;
            bias2RSDPrePAUC(alphaInd, lambdaInd) = 0;
            cmpBiasPrePAUC(alphaInd, lambdaInd) = 0;
            cmpRSDPrePAUC(alphaInd, lambdaInd) = 0; 
            end
    end
end

bestRSD2RSDPlusAUC = zeros(numOfAlphaValue, 2);
bestRSD2Exp13PlusAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13PlusAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13NoBPlusAUC = zeros(numOfAlphaValue, 2);
bestBias2RSDPlusAUC = zeros(numOfAlphaValue, 2);
bestCmpBiasPlusAUC = zeros(numOfAlphaValue, 2);
bestCmpRSDPlusAUC = zeros(numOfAlphaValue, 2);
bestRSD2RSDPrePAUC = zeros(numOfAlphaValue, 2);
bestRSD2Exp13PrePAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13PrePAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13NoBPrePAUC = zeros(numOfAlphaValue, 2);
bestBias2RSDPrePAUC = zeros(numOfAlphaValue, 2);
bestCmpBiasPrePAUC = zeros(numOfAlphaValue, 2);
bestCmpRSDPrePAUC = zeros(numOfAlphaValue, 2);

bestRSD2RSDPlusAUC(:,1) = max(RSD2RSDPlusAUC,[],2);
bestRSD2Exp13PlusAUC(:,1) = max(RSD2Exp13PlusAUC,[],2);
bestBias2Exp13PlusAUC(:,1) = max(bias2Exp13PlusAUC,[],2);
bestBias2Exp13NoBPlusAUC(:,1) = max(bias2Exp13NoBPlusAUC,[],2);
bestBias2RSDPlusAUC(:,1) = max(bias2RSDPlusAUC,[],2);
bestCmpBiasPlusAUC(:,1) = max(cmpBiasPlusAUC,[],2);
bestCmpRSDPlusAUC(:,1) = max(cmpRSDPlusAUC,[],2);

bestRSD2RSDPrePAUC(:,1) = max(RSD2RSDPrePAUC,[],2);
bestRSD2Exp13PrePAUC(:,1) = max(RSD2Exp13PrePAUC,[],2);
bestBias2Exp13PrePAUC(:,1) = max(bias2Exp13PrePAUC,[],2);
bestBias2Exp13NoBPrePAUC(:,1) = max(bias2Exp13NoBPrePAUC,[],2);
bestBias2RSDPrePAUC(:,1) = max(bias2RSDPrePAUC,[],2);
bestCmpBiasPrePAUC(:,1) = max(cmpBiasPrePAUC,[],2);
bestCmpRSDPrePAUC(:,1) = max(cmpRSDPrePAUC,[],2);

for countAlphaInd = 1:1:numOfAlphaValue
    [bestRSD2RSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestRSD2RSDPlusAUC(countAlphaInd,1),labelRSDPlus,CIParameter);
    [bestRSD2Exp13PlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestRSD2Exp13PlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestBias2Exp13PlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13PlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestBias2Exp13NoBPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13NoBPlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestBias2RSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2RSDPlusAUC(countAlphaInd,1),labelRSDPlus,CIParameter);
    [bestCmpBiasPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpBiasPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpRSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpRSDPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);

    [bestRSD2RSDPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestRSD2RSDPrePAUC(countAlphaInd,1),labelRSDPreP,CIParameter);
    [bestRSD2Exp13PrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestRSD2Exp13PrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestBias2Exp13PrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13PrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestBias2Exp13NoBPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13NoBPrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestBias2RSDPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2RSDPrePAUC(countAlphaInd,1),labelRSDPreP,CIParameter);
    [bestCmpBiasPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpBiasPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpRSDPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpRSDPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
end

save(savefileName,...
    'bestRSD2RSDPlusAUC','bestRSD2Exp13PlusAUC','bestBias2Exp13PlusAUC','bestBias2Exp13NoBPlusAUC',...
    'bestBias2RSDPlusAUC','bestCmpBiasPlusAUC','bestCmpRSDPlusAUC',...
    'bestRSD2RSDPrePAUC','bestRSD2Exp13PrePAUC','bestBias2Exp13PrePAUC','bestBias2Exp13NoBPrePAUC',...
    'bestBias2RSDPrePAUC','bestCmpBiasPrePAUC','bestCmpRSDPrePAUC') 





