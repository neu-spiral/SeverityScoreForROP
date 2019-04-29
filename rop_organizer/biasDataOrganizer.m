% This script is to generate AUC and the CI for the modelSelectionL1.py
% The method inside the script contains a global unique beta(Training the RSD labels?, beta with bias, beta train with bias do no use bias.
% To goal is to generate the result on AUC for 13 expert absolute labels.
% Input:
% Output:
% 
% Date: Jan 2017
% Author: Peng Tian

clear all;
close all;
clc;
%% Parameter Setting
fileName = '../../../Projects/C_Tian_ProbalisticComparison_ICML2017/data/SVMAll/MS_SVMall_l1_CV1_auto_';
fileName = '../../../Data/Result/L1_Auto_SVMAll/MS_SVMAll_l1_CV1_auto_';
% fileName = '../../../Projects/C_Tian_ProbalisticComparison_ICML2017/data/SVMAll/MS_SVMAll_l1_CV1_auto_';
savefileName = 'SVMAllNew.mat';
isSaving = 1;
CIParameter = 0.10;
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

addpath('../../../ClinicalLabErrors/');
numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2);
abs2Exp13PlusAUC = zeros(numOfAlphaValue,numOfLambdaValue); 
bias2Exp13PlusAUC= zeros(numOfAlphaValue,numOfLambdaValue); 
mdl2Exp13PlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpAbsPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpBiasPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpMdlPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);

abs2Exp13PrePAUC= zeros(numOfAlphaValue,numOfLambdaValue);
bias2Exp13PrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
mdl2Exp13PrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpBiasPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpAbsPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
cmpMdlPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
%             try 
            file = load([fileName, num2str(alphaValue(alphaInd),'%.1f'), '_',lambdaValue{lambdaInd},'.mat']);
            % For the plot in Plus vs Not Plus Image
            abs2Exp13PlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusAbs,2);
            bias2Exp13PlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusBias,2);
%             mdl2Exp13PlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusUnique,2);
            cmpBiasPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusBias,2);
            cmpAbsPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusAbs,2);
%             cmpMdlPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusUnique,2);

            abs2Exp13PrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePAbs,2);
            bias2Exp13PrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePBias,2);
%             mdl2Exp13PrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePUnique,2);
            cmpBiasPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePBias,2);
            cmpAbsPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePAbs,2);
%             cmpMdlPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePUnique,2);
%             catch 
%             abs2Exp13PlusAUC(alphaInd, lambdaInd) = 0;
%             bias2Exp13PlusAUC(alphaInd, lambdaInd) = 0;
%             mdl2Exp13PlusAUC(alphaInd, lambdaInd) = 0;
%             cmpBiasPlusAUC(alphaInd, lambdaInd) = 0;
%             cmpAbsPlusAUC(alphaInd, lambdaInd) = 0;
%             cmpMdlPlusAUC(alphaInd, lambdaInd) = 0;
% 
%             abs2Exp13PrePAUC(alphaInd, lambdaInd) = 0;
%             bias2Exp13PrePAUC(alphaInd, lambdaInd) = 0;
%             mdl2Exp13PrePAUC(alphaInd, lambdaInd) = 0;
%             cmpAbsPrePAUC(alphaInd, lambdaInd) = 0;
%             cmpBiasPrePAUC(alphaInd, lambdaInd) = 0;
%             cmpMdlPrePAUC(alphaInd, lambdaInd) = 0;
%             end
    end
end

bestAbs2Exp13PlusAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13PlusAUC = zeros(numOfAlphaValue, 2);
bestMdl2Exp13PlusAUC = zeros(numOfAlphaValue, 2);
bestCmpAbsPlusAUC = zeros(numOfAlphaValue, 2);
bestCmpBiasPlusAUC = zeros(numOfAlphaValue, 2);
bestCmpMdlPlusAUC = zeros(numOfAlphaValue, 2);
bestAbs2Exp13PrePAUC = zeros(numOfAlphaValue, 2);
bestBias2Exp13PrePAUC = zeros(numOfAlphaValue, 2);
bestMdl2Exp13PrePAUC = zeros(numOfAlphaValue, 2);
bestCmpAbsPrePAUC = zeros(numOfAlphaValue, 2);
bestCmpBiasPrePAUC = zeros(numOfAlphaValue, 2);
bestCmpMdlPrePAUC = zeros(numOfAlphaValue, 2);

bestAbs2Exp13PlusAUC(:,1) = max(abs2Exp13PlusAUC,[],2);
bestBias2Exp13PlusAUC(:,1) = max(bias2Exp13PlusAUC,[],2);
bestMdl2Exp13PlusAUC(:,1) = max(mdl2Exp13PlusAUC,[],2);
bestCmpAbsPlusAUC(:,1) = max(cmpAbsPlusAUC,[],2);
bestCmpBiasPlusAUC(:,1) = max(cmpBiasPlusAUC,[],2);
bestCmpMdlPlusAUC(:,1) = max(cmpMdlPlusAUC,[],2);

bestAbs2Exp13PrePAUC(:,1) = max(abs2Exp13PrePAUC,[],2);
bestBias2Exp13PrePAUC(:,1) = max(bias2Exp13PrePAUC,[],2);
bestMdl2Exp13PrePAUC(:,1) = max(mdl2Exp13PrePAUC,[],2);
bestCmpAbsPrePAUC(:,1) = max(cmpAbsPrePAUC,[],2);
bestCmpBiasPrePAUC(:,1) = max(cmpBiasPrePAUC,[],2);
bestCmpMdlPrePAUC(:,1) = max(cmpMdlPrePAUC,[],2);

for countAlphaInd = 1:1:numOfAlphaValue
    [bestAbs2Exp13PlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2Exp13PlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestBias2Exp13PlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13PlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestMdl2Exp13PlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestMdl2Exp13PlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestCmpAbsPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpAbsPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpBiasPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpBiasPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpMdlPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpMdlPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);

    [bestAbs2Exp13PrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2Exp13PrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestBias2Exp13PrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2Exp13PrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestMdl2Exp13PrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestMdl2Exp13PrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestCmpAbsPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpAbsPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpBiasPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpBiasPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestCmpMdlPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestCmpMdlPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    
end

save(savefileName,...
    'bestAbs2Exp13PlusAUC','bestBias2Exp13PlusAUC','bestMdl2Exp13PlusAUC',...
    'bestCmpAbsPlusAUC','bestCmpBiasPlusAUC','bestCmpMdlPlusAUC',...
    'bestAbs2Exp13PrePAUC','bestBias2Exp13PrePAUC','bestMdl2Exp13PrePAUC',...
    'bestCmpAbsPrePAUC','bestCmpBiasPrePAUC','bestCmpMdlPrePAUC') 




