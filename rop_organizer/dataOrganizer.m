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


%% Parameter Setting
CIParameter = 0.10;
% inputFileName = '../../../Data/Result/icml_auto/MS_SVMAll_l1_CV1_auto_';
% inputFileName = '../../../Data/Result/icml_auto/MS_SVM_l1_CV1_auto_';
% inputFileName = '../../../Data/Result/icml_auto/MS_l1_CV1_auto_';

% inputFileName = '../../../Data/Result/icml_manual/MS_l1_CV1_manual_';
% inputFileName = '../../../Data/Result/icml_manual/MS_SVM_l1_CV1_manual_';
% inputFileName = '../../../Data/Result/icml_manual/MS_SVMAll_l1_CV1_manual_';



inputFileName = '../../../Projects/ICML2017/data/auto_full/MS_SVMLog_l1_CV1_auto_';

% saveFileName = '../../../Projects/ICML2017/data/icml_manual_Logistic.mat';
isSaveFigure = 0;
alphaValue = 0.0:0.1:1.0;
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
parameters(:,1) = {'abs2RSDPlusAUC','bias2RSDPlusAUC','abs2AbsPlusAUC','bias2AbsPlusAUC',...
                     'unique2AbsPlusAUC','abs2CmpPlusAUC','bias2CmpPlusAUC','unique2CmpPlusAUC',...
                     'abs2RSDPrePAUC','bias2RSDPrePAUC','abs2AbsPrePAUC','bias2AbsPrePAUC',...
                     'unique2AbsPrePAUC','abs2CmpPrePAUC','bias2CmpPrePAUC','unique2CmpPrePAUC'};

numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2); 
abs2RSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2AbsPlusAUC= zeros(numOfAlphaValue,numOfLambdaValue); 
unique2AbsPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2CmpPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2CmpPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);
unique2CmpPlusAUC = zeros(numOfAlphaValue,numOfLambdaValue);

abs2RSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2AbsPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2AbsPrePAUC= zeros(numOfAlphaValue,numOfLambdaValue); 
unique2AbsPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2CmpPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2CmpPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
unique2CmpPrePAUC = zeros(numOfAlphaValue,numOfLambdaValue);
for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
            try 
            file = load([inputFileName, num2str(alphaValue(alphaInd),'%.1f'), '_',lambdaStr{lambdaInd},'.mat']);
            % For the plot in Plus vs Not Plus Image
            abs2RSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucRSDPlusAbs,2);
            bias2RSDPlusAUC(alphaInd, lambdaInd) = mean(file.aucRSDPlusBias,2);
            abs2AbsPlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusAbs,2);
            bias2AbsPlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusBias,2);
            unique2AbsPlusAUC(alphaInd, lambdaInd) = mean(file.aucLblPlusUnique,2);
            abs2CmpPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusAbs,2);
            bias2CmpPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusBias,2);
            unique2CmpPlusAUC(alphaInd, lambdaInd) = mean(file.aucCmpPlusUnique,2);

            abs2RSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucRSDPrePAbs,2);
            bias2RSDPrePAUC(alphaInd, lambdaInd) = mean(file.aucRSDPrePBias,2);
            abs2AbsPrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePAbs,2);
            bias2AbsPrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePBias,2);
            unique2AbsPrePAUC(alphaInd, lambdaInd) = mean(file.aucLblPrePUnique,2);
            abs2CmpPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePAbs,2);
            bias2CmpPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePBias,2);
            unique2CmpPrePAUC(alphaInd, lambdaInd) = mean(file.aucCmpPrePUnique,2);
            catch 
            abs2RSDPlusAUC(alphaInd, lambdaInd) = 0;
            bias2RSDPlusAUC(alphaInd, lambdaInd) = 0;
            abs2AbsPlusAUC(alphaInd, lambdaInd) = 0;
            bias2AbsPlusAUC(alphaInd, lambdaInd) = 0;
            unique2AbsPlusAUC(alphaInd, lambdaInd) = 0;
            abs2CmpPlusAUC(alphaInd, lambdaInd) = 0;
            bias2CmpPlusAUC(alphaInd, lambdaInd) = 0;
            unique2CmpPlusAUC(alphaInd, lambdaInd) = 0;

            abs2RSDPrePAUC(alphaInd, lambdaInd) = 0;
            bias2RSDPrePAUC(alphaInd, lambdaInd) = 0;
            abs2AbsPrePAUC(alphaInd, lambdaInd) = 0;
            bias2AbsPrePAUC(alphaInd, lambdaInd) = 0;
            unique2AbsPrePAUC(alphaInd, lambdaInd) = 0;
            abs2CmpPrePAUC(alphaInd, lambdaInd) = 0;
            bias2CmpPrePAUC(alphaInd, lambdaInd) = 0;
            unique2CmpPrePAUC(alphaInd, lambdaInd) = 0;
            end
    end
end
          

bestAbs2RSDPlusAUC = zeros(numOfAlphaValue, 2);
bestBias2RSDPlusAUC = zeros(numOfAlphaValue, 2);
bestAbs2AbsPlusAUC = zeros(numOfAlphaValue, 2);
bestBias2AbsPlusAUC = zeros(numOfAlphaValue, 2);
bestUnique2AbsPlusAUC = zeros(numOfAlphaValue, 2);
bestAbs2CmpPlusAUC = zeros(numOfAlphaValue, 2);
bestBias2CmpPlusAUC = zeros(numOfAlphaValue, 2);
bestUnique2CmpPlusAUC = zeros(numOfAlphaValue, 2);

bestAbs2RSDPrePAUC = zeros(numOfAlphaValue, 2);
bestBias2RSDPrePAUC = zeros(numOfAlphaValue, 2);
bestAbs2AbsPrePAUC = zeros(numOfAlphaValue, 2);
bestBias2AbsPrePAUC = zeros(numOfAlphaValue, 2);
bestUnique2AbsPrePAUC = zeros(numOfAlphaValue, 2);
bestAbs2CmpPrePAUC = zeros(numOfAlphaValue, 2);
bestBias2CmpPrePAUC = zeros(numOfAlphaValue, 2);
bestUnique2CmpPrePAUC = zeros(numOfAlphaValue, 2);


% [a,a1]=selectParameter(abs2RSDPlusAUC,alphaValue,lambdaValue)
parameters(2,3:4) = {selectParameter(bias2RSDPlusAUC,alphaValue,lambdaValue)};
parameters(3,3:4) = {selectParameter(abs2AbsPlusAUC,alphaValue,lambdaValue)};
parameters(4,3:4) = {selectParameter(bias2AbsPlusAUC,alphaValue,lambdaValue)};
parameters(5,3:4) = {selectParameter(unique2AbsPlusAUC,alphaValue,lambdaValue)};
parameters(6,3:4) = {selectParameter(abs2CmpPlusAUC,alphaValue,lambdaValue)};
parameters(7,3:4) = {selectParameter(bias2CmpPlusAUC,alphaValue,lambdaValue)};
parameters(8,3:4) = {selectParameter(unique2CmpPlusAUC,alphaValue,lambdaValue)};

parameters(9,3:4) = {selectParameter(abs2RSDPrePAUC,alphaValue,lambdaValue)};
parameters(10,3:4) = {selectParameter(bias2RSDPrePAUC,alphaValue,lambdaValue)};
parameters(11,3:4) = {selectParameter(abs2AbsPrePAUC,alphaValue,lambdaValue)};
parameters(12,3:4) = {selectParameter(bias2AbsPrePAUC,alphaValue,lambdaValue)};
parameters(13,3:4) = {selectParameter(unique2AbsPrePAUC,alphaValue,lambdaValue)};
parameters(14,3:4) = {selectParameter(abs2CmpPrePAUC,alphaValue,lambdaValue)};
parameters(15,3:4) = {selectParameter(bias2CmpPrePAUC,alphaValue,lambdaValue)};
parameters(16,3:4) = {selectParameter(unique2CmpPrePAUC,alphaValue,lambdaValue)};



bestAbs2RSDPlusAUC(:,1) = max(abs2RSDPlusAUC,[],2);
bestBias2RSDPlusAUC(:,1) = max(bias2RSDPlusAUC,[],2);
bestAbs2AbsPlusAUC(:,1) = max(abs2AbsPlusAUC,[],2);
bestBias2AbsPlusAUC(:,1) = max(bias2AbsPlusAUC,[],2);
bestUnique2AbsPlusAUC(:,1) = max(unique2AbsPlusAUC,[],2);
bestAbs2CmpPlusAUC(:,1) = max(abs2CmpPlusAUC,[],2);
bestBias2CmpPlusAUC(:,1) = max(bias2CmpPlusAUC,[],2);
bestUnique2CmpPlusAUC(:,1) = max(unique2CmpPlusAUC,[],2);

bestAbs2RSDPrePAUC(:,1) = max(abs2RSDPrePAUC,[],2);
bestBias2RSDPrePAUC(:,1) = max(bias2RSDPrePAUC,[],2);
bestAbs2AbsPrePAUC(:,1) = max(abs2AbsPrePAUC,[],2);
bestBias2AbsPrePAUC(:,1) = max(bias2AbsPrePAUC,[],2);
bestUnique2AbsPrePAUC(:,1) = max(unique2AbsPrePAUC,[],2);
bestAbs2CmpPrePAUC(:,1) = max(abs2CmpPrePAUC,[],2);
bestBias2CmpPrePAUC(:,1) = max(bias2CmpPrePAUC,[],2);
bestUnique2CmpPrePAUC(:,1) = max(unique2CmpPrePAUC,[],2);

for countAlphaInd = 1:numOfAlphaValue
    [bestAbs2RSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2RSDPlusAUC(countAlphaInd,1),labelRSDPlus,CIParameter);
    [bestBias2RSDPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2RSDPlusAUC(countAlphaInd,1),labelRSDPlus,CIParameter);
    [bestAbs2AbsPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2AbsPlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestBias2AbsPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2AbsPlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestUnique2AbsPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestUnique2AbsPlusAUC(countAlphaInd,1),labelExp13Plus,CIParameter);
    [bestAbs2CmpPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2CmpPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestBias2CmpPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2CmpPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestUnique2CmpPlusAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestUnique2CmpPlusAUC(countAlphaInd,1),cmpLabels,CIParameter);

    [bestAbs2RSDPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2RSDPrePAUC(countAlphaInd,1),labelRSDPreP,CIParameter);
    [bestBias2RSDPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2RSDPrePAUC(countAlphaInd,1),labelRSDPreP,CIParameter);
    [bestAbs2AbsPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2AbsPrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestBias2AbsPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2AbsPrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestUnique2AbsPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestUnique2AbsPrePAUC(countAlphaInd,1),labelExp13PreP,CIParameter);
    [bestAbs2CmpPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestAbs2CmpPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestBias2CmpPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestBias2CmpPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
    [bestUnique2CmpPrePAUC(countAlphaInd,2),~] = computeCIBasedOnAUC(bestUnique2CmpPrePAUC(countAlphaInd,1),cmpLabels,CIParameter);
end

% save(saveFileName,...
%     'bestAbs2RSDPlusAUC','bestBias2RSDPlusAUC','bestAbs2AbsPlusAUC','bestBias2AbsPlusAUC',...
%     'bestUnique2AbsPlusAUC','bestAbs2CmpPlusAUC','bestBias2CmpPlusAUC','bestUnique2CmpPlusAUC',...
%     'bestAbs2RSDPrePAUC','bestBias2RSDPrePAUC','bestAbs2AbsPrePAUC','bestBias2AbsPrePAUC',...
%     'bestUnique2AbsPrePAUC','bestAbs2CmpPrePAUC','bestBias2CmpPrePAUC','bestUnique2CmpPrePAUC','parameters'); 





