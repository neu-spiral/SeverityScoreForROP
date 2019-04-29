% This script is to compute the correlation coefficient between severity
% rank and the 5 expert consensus rank.
% input:
% output:
% Author: Peng Tian
% Date : Feb 2017
clear all;
close all;
clc;

inputFileName = '../../../Data/Result/icml_auto/MS_SVMAll_l1_CV1_auto_';
saveFileName = '../data/icml_auto_SVMAll_CC.mat';
isSaveFigure = 0;
alphaValue = 0.0:0.1:1.0;
lambdaStr = {'10000.0' '1000.0' '100.0' '10.0' '9.0' '7.0' '5.0' '3.0' '1.0' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1' '0.01' '0.001' '0.0001' '1e-05' '1e-06' };

%% Prepare Label Data
lambdaValue = cellfun(@str2num,lambdaStr(1:end));

RSDlabelFile = load('../../../Data/ProbalisticModel/iROPData_6DD.mat');
ExpertRank = RSDlabelFile.ExpertsRank1st;

parameters = cell(4,3);
parameters(:,1) = {'abs2RSDPlusCC','bias2RSDPlusCC',...
                     'abs2RSDPrePCC','bias2RSDPrePCC'};

numOfAlphaValue = size(alphaValue,2);
numOfLambdaValue = size(lambdaValue,2); 
abs2RSDPlusCC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPlusCC = zeros(numOfAlphaValue,numOfLambdaValue);
abs2RSDPrePCC = zeros(numOfAlphaValue,numOfLambdaValue);
bias2RSDPrePCC = zeros(numOfAlphaValue,numOfLambdaValue);
for alphaInd = 1:numOfAlphaValue
    for lambdaInd = 1:numOfLambdaValue
%             try 
            file = load([inputFileName, num2str(alphaValue(alphaInd),'%.1f'), '_',lambdaStr{lambdaInd},'.mat']);
            % For the plot in Plus vs Not Plus Image
            abs2RSDPlusCC(alphaInd, lambdaInd) = CC(rankScore(mean(file.scoreRSDPlusAbs,2)),ExpertRank);
            bias2RSDPlusCC(alphaInd, lambdaInd) = CC(rankScore(mean(file.scoreRSDPlusBias,2)),ExpertRank);
            

            abs2RSDPrePCC(alphaInd, lambdaInd) = CC(rankScore(mean(file.scoreRSDPrePAbs,2)),ExpertRank);
            bias2RSDPrePCC(alphaInd, lambdaInd) = CC(rankScore(mean(file.scoreRSDPrePBias,2)),ExpertRank);
            
%             catch 
%             abs2RSDPlusCC(alphaInd, lambdaInd) = 0;
%             bias2RSDPlusCC(alphaInd, lambdaInd) = 0;
%             abs2AbsPlusCC(alphaInd, lambdaInd) = 0;
%             bias2AbsPlusCC(alphaInd, lambdaInd) = 0;
%             unique2AbsPlusCC(alphaInd, lambdaInd) = 0;
%             abs2CmpPlusCC(alphaInd, lambdaInd) = 0;
%             bias2CmpPlusCC(alphaInd, lambdaInd) = 0;
%             unique2CmpPlusCC(alphaInd, lambdaInd) = 0;
% 
%             abs2RSDPrePCC(alphaInd, lambdaInd) = 0;
%             bias2RSDPrePCC(alphaInd, lambdaInd) = 0;
%             abs2AbsPrePCC(alphaInd, lambdaInd) = 0;
%             bias2AbsPrePCC(alphaInd, lambdaInd) = 0;
%             unique2AbsPrePCC(alphaInd, lambdaInd) = 0;
%             abs2CmpPrePCC(alphaInd, lambdaInd) = 0;
%             bias2CmpPrePCC(alphaInd, lambdaInd) = 0;
%             unique2CmpPrePCC(alphaInd, lambdaInd) = 0;
%             end
    end
end

bestAbs2RSDPlusCC = zeros(numOfAlphaValue, 2);
bestBias2RSDPlusCC = zeros(numOfAlphaValue, 2);

bestAbs2RSDPrePCC = zeros(numOfAlphaValue, 2);
bestBias2RSDPrePCC = zeros(numOfAlphaValue, 2);

parameters(1,3:4) = {selectParameter(bestAbs2RSDPlusCC,alphaValue,lambdaValue)};
parameters(2,3:4) = {selectParameter(bestBias2RSDPlusCC,alphaValue,lambdaValue)};
parameters(3,3:4) = {selectParameter(bestAbs2RSDPrePCC,alphaValue,lambdaValue)};
parameters(4,3:4) = {selectParameter(bestBias2RSDPrePCC,alphaValue,lambdaValue)};




bestAbs2RSDPlusCC(:,1) = max(abs2RSDPlusCC,[],2);
bestBias2RSDPlusCC(:,1) = max(bias2RSDPlusCC,[],2);


bestAbs2RSDPrePCC(:,1) = max(abs2RSDPrePCC,[],2);
bestBias2RSDPrePCC(:,1) = max(bias2RSDPrePCC,[],2);


save(saveFileName,...
    'bestAbs2RSDPlusCC','bestBias2RSDPlusCC',...
    'bestAbs2RSDPrePCC','bestBias2RSDPrePCC','parameters'); 
