function [ score ] = getScores(featPre,isManual,varargin)
% This function is to generate the score for an image feature.
% Input:
%       featPre:  1 by d feature vector without normalized.
%       isManual: 0 - the feature is an Automatic Segmented Feature; 1 - the feature
%       is manual segmented feature.
%       isPlus: 1 - Predict the score for Plus category. 0 - Predict the
%       score for PrePlus Category. Default is 0.
% Output:
%       score: a float number between 0 and 100. The higher represent the severe of the disease.
% 
% Author: Peng Tian
% Date: December 2016

numvarargs = length(varargin);
if numvarargs > 3
    error('get Scores has maximum 3 input variables');
end
% The optional argument is isAuto and isPlus
optargs = {0};
optargs(1:numvarargs) = varargin;
[isPlus] = optargs{:};
% featFile = load(featFileName);
% featPre = featFile.feat; % This feature has not been normalized.

% Load the manualBeta
getScoreDataFile = load('getScoreData.mat');
manualPlus = getScoreDataFile.manualRSDPlus;
manualPreP = getScoreDataFile.manualRSDPreP;
manualNormalized = getScoreDataFile.manualNormalized;
autoPlus = getScoreDataFile.autoRSDPlus;
autoPreP = getScoreDataFile.autoRSDPreP;
autoNormalized = getScoreDataFile.autoNormalized;

switch isManual 
    case 1 % manual segmented Feature
        feat = featPre./manualNormalized;
        switch isPlus
            case 0 % Predict manual feature Plus disease.
                scorePre = feat * manualPlus{1} + manualPlus{2};
            case 1 % Predict manual feature PrePlus or higher disease.
                scorePre = feat * manualPreP{1} + manualPreP{2};
            otherwise 
                error('isPlus variable is either 0 or 1. Other value is not acceptable');
        end
    case 0 % automatic segmented Feature
        feat = featPre./autoNormalized;
        switch isPlus
            case 0 % Predict manual feature Plus disease.
                scorePre = feat * autoPlus{1} + autoPlus{2};
            case 1 % Predict automatic feature PrePlus or higher disease.
                scorePre = feat * autoPreP{1} + autoPreP{2};
            otherwise
                error('isPlus variable is either 0 or 1. Other value is not acceptable');
        end
    otherwise 
        error('isAuto variable is either 0 or 1. Other value is not acceptable')
end

%% Mapping the scorePre from (-inf,+inf) to (0,1)
score = 100./(1+exp(-scorePre));


end

