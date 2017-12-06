% Facial Keypoints Detection - Convolution Neural Network Analysis
% References:
%   1.  Create Simple Deep Learning Network for Classification - https://www.mathworks.com/help/nnet/examples/create-simple-deep-learning-network-for-classification.html.
%   2.  'trainNetwork' - https://www.mathworks.com/help/nnet/ref/trainnetwork.html.
%   3.  'trainingOptions' - https://www.mathworks.com/help/nnet/ref/trainingoptions.html.
%   4.  'augmentedImageSource' - https://www.mathworks.com/help/nnet/ref/augmentedimagesource.html.
%   5.  'imageDataAugmenter' - https://www.mathworks.com/help/nnet/ref/imagedataaugmenter.html.
%   6.  Neural Network Toolbox Functions - https://www.mathworks.com/help/nnet/functionlist.html.
% Remarks:
%   1.  sa
% Known Issues:
%   1.  s
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     04/12/2016  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

addpath(genpath('./AuxiliaryFunctions'));

dataFolderPath              = 'Data/';
netFolderPath               = 'Data/';
trainImageFileName          = 'tTrainImagesValid.mat'; %<! tTrainImages
trainFeaturesCoordFileName  = 'tTrainFeaturesCoordValid.mat'; %<! tTrainFeaturesCoord
featuresNameFileName        = 'cFeaturesName.mat';
testImageFileName           = 'tTestImages.mat'; %<! tTestImages


%% Load Data

load([dataFolderPath, trainImageFileName]); %<! tTrainImages
load([dataFolderPath, trainFeaturesCoordFileName]); %<! tRefFeaturesCoord
load([dataFolderPath, featuresNameFileName]); %<! cFeaturesName


%% Training Settings

normalizeData       = OFF;
dataAugmentation    = OFF;
netLayerModelIdx    = 1;
validRatio          = 0.1;


%% Test Data

numRows     = size(tTrainImages, 1);
numCols     = size(tTrainImages, 2);
numChannels = 1;
numSamples  = size(tTrainImages, 3);

% Data Shape - Height, Width, Number of Channels, Number of Samples
mImageData = reshape(tTrainImages, [numRows, numCols, numChannels, numSamples]);

meanVal = mean(mImageData(:));
stdVal = std(mImageData(:));

if(normalizeData == ON)
    mImageData = (mImageData - meanVal) / stdVal;
end

vY = tTrainFeaturesCoord(1, 1, :); %<! Regression response
vY = reshape(vY, [numSamples, 1]);

mTrainData = mImageData(:, :, :, 1:2000);
vYTrain = vY(1:2000);

mValidationData = mImageData(:, :, :, 2001:numSamples);
vYValidation    = vY(2001:numSamples);

if(dataAugmentation == ON)
    imageSource = augmentedImageSource([numRows, numCols], mTrainData, vYTrain, 'DataAugmentation', imageDataAugmenter('RandXReflection', ON));
else
    imageSource = augmentedImageSource([numRows, numCols], mTrainData, vYTrain);
end


%% Define Network

hNetLayerModel = SelectNetLayerModel(netLayerModelIdx, numRows, numCols, numChannels);

% Pre Processing


%% Training

% trainingOptions = trainingOptions('sgdm',...
%     'MaxEpochs', 3, ...
%     'Verbose', true,...
%     'Plots', 'training-progress');

trainingOptions = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.00025, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.99, ...
    'LearnRateDropPeriod', 2, ...
    'L2Regularization', 0.00001, ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 200, ...
    'Momentum', 0.65, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {mValidationData, vYValidation}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5000, ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'training-progress');

[hCnnNet, sTrainInfo] = trainNetwork(imageSource, hNetLayerModel, trainingOptions);


%% Save Data

sTrainParams.subStreamNumber    = subStreamNumber;
sTrainParams.normalizeData      = normalizeData;
sTrainParams.netLayerModelIdx   = netLayerModelIdx;
sTrainParams.dataAugmentation   = dataAugmentation;
sTrainParams.meanVal            = meanVal;
sTrainParams.stdVal             = stdVal;

save([netFolderPath, 'hNetModel', num2str(netLayerModelIdx, '%03d')], 'hCnnNet', 'sTrainInfo', 'trainingOptions', 'sTrainParams');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

