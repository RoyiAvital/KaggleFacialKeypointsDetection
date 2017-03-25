% Facial Keypoints Detection - Neural Network Analysis
% References:
%   1.  Fit Data with a Neural Network - https://www.mathworks.com/help/nnet/gs/fit-data-with-a-neural-network.html
%   2.  Improve Neural Network Generalization and Avoid Overfitting - https://www.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html
%   3.  'fitnet' - https://www.mathworks.com/help/nnet/ref/fitnet.html
%   4.  'train' - https://www.mathworks.com/help/nnet/ref/train.html
%   5.  Neural Network Object Properties - https://www.mathworks.com/help/nnet/ug/neural-network-object-properties.html
%   6.  Neural Network Subobject Properties - https://www.mathworks.com/help/nnet/ug/neural-network-subobject-properties.html
%   7.  Train and Apply Multilayer Neural Networks - https://www.mathworks.com/help/nnet/ug/train-and-apply-multilayer-neural-networks.html
%   8.  Using Convolutional Neural Nets to Detect Facial Keypoints Tutorial - http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     08/11/2016  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

addpath(genpath('./AuxiliaryFunctions'));

dataFolderPath              = 'Data/';
refImageFileName            = 'tRefImages.mat';
refFeaturesCoordFileName    = 'tRefFeaturesCoord.mat';
featuresNameFileName        = 'cFeaturesName.mat';
testImageFileName           = 'tTestImage.mat';


%% Load Data

load([dataFolderPath, refImageFileName]); %<! tRefImage
load([dataFolderPath, refFeaturesCoordFileName]); %<! tRefFeaturesCoord
load([dataFolderPath, featuresNameFileName]); %<! cFeaturesName


%% Analysis Settings

generateReflectedImages = OFF;
normalizeImage          = ON;

% Neural Network Settings
vHiddenLayesSize    = [100, 25, 10];
% trainFcn            = 'trainrp'; %<! Resilient Backpropagation
trainFcn            = 'trainscg'; %<! Scaled Conjugate Gradient
% trainFcn            = 'traingdx'; %<! Variable Learning Rate Gradient Descent
% trainFcn            = 'traingdm'; %<! Gradient Descent with Momentum
% trainFcn            = 'traingd'; %<! Gradient Descent
% transferFcn         = 'tansig';
transferFcn         = 'logsig';
% transferFcn         = 'poslin'; %<! RELU
weightsRegFactr     = 0.035;
numFails            = 500;
numEpochs           = 2500;
useGpu              = OFF;

trainRatio      = 0.80;
validationRatio = 0.2;
testRatio       = 0.0;

% Test Data
runTest = ON;


%% Data Parameters & Pre Processing

numFeatures = size(tRefFeaturesCoord, 1);
numCoord    = size(tRefFeaturesCoord, 2);
numImages   = size(tRefFeaturesCoord, 3);
numRows     = size(tRefImages, 1);
numCols     = size(tRefImages, 2);

if(generateReflectedImages == ON)
    tRefImages(:, :, (2 * numImages))           = 0;
    tRefFeaturesCoord(:, :, (2 * numImages))    = 0;
    
    for ii = 1:numImages
        tRefImages(:, :, (ii + numImages))          = fliplr(tRefImages(:, :, ii));
        tRefFeaturesCoord(:, 1, (ii + numImages))   = (numCols + 1) - tRefFeaturesCoord(:, 1, ii);
        tRefFeaturesCoord(:, 2, (ii + numImages))   = tRefFeaturesCoord(:, 2, ii);
    end
    
    numImages = 2 * numImages;
end

if(normalizeImage == ON)
    for ii = 1:numImages
        mRefImage = tRefImages(:, :, ii);
        mRefImage = (mRefImage - min(mRefImage(:))) ./ (max(mRefImage(:)) - min(mRefImage(:)));
        tRefImages(:, :, ii) = mRefImage;
    end
end


%% Build Neural Network

numLayers = length(vHiddenLayesSize);

hRegNet = fitnet(vHiddenLayesSize, trainFcn);

for ii = 1:numLayers
    hRegNet.layers{ii}.transferFcn = transferFcn;
end

hRegNet.performParam.regularization = weightsRegFactr;

hRegNet.trainParam.max_fail = numFails;
hRegNet.trainParam.epochs   = numEpochs;

hRegNet.divideParam.trainRatio  = trainRatio;
hRegNet.divideParam.valRatio    = validationRatio;
hRegNet.divideParam.testRatio   = testRatio;


%% Train the Network

mDataSamples    = reshape(tRefImages(:, :, :), [(numRows * numCols), numImages]); %<! Each Example as a Column
mRegVal         = reshape(tRefFeaturesCoord ./ [numCols, numRows], [(numCoord * numFeatures), numImages]);

% Configure Net
hRegNet = configure(hRegNet, mDataSamples, mRegVal);

% Trin Net
switch(useGpu)
    case(OFF)
        [hRegNet, sTrainRecord] = train(hRegNet, mDataSamples, mRegVal);
    case(ON)
        [hRegNet, sTrainRecord] = train(hRegNet, mDataSamples, mRegVal, 'useGPU', 'yes');
end

trainPerfString = ['Train_RMS_', num2str(round(sTrainRecord.best_perf * 1e5), '%03d')];
validPerfString = ['Validation_RMS_', num2str(round(sTrainRecord.best_vperf * 1e5), '%03d')];
testPerfString  = ['Test_RMS_', num2str(round(sTrainRecord.best_tperf * 1e5), '%03d')];

save([dataFolderPath, 'RegNetData_', trainPerfString, '_', validPerfString, '_', testPerfString], 'hRegNet', 'sTrainRecord');


%% Performance Analysis

% Display Training
figure();
plotperform(sTrainRecord);

% Display Train Result
imageIdx    = randi([1, numImages], [1, 1]);
mTestImage  = tRefImages(:, :, imageIdx);

mPredFeatureCoord = hRegNet(mTestImage(:));
mPredFeatureCoord = reshape(mPredFeatureCoord, [numFeatures, 2]);
mPredFeatureCoord = mPredFeatureCoord .* [numCols, numRows];

figure();
imshow(tRefImages(:, :, imageIdx));
hold('on');
plot(mPredFeatureCoord(:, 1), mPredFeatureCoord(:, 2), '*');


%% Prediction

if(runTest)
    
    load([dataFolderPath, testImageFileName]); %<! tTestImage
    numTestImages = size(tTestImage, 3);
    
    if(normalizeImage == ON)
        for ii = 1:numTestImages
            mTestImage = tTestImage(:, :, ii);
            mTestImage = (mTestImage - min(mTestImage(:))) ./ (max(mTestImage(:)) - min(mTestImage(:)));
            tTestImage(:, :, ii) = mTestImage;
        end
    end
    
    mDataSamples    = reshape(tTestImage, [(numRows * numCols), numTestImages]); %<! Each Example as a Column
    
    tPredtFeaturesCoord = hRegNet(mDataSamples);
    tPredtFeaturesCoord = reshape(tPredtFeaturesCoord, [numFeatures, numCoord, numTestImages]);
    tPredtFeaturesCoord = tPredtFeaturesCoord .* [numCols, numRows];
    
    % Display Test Result
    imageIdx    = randi([1, numTestImages], [1, 1]);
    figure();
    imshow(tTestImage(:, :, imageIdx));
    hold('on');
    plot(tPredtFeaturesCoord(:, 1, imageIdx), tPredtFeaturesCoord(:, 2, imageIdx), '*');
    
    save([dataFolderPath, 'RegNetData_', trainPerfString, '_', validPerfString, '_', testPerfString], 'hRegNet', 'sTrainRecord', 'tPredtFeaturesCoord');
    
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

