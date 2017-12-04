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
dataAugmentation    = ON;
netLayerModelIdx    = 2;


generateReflectedImages = OFF;
normalizeImage          = ON;

% Local Binary Pattern (LPB) Descriptor Settings
numNeighbors    = 8;
patternRadius   = 1;
rotVariantFlag  = FALSE;
cellSize        = 8;

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
weightsRegFctr      = 0.035;
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
    
%     figure();
%     imshow(tRefImages(:, :, ii));
%     hold('on');
%     plot(tRefFeaturesCoord(:, 1, ii), tRefFeaturesCoord(:, 2, ii), '*');
%     
%     figure();
%     imshow(tRefImages(:, :, (ii + numImages)));
%     hold('on');
%     plot(tRefFeaturesCoord(:, 1, (ii + numImages)), tRefFeaturesCoord(:, 2, (ii + numImages)), '*');
    
    numImages = 2 * numImages;
end

if(normalizeImage == ON)
    for ii = 1:numImages
        mRefImage = tRefImages(:, :, ii);
        mRefImage = (mRefImage - min(mRefImage(:))) ./ (max(mRefImage(:)) - min(mRefImage(:)));
        tRefImages(:, :, ii) = mRefImage;
    end
end


%% Extract LPB Features

tLpbDescriptor = ExtractLpbDescriptor(tRefImages, numNeighbors, patternRadius, rotVariantFlag, cellSize, ON);


%% Build Neural Network

numLayers = length(vHiddenLayesSize);

hRegNet = fitnet(vHiddenLayesSize, trainFcn);

for ii = 1:numLayers
    hRegNet.layers{ii}.transferFcn = transferFcn;
end

hRegNet.performParam.regularization = weightsRegFctr;

hRegNet.trainParam.max_fail = numFails;
hRegNet.trainParam.epochs   = numEpochs;

hRegNet.divideParam.trainRatio  = trainRatio;
hRegNet.divideParam.valRatio    = validationRatio;
hRegNet.divideParam.testRatio   = testRatio;


%% Train the Network

mDataSamples    = reshape(tRefImages(:, :, :), [(numRows * numCols), numImages]); %<! Each Example as a Column
mDataSamples    = [mDataSamples; permute(tLpbDescriptor, [2, 3, 1])];
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

mPredFeatureCoord = hRegNet([mTestImage(:); ExtractLpbDescriptor(mTestImage, numNeighbors, patternRadius, rotVariantFlag, cellSize, ON).']);
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
    
    tLpbDescriptor = ExtractLpbDescriptor(tTestImage, numNeighbors, patternRadius, rotVariantFlag, cellSize, OFF);
    
    mDataSamples    = reshape(tTestImage, [(numRows * numCols), numTestImages]); %<! Each Example as a Column
    mDataSamples    = [mDataSamples; permute(tLpbDescriptor, [2, 3, 1])];
    
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

