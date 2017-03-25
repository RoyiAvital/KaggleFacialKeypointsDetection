% Facial Keypoints Detection - Neural Network Analysis
% References:
%   1.  Ensemble Methods - https://www.mathworks.com/help/stats/ensemble-methods.html.
%   2.  fitrensemble - https://www.mathworks.com/help/stats/fitrensemble.html.
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

FEATURE_IDX_LEFT_EYE_CENTER             = 1;
FEATURE_IDX_RIGHT_EYE_CENTER            = 2;
FEATURE_IDX_LEFT_EYE_INNER_CORNER       = 3;
FEATURE_IDX_LEFT_EYE_OUTER_CORNER       = 4;
FEATURE_IDX_RIGHT_EYE_INNER_CORNER      = 5;
FEATURE_IDX_RIGHT_EYE_OUTER_CORNER      = 6;
FEATURE_IDX_LEFT_EYEBROW_INNER_END      = 7;
FEATURE_IDX_LEFT_EYEBROW_OUTER_END      = 8;
FEATURE_IDX_RIGHT_EYEBROW_INNER_END     = 9;
FEATURE_IDX_RIGHT_EYEBROW_OUTER_END     = 10;
FEATURE_IDX_NOSE_TIP                    = 11;
FEATURE_IDX_MOUTH_LEFT_CORNER           = 12;
FEATURE_IDX_MOUTH_RIGHT_CORNER          = 13;
FEATURE_IDX_MOUTH_CENTER_TOP_LIP        = 14;
FEATURE_IDX_MOUTH_CENTER_BUTTOM_LIP     = 15;


%% Analysis Settings

generateReflectedImages = OFF;
trainImagesRatio        = 0.95;
normalizeImage          = ON;

% Radnom Forest Settings

featureIdx = FEATURE_IDX_MOUTH_LEFT_CORNER;

trainMethod         = 'LSBoost';
numLearningCycles   = 50;
learnRate           = 0.1;

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
        tRefFeaturesCoord(:, 1, (ii + numImages))   = tRefFeaturesCoord(:, 2, ii);
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




%% Train Radnom Forest

mDataSamples    = permute(reshape(tRefImages(:, :, :), [(numRows * numCols), numImages]), [2, 1]); %<! Each Example as a Row
vRegValX        = reshape(tRefFeaturesCoord(featureIdx, 1, :) ./ numCols, [numImages, 1]);
vRegValY        = reshape(tRefFeaturesCoord(featureIdx, 2, :) ./ numRows, [numImages, 1]);


hEnsRegTreeMdlX = fitrensemble(mDataSamples, vRegValX, 'Method', trainMethod, 'NumLearningCycles', numLearningCycles, 'LearnRate', learnRate);
hEnsRegTreeMdlY = fitrensemble(mDataSamples, vRegValY, 'Method', trainMethod, 'NumLearningCycles', numLearningCycles, 'LearnRate', learnRate);

vRegValPredX  = predict(hEnsRegTreeMdlX, mDataSamples);
vRegValPredY  = predict(hEnsRegTreeMdlY, mDataSamples);



%% Train the Network



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

