% Adaptive Signal Processing - Exercise 001
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


%% Load Data

load([dataFolderPath, refImageFileName]); %<! tRefImage
load([dataFolderPath, refFeaturesCoordFileName]); %<! tRefFeaturesCoord
load([dataFolderPath, featuresNameFileName]); %<! cFeaturesName


%% Data Parameters

numFeatures = size(tRefFeaturesCoord, 1);
numImages   = size(tRefFeaturesCoord, 3);
numRows     = size(tRefImages, 1);
numCols     = size(tRefImages, 2);

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
FEATURE_IDX_MOUTH_CENTERT_TOP_LIP       = 14;
FEATURE_IDX_MOUTH_CENTERT_BUTTOM_LIP    = 15;


%% Analysis Settings

featureIdx          = FEATURE_IDX_NOSE_TIP;
numImagesAnalyze    = 2000;
refImageStartIdx    = 1;


%% Feature Location Analysis

mFeatureCoord       = zeros([numImagesAnalyze, 2], 'single');
vImagesAnalyzIdx    = refImageStartIdx:(refImageStartIdx + numImagesAnalyze - 1);

for ii = 1:numImagesAnalyze
    imageIdx                = refImageStartIdx + ii - 1;
    mFeatureCoord(ii, :)    = round(tRefFeaturesCoord(featureIdx, :, imageIdx)); %<! Rounding has RMSE of ~0.2
end

mMeanImage = mean(tRefImages(:, :, vImagesAnalyzIdx), 3);

supportRectRowStartIdx  = min(mFeatureCoord(:, 2));
supportRectRowEndIdx    = max(mFeatureCoord(:, 2));
supportRectColStartIdx  = min(mFeatureCoord(:, 1));
supportRectColEndIdx    = max(mFeatureCoord(:, 1));

vRectSize       = [supportRectRowEndIdx - supportRectRowStartIdx + 1, supportRectColEndIdx - supportRectColStartIdx + 1];
numPixelsRect   = (supportRectRowEndIdx - supportRectRowStartIdx + 1) * (supportRectColEndIdx - supportRectColStartIdx + 1);

mFeatureCoordRel    = mFeatureCoord - [supportRectColStartIdx - 1, supportRectRowStartIdx - 1];
vFeaturePxIdxRel    = sub2ind(vRectSize, mFeatureCoordRel(:, 2), mFeatureCoordRel(:, 1));
vBinCounts          = hist(vFeaturePxIdxRel, 1:numPixelsRect);

% TODO: Remove outliers

vSupportRectTopLeft     = [supportRectColStartIdx, supportRectRowStartIdx];
vSupportRectBottomLeft  = [supportRectColStartIdx, supportRectRowEndIdx];
vSupportRectTopRight    = [supportRectColEndIdx, supportRectRowStartIdx];
vSupportRectBottomRight = [supportRectColEndIdx, supportRectRowEndIdx];

mPlotRect = [vSupportRectTopLeft; vSupportRectBottomLeft; vSupportRectBottomRight; vSupportRectTopRight; vSupportRectTopLeft];

% Displaying Results
figureIdx   = figureIdx + 1;
hFigure     = figure('Position', figPosMedium);
hAxes       = axes();
set(hAxes, 'DataAspectRatio', [1, 1, 1], 'NextPlot', 'add');
set(hAxes, 'YDir', 'reverse');
set(hAxes, 'XLim', [0.5, numCols + 0.5], 'Ylim', [0.5, numRows + 0.5]);
hImageObject = image(repmat(mMeanImage,  [1, 1, 3]));
hLineSeries = line(mFeatureCoord(:, 1), mFeatureCoord(:, 2));
set(hLineSeries, 'LineStyle', 'none');
set(hLineSeries, 'Marker', 'o');
% set(hAxes, 'ColorOrderIndex', 2);
hLineSeries = line(mPlotRect(:, 1), mPlotRect(:, 2));
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(2, :));

figureIdx   = figureIdx + 1;
hFigure     = figure('Position', figPosMedium);
hAxes       = axes();
hBarObjext = bar(1:numPixelsRect, vBinCounts);
set(get(hAxes, 'Title'), 'String', ['Feature Pixel Index Histogram'], ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', 'Pixel Index', ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', 'Counts', ...
    'FontSize', fontSizeAxis);


%% Build Neural Network

vHiddenLayesSize    = [10];
trainFcn            = 'traingdm';

% numInputs = numRows * numCols;
% numLayers = 2;

hRegNet = fitnet(vHiddenLayesSize, trainFcn);

% hRegNet.layers{1}.transferFcn = 'tansig';
% hRegNet.layers{1}.transferFcn = 'logsig';
% hRegNet.layers{1}.transferFcn = 'poslin'; %<! RELU

% hRegNet.performParam.regularization = 0.1;

hRegNet.trainParam.max_fail = 45;
hRegNet.trainParam.epochs = 750;


%% Train the Network

mDataSamples    = reshape(tRefImages(:, :, vImagesAnalyzIdx), [(numRows * numCols), numImagesAnalyze]); %<! Each Example as a Column
mDataSamples    = (mDataSamples - min(mDataSamples, [], 1)) ./ (max(mDataSamples, [], 1) - min(mDataSamples, [], 1));
vRegVal         = vFeaturePxIdxRel ./ numPixelsRect;
vRegVal         = vRegVal(:).'; %<! Expected as a Row Vector

numImagesTrain  = round(0.95 * numImagesAnalyze);

mDataSamplesTrain   = mDataSamples(:, 1:numImagesTrain);
vRegValTrain        = vRegVal(1:numImagesTrain);
mDataSamplesTest    = mDataSamples(:, (numImagesTrain + 1):numImagesAnalyze);
vRegValTest         = vRegVal((numImagesTrain + 1):numImagesAnalyze);


hRegNet = configure(hRegNet, mDataSamplesTrain, vRegValTrain);
hRegNet = train(hRegNet, mDataSamplesTrain, vRegValTrain);

yy = hRegNet(mDataSamplesTest);

vE = yy - mDataSamplesTest;
vE = vE(:) * numPixelsRect;


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

