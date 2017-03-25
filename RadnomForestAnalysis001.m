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
vFeaturePxIdexRel   = sub2ind(vRectSize, mFeatureCoordRel(:, 2), mFeatureCoordRel(:, 1));
vBinCounts          = hist(vFeaturePxIdexRel, 1:numPixelsRect);

% vRectSupport = FindRectSupport(reshape(vBinCounts, vRectSize), 0.9);
% 
% supportRectRowEndIdx    = vRectSupport(2) + supportRectRowStartIdx - 1;
% supportRectColEndIdx    = vRectSupport(4) + supportRectColStartIdx - 1;
% supportRectRowStartIdx  = vRectSupport(1) + supportRectRowStartIdx - 1;
% supportRectColStartIdx  = vRectSupport(3) + supportRectColStartIdx - 1;
% 
% vRectSize       = [supportRectRowEndIdx - supportRectRowStartIdx + 1, supportRectColEndIdx - supportRectColStartIdx + 1];
% numPixelsRect   = (supportRectRowEndIdx - supportRectRowStartIdx + 1) * (supportRectColEndIdx - supportRectColStartIdx + 1);

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


%% Extracting HOG Descriptor

hogCellSize     = 8;
hogBlockSize    = 2;
hogNumBins      = 9;

hogDescDim = hogBlockSize * hogBlockSize * hogNumBins;

tPxHogDescriptor    = zeros([numPixelsRect, hogDescDim, numImagesAnalyze]);
mPxClass            = zeros([numPixelsRect, numImagesAnalyze]);

runTime = 0;
for ii = 1:numImagesAnalyze
    hProcImageTimer = tic();
    imageIdx    = refImageStartIdx + ii - 1;
    pxIdx       = 0;
    for iRow = supportRectRowStartIdx:supportRectRowEndIdx
        for jCol = supportRectColStartIdx:supportRectColEndIdx
            pxIdx = pxIdx + 1;
            tPxHogDescriptor(pxIdx, :, ii) = extractHOGFeatures(tRefImages(:, :, imageIdx), [jCol, iRow], ...
                'CellSize', [hogCellSize hogCellSize], 'BlockSize', [hogBlockSize, hogBlockSize], 'NumBins', hogNumBins, 'UseSignedOrientation', false);
            if((iRow == mFeatureCoord(ii, 2)) && (jCol == mFeatureCoord(ii, 1)))
                mPxClass(pxIdx, ii) = 1;
            end
        end
    end
    procImagTime = toc(hProcImageTimer);
    runTime = runTime + procImagTime;
    disp(['Finished processing Image #', num2str(ii, '%04d'), ' out of ', num2str(numImagesAnalyze), ' images']);
    disp(['Processig Time       - ', num2str(procImagTime, '%08.3f'), ' [Sec]']);
    disp(['Total Run Time       - ', num2str(runTime, '%08.3f'), ' [Sec]']);
    disp(['Expected Run Time    - ', num2str((numImagesAnalyze / ii) * runTime, '%08.3f'), ' [Sec]']);
    disp([' ']);
end

% mA = [reshape(tPxHogDescriptor, [(numPixelsRect * numImagesAnalyze), hogDescDim]), mPxClass(:)];

% mappedX = tsne(reshape(tPxHogDescriptor, [(numPixelsRect * numImagesAnalyze), hogDescDim]), mPxClass(:), 3, 12, 2);
% figure();
% % gscatter(mappedX(:, 1), mappedX(:, 2), mPxClass(:));
% scatter3(mappedX(:, 1), mappedX(:, 2), mappedX(:, 3), [], mPxClass(:));

% svmOptions = '-s 0 -t 0 -c 1 -e 0.001 -h 1 -b 0 -w0 1 -w1 450';
% hSvmModel = svmtrain(mPxClass(:), mA, svmOptions);


numImagesTrain  = 450;
numImagesTest   = 50;

vRegVal = (vFeaturePxIdexRel - 1) / numPixelsRect;

tPxHogDescriptorTrain   = tPxHogDescriptor(:, :, 1:numImagesTrain);
vRegValTrain            = vRegVal(1:numImagesTrain);

mA = permute(tPxHogDescriptorTrain, [2, 1, 3]);
mA = reshape(mA, [(numPixelsRect * hogDescDim), numImagesTrain]).';

tPxHogDescriptorTest    = tPxHogDescriptor(:, :, (numImagesTrain + 1):(numImagesTrain + numImagesTest));
vRegValTest             = vRegVal((numImagesTrain + 1):(numImagesTrain + numImagesTest));

mB = permute(tPxHogDescriptorTest, [2, 1, 3]);
mB = reshape(mB, [(numPixelsRect * hogDescDim), numImagesTest]).';


% SVM
svmOptions      = '-s 3 -t 0 -c 1 -p 0.1 -e 0.001 -h 1';
hSvmModel       = svmtrain(double(vRegValTrain), mA, svmOptions);
vRegValPredSvm  = svmpredict(double(vRegValTest), mB, hSvmModel);
vErrSvm         = numPixelsRect * (vRegValPredSvm - vRegValTest);

% Random Forests
hEnsRegTreeMdl  = fitrensemble(mA, vRegValTrain, 'Method', 'LSBoost', 'NumLearningCycles', 500);
vRegValPredEns  = predict(hEnsRegTreeMdl, mB);
vErrEns         = numPixelsRect * (vRegValPredEns - vRegValTest);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

