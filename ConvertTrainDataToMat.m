% Convert Data to MATLAB (MAT) Format
% Remarks:
%   1.  sa
% TODO:
% 	1.  Add vector of number of images per feature to desginate each image
%       for validity for training for specific feature.
% Release Notes
% - 1.1.000     04/12/2017  Royi Avital
%   *   Added 'mTrainFeatureValid' to flag which image is valid for which
%       feature training.
% - 1.0.000     23/03/2017  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

addpath(genpath('./AuxiliaryFunctions'));

INVALID_VAL = -999;

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Data Parameters

dataFolderPath      = './Data/';
trainDataFileName   = 'training.csv';
testDataFileName    = 'test.csv';

numFeatures = 15;
numRows     = 96;
numCols     = 96;

imageDataIdx = ((2 * numFeatures) + 1); %<! X, Y per Features, Image Data


%% Training Data

hCsvParsingTimer    = tic();
cRawData            = csvimport([dataFolderPath, trainDataFileName]);
csvParsingTime      = toc(hCsvParsingTimer);

disp(['Parsing CSV Run Time - ', num2str(csvParsingTime), ' [Sec]']);

numImages       = size(cRawData, 1) - 1; %<! Header row
cFeaturesName   = cRawData(1, (1:(2 * numFeatures)));
cFeaturesName   = cFeaturesName(:);

tTrainImages        = zeros([numRows, numCols, numImages], 'single'); %<! Data is in UINT8
tTrainFeaturesCoord = zeros([numFeatures, 2, numImages], 'single'); %<! [featureIdx, xyIdx, imageIdx]
mTrainFeatureFlag   = ones([numFeatures, numImages]);

runTime = 0;

for ii = 1:numImages
    hProcImageTimer = tic();
    imageIdx = ii + 1;
    for jj = 1:imageDataIdx
        if(jj == imageDataIdx)
            cImageData = strsplit(cRawData{imageIdx, jj});
            for iRow = 1:numRows
                for jCol = 1:numCols
                    pxIdx = ((iRow - 1) * numCols) + jCol;
                    tTrainImages(iRow, jCol, ii) = single(str2double(cImageData{pxIdx})) / 255;
                end
            end
        elseif(~isempty(cRawData{imageIdx, jj}))
            featureIdx = ceil(jj / 2);
            coordIx = mod(jj, 2) + ((1 - mod(jj, 2)) * 2);
            if(ischar(cRawData{imageIdx, jj}))
                tTrainFeaturesCoord(featureIdx, coordIx, ii) = single(str2double(cRawData{imageIdx, jj}));
            end
            if(isnumeric(cRawData{imageIdx, jj}))
                tTrainFeaturesCoord(featureIdx, coordIx, ii) = single(cRawData{imageIdx, jj});
            end
        else
            tTrainFeaturesCoord(featureIdx, coordIx, ii) = single(INVALID_VAL);
            mTrainFeatureFlag(featureIdx, ii) = 0; %<! Image is invalid for training on this feature
        end
    end
    procImagTime = toc(hProcImageTimer);
    runTime = runTime + procImagTime;
    disp(['Finished processing Image #', num2str(ii, '%04d'), ' out of ', num2str(numImages), ' images']);
    disp(['Processig Time       - ', num2str(procImagTime, '%08.3f'), ' [Sec]']);
    disp(['Total Run Time       - ', num2str(runTime, '%08.3f'), ' [Sec]']);
    disp(['Expected Run Time    - ', num2str((numImages / ii) * runTime, '%08.3f'), ' [Sec]']);
    disp([' ']);
end

vInvalidImageIdx = [];
for ii = 1:numImages
    mTrainFeaturesCoord = tTrainFeaturesCoord(:, :, ii);
    if(any(mTrainFeaturesCoord(:) == INVALID_VAL))
        vInvalidImageIdx = [vInvalidImageIdx, ii];
    end
end

tTrainFeaturesCoordValid    = tTrainFeaturesCoord;
tTrainImagesValid           = tTrainImages;

tTrainFeaturesCoordValid(:, :, vInvalidImageIdx) = [];
tTrainImagesValid(:, :, vInvalidImageIdx)        = [];

save([dataFolderPath, 'cFeaturesName'], 'cFeaturesName');
save([dataFolderPath, 'tTrainFeaturesCoord'], 'tTrainFeaturesCoord');
save([dataFolderPath, 'tTrainImages'], 'tTrainImages');
save([dataFolderPath, 'mTrainFeatureFlag'], 'mTrainFeatureFlag');

mTrainFeatureFlag   = ones([numFeatures, numImages]);
tTrainFeaturesCoord = tTrainFeaturesCoordValid;
tTrainImages        = tTrainImagesValid;

save([dataFolderPath, 'tTrainFeaturesCoordValid'], 'tTrainFeaturesCoord');
save([dataFolderPath, 'tTrainImagesValid'], 'tTrainImages');
save([dataFolderPath, 'mTrainFeatureFlagValid'], 'mTrainFeatureFlag');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

