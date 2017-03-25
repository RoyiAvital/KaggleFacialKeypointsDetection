% Convert Data to MAT
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     23/03/2017  Royi Avital
%   *   First release.
%

%% General Parameters

run('InitScript.m');

addpath(genpath('./AuxiliaryFunctions'));

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

cRawData = csvimport([dataFolderPath, trainDataFileName]);
numImages = size(cRawData, 1) - 1; %<! Header row

cFeaturesName = cRawData(1, (1:(2 * numFeatures)));
cFeaturesName = cFeaturesName(:);

tRefImages          = zeros([numRows, numCols, numImages], 'single'); %<! Data is in UINT8
tRefFeaturesCoord   = zeros([numFeatures, 2, numImages], 'single'); %<! [featureIdx, imageIdx, xyIdx]

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
                    tRefImages(iRow, jCol, ii) = single(str2double(cImageData{pxIdx})) / 255;
                end
            end
        elseif(~isempty(cRawData{imageIdx, jj}))
            featureIdx = ceil(jj / 2);
            coordIx = mod(jj, 2) + ((1 - mod(jj, 2)) * 2);
            if(ischar(cRawData{imageIdx, jj}))
            tRefFeaturesCoord(featureIdx, coordIx, ii) = single(str2double(cRawData{imageIdx, jj}));
            end
            if(isnumeric(cRawData{imageIdx, jj}))
                tRefFeaturesCoord(featureIdx, coordIx, ii) = single(cRawData{imageIdx, jj});
            end
        else
            tRefFeaturesCoord(featureIdx, coordIx, ii) = single(0);
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
    if(any(tRefFeaturesCoord(:, :, ii) == 0))
        vInvalidImageIdx = [vInvalidImageIdx, ii];
    end
end

tRefFeaturesCoord(:, :, vInvalidImageIdx) = [];
tRefImages(:, :, vInvalidImageIdx)        = [];



save([dataFolderPath, 'cFeaturesName'], 'cFeaturesName');
save([dataFolderPath, 'tRefFeaturesCoord'], 'tRefFeaturesCoord');
save([dataFolderPath, 'tRefImages'], 'tRefImages');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

