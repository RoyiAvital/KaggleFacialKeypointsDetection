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
testDataFileName    = 'test.csv';

numFeatures = 15;
numRows     = 96;
numCols     = 96;

imageDataIdx = 2; %<! First Index - Image Index


%% Test Data

cRawData    = csvimport([dataFolderPath, testDataFileName]);
numImages   = size(cRawData, 1) - 1; %<! Header row

tTestImage          = zeros([numRows, numCols, numImages], 'single'); %<! Data is in UINT8

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
                    tTestImage(iRow, jCol, ii) = single(str2double(cImageData{pxIdx})) / 255;
                end
            end
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

save([dataFolderPath, 'tTestImage'], 'tTestImage');


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

