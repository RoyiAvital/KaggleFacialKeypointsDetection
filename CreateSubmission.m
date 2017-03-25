% Facial Keypoints Detection - Create Submission
% References:
%   1.  https://www.kaggle.com/c/facial-keypoints-detection/discussion/4960
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

dataFolderPath              = 'Data/';
idLookUpTableFileName       = 'cIdLookUpTable.mat';
featuresNameFileName        = 'cFeaturesName.mat';
predictorFileName           = 'RegNetData_Train_RMS_010_Validation_RMS_036_Test_RMS_NaN.mat';

fileName        = 'SubmissionData.csv';
headerRowString = 'RowId,Location';


%% Loading Data

load([dataFolderPath, idLookUpTableFileName]); %<! cIdLookUpTable
load([dataFolderPath, featuresNameFileName]); %<! cFeaturesName
load([dataFolderPath, predictorFileName]); %<! tPredtFeaturesCoord

numRows         = size(cIdLookUpTable, 1) - 1;
numFeaturesName = size(cFeaturesName, 1);


%% Writing Data

hFileId = fopen([dataFolderPath, fileName], 'w');
fprintf(hFileId, [headerRowString, '\n']);

for ii = 1:numRows
    lookUpTableIdx  = ii + 1;
    % imageIdx        = str2double(cIdLookUpTable{lookUpTableIdx, 2});
    imageIdx        = cIdLookUpTable{lookUpTableIdx, 2};
    featureName     = cIdLookUpTable{lookUpTableIdx, 2};
    for jj = 1:numFeaturesName
        if(strcmp(cIdLookUpTable{lookUpTableIdx, 3}, cFeaturesName{jj}))
            featureIdx = jj;
            break;
        end
        if(jj == numFeaturesName)
            disp(['Error, Couldn''t find feature']);
        end
        
    end
    featureRowIdx = ceil(featureIdx / 2);
    featureColIdx = mod(featureIdx, 2) + ((1 - mod(featureIdx, 2)) * 2);
    featureCoord = tPredtFeaturesCoord(featureRowIdx, featureColIdx, imageIdx);
    
    featureCoord = max(min(featureCoord, 96), 1);
    
    fprintf(hFileId, [num2str(ii), ',']);
    fprintf(hFileId, [num2str(featureCoord), '\n']);
end


fclose(hFileId);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

