function [ tHogDescriptor ] = ExtractHogDescriptor( tInputImage, hogCellSize, hogBlockSize, blockOverlap, hogNumBins, verboseMode )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

numRows     = size(tInputImage, 1);
numCols     = size(tInputImage, 2);
numImages   = size(tInputImage, 3);

vNumHogPixels   = floor(([numRows, numCols] ./ hogCellSize - hogBlockSize) ./ (hogBlockSize - blockOverlap) + 1);
hogDescDim      = vNumHogPixels(1) *  vNumHogPixels(2) * hogBlockSize * hogBlockSize * hogNumBins;

tHogDescriptor = zeros([1, hogDescDim, numImages]);

runTime = 0;
for ii = 1:numImages
    hProcImageTimer = tic();
    
    tHogDescriptor(1, :, ii) = extractHOGFeatures(tInputImage(:, :, ii), ...
        'CellSize', [hogCellSize hogCellSize], 'BlockSize', [hogBlockSize, hogBlockSize], ...
        'BlockOverlap', [blockOverlap, blockOverlap], 'NumBins', hogNumBins, 'UseSignedOrientation', false);
    
    procImagTime = toc(hProcImageTimer);
    runTime = runTime + procImagTime;
    
    if(verboseMode == ON)
        disp(['Finished processing Image #', num2str(ii, '%04d'), ' out of ', num2str(numImages), ' images']);
        disp(['Processig Time       - ', num2str(procImagTime, '%08.3f'), ' [Sec]']);
        disp(['Total Run Time       - ', num2str(runTime, '%08.3f'), ' [Sec]']);
        disp(['Expected Run Time    - ', num2str((numImages / ii) * runTime, '%08.3f'), ' [Sec]']);
        disp([' ']);
    end
end


end

