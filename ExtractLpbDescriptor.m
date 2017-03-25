function [ tLpbDescriptor ] = ExtractLpbDescriptor( tInputImage, numNeighbors, patternRadius, rotVariantFlag, cellSize, verboseMode )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

numRows     = size(tInputImage, 1);
numCols     = size(tInputImage, 2);
numImages   = size(tInputImage, 3);

numCells = prod(floor([numRows, numCols] ./ cellSize));

switch(rotVariantFlag)
    case(FALSE)
        numBins         = numNeighbors + 2;
        rotVariantFlag  = false();
    case(TRUE)
        numBins         = (numNeighbors * (numNeighbors - 1)) + 3;
        rotVariantFlag  = true();
end

lpbDescDim      = numCells * numBins;

tLpbDescriptor = zeros([1, lpbDescDim, numImages]);

runTime = 0;
for ii = 1:numImages
    hProcImageTimer = tic();
    
    tLpbDescriptor(1, :, ii) = extractLBPFeatures(tInputImage(:, :, ii), ...
        'NumNeighbors', numNeighbors, 'Radius', patternRadius, ...
        'Upright', rotVariantFlag, 'CellSize', [cellSize, cellSize]);
    
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

