function [ vRectSupport ] = FindRectSupport( mBinsCount, relSumFctr )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

FALSE   = 0;
TRUE    = 1;

rowStartIdx = 1;
rowEndIdx   = size(mBinsCount, 1);
colStartIdx = 1;
colEndIdx   = size(mBinsCount, 2);

totalSum    = sum(mBinsCount(:));
mRelRect    = mBinsCount(rowStartIdx:rowEndIdx, colStartIdx:colEndIdx);

validFlag = TRUE;

% Seems to be like a Dynamic Programming problem
while(validFlag == TRUE)
    mRelRect1 = mBinsCount((rowStartIdx + 1):rowEndIdx, colStartIdx:colEndIdx);
    mRelRect2 = mBinsCount(rowStartIdx:(rowEndIdx - 1), colStartIdx:colEndIdx);
    mRelRect3 = mBinsCount(rowStartIdx:rowEndIdx, (colStartIdx + 1):colEndIdx);
    mRelRect4 = mBinsCount(rowStartIdx:rowEndIdx, colStartIdx:(colEndIdx - 1));
    
    relSum1 = sum(mRelRect1(:));
    relSum2 = sum(mRelRect2(:));
    relSum3 = sum(mRelRect3(:));
    relSum4 = sum(mRelRect4(:));
    
    numElmnts1 = numel(mRelRect1);
    numElmnts2 = numel(mRelRect2);
    numElmnts3 = numel(mRelRect3);
    numElmnts4 = numel(mRelRect4);
    
    relMean1 = relSum1 ./ numElmnts1;
    relMean2 = relSum2 ./ numElmnts2;
    relMean3 = relSum3 ./ numElmnts3;
    relMean4 = relSum4 ./ numElmnts4;
    
    vRelMean    = [relMean1, relMean2, relMean3, relMean4];
    vRelSum     = [relSum1, relSum2, relSum3, relSum4];
    [~, vSortIdx] = sort(vRelMean, 'descend');
    
    validFlag = FALSE;
    for ii = 1:4
        relSum = vRelSum(vSortIdx(ii));
        if((relSum / totalSum) > relSumFctr)
            validFlag = TRUE;
            switch(vSortIdx(ii))
                case(1)
                    rowStartIdx = rowStartIdx + 1;
                case(2)
                    rowEndIdx = rowEndIdx - 1;
                case(3)
                    colStartIdx = colStartIdx + 1;
                case(4)
                    colEndIdx = colEndIdx - 1;
            end
        end
        if(validFlag == TRUE)
            break;
        end
    end
    
end

vRectSupport = [rowStartIdx, rowEndIdx, colStartIdx, colEndIdx];


end

