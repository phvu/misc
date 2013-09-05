function [H, inliners] = ransac(imgLoc1, imgLoc2, N, ...
    inlinerDistThres, inlinerCntThres)
% imgLoc1 - N x 2 locations of N points
% imgLoc2 - N x 2 locations of N points

    pointCnt = size(imgLoc1, 1);
    assert(size(imgLoc2, 1) == pointCnt);
    
    homoImgLoc1 = [imgLoc1'; ones(1, pointCnt)];
    bestInlinerFrac = -1;
    bestH = [];
    inliners = [];
    for i=1:N
       pointIdx = randsample(pointCnt, 4);
       Hi = computeH(imgLoc1(pointIdx, :), imgLoc2(pointIdx, :));
       
       tmp = Hi * homoImgLoc1;
       tmp = [tmp(1, :) ./ tmp(3, :); tmp(2, :) ./ tmp(3, :)];
       diff = sqrt(sum((tmp - imgLoc2').^2, 1));
       inliners = diff <= inlinerDistThres;
       inlinerFrac = sum(inliners) / pointCnt;
       if inlinerFrac >= inlinerCntThres
           bestH = Hi;
           bestInlinerFrac = inlinerFrac;
           break;
       end
       if inlinerFrac >= bestInlinerFrac
           bestInlinerFrac = inlinerFrac;
           bestH = Hi;
       end
    end
    
    if bestInlinerFrac < inlinerCntThres 
        fprintf( ...
            sprintf('RANSAC can only match %f, while threshold is %f\n', ...
            bestInlinerFrac,  inlinerCntThres));
    end
    % re-estimate
    if isempty(bestH)
        H = [];
    else
        H = computeH(imgLoc1(inliners, :), imgLoc2(inliners, :));
    end
end