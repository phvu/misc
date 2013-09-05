function [H, corres1, corres2, inlinersIdx ] = ...
    getHomography(imgFeatures1, imgLoc1, imgFeatures2, imgLoc2)
%%%%
% H - homography 3x3 matrix
% corres1 - Nx2 matrix, location of N tentative correspondences
% corres2 - Nx2 matrix, location of N tentative correspondences
% inlinersIdx - Mx1 vector, index of inliners

    MATCH_THRESHOLD = 0.5;
    RANSAC_N = 200;
    RANSAC_EPSILON = 3;
    RANSAC_FINISH_THRES = 0.6;
    
    matches = match(imgFeatures1, imgFeatures2, MATCH_THRESHOLD);
    corres1 = imgLoc1(matches(:, 1), :);
    corres2 = imgLoc2(matches(:, 2), :);
    [H, inlinersIdx] = ransac(corres1, corres2, ...
        RANSAC_N, RANSAC_EPSILON, RANSAC_FINISH_THRES);

    if isempty(H)
        error('stitch:Homography', 'Impossible to compute Homography');
    end
end