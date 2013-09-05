function matches = match(imgFeatures1, imgFeatures2, matchThres)
% imgFeatures1 - M x N with M feature vectors
% imgFeatures2 - L x N with L feature vectors

    L = size(imgFeatures2, 1);
    dist = dist2(imgFeatures1, imgFeatures2);
    
    % sort by columns
    [dist, idx] = sort(dist, 'ascend');
    
    selected = (dist(1, :) ./ dist(2, :)) < matchThres;
    matches = [];
    
    if (sum(selected) > 0)
        matches = zeros(sum(selected), 2);
        tmpIdx = 1:L;
        matches(:, 2) = tmpIdx(selected);
        matches(:, 1) = idx(1, selected);
    end
end