function homoInfo = computeHomoInfo(sifts, siftLoc, images)

    nImages = length(sifts);

    assert(length(siftLoc) == nImages);
    assert(length(images) == nImages);
    
    iReference = ceil(nImages / 2);
    homoInfo = cell(nImages, 1);
    
    % compute homo for neighbors
    for i=1:nImages
        if i == iReference
            H = eye(3);
        else
            if i < iReference
                j = i + 1;
            else
                j = i - 1;
            end
            [H, corres_1, corres_2, inlierIdx] = ...
            getHomography(sifts{i}, siftLoc{i}, sifts{j}, siftLoc{j});
        
            %homoStruct.corres1 = corres_1;
            %homoStruct.corres2 = corres_2;
            %homoStruct.inlierIdx = inlierIdx;
            visualize(images{i}, corres_1', corres_2', inlierIdx, ...
                sprintf('Image %d - %d', i, j));
        end
        homoInfo{i} = H;
    end
    
    % update homo
    for i=iReference-2:-1:1
        homoInfo{i} = homoInfo{i} * homoInfo{i + 1};
    end
    for i=iReference+2:nImages
        homoInfo{i} = homoInfo{i} * homoInfo{i - 1};
    end 
end