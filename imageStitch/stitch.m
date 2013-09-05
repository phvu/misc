function stitch(imgFiles)
   
    % compute sift feature vectors
    info = getSift(imgFiles);
    
    homoInfo = computeHomoInfo(info.features, info.featureLocations, info.images);
    
    warp(homoInfo, info.images);
end