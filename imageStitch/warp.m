function warp(homoInfo, images)

    nImages = length(images);
    assert(length(homoInfo) == nImages);
    
    % image space for mosaic
    bbox = getBoundingBox([0 0 0 0], images, homoInfo);
    %bbox = bbox + [-30 -20 10 250];
    %bbox = bbox + [-30 200 -30 -120];
    bbox = bbox + [-800 -1000 100 800];
    
    % warp image 1 to mosaic image
    ImWarpMax = [];
    for i=1:nImages
        Imw = vgg_warp_H(double(images{i}), homoInfo{i}, 'linear', bbox);
        Imw(isnan(Imw)) = 0;
        if isempty(ImWarpMax)
            ImWarpMax = Imw;
        end
        ImWarpMax = max(ImWarpMax, Imw);
    end
    ImWarpMax = ImWarpMax / 255;
    figure, imagesc(ImWarpMax);
    
    %{
    mask1 = double(sum(Im2w, 3) > 0);
    mask1 = cat(3, mask1, mask1, mask1);
    Im1 = pyrBlend(Im2w, Im1w, mask1, 5);
    figure, imagesc(Im1);

    mask1 = double(sum(Im1, 3) > 0);
    mask1 = cat(3, mask1, mask1, mask1);
    Im2 = pyrBlend(Im1, Im3w, mask1, 5);
    figure, imagesc(max(Im1, Im2));
    %}
end

function newBox = getBoundingBox(bbox, images, H)
    assert(length(images) == length(H));
    n = length(images);
    newBox = bbox;
    for i=1:n
        [w, h, ~] = size(images{i});
        imgBox = H{i}*[1 w 1 w; 1 1 h h; 1 1 1 1];
        imgBox = [imgBox(1, :) ./ imgBox(3, :); ...
                  imgBox(2, :) ./ imgBox(3, :)];
        newBox = [  ceil(min(newBox(1), min(imgBox(1, :)))) ...
                    ceil(max(newBox(2), max(imgBox(1, :)))) ...
                    ceil(min(newBox(3), min(imgBox(2, :)))) ...
                    ceil(max(newBox(4), max(imgBox(2, :))))];
    end
end