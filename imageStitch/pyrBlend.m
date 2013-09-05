function imgo = pyrBlend(imga, imgb, maska, level)

    if any(size(imgb) ~= size(maska))
        imgb = imresize(imgb, [size(imga, 1) size(imga, 2)]);
    end
    assert(all(size(imga) == size(imgb)) && all(size(imga) == size(maska)));

    %[M N ~] = size(imga);

    limga = genPyr(imga,'lap',level); % the Laplacian pyramid
    limgb = genPyr(imgb,'lap',level);

    maskb = 1-maska;
    blurh = fspecial('gauss',30,15); % feather the border
    maska = imfilter(maska,blurh,'replicate');
    maskb = imfilter(maskb,blurh,'replicate');

    limgo = cell(1,level); % the blended pyramid
    for p = 1:level
        [Mp Np ~] = size(limga{p});
    	maskap = imresize(maska,[Mp Np]);
        maskbp = imresize(maskb,[Mp Np]);
        limgo{p} = limga{p}.*maskap + limgb{p}.*maskbp;
    end
    imgo = min(1, max(double(pyrReconstruct(limgo))/255, 0));
    %figure,imagesc(imgo); % blend by pyramid
    
    %imgo1 = maska.*imga+maskb.*imgb;
    %figure,imshow(imgo1) % blend by feathering
end