function H = computeH(im1_pts, im2_pts)

    n = size(im1_pts, 1);
    assert(size(im2_pts, 1) == n);
    
    A = zeros(2*n, 9);
    for i=1:n
        p1 = im1_pts(i, :);
        p2 = im2_pts(i, :);
        A(2*i-1, :) = [-p1(1) -p1(2) -1 0 0 0 p2(1)*p1(1) p2(1)*p1(2) p2(1)];
        A(2*i, :) = [0 0 0 -p1(1) -p1(2) -1 p2(2)*p1(1) p2(2)*p1(2) p2(2)];
    end
    [~, ~, V] = svd(A);
    x = V(:,end);
    x = x / norm(x);
    %assert(norm(x) == 1);
    H = reshape(x, 3, 3)';
end