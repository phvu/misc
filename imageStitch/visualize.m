function visualize(im1rgb, im1_pts, im2_pts, inliers, sTitle)
    figure('Name', sTitle);
    
    subplot 121;
    imagesc(im1rgb);
    hold on;
    
    % show features detected in image 1
    plot(im1_pts(1,:),im1_pts(2,:),'+g');

    % show displacements
    line([im1_pts(1,:); im2_pts(1,:)], ...
        [im1_pts(2,:); im2_pts(2,:)],'color','y');
    title(gca, 'Tentative correspondences');
    %set(gca, 'Title', text('String','Tentative correspondences','Color','r'));
    hold off;
    
    subplot 122;
    imagesc(im1rgb);
    hold on;
    
    % show features detected in image 1
    plot(im1_pts(1,inliers),im1_pts(2,inliers),'+g');

    % show displacements
    line([im1_pts(1,inliers); im2_pts(1,inliers)], ...
        [im1_pts(2,inliers); im2_pts(2,inliers)],'color','y');
    title(gca, 'Inliers');
    %set(gca, 'Title', text('String','Inliers','Color','r'));
    hold off;
end