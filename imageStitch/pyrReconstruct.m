function [ img ] = pyrReconstruct( pyr )
%PYRRECONSTRUCT Uses a Laplacian pyramid to reconstruct a image
%   IMG = PYRRECONSTRUCT(PYR) PYR should be a 1*level cell array containing
%   the pyramid, SIZE(PYR{i}) = SIZE(PYR{i-1})*2-1
%		Yan Ke @ THUEE, xjed09@gmail.com

for p = length(pyr)-1:-1:1
	pyr{p} = pyr{p}+pyr_expand(pyr{p+1});
end
img = pyr{1};

end

