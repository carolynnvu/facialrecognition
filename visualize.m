function visualize(scores, faces)
%   The function for visualization part, 
%   which put the image at the coordinates given by their coefficients of 
%   the first two principal components (with translation and scaling).
% 
%   scores: n x 2 array, where each row contains the first 2 principal component scores of each face
%   faces: n x 4096 array
    scores = scores(:, 1:2);
    min_score = min(scores);
    max_score = max(scores);
    scores = 800*bsxfun(@rdivide, bsxfun(@minus, scores, min_score), (max_score-min_score));
    colormap(gray); 
    box on; axis ij; hold on
    for i = 1:size(faces, 1)
        imagesc(scores(i, 1), scores(i, 2), reshape(faces(i, :), 64, 64));
    end
    hold off
end

