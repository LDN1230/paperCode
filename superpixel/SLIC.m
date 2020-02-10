function [labels_1, numlabels, seedx, seedy] = SLIC( img, numSuperpixels, compactness, dist_type, seg_all)
    % labels: segmentation results
    % numlabels: the final number of superpixels
    % seedx: the x indexing of seeds
    % seedy: the y indexing of seeds
    
    [i_row, i_col, nb] = size(img);
    % reshape the data into a vector
    input_img = zeros(1, i_row * i_col * nb);
    startpos = 1;
    for i = 1 : i_row % lines
        for j = 1 : i_col % columes
            input_img(startpos : startpos + nb  - 1) = img(i, j, :); % bands
            startpos = startpos + nb;
        end
    end
    [labels_1, numlabels, seedx, seedy] = hybridseg(input_img, i_row, i_col, nb, numSuperpixels, compactness, dist_type, seg_all);%numlabels is the same as number of superpixels

end