function neighbors = findNeighbor(img, position)
    % 找出img中索引为position的点的邻域4点的索引
    
    [r, c] = size(img);
    total = r*c;
    len = length(position);
    neighbors = zeros(len,5); % 第1列是当前点的索引，其他4列是邻域点的索引，不够4个邻域，0填充
    for i = 1:len
        index = position(i);
        neighbors_temp = zeros(1,5);
        neighbors_temp(1,1) = index;
        n = 2;
        if mod(index,r)>1 && (img(index-1)~=img(index))% 除了第一行以外
            neighbors_temp(1,n) = index-1; 
            n = n+1;
        end
        if index+r<=total && (img(index+r)~=img(index))% 除了最后一列以外
            neighbors_temp(1,n) = index+r;
            n = n+1;
        end
        if mod(index,r)>0 && (img(index+1)~=img(index))% 除了最后一行以外
            neighbors_temp(1,n) = index+1;
            n = n+1;
        end
        if index-r>=1 && (img(index-r)~=img(index))% 除了第一列以外
            neighbors_temp(1,n) = index-r;
            n = n+1;
        end
        neighbors(i,:) = neighbors_temp;
    end
end