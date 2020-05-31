function Ps = pixelInSameSuperpixel(Ps, superpixelMap, currentPosition,  nList)
    % 找出与当前点在同一超像素块的点
    % superpixelMap: 超像素分割的Map
    % currentPosition: 当前点的索引
    % nList: function [numN, nList] = getNeighFromGrid(rows,cols)的输出
    % Ps：输出满足条件的点的索引
    
    currentLabel = superpixelMap(currentPosition);
    neighbors = nList(currentPosition, :);
    for i = 1: length(neighbors)
        neigh = neighbors(i);
        if neigh~=0 && superpixelMap(uint32(neigh)) == currentLabel && isempty(intersect(Ps, neigh))
            Ps = [Ps neigh];
            Ps = pixelInSameSuperpixel(Ps, superpixelMap, neigh,  nList);
            
        end
    end
end