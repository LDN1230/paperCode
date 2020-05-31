function Ps = pixelInSameSuperpixel(Ps, superpixelMap, currentPosition,  nList)
    % �ҳ��뵱ǰ����ͬһ�����ؿ�ĵ�
    % superpixelMap: �����طָ��Map
    % currentPosition: ��ǰ�������
    % nList: function [numN, nList] = getNeighFromGrid(rows,cols)�����
    % Ps��������������ĵ������
    
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