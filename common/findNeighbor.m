function neighbors = findNeighbor(img, position)
    % �ҳ�img������Ϊposition�ĵ������4�������
    
    [r, c] = size(img);
    total = r*c;
    len = length(position);
    neighbors = zeros(len,5); % ��1���ǵ�ǰ�������������4��������������������4������0���
    for i = 1:len
        index = position(i);
        neighbors_temp = zeros(1,5);
        neighbors_temp(1,1) = index;
        n = 2;
        if mod(index,r)>1 && (img(index-1)~=img(index))% ���˵�һ������
            neighbors_temp(1,n) = index-1; 
            n = n+1;
        end
        if index+r<=total && (img(index+r)~=img(index))% �������һ������
            neighbors_temp(1,n) = index+r;
            n = n+1;
        end
        if mod(index,r)>0 && (img(index+1)~=img(index))% �������һ������
            neighbors_temp(1,n) = index+1;
            n = n+1;
        end
        if index-r>=1 && (img(index-r)~=img(index))% ���˵�һ������
            neighbors_temp(1,n) = index-r;
            n = n+1;
        end
        neighbors(i,:) = neighbors_temp;
    end
end