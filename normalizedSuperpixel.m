function [mergedSuperpixel, n] = normalizedSuperpixel(segment_result, nList)
% ��segment_result�ĳ�����ÿ�������ؿ鸳��Ψһ�ı�ǩ
% mergedSuperpixel: ����ĳ�����ͼ��
% n: mergedSuperpixel�ܵĳ���������

    [nRow, nCol] = size(segment_result);
    mergedSuperpixel = zeros(nRow, nCol);
    n = 0;
    for i = 1: nRow
        for j = 1: nCol
                if segment_result(i, j) ~= 0
                        n = n+1;
                        currentIndex = sub2ind([nRow nCol], i, j);
                        indexes = findPixelInSameSuperpixel(segment_result, currentIndex, nList);
                        for k = 1: length(indexes)
                            index = indexes(k);
                            [r, c] = ind2sub([nRow nCol], index);
                            segment_result(r, c) = 0;
                            mergedSuperpixel(r, c) = n;
                        end   
                end
        end
    end
end