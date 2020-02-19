function paintSuperpixelAdge(RgbImg, superpixelLabel)
    % ��ԭͼ���ϵ��ӻ����������صı�Ե
    % RgbImg�� ԭ��ͼ��RGBͼ��
    % superpixelLabel�������ؿ��label
    
    [nRow, nCol] = size(superpixelLabel);
    superpixelImg = RgbImg;
    for i = 1: nRow
        for j = 1: nCol
            if i > 1 && superpixelLabel(i-1, j) ~= superpixelLabel(i, j)
                superpixelImg(i, j, 1) = 255; superpixelImg(i, j, 2) = 0; superpixelImg(i, j, 3) = 0;
            else
                if j > 1 && superpixelLabel(i, j-1) ~= superpixelLabel(i, j)
                    superpixelImg(i, j, 1) = 255; superpixelImg(i, j, 2) = 0; superpixelImg(i, j, 3) = 0;
                else
                    if i < nRow && superpixelLabel(i+1, j) ~= superpixelLabel(i, j)
                        superpixelImg(i, j, 1) = 255; superpixelImg(i, j, 2) = 0; superpixelImg(i, j, 3) = 0;
                    else
                        if j < nCol && superpixelLabel(i, j+1) ~= superpixelLabel(i, j)
                            superpixelImg(i, j, 1) = 255; superpixelImg(i, j, 2) = 0; superpixelImg(i, j, 3) = 0;
                        end
                    end
                end
            end
        end
    end
    figure('Name','ͼ�����ر�Ե');
    subplot(121);  imshow(RgbImg);
    subplot(122);  imshow(superpixelImg);
end