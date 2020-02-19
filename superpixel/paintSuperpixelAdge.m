function paintSuperpixelAdge(RgbImg, superpixelLabel)
    % ÔÚÔ­Í¼ÏñÉÏµþ¼Ó»­¸÷¸ö³¬ÏñËØµÄ±ßÔµ
    % RgbImg£º Ô­À´Í¼Ïñ£¨RGBÍ¼Ïñ£©
    % superpixelLabel£º³¬ÏñËØ¿éµÄlabel
    
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
    figure('Name','Í¼Ïñ³¬ÏñËØ±ßÔµ');
    subplot(121);  imshow(RgbImg);
    subplot(122);  imshow(superpixelImg);
end