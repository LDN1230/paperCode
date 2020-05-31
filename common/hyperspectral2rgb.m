function RgbImg = hyperspectral2rgb(hyperspectralImg, selectedBands)
    % 将高光谱图像转化为伪彩色图像
    % hyperspectralImg：高光谱图像
    % selectedBands： 选择作为伪彩色图像的三个光谱带[b1 b2 b3]
    
    [nRow, nCol, nBand] = size(hyperspectralImg);
    RgbImg = hyperspectralImg(:,:,selectedBands);
    % 将高光谱图像像素归一化
    for n = 1:1:size(RgbImg,3)
        simpleBand = RgbImg(:,:,n);
        simpleBandVector = simpleBand(:);
        minV = min(simpleBandVector);
        maxV = max(simpleBandVector);
        for i = 1:size(RgbImg,1)
            for j = 1:size(RgbImg,2)
                RgbImg(i,j,n) = (RgbImg(i,j,n)-minV)/(maxV-minV);
            end
        end
    end
    RgbImg = 255*RgbImg;
    RgbImg = uint8(RgbImg);
end