function RgbImg = hyperspectral2rgb(hyperspectralImg, selectedBands)
    % ���߹���ͼ��ת��Ϊα��ɫͼ��
    % hyperspectralImg���߹���ͼ��
    % selectedBands�� ѡ����Ϊα��ɫͼ����������״�[b1 b2 b3]
    
    [nRow, nCol, nBand] = size(hyperspectralImg);
    RgbImg = hyperspectralImg(:,:,selectedBands);
    % ���߹���ͼ�����ع�һ��
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