function index = findPixelInSameSuperpixel(superpixel_label, currentIndex, nList)
        index = [currentIndex];
        postIndex = index;
        preIndex = [];
        label = superpixel_label(currentIndex);
        while ~isequal(preIndex, postIndex)
                list = nList(postIndex, :);
                list1 = unique(list);
                if list1(1) == 0
                    list1 = list1(2:end);
                end
                preIndex = postIndex;
                for i = 1: length(list1)
                    if superpixel_label(list1(i)) == label
                            postIndex =[postIndex list1(i)];
                    end
                end
                postIndex = unique(postIndex);
        end
       index = postIndex;
end