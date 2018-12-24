function [Video] = ReadDataFromImage(ImagePath,w,h)
    ImageList=dir(ImagePath);
    Video=zeros(length(ImageList)-2,w,h,3);
    for j=3:length(ImageList)
        Image = imread(fullfile(ImagePath,ImageList(j).name));
        Image = imresize(Image,[h,w]);
        Video(j-2,:,:,:) = Image; 
    end
end

