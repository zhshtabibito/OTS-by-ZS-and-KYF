close all; clear; clc;
% config
nBlk = 20;
threshold = 0.6;
videoPaths = dir('F:\\ÊÓÌýÊý¾Ý\\dataset\\images\\solo\\*\\');
for nP = 1:length(videoPaths)
    clear('imgs');
    if(isequal(videoPaths(nP).name,'.')||... 
       isequal(videoPaths(nP).name,'..')||...
       ~videoPaths(nP).isdir)
        continue
    end
    if(~isempty(strfind(videoPaths(nP).folder, 'accordion')))
        continue;
    end
    if(~isempty(strfind(videoPaths(nP).folder, 'acoustic_guitar')))
        continue;
    end
    if(~isempty(strfind(videoPaths(nP).folder, 'cello')))
        continue;
    end
    if(~isempty(strfind(videoPaths(nP).folder, 'flute')))
        continue;
    end
    if(~isempty(strfind(videoPaths(nP).folder, 'saxphone')))
        continue;
    end
    videoPath = [videoPaths(nP).folder,'\\',videoPaths(nP).name,'\\'];
    nImgs = length(dir([videoPath '*.jpg']));
    for i = 1:10:nImgs
        imgPath = [videoPath, num2str(i,'%06d'), '.jpg'];
        img = imread(imgPath);
        imgs(:,:,floor((nImgs-i)/10)+1) = double(rgb2gray(img));
    end
    [lenX, lenY, ~] = size(imgs);
    lenBlkX = floor(lenX/nBlk);
    lenBlkY = floor(lenY/nBlk);
    matD = zeros(nBlk, nBlk);
    for m = 1:nBlk
        for n = 1:nBlk
            tmp = zeros(lenBlkX, lenBlkY);
            for x = 1:4:lenBlkX
                for y = 1:4:lenBlkY
                    tmp((x-1)/4+1,(y-1)/4+1) = var(imgs(m*lenBlkX-lenBlkX+x,n*lenBlkY-lenBlkY+y,:));
                end
            end
            matD(m,n) = mean2(tmp);
        end
    end
    maxD = max(max(matD));
    matFlag = matD>0.5*maxD;

    y1 = nBlk; y2=1;
    for m = 1:nBlk
        tmp1 = find(matFlag(m,:),1);
        tmp2 = find(matFlag(m,:),1,'last');
        if(tmp1<y1)
            y1=tmp1;
        end
        if(tmp2>y2)
            y2=tmp2;
        end
    end
    x1 = nBlk; x2=1;
    for n = 1:nBlk
        tmp1 = find(matFlag(:,n),1);
        tmp2 = find(matFlag(:,n),1,'last');
        if(tmp1<x1)
            x1=tmp1;
        end
        if(tmp2>x2)
            x2=tmp2;
        end
    end
    for i = 1:10:nImgs
        imgPath = [videoPath, num2str(i,'%06d'), '.jpg'];
        img0 = imread(imgPath);
        img1 = imcrop(img0, [y1*lenBlkY-lenBlkY+1,x1*lenBlkX-lenBlkX+1,y2*lenBlkY,x2*lenBlkX]);       
        if ~exist(strrep(videoPath, 'images', 'images_cut')) 
            mkdir(strrep(videoPath, 'images', 'images_cut'))
        end 
        imwrite(img1, strrep(imgPath, 'images', 'images_cut'));
    end
    imshow(img1);
end
