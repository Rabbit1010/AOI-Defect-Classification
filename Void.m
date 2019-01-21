% Read image
fid = fopen('train.csv');
readheader = textscan(fid,'%s %s',2529,'Headerlines',0,'Delimiter',',');

cd 'F:\影像處理final\train_images';


list = string(readheader{1,1});
number = str2double(string(readheader{1,2}));

[train, map] = imread(char(list(22)));
figure
imshow(train);
colormap('gray')
         
% Remove some noise
train = double(train);
img_gauss = imfilter(train,fspecial('gaussian',10,2));
figure
imshow(mat2gray(img_gauss));
colormap('gray')
img_filtered = medfilt2(img_gauss,[31,31],'symmetric');
figure
imshow(mat2gray(img_filtered));
colormap('gray')

% Threshold
[N,edges] = histcounts(img_filtered);
N = smooth(N);
N = N(round(2/10*length(N)):round(8/10*length(N)));
edges = edges(round(2/10*length(edges)):round(8/10*length(edges)));
[pks,loc] = findpeaks(-N);
threshold = edges(loc(pks==max(pks)));
             
product = img_filtered>threshold;
product = imfill(product,'holes');
figure
subplot(1,2,1);
imshow(train/256);
colormap('gray')
title('Original Image(Void)')
subplot(1,2,2);
imshow(product);
colormap('gray')
title('Segmentation')
figure
imshow(product);
colormap('gray')

cd 'F:\影像處理final\';