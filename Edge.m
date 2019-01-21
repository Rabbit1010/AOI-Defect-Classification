% Read image
fid = fopen('train.csv');
readheader = textscan(fid,'%s %s',2529,'Headerlines',0,'Delimiter',',');

cd 'F:\影像處理final\train_images';

[train, map] = imread(char(list(30)));
figure
imshow(train);
colormap('gray')
 
% Remove some noise
img_filtered = medfilt2(train,[20,20],'symmetric');
figure
imshow(img_filtered);
colormap('gray')
    
% Canny
product = edge( img_filtered,'Canny',[],25);
figure
imshow(product);
colormap('gray')
product = conv2(product,[1 1 1,1 1 1,1 1 1],'same');
       
figure
subplot(1,2,1);
imshow(train);
colormap('gray')
title('Original Image(Edge Defect)')
subplot(1,2,2);
imshow(product);
colormap('gray')
title('Segmentation')
figure
imshow(product);
colormap('gray')

cd 'F:\影像處理final\';